"""
rag/ingestion.py — Ingestor v3.1 (dual-parser: pymupdf ou LlamaParse)
=======================================================================

DOIS PARSERS DISPONÍVEIS — escolhe no .env:
─────────────────────────────────────────────
  PDF_PARSER=pymupdf      ← local, gratuito, rápido (~50ms/página)
                             bom para PDFs simples e semi-estruturados
  PDF_PARSER=llamaparse   ← cloud, pago por página (~$0.003/pág)
                             melhor para tabelas complexas e layouts difíceis

  O default é pymupdf. Para activar o LlamaParse:
    1. Adiciona ao .env:
         PDF_PARSER=llamaparse
         LLAMA_CLOUD_API_KEY=llx-...
    2. Adiciona ao requirements.txt:
         llama-parse
    3. Faz docker-compose restart bot

COMO FUNCIONA A SELECÇÃO:
─────────────────────────
  No startup, _parsear_pdf() lê settings.PDF_PARSER e despacha para
  _parsear_com_pymupdf() ou _parsear_com_llamaparse().
  Os TXTs são sempre lidos directamente (sem parser externo).

  Podes também configurar por ficheiro individual no PDF_CONFIG,
  adicionando a chave "parser": "llamaparse" só nos PDFs complexos.
  Se "parser" não estiver definido no config, usa o default do settings.

O RESTO NÃO MUDOU:
──────────────────
  - Chunking Hierárquico com prefixo anti-alucinação
  - Verificação por ficheiro (só ingere os novos)
  - Embeddings BAAI/bge-m3 locais (CPU)
  - Persistência no Redis Stack
"""
from __future__ import annotations

import glob
import hashlib
import json
import logging
import os
import re
from dataclasses import dataclass, field
from typing import Iterator

from src.infrastructure.redis_client import (
    PREFIX_CHUNKS,
    get_redis,
    salvar_chunk,
)
from src.infrastructure.settings import settings

logger = logging.getLogger(__name__)

# ─────────────────────────────────────────────────────────────────────────────
# Configuração por ficheiro
# A chave "parser" é opcional — substitui o PDF_PARSER do settings por ficheiro
# ─────────────────────────────────────────────────────────────────────────────

PDF_CONFIG: dict[str, dict] = {
    # ── PDFs ──────────────────────────────────────────────────────────────────
    "calendario-academico-2026.pdf": {
        "doc_type":   "calendario",
        "titulo":     "Calendário Acadêmico UEMA 2026",
        "chunk_size": 350,
        "overlap":    60,
        "label":      "CALENDÁRIO ACADÊMICO UEMA 2026",
        # "parser": "llamaparse",  # descomenta para forçar LlamaParse neste ficheiro
    },
    "edital_paes_2026.pdf": {
        "doc_type":   "edital",
        "titulo":     "Edital PAES 2026 — Processo Seletivo UEMA",
        "chunk_size": 550,
        "overlap":    80,
        "label":      "EDITAL PAES 2026",
        # "parser": "llamaparse",  # descomenta se as tabelas de vagas ficarem mal
    },
    "guia_contatos_2025.pdf": {
        "doc_type":   "contatos",
        "titulo":     "Guia de Contatos UEMA 2025",
        "chunk_size": 280,
        "overlap":    30,
        "label":      "CONTATOS UEMA 2025",
    },
    # ── TXTs (sempre lidos directamente — parser ignorado) ────────────────────
    "contatos_saoluis.txt": {
        "doc_type":   "contatos",
        "titulo":     "Contatos São Luís — UEMA",
        "chunk_size": 280,
        "overlap":    30,
        "label":      "CONTATOS SÃO LUÍS",
    },
    "regras_ru.txt": {
        "doc_type":   "geral",
        "titulo":     "Regras do Restaurante Universitário",
        "chunk_size": 350,
        "overlap":    50,
        "label":      "RESTAURANTE UNIVERSITÁRIO",
    },
}

# Valores válidos para PDF_PARSER
_PARSERS_VALIDOS = {"pymupdf", "llamaparse"}

# ─────────────────────────────────────────────────────────────────────────────
# Manifesto de ingestão (ficheiro em disco — independente do Redis)
#
# Problema que resolve:
#   _sources_no_redis() depende do Redis ter o índice FT criado.
#   No startup, o Redis pode estar a iniciar e o índice ainda não existe.
#   Resultado: devolve set() vazio → re-ingere tudo desnecessariamente.
#
# Solução:
#   Guardamos um ficheiro JSON em DATA_DIR/.ingest_manifest.json com:
#     { "calendario-academico-2026.pdf": { "hash": "abc123", "chunks": 45 }, ... }
#   O hash é do conteúdo do ficheiro — se o PDF mudar, o hash muda e re-ingere.
#   Se o Redis perder dados mas o ficheiro existir, o manifesto protege.
#
# Ciclo de vida:
#   1. Startup → lê manifesto → compara com ficheiros em disco
#   2. Se hash igual → skip (mesmo que Redis esteja vazio temporariamente)
#   3. Após ingestão bem-sucedida → actualiza manifesto
#   4. Para forçar re-ingestão: apaga DATA_DIR/.ingest_manifest.json
# ─────────────────────────────────────────────────────────────────────────────

def _caminho_manifesto() -> str:
    return os.path.join(settings.DATA_DIR, ".ingest_manifest.json")


def _ler_manifesto() -> dict:
    """Lê o manifesto de disco. Devolve {} se não existir ou estiver corrompido."""
    path = _caminho_manifesto()
    try:
        if os.path.exists(path):
            with open(path, "r", encoding="utf-8") as f:
                return json.load(f)
    except Exception as e:
        logger.warning("⚠️  Manifesto corrompido, ignorado: %s", e)
    return {}


def _guardar_manifesto(manifesto: dict) -> None:
    """Guarda o manifesto em disco atomicamente."""
    path = _caminho_manifesto()
    tmp  = path + ".tmp"
    try:
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(manifesto, f, indent=2, ensure_ascii=False)
        os.replace(tmp, path)  # atómico — nunca fica ficheiro meio-escrito
        logger.debug("💾 Manifesto actualizado: %s", path)
    except Exception as e:
        logger.warning("⚠️  Falha ao guardar manifesto: %s", e)


def _hash_ficheiro(caminho: str) -> str:
    """SHA-256 dos primeiros 64KB do ficheiro (rápido, detecta mudanças)."""
    h = hashlib.sha256()
    try:
        with open(caminho, "rb") as f:
            h.update(f.read(65536))
    except Exception:
        pass
    return h.hexdigest()[:16]  # 16 chars são suficientes para detectar mudanças


# ─────────────────────────────────────────────────────────────────────────────
# Tipos internos
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class ChunkBruto:
    """Chunk de texto antes de gerar o embedding."""
    texto_puro:  str
    texto_final: str
    source:      str
    doc_type:    str
    chunk_index: int
    metadata:    dict = field(default_factory=dict)


# ─────────────────────────────────────────────────────────────────────────────
# Ingestor principal
# ─────────────────────────────────────────────────────────────────────────────

class Ingestor:
    """
    Ingestor com suporte a dois parsers de PDF.
    Interface pública idêntica ao ingestor.py original.
    """

    def __init__(self):
        from src.rag.embeddings import get_embeddings
        self._embeddings = get_embeddings()

        # Valida e loga o parser activo no startup
        parser = settings.PDF_PARSER.lower()
        if parser not in _PARSERS_VALIDOS:
            logger.warning(
                "⚠️  PDF_PARSER='%s' inválido. A usar 'pymupdf'. "
                "Valores válidos: %s",
                parser, _PARSERS_VALIDOS,
            )
        elif parser == "llamaparse":
            if not settings.LLAMA_CLOUD_API_KEY:
                logger.error(
                    "❌ PDF_PARSER=llamaparse mas LLAMA_CLOUD_API_KEY não está definida no .env! "
                    "A fazer fallback para pymupdf."
                )
            else:
                logger.info("🦙 Parser activo: LlamaParse (cloud — pago por página)")
        else:
            logger.info("📄 Parser activo: pymupdf (local — gratuito)")

    # ── API pública ───────────────────────────────────────────────────────────

    def ingerir_se_necessario(self) -> None:
        """
        Verifica ficheiro a ficheiro quais precisam de ser ingeridos.

        FONTE DE VERDADE: manifesto em disco (DATA_DIR/.ingest_manifest.json)
        ─────────────────────────────────────────────────────────────────────
        Usa o hash do conteúdo do ficheiro para decidir se deve re-ingerir:
          - Hash igual ao manifesto → skip (mesmo que Redis esteja a reiniciar)
          - Hash diferente           → ficheiro mudou → re-ingere
          - Não está no manifesto    → ficheiro novo → ingere

        Vantagem sobre verificar só o Redis:
          O Redis pode estar a iniciar no startup e o índice FT ainda não existe.
          O manifesto em disco está sempre disponível imediatamente.

        Para forçar re-ingestão:
          - De um ficheiro: apaga a entrada no .ingest_manifest.json
          - De tudo:        rm dados/.ingest_manifest.json
        """
        data_dir  = settings.DATA_DIR
        ficheiros = self._listar_ficheiros(data_dir)

        if not ficheiros:
            logger.warning("⚠️  Nenhum ficheiro em %s", data_dir)
            return

        manifesto  = _ler_manifesto()
        pendentes  = []   # (caminho, motivo)
        ignorados  = []

        for caminho in ficheiros:
            nome = os.path.basename(caminho)
            if nome not in PDF_CONFIG:
                continue

            hash_actual = _hash_ficheiro(caminho)
            entrada     = manifesto.get(nome, {})

            if entrada.get("hash") == hash_actual:
                ignorados.append(nome)
                logger.info("💾 '%s' já ingerido (hash ok). Skip.", nome)
            else:
                motivo = "novo" if nome not in manifesto else "modificado"
                pendentes.append((caminho, motivo))

        if not pendentes:
            logger.info("✅ Todos os ficheiros já estão no manifesto. Nada a fazer.")
            # Garante que o Redis também tem os dados (por se houve perda de dados)
            self._verificar_redis_vs_manifesto(manifesto, ficheiros)
            return

        logger.info(
            "📭 %d ficheiro(s) para ingerir: %s",
            len(pendentes),
            [(os.path.basename(p), m) for p, m in pendentes],
        )

        for caminho, _ in pendentes:
            nome   = os.path.basename(caminho)
            chunks = self._ingerir_ficheiro(caminho)
            if chunks > 0:
                manifesto[nome] = {
                    "hash":   _hash_ficheiro(caminho),
                    "chunks": chunks,
                }
                _guardar_manifesto(manifesto)  # guarda após cada ficheiro
                logger.info("✅ '%s': %d chunks → manifesto actualizado.", nome, chunks)

        self.diagnosticar()

    def _verificar_redis_vs_manifesto(self, manifesto: dict, ficheiros: list) -> None:
        """
        Verifica se o Redis tem os dados que o manifesto diz existirem.
        Se o Redis perdeu dados (ex: volume apagado), re-ingere silenciosamente.
        Chamado apenas quando o manifesto diz que tudo está ok.
        """
        try:
            sources_redis = _sources_no_redis()
            em_falta      = []

            for caminho in ficheiros:
                nome = os.path.basename(caminho)
                if nome in manifesto and nome not in sources_redis:
                    em_falta.append(caminho)

            if not em_falta:
                return

            logger.warning(
                "⚠️  Manifesto diz ok mas Redis perdeu %d ficheiro(s): %s\n"
                "   (Redis foi reiniciado sem volume persistente?)\n"
                "   A re-ingerir no Redis sem actualizar manifesto...",
                len(em_falta),
                [os.path.basename(p) for p in em_falta],
            )
            for caminho in em_falta:
                self._ingerir_ficheiro(caminho)

        except Exception as e:
            logger.debug("ℹ️  Verificação Redis vs manifesto falhou (ignorado): %s", e)

    def ingerir_tudo(self) -> None:
        """Força re-ingestão de todos os ficheiros (ignora o que já existe)."""
        data_dir  = settings.DATA_DIR
        ficheiros = self._listar_ficheiros(data_dir)

        if not ficheiros:
            logger.warning("⚠️  Nenhum ficheiro em %s", data_dir)
            return

        logger.info("🕵️  Ingestão em: %s", data_dir)
        logger.info("📁 Ficheiros: %s", [os.path.basename(f) for f in ficheiros])

        total_chunks = 0
        for ficheiro in ficheiros:
            total_chunks += self._ingerir_ficheiro(ficheiro)

        logger.info("✅ Ingestão concluída: %d chunks guardados no Redis.", total_chunks)
        self.diagnosticar()

    def diagnosticar(self) -> set[str]:
        """Retorna e loga os sources presentes no Redis."""
        sources = _sources_no_redis()
        print("=" * 60)
        print("🔍 DIAGNÓSTICO — Redis Stack")
        print(f"   Parser activo: {settings.PDF_PARSER}")
        print(f"   Sources presentes: {sources}")
        print(f"   Esperados (PDF_CONFIG): {list(PDF_CONFIG.keys())}")
        faltam = set(PDF_CONFIG.keys()) - sources
        if faltam:
            print(f"   ❌ NÃO INGERIDOS: {faltam}")
        else:
            print("   ✅ Todos os ficheiros estão no Redis.")
        print("=" * 60)
        return sources

    # ── Ingestão por ficheiro ─────────────────────────────────────────────────

    def _ingerir_ficheiro(self, caminho: str) -> int:
        """Processa um ficheiro e guarda os chunks no Redis. Retorna nº de chunks."""
        nome   = os.path.basename(caminho)
        config = PDF_CONFIG.get(nome)

        if not config:
            logger.warning("⚠️  '%s' não está no PDF_CONFIG. Ignorado.", nome)
            return 0

        logger.info("📦 Processando '%s'...", nome)
        eh_txt = nome.lower().endswith(".txt")

        try:
            # 1. Extrai texto — TXT directo, PDF via parser configurado
            if eh_txt:
                texto_raw = _ler_txt(caminho)
            else:
                texto_raw = _parsear_pdf(caminho, config)

            if not texto_raw.strip():
                logger.warning("⚠️  '%s' está vazio após parsing.", nome)
                return 0

            # 2. Limpa
            texto_limpo = _limpar_texto(texto_raw)

            # 3. Chunks com metadados hierárquicos
            chunks = list(_criar_chunks(texto_limpo, nome, config))
            if not chunks:
                logger.warning("⚠️  Nenhum chunk gerado para '%s'.", nome)
                return 0

            # 4. Embeddings em batch
            textos_para_embed = [c.texto_puro for c in chunks]
            embeddings = self._embeddings.embed_documents(textos_para_embed)

            # 5. Guarda no Redis
            for chunk, embedding in zip(chunks, embeddings):
                chunk_id = _gerar_chunk_id(nome, chunk.chunk_index)
                salvar_chunk(
                    chunk_id=chunk_id,
                    content=chunk.texto_final,
                    source=chunk.source,
                    doc_type=chunk.doc_type,
                    embedding=embedding,
                    chunk_index=chunk.chunk_index,
                    metadata=chunk.metadata,
                )

            logger.info("✅ '%s': %d chunks guardados.", nome, len(chunks))
            return len(chunks)

        except Exception as e:
            logger.exception("❌ Erro ao ingerir '%s': %s", nome, e)
            return 0

    # ── Helpers ───────────────────────────────────────────────────────────────

    def _listar_ficheiros(self, data_dir: str) -> list[str]:
        pdfs = glob.glob(os.path.join(data_dir, "*.[pP][dD][fF]"))
        txts = glob.glob(os.path.join(data_dir, "*.[tT][xX][tT]"))
        return sorted(pdfs + txts)


# ─────────────────────────────────────────────────────────────────────────────
# Parsers de PDF
# ─────────────────────────────────────────────────────────────────────────────

def _parsear_pdf(caminho: str, config: dict) -> str:
    """
    Despacha para o parser correcto.

    Ordem de prioridade:
      1. config["parser"] — parser específico para este ficheiro (se definido)
      2. settings.PDF_PARSER — parser global do .env
      3. fallback para pymupdf se o LlamaParse falhar ou não tiver API key
    """
    # Parser por ficheiro tem prioridade sobre o global
    parser_nome = config.get("parser") or settings.PDF_PARSER.lower()

    if parser_nome == "llamaparse":
        if not settings.LLAMA_CLOUD_API_KEY:
            logger.warning(
                "⚠️  LlamaParse pedido mas LLAMA_CLOUD_API_KEY ausente. "
                "A usar pymupdf como fallback."
            )
            return _parsear_com_pymupdf(caminho)
        return _parsear_com_llamaparse(caminho, config)

    return _parsear_com_pymupdf(caminho)


def _parsear_com_pymupdf(caminho: str) -> str:
    """
    Extrai texto com pymupdf (fitz).

    PRÓS:  local, gratuito, ~50ms/página, sem limite
    CONTRAS: PDFs baseados em imagem/scan devolvem texto vazio.
             Nesse caso activa PDF_PARSER=llamaparse no .env.

    INSTALA: pip install pymupdf
    """
    try:
        import fitz
        doc       = fitz.open(caminho)
        paginas   = []
        nome      = os.path.basename(caminho)
        n_total   = len(doc)

        for pagina in doc:
            texto = pagina.get_text("text")
            if texto.strip():
                paginas.append(texto)
        doc.close()
        if not paginas:
            logger.warning( "pymupdf: 0 páginas com texto em '%s' ('%d' págs)."

                            ,"PDF provavelmente baseado em imagem/scan.",

                            "Solução: adiciona PDF_PARSER=llamaparse ao .env e LLAMA_CLOUD_API_KEY=llx-...,"

                            ,nome, n_total,

                                )
        else:
            logger.debug("📄 pymupdf: %d/%d páginas | '%s'", len(paginas), n_total, nome)

        return "\n\n".join(paginas)

    except ImportError:
        logger.error("❌ pymupdf não instalado. Executa: pip install pymupdf")
        raise
    except Exception as e:
        logger.exception("❌ pymupdf falhou em '%s': %s", caminho, e)
        return ""


def _parsear_com_llamaparse(caminho: str, config: dict) -> str:
    """
    Extrai texto com LlamaParse (API cloud Llama Index).

    PRÓS:  excelente com tabelas complexas, colunas, layouts difíceis
    CONTRAS: pago (~$0.003/página), lento (round-trip cloud), precisa de API key

    ACTIVA NO .env:
      PDF_PARSER=llamaparse
      LLAMA_CLOUD_API_KEY=llx-...

    INSTALA: pip install llama-parse

    A parsing_instruction é lida do PDF_CONFIG["parsing_instruction"].
    Se não estiver definida, usa uma instrução genérica.
    """
    try:
        from llama_parse import LlamaParse
    except ImportError:
        logger.error(
            "❌ llama-parse não instalado. Executa: pip install llama-parse\n"
            "   A fazer fallback para pymupdf."
        )
        return _parsear_com_pymupdf(caminho)

    instrucao = config.get("parsing_instruction") or (
        "Extrai todo o texto preservando a estrutura de tabelas. "
        "Para tabelas, usa o formato: COLUNA1: valor | COLUNA2: valor. "
        "Responde em português."
    )

    try:
        parser = LlamaParse(
            api_key=settings.LLAMA_CLOUD_API_KEY,
            result_type="markdown",
            language="pt",
            verbose=False,
            parsing_instruction=instrucao,
        )
        docs    = parser.load_data(caminho)
        paginas = [doc.text for doc in docs if doc.text.strip()]
        logger.debug(
            "🦙 LlamaParse: %d páginas extraídas de '%s'",
            len(paginas), os.path.basename(caminho),
        )
        return "\n\n".join(paginas)
    except Exception as e:
        logger.exception(
            "❌ LlamaParse falhou em '%s': %s\n   A fazer fallback para pymupdf.",
            caminho, e,
        )
        return _parsear_com_pymupdf(caminho)


def _ler_txt(caminho: str) -> str:
    """Lê ficheiro .txt com detecção de encoding."""
    for encoding in ("utf-8", "latin-1", "cp1252"):
        try:
            with open(caminho, "r", encoding=encoding) as f:
                return f.read()
        except UnicodeDecodeError:
            continue
    logger.warning("⚠️  Não foi possível ler '%s' com encodings comuns.", caminho)
    return ""


# ─────────────────────────────────────────────────────────────────────────────
# Chunking Hierárquico
# ─────────────────────────────────────────────────────────────────────────────

def _criar_chunks(texto: str, nome_ficheiro: str, config: dict) -> Iterator[ChunkBruto]:
    """
    Divide o texto em chunks com prefixo hierárquico anti-alucinação.

    Cada chunk começa com:
      [EDITAL PAES 2026 | edital]
      CURSO: Engenharia Civil | AC: 40 | PcD: 2 | TOTAL: 42

    O LLM vê a fonte ANTES do conteúdo → ancora a resposta.
    """
    chunk_size  = config["chunk_size"]
    overlap     = config["overlap"]
    label       = config["label"]
    doc_type    = config["doc_type"]
    prefixo     = f"[{label} | {doc_type}]\n"

    paragrafos  = _dividir_em_paragrafos(texto)
    buffer      = ""
    chunk_index = 0

    for paragrafo in paragrafos:
        paragrafo = paragrafo.strip()
        if not paragrafo:
            continue

        if len(paragrafo) > chunk_size * 1.5:
            if buffer.strip():
                yield _fazer_chunk(buffer, prefixo, nome_ficheiro, doc_type, chunk_index)
                chunk_index += 1
                buffer = buffer[-overlap:] if overlap else ""
            for parte in _dividir_em_sentencas(paragrafo, chunk_size, overlap):
                yield _fazer_chunk(parte, prefixo, nome_ficheiro, doc_type, chunk_index)
                chunk_index += 1
            continue

        candidato = buffer + ("\n" if buffer else "") + paragrafo
        if len(candidato) <= chunk_size:
            buffer = candidato
        else:
            if buffer.strip():
                yield _fazer_chunk(buffer, prefixo, nome_ficheiro, doc_type, chunk_index)
                chunk_index += 1
            buffer = buffer[-overlap:] + "\n" + paragrafo if overlap else paragrafo

    if buffer.strip():
        yield _fazer_chunk(buffer, prefixo, nome_ficheiro, doc_type, chunk_index)


def _fazer_chunk(texto_puro, prefixo, source, doc_type, chunk_index) -> ChunkBruto:
    return ChunkBruto(
        texto_puro=texto_puro.strip(),
        texto_final=prefixo + texto_puro.strip(),
        source=source,
        doc_type=doc_type,
        chunk_index=chunk_index,
        metadata={"titulo_fonte": prefixo.strip("[]").split("|")[0].strip()},
    )


def _dividir_em_paragrafos(texto: str) -> list[str]:
    blocos = re.split(r"\n{2,}", texto)
    if len(blocos) > 3:
        return blocos
    return texto.split("\n")


def _dividir_em_sentencas(texto: str, chunk_size: int, overlap: int) -> list[str]:
    partes, inicio = [], 0
    while inicio < len(texto):
        fim    = inicio + chunk_size
        if fim >= len(texto):
            partes.append(texto[inicio:])
            break
        espaco = texto.rfind(" ", inicio, fim)
        if espaco > inicio:
            fim = espaco
        partes.append(texto[inicio:fim].strip())
        inicio = fim - overlap if overlap else fim
    return [p for p in partes if p.strip()]


# ─────────────────────────────────────────────────────────────────────────────
# Limpeza de texto
# ─────────────────────────────────────────────────────────────────────────────

def _limpar_texto(texto: str) -> str:
    if not texto:
        return ""
    texto = re.sub(
        r"UNIVERSIDADE ESTADUAL DO MARANHÃO|www\.uema\.br|UEMA\s*[-–]\s*Campus",
        "", texto, flags=re.IGNORECASE,
    )
    texto = re.sub(r"^[-|=\s]+$", "", texto, flags=re.MULTILINE)
    texto = re.sub(r"^\s*\|?\s*\d+\s*\|?\s*$", "", texto, flags=re.MULTILINE)
    texto = re.sub(r"\n{3,}", "\n\n", texto)
    texto = "\n".join(linha.rstrip() for linha in texto.splitlines())
    return texto.strip()


# ─────────────────────────────────────────────────────────────────────────────
# Utilitários
# ─────────────────────────────────────────────────────────────────────────────

def _gerar_chunk_id(source: str, index: int) -> str:
    return hashlib.md5(f"{source}:{index}".encode()).hexdigest()[:16]


def _sources_no_redis() -> set[str]:
    """Sources únicos no Redis — usa FT.AGGREGATE (rápido) com fallback SCAN."""
    from src.infrastructure.redis_client import IDX_CHUNKS
    from redis.commands.search.aggregations import AggregateRequest
    from redis.commands.search.reducers import count as ft_count

    r = get_redis()

    try:
        req      = AggregateRequest("*").group_by("@source", ft_count().alias("n"))
        resultado = r.ft(IDX_CHUNKS).aggregate(req)
        sources  = set()
        for row in resultado.rows:
            it       = iter(row)
            row_dict = {k: v for k, v in zip(it, it)}
            fonte    = row_dict.get(b"source") or row_dict.get("source")
            if fonte:
                sources.add(fonte.decode() if isinstance(fonte, bytes) else str(fonte))
        return sources
    except Exception:
        pass

    sources: set[str] = set()
    cursor = 0
    while True:
        cursor, keys = r.scan(cursor, match=f"{PREFIX_CHUNKS}*", count=200)
        for key in keys:
            try:
                doc   = r.json().get(key, "$.source")
                fonte = (doc[0] if isinstance(doc, list) else doc) if doc else None
                if fonte:
                    sources.add(str(fonte))
            except Exception:
                pass
        if cursor == 0:
            break

    return sources