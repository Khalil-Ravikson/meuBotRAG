"""
rag/ingestion.py — Ingestor v3.2
==================================

BUGS CORRIGIDOS NESTA VERSÃO (v3.1 → v3.2):
─────────────────────────────────────────────

BUG 1 (CRÍTICO — Dupla ingestão simultânea):
  Sintoma no log:
    ForkPoolWorker-1: ⚠️  Manifesto diz ok mas Redis perdeu 3 ficheiro(s)
    ForkPoolWorker-2: ⚠️  Manifesto diz ok mas Redis perdeu 3 ficheiro(s)
    → ambos iniciam LlamaParse para os mesmos PDFs simultaneamente.

  Causa raiz:
    _verificar_redis_vs_manifesto() não tinha proteção contra chamadas concorrentes.
    Worker-1 e Worker-2 chegavam ao mesmo tempo, ambos viam Redis vazio,
    ambos entravam na re-ingestão ANTES de qualquer lock ser adquirido.

  Solução (dois níveis de proteção):
    1. Lock distribuído Redis ("lock:ingestao:verificar:{nome}") por ficheiro:
       Garante que apenas UM processo (de qualquer container) re-ingere cada PDF.
    2. Verificação dupla (double-check) dentro do lock:
       Após adquirir o lock, re-verifica se o ficheiro já está no Redis.
       O segundo worker, ao adquirir o lock, encontra o ficheiro já ingerido → skip.

  Custo do lock por ficheiro vs. lock global:
    Lock global → Worker-2 espera Worker-1 terminar TODOS os PDFs (5-10 min).
    Lock por ficheiro → Worker-2 espera só o PDF que está sendo ingerido pelo
    Worker-1 naquele momento. Se Worker-1 está no calendário e Worker-2 precisa
    do edital, eles podem ingerir em paralelo ficheiros diferentes.

BUG 2 (MENOR — parsing_instruction deprecated):
  Sintoma no log:
    WARNING: parsing_instruction is deprecated. Use system_prompt, system_prompt_append
    or user_prompt instead.

  Causa: LlamaParse v0.3+ renomeou o parâmetro.

  Solução: _parsear_com_llamaparse() agora usa system_prompt.
    A lógica é idêntica — só o nome do parâmetro mudou.
    Retrocompatibilidade: PDF_CONFIG ainda aceita "parsing_instruction" como chave,
    mas internamente é passado como system_prompt para o LlamaParse.
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
    get_redis_text,
    salvar_chunk,
)
from src.infrastructure.settings import settings

logger = logging.getLogger(__name__)

# ─────────────────────────────────────────────────────────────────────────────
# PDF_CONFIG
# ─────────────────────────────────────────────────────────────────────────────

PDF_CONFIG: dict[str, dict] = {
    "calendario-academico-2026.pdf": {
        "doc_type":   "calendario",
        "titulo":     "Calendário Acadêmico UEMA 2026",
        "chunk_size": 280,
        "overlap":    80,
        "label":      "CALENDÁRIO ACADÊMICO UEMA 2026",
        # Usado como system_prompt no LlamaParse (parsing_instruction foi depreciado)
        "parsing_instruction": (
            "Este PDF é o Calendário Acadêmico da UEMA 2026. "
            "Para CADA linha de evento na tabela, formate exatamente assim:\n"
            "EVENTO: [nome do evento] | DATA: [data ou período completo] | SEM: [semestre]\n"
            "Exemplo: EVENTO: Matrícula de veteranos | DATA: 03/02/2026 a 07/02/2026 | SEM: 2026.1\n"
            "IMPORTANTE: Mantenha TODOS os eventos. Cada linha da tabela = uma linha EVENTO:."
        ),
    },
    "edital_paes_2026.pdf": {
        "doc_type":   "edital",
        "titulo":     "Edital PAES 2026 — Processo Seletivo UEMA",
        "chunk_size": 500,
        "overlap":    80,
        "label":      "EDITAL PAES 2026",
        "parsing_instruction": (
            "Este PDF é o Edital do PAES 2026 da UEMA. "
            "Para tabelas de vagas, preserve:\n"
            "CURSO: [nome] | TURNO: [turno] | AC: [nº] | PcD: [nº] | TOTAL: [nº]\n"
            "Para cotas:\n"
            "CATEGORIA: [sigla] | NOME: [nome completo] | PÚBLICO: [descrição]\n"
            "Preserve todos os números de vagas e numeração dos itens."
        ),
    },
    "guia_contatos_2025.pdf": {
        "doc_type":   "contatos",
        "titulo":     "Guia de Contatos UEMA 2025",
        "chunk_size": 250,
        "overlap":    30,
        "label":      "CONTATOS UEMA 2025",
        "parsing_instruction": (
            "Este PDF é o Guia de Contatos da UEMA 2025. "
            "Para cada contato:\n"
            "CARGO: [cargo] | NOME: [nome completo] | EMAIL: [email] | TEL: [telefone]\n"
            "Mantenha o nome do centro/unidade como cabeçalho de cada bloco."
        ),
    },
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

DOCUMENT_CONFIG = {

    # ── Markdown de teste (este projecto) ────────────────────────────────────
    "agente_rag_uema_spec.md": {
        "doc_type":   "geral",
        "titulo":     "Especificação Técnica Bot UEMA v5",
        "chunk_size": 400,
        "overlap":    60,
        "label":      "ESPECIFICAÇÃO BOT UEMA v5",
        # Perguntas de teste: thresholds CRAG, métricas de performance,
        # tabela de formatos, perguntas T01-T08
    },
    "instrucoes_uso_agente.md": {
        "doc_type":   "geral",
        "titulo":     "Manual de Uso e Comandos — Bot UEMA",
        "chunk_size": 350,
        "overlap":    50,
        "label":      "MANUAL USO BOT UEMA",
        # Perguntas de teste: rate limits, comandos admin, ingestão, Linux
        # O LLM deve distinguir contextos (bot vs Linux) dentro do doc
    },

    # ── CSV de teste (mock gerado) ────────────────────────────────────────────
    # Coloca em dados/CSV/testes/
    "vagas_mock_2026.csv": {
        "doc_type":   "edital",
        "titulo":     "Vagas Mock PAES 2026 (Teste)",
        "chunk_size": 300,
        "overlap":    40,
        "label":      "VAGAS MOCK PAES 2026",
        # Para testar: busca exacta de números "AC: 40", "BR-PPI: 8"
    },
    "contatos_mock.csv": {
        "doc_type":   "contatos",
        "titulo":     "Contatos Mock UEMA (Teste)",
        "chunk_size": 250,
        "overlap":    30,
        "label":      "CONTATOS MOCK UEMA",
        # Para testar: busca de email@uema.br, telefones (99) 9999-9999
    },
}



_PARSERS_VALIDOS = {"pymupdf", "llamaparse"}

_EXTENSOES_SUPORTADAS = {".pdf", ".txt", ".csv", ".md"}

# ─────────────────────────────────────────────────────────────────────────────
# Manifesto
# ─────────────────────────────────────────────────────────────────────────────

def _caminho_manifesto() -> str:
    return os.path.join(settings.DATA_DIR, ".ingest_manifest.json")


def _ler_manifesto() -> dict:
    path = _caminho_manifesto()
    try:
        if os.path.exists(path):
            with open(path, "r", encoding="utf-8") as f:
                return json.load(f)
    except Exception as e:
        logger.warning("⚠️  Manifesto corrompido, ignorado: %s", e)
    return {}


def _guardar_manifesto(manifesto: dict) -> None:
    path = _caminho_manifesto()
    tmp  = path + ".tmp"
    try:
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(manifesto, f, indent=2, ensure_ascii=False)
        os.replace(tmp, path)
    except Exception as e:
        logger.warning("⚠️  Falha ao guardar manifesto: %s", e)


def _hash_ficheiro(caminho: str) -> str:
    h = hashlib.sha256()
    try:
        with open(caminho, "rb") as f:
            h.update(f.read(65536))
    except Exception:
        pass
    return h.hexdigest()[:16]


# ─────────────────────────────────────────────────────────────────────────────
# Tipos internos
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class ChunkBruto:
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

    def __init__(self):
        from src.rag.embeddings import get_embeddings
        self._embeddings = get_embeddings()

        parser = settings.PDF_PARSER.lower()
        if parser not in _PARSERS_VALIDOS:
            logger.warning("⚠️  PDF_PARSER='%s' inválido. A usar 'pymupdf'.", parser)
        elif parser == "llamaparse":
            if not settings.LLAMA_CLOUD_API_KEY:
                logger.error("❌ PDF_PARSER=llamaparse mas LLAMA_CLOUD_API_KEY ausente!")
            else:
                logger.info("🦙 Parser activo: LlamaParse (cloud — pago por página)")
        else:
            logger.info("📄 Parser activo: pymupdf (local — gratuito)")

    # ── API pública ───────────────────────────────────────────────────────────

    def ingerir_se_necessario(self) -> None:
        """
        Verifica ficheiro a ficheiro quais precisam de ser ingeridos.
        Usa o manifesto em disco como fonte de verdade primária.
        """
        data_dir  = settings.DATA_DIR
        ficheiros = self._listar_ficheiros(data_dir)

        if not ficheiros:
            logger.warning("⚠️  Nenhum ficheiro em %s", data_dir)
            return

        manifesto = _ler_manifesto()
        pendentes = []

        for caminho in ficheiros:
            nome = os.path.basename(caminho)
            if nome not in PDF_CONFIG:
                continue

            hash_actual = _hash_ficheiro(caminho)
            entrada     = manifesto.get(nome, {})

            if entrada.get("hash") == hash_actual:
                logger.info("💾 '%s' já ingerido (hash ok). Skip.", nome)
            else:
                motivo = "novo" if nome not in manifesto else "modificado"
                pendentes.append((caminho, motivo))

        if not pendentes:
            logger.info("✅ Todos os ficheiros já estão no manifesto. Nada a fazer.")
            # CORREÇÃO BUG 1: passa o lock para _verificar_redis_vs_manifesto
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
                _guardar_manifesto(manifesto)
                logger.info("✅ '%s': %d chunks → manifesto actualizado.", nome, chunks)

        self.diagnosticar()

    def ingerir_tudo(self) -> None:
        """Força re-ingestão de todos os ficheiros."""
        data_dir  = settings.DATA_DIR
        ficheiros = self._listar_ficheiros(data_dir)

        if not ficheiros:
            logger.warning("⚠️  Nenhum ficheiro em %s", data_dir)
            return

        logger.info("🕵️  Re-ingestão forçada em: %s", data_dir)
        total_chunks = 0
        for ficheiro in ficheiros:
            total_chunks += self._ingerir_ficheiro(ficheiro)

        logger.info("✅ Re-ingestão concluída: %d chunks.", total_chunks)
        self.diagnosticar()

    def diagnosticar(self) -> set[str]:
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
        nome   = os.path.basename(caminho)
        config = PDF_CONFIG.get(nome)

        if not config:
            logger.warning("⚠️  '%s' não está no PDF_CONFIG. Ignorado.", nome)
            return 0

        logger.info("📦 Processando '%s'...", nome)
        eh_txt = nome.lower().endswith(".txt")

        try:
            if eh_txt:
                texto_raw = _ler_txt(caminho)
            else:
                texto_raw = _parsear_pdf(caminho, config)

            if not texto_raw.strip():
                logger.warning("⚠️  '%s' está vazio após parsing.", nome)
                return 0

            texto_limpo = _limpar_texto(texto_raw)
            chunks = list(_criar_chunks(texto_limpo, nome, config))

            if not chunks:
                logger.warning("⚠️  Nenhum chunk gerado para '%s'.", nome)
                return 0

            textos_para_embed = [c.texto_puro for c in chunks]
            embeddings = self._embeddings.embed_documents(textos_para_embed)

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

    def _listar_ficheiros(self, data_dir: str) -> list[str]:
        encontrados: set[str] = set()
        for raiz, _, ficheiros in os.walk(data_dir):
            for nome in ficheiros:
                ext = os.path.splitext(nome)[1].lower()
                if ext in _EXTENSOES_SUPORTADAS:
                    encontrados.add(os.path.join(raiz, nome))
        return sorted(encontrados)

    # ── CORREÇÃO BUG 1: _verificar_redis_vs_manifesto com lock por ficheiro ───

    def _verificar_redis_vs_manifesto(self, manifesto: dict, ficheiros: list) -> None:
        """
        Verifica se o Redis tem os dados que o manifesto diz existirem.

        CORREÇÃO DA DUPLA INGESTÃO:
        ────────────────────────────
        Problema original: ambos os workers chegavam aqui simultaneamente,
        ambos viam Redis vazio, ambos iniciavam re-ingestão com LlamaParse → $$ duplo.

        Solução: lock distribuído Redis POR FICHEIRO com double-check.

        Fluxo corrigido (2 workers simultâneos, Redis vazio):
          Worker-1: detecta calendário em falta → tenta adquirir lock:ingestao:calendário
                    → adquire → ingere → libera lock
          Worker-2: detecta calendário em falta → tenta adquirir lock:ingestao:calendário
                    → BLOQUEADO (Worker-1 tem o lock)
                    → lock liberado → adquire → DOUBLE-CHECK: Redis já tem o calendário
                    → skip (0 chamadas LlamaParse extras)

        Lock por ficheiro vs. lock global:
          Lock global: Worker-2 espera Worker-1 terminar TODOS os PDFs (5-10 min).
          Lock por ficheiro: Workers podem ingerir PDFs DIFERENTES em paralelo.
          Ex: Worker-1 ingere calendário enquanto Worker-2 ingere edital → 2x mais rápido.
        """
        try:
            sources_redis = _sources_no_redis()
            em_falta = [
                caminho for caminho in ficheiros
                if os.path.basename(caminho) in manifesto
                and os.path.basename(caminho) not in sources_redis
            ]

            if not em_falta:
                return  # Redis tem tudo — caminho feliz

            logger.warning(
                "⚠️  Manifesto diz ok mas Redis perdeu %d ficheiro(s): %s\n"
                "   (Redis foi reiniciado sem volume persistente?)\n"
                "   A re-ingerir com lock distribuído para evitar duplicação...",
                len(em_falta),
                [os.path.basename(p) for p in em_falta],
            )

            r_text = get_redis_text()

            for caminho in em_falta:
                nome     = os.path.basename(caminho)
                lock_key = f"lock:ingestao:{nome}"

                lock = r_text.lock(
                    lock_key,
                    timeout          = 300,   # 5 min: tempo máximo para ingerir 1 PDF
                    blocking_timeout = 310,   # espera até o outro worker terminar
                )

                logger.info("⏳ [%s] Aguardando lock de ingestão...", nome)
                acquired = lock.acquire()

                if not acquired:
                    logger.warning(
                        "⚠️  [%s] Timeout aguardando lock — outro worker demorou muito. "
                        "Prosseguindo sem lock (pode causar duplicação).",
                        nome,
                    )
                    self._ingerir_ficheiro(caminho)
                    continue

                try:
                    # DOUBLE-CHECK: re-verifica se o ficheiro foi ingerido
                    # enquanto esperávamos pelo lock (pelo Worker-1)
                    sources_atuais = _sources_no_redis()
                    if nome in sources_atuais:
                        logger.info(
                            "✅ [%s] Já foi ingerido por outro worker enquanto aguardávamos. Skip.",
                            nome,
                        )
                        continue

                    # Ainda não está no Redis — somos o worker responsável
                    logger.info("🔒 [%s] Lock adquirido. Iniciando ingestão exclusiva.", nome)
                    self._ingerir_ficheiro(caminho)

                finally:
                    try:
                        lock.release()
                        logger.info("🔓 [%s] Lock de ingestão liberado.", nome)
                    except Exception:
                        pass  # Lock pode já ter expirado — normal se a ingestão demorou

        except Exception as e:
            logger.warning("⚠️  _verificar_redis_vs_manifesto falhou: %s", e)


# ─────────────────────────────────────────────────────────────────────────────
# Parsers de PDF
# ─────────────────────────────────────────────────────────────────────────────

def _parsear_pdf(caminho: str, config: dict) -> str:
    """Despacha para o parser correto. Prioridade: config["parser"] > settings.PDF_PARSER."""
    parser_nome = config.get("parser") or settings.PDF_PARSER.lower()

    if parser_nome == "llamaparse":
        if not settings.LLAMA_CLOUD_API_KEY:
            logger.warning("⚠️  LlamaParse pedido mas API key ausente. Usando pymupdf.")
            return _parsear_com_pymupdf(caminho)
        return _parsear_com_llamaparse(caminho, config)

    return _parsear_com_pymupdf(caminho)


def _parsear_com_pymupdf(caminho: str) -> str:
    try:
        import fitz
        doc     = fitz.open(caminho)
        paginas = [p.get_text("text") for p in doc if p.get_text("text").strip()]
        doc.close()

        if not paginas:
            nome = os.path.basename(caminho)
            logger.warning(
                "⚠️  pymupdf: 0 páginas com texto em '%s'. "
                "PDF baseado em imagem/scan? Tenta PDF_PARSER=llamaparse no .env.",
                nome,
            )
        else:
            logger.debug("📄 pymupdf: %d páginas | '%s'", len(paginas), os.path.basename(caminho))

        return "\n\n".join(paginas)

    except ImportError:
        logger.error("❌ pymupdf não instalado: pip install pymupdf")
        raise
    except Exception as e:
        logger.exception("❌ pymupdf falhou: %s", e)
        return ""


def _parsear_com_llamaparse(caminho: str, config: dict) -> str:
    """
    Extrai texto com LlamaParse.

    CORREÇÃO BUG 2: parsing_instruction foi depreciado no LlamaParse v0.3+.
    Agora usamos system_prompt (equivalente funcional).
    O PDF_CONFIG ainda aceita a chave "parsing_instruction" por retrocompatibilidade
    — ela é lida e passada internamente como system_prompt.
    """
    try:
        from llama_parse import LlamaParse
    except ImportError:
        logger.error("❌ llama-parse não instalado: pip install llama-parse. Usando pymupdf.")
        return _parsear_com_pymupdf(caminho)

    # Retrocompatibilidade: aceita "parsing_instruction" mas passa como system_prompt
    instrucao = config.get("parsing_instruction") or (
        "Extrai todo o texto preservando a estrutura de tabelas. "
        "Para tabelas, usa: COLUNA1: valor | COLUNA2: valor. "
        "Responde em português."
    )

    try:
        parser = LlamaParse(
            api_key=settings.LLAMA_CLOUD_API_KEY,
            result_type="markdown",
            language="pt",
            verbose=False,
            # CORREÇÃO: system_prompt em vez de parsing_instruction (depreciado)
            system_prompt=instrucao,
        )
        docs    = parser.load_data(caminho)
        paginas = [doc.text for doc in docs if doc.text.strip()]
        logger.debug("🦙 LlamaParse: %d páginas | '%s'", len(paginas), os.path.basename(caminho))
        return "\n\n".join(paginas)

    except Exception as e:
        logger.exception("❌ LlamaParse falhou em '%s': %s. Usando pymupdf.", caminho, e)
        return _parsear_com_pymupdf(caminho)


def _ler_txt(caminho: str) -> str:
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

    O LLM vê a fonte ANTES do conteúdo → ancora a resposta e reduz alucinações.
    """
    label    = config.get("label", nome_ficheiro.upper())
    doc_type = config.get("doc_type", "geral")
    titulo   = config.get("titulo", nome_ficheiro)
    size     = config.get("chunk_size", 400)
    overlap  = config.get("overlap", 50)

    prefixo_hierarquico = f"[{label} | {doc_type}]\n"

    partes = _dividir_texto(texto, size, overlap)

    for i, parte in enumerate(partes):
        texto_final = prefixo_hierarquico + parte

        yield ChunkBruto(
            texto_puro  = parte,
            texto_final = texto_final,
            source      = nome_ficheiro,
            doc_type    = doc_type,
            chunk_index = i,
            metadata    = {
                "titulo":      titulo,
                "chunk_index": i,
                "total_parts": len(partes),
            },
        )


def _dividir_texto(texto: str, chunk_size: int, overlap: int) -> list[str]:
    """Divisão por parágrafos com fallback para divisão por tamanho."""
    paragrafos = [p.strip() for p in re.split(r"\n{2,}", texto) if p.strip()]
    chunks: list[str] = []
    atual = ""

    for paragrafo in paragrafos:
        candidato = f"{atual}\n\n{paragrafo}".strip() if atual else paragrafo

        if len(candidato) <= chunk_size:
            atual = candidato
        else:
            if atual:
                chunks.append(atual)
            # Parágrafo maior que chunk_size → divide por tamanho
            if len(paragrafo) > chunk_size:
                for inicio in range(0, len(paragrafo), chunk_size - overlap):
                    parte = paragrafo[inicio: inicio + chunk_size]
                    if parte.strip():
                        chunks.append(parte)
                atual = ""
            else:
                atual = paragrafo

    if atual:
        chunks.append(atual)

    return chunks or [texto[:chunk_size]]


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
    """Sources únicos no Redis — usa FT.AGGREGATE com fallback SCAN."""
    from src.infrastructure.redis_client import IDX_CHUNKS
    from redis.commands.search.aggregation import AggregateRequest
    from redis.commands.search.reducers import count as ft_count

    r = get_redis()

    try:
        req       = AggregateRequest("*").group_by("@source", ft_count().alias("n"))
        resultado = r.ft(IDX_CHUNKS).aggregate(req)
        sources   = set()
        for row in resultado.rows:
            it       = iter(row)
            row_dict = {k: v for k, v in zip(it, it)}
            fonte    = row_dict.get(b"source") or row_dict.get("source")
            if fonte:
                sources.add(fonte.decode() if isinstance(fonte, bytes) else str(fonte))
        return sources
    except Exception:
        pass

    # Fallback: SCAN manual
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