"""
rag/ingestion.py — Ingestor v3 (Redis Stack + Chunking Hierárquico)
=====================================================================

SUBSTITUI: src/rag/ingestor.py

O QUE MUDOU vs ingestor.py:
─────────────────────────────
  REMOVIDO:
    - pgvector / PGVector / langchain_postgres → tudo vai para o Redis
    - LlamaParse (API externa, paga por página) → substituído por pymupdf (local, gratuito)
    - langchain_text_splitters.RecursiveCharacterTextSplitter → chunker próprio

  ADICIONADO:
    - Chunking Hierárquico: chunks têm metadados estruturados inline
    - Prefixo de fonte no início de cada chunk (âncora anti-alucinação)
    - Embeddings computados localmente com BAAI/bge-m3 (CPU — sem CUDA)
    - Persistência no Redis via redis_client.salvar_chunk()
    - Verificação de re-ingestão por count de chunks no Redis (não por query)

  MANTIDO:
    - PDF_CONFIG com as mesmas chaves (compatível com as tools existentes)
    - diagnosticar() — retorna sources presentes no Redis
    - ingerir_se_necessario() e ingerir_tudo() — interface pública igual

COMO O CHUNKING HIERÁRQUICO FUNCIONA:
────────────────────────────────────────
  O chunking "flat" do sistema anterior produzia chunks sem contexto:
    ❌ "03/02/2026 a 07/02/2026"   ← data sem contexto = alucinação garantida

  O chunking hierárquico prefazia cada chunk com a sua fonte:
    ✓ "[CALENDÁRIO ACADÊMICO UEMA 2026 | evento_academico]\n
       EVENTO: Matrícula de veteranos | DATA: 03/02/2026 a 07/02/2026 | SEM: 2026.1"

  Resultado: o LLM vê a fonte antes do conteúdo → resposta ancorada.

  O PREFIXO NÃO CONTA para o chunk_size — é adicionado depois do split,
  por isso o texto útil por chunk mantém-se dentro do limite configurado.

ESTRUTURA NO REDIS (após ingestão):
─────────────────────────────────────
  rag:chunk:calendario-academico-2026.pdf:0000
  rag:chunk:calendario-academico-2026.pdf:0001
  ...
  rag:chunk:edital_paes_2026.pdf:0000
  ...

  Cada JSON:
  {
    "content":     "[CALENDÁRIO ACADÊMICO UEMA 2026 | ...]\nEVENTO: ...",
    "source":      "calendario-academico-2026.pdf",
    "doc_type":    "calendario",
    "chunk_index": 0,
    "embedding":   [0.023, -0.041, ...],   ← 1024 floats (BAAI/bge-m3)
    "metadata": {
      "titulo_fonte": "Calendário Acadêmico UEMA 2026",
      "pagina":       1
    }
  }

PARSE DE PDFs SEM LLAMAPARSE:
───────────────────────────────
  Usamos pymupdf (fitz) — instala com: pip install pymupdf
  É local, gratuito, rápido (~50ms por página) e sem limite de páginas.
  Para os PDFs da UEMA (tabelas de calendário e edital), o output do pymupdf
  é suficiente. Para PDFs com layouts muito complexos, considera manter
  LlamaParse apenas para esses ficheiros específicos.
"""
from __future__ import annotations

import glob
import hashlib
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
# Configuração por ficheiro (mesmas chaves do ingestor.py original)
# SOURCE_* nas tools deve bater EXACTAMENTE com estas chaves
# ─────────────────────────────────────────────────────────────────────────────

PDF_CONFIG: dict[str, dict] = {
    # ── PDFs ──────────────────────────────────────────────────────────────────
    "calendario-academico-2026.pdf": {
        "doc_type":    "calendario",
        "titulo":      "Calendário Acadêmico UEMA 2026",
        "chunk_size":  350,
        "overlap":     60,
        "label":       "CALENDÁRIO ACADÊMICO UEMA 2026",
    },
    "edital_paes_2026.pdf": {
        "doc_type":    "edital",
        "titulo":      "Edital PAES 2026 — Processo Seletivo UEMA",
        "chunk_size":  550,
        "overlap":     80,
        "label":       "EDITAL PAES 2026",
    },
    "guia_contatos_2025.pdf": {
        "doc_type":    "contatos",
        "titulo":      "Guia de Contatos UEMA 2025",
        "chunk_size":  280,
        "overlap":     30,
        "label":       "CONTATOS UEMA 2025",
    },
    # ── TXTs ──────────────────────────────────────────────────────────────────
    "contatos_saoluis.txt": {
        "doc_type":    "contatos",
        "titulo":      "Contatos São Luís — UEMA",
        "chunk_size":  280,
        "overlap":     30,
        "label":       "CONTATOS SÃO LUÍS",
    },
    "regras_ru.txt": {
        "doc_type":    "geral",
        "titulo":      "Regras do Restaurante Universitário",
        "chunk_size":  350,
        "overlap":     50,
        "label":       "RESTAURANTE UNIVERSITÁRIO",
    },
}


# ─────────────────────────────────────────────────────────────────────────────
# Tipos internos
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class ChunkBruto:
    """Chunk de texto antes de gerar o embedding."""
    texto_puro:    str          # Texto sem prefixo (para o splitter)
    texto_final:   str          # Texto com prefixo hierárquico (vai para o Redis)
    source:        str
    doc_type:      str
    chunk_index:   int
    metadata:      dict = field(default_factory=dict)


# ─────────────────────────────────────────────────────────────────────────────
# Ingestor principal
# ─────────────────────────────────────────────────────────────────────────────

class Ingestor:
    """
    Substitui o Ingestor do ingestor.py original.
    Interface pública idêntica para compatibilidade com main.py.
    """

    def __init__(self):
        # Importa o modelo de embedding aqui (singleton via lru_cache)
        # Evita re-carregar o BAAI/bge-m3 em cada instância
        from src.rag.embeddings import get_embeddings
        self._embeddings = get_embeddings()

    # ── API pública (igual ao ingestor.py original) ───────────────────────────

    def ingerir_se_necessario(self) -> None:
        """
        Verifica ficheiro a ficheiro quais já estão no Redis.
        Só processa os que ainda não existem — os já ingeridos são ignorados.

        LÓGICA:
          Para cada ficheiro no PDF_CONFIG, verifica se existe pelo menos
          1 chunk com esse source no Redis. Se não existe → ingere.
          Se existe → pula (já está feito).

        VANTAGEM vs verificação binária anterior:
          Anterior: "tem algum chunk?" → se sim, ignora TUDO (incluindo novos ficheiros)
          Actual:   verifica cada ficheiro individualmente → só processa os novos

        EXEMPLO:
          Redis tem: calendario, edital
          Pasta tem: calendario, edital, guia_contatos (novo)
          Resultado: só guia_contatos é processado → economia de tokens de embedding
        """
        data_dir   = settings.DATA_DIR
        ficheiros  = self._listar_ficheiros(data_dir)

        if not ficheiros:
            logger.warning("⚠️  Nenhum ficheiro em %s", data_dir)
            return

        sources_existentes = _sources_no_redis()
        pendentes = []

        for caminho in ficheiros:
            nome = os.path.basename(caminho)
            if nome not in PDF_CONFIG:
                # Não está na config → ignora silenciosamente
                continue
            if nome in sources_existentes:
                logger.info("💾 '%s' já está no Redis. Ignorado.", nome)
            else:
                pendentes.append(caminho)

        if not pendentes:
            logger.info("✅ Todos os ficheiros já estão no Redis. Nada a fazer.")
            return

        logger.info(
            "📭 %d ficheiro(s) novo(s) para ingerir: %s",
            len(pendentes),
            [os.path.basename(p) for p in pendentes],
        )

        total_chunks = 0
        for caminho in pendentes:
            total_chunks += self._ingerir_ficheiro(caminho)

        logger.info("✅ Ingestão parcial concluída: %d chunks novos.", total_chunks)
        self.diagnosticar()

    def ingerir_tudo(self) -> None:
        """Força re-ingestão de todos os ficheiros, mesmo se o Redis não estiver vazio."""
        data_dir = settings.DATA_DIR
        logger.info("🕵️  Ingestão em: %s", data_dir)

        ficheiros = self._listar_ficheiros(data_dir)
        if not ficheiros:
            logger.warning("⚠️  Nenhum ficheiro em %s", data_dir)
            return

        logger.info("📁 Ficheiros: %s", [os.path.basename(f) for f in ficheiros])

        total_chunks = 0
        for ficheiro in ficheiros:
            n = self._ingerir_ficheiro(ficheiro)
            total_chunks += n

        logger.info("✅ Ingestão concluída: %d chunks guardados no Redis.", total_chunks)
        self.diagnosticar()

    def diagnosticar(self) -> set[str]:
        """Retorna e loga os sources presentes no Redis."""
        sources = _sources_no_redis()
        print("=" * 60)
        print("🔍 DIAGNÓSTICO — Redis Stack")
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
            # 1. Extrai texto
            if eh_txt:
                texto_raw = _ler_txt(caminho)
            else:
                texto_raw = _parsear_pdf(caminho)

            if not texto_raw.strip():
                logger.warning("⚠️  '%s' está vazio após parsing.", nome)
                return 0

            # 2. Limpa texto
            texto_limpo = _limpar_texto(texto_raw)

            # 3. Cria chunks com metadados hierárquicos
            chunks = list(_criar_chunks(texto_limpo, nome, config))

            if not chunks:
                logger.warning("⚠️  Nenhum chunk gerado para '%s'.", nome)
                return 0

            # 4. Gera embeddings em batch (mais eficiente que um a um)
            textos_para_embed = [c.texto_puro for c in chunks]
            embeddings = self._embeddings.embed_documents(textos_para_embed)

            # 5. Guarda no Redis
            for chunk, embedding in zip(chunks, embeddings):
                chunk_id = _gerar_chunk_id(nome, chunk.chunk_index)
                salvar_chunk(
                    chunk_id=chunk_id,
                    content=chunk.texto_final,   # Com prefixo hierárquico
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
# Chunking Hierárquico
# ─────────────────────────────────────────────────────────────────────────────

def _criar_chunks(
    texto: str,
    nome_ficheiro: str,
    config: dict,
) -> Iterator[ChunkBruto]:
    """
    Divide o texto em chunks e adiciona prefixo hierárquico a cada um.

    ESTRUTURA DO CHUNK FINAL (o que vai para o Redis e depois para o prompt):
    ──────────────────────────────────────────────────────────────────────────
      [CALENDÁRIO ACADÊMICO UEMA 2026 | calendario]
      EVENTO: Matrícula de veteranos | DATA: 03/02/2026 a 07/02/2026 | SEM: 2026.1

    POR QUÊ O PREFIXO É ANTI-ALUCINAÇÃO:
      O Gemini vê explicitamente de onde vem a informação ANTES de a ler.
      Estudos de grounding mostram que o LLM "ancora" a resposta na fonte
      declarada → menos probabilidade de misturar com conhecimento interno.

    OVERLAP:
      O overlap entre chunks garante que frases no limite não são cortadas
      a meio. Para tabelas (edital, calendário), usamos overlap maior.
    """
    chunk_size = config["chunk_size"]
    overlap    = config["overlap"]
    label      = config["label"]
    doc_type   = config["doc_type"]
    prefixo    = f"[{label} | {doc_type}]\n"

    # Divide em parágrafos primeiro (preserva estrutura de tabelas)
    paragrafos = _dividir_em_paragrafos(texto)

    buffer       = ""
    chunk_index  = 0

    for paragrafo in paragrafos:
        paragrafo = paragrafo.strip()
        if not paragrafo:
            continue

        # Se o parágrafo sozinho excede chunk_size, divide-o
        if len(paragrafo) > chunk_size * 1.5:
            # Primeiro emite o buffer acumulado
            if buffer.strip():
                yield _fazer_chunk(buffer, prefixo, nome_ficheiro, doc_type, chunk_index)
                chunk_index += 1
                buffer = buffer[-overlap:] if overlap else ""

            # Divide o parágrafo longo em sentenças
            for parte in _dividir_em_sentencas(paragrafo, chunk_size, overlap):
                yield _fazer_chunk(parte, prefixo, nome_ficheiro, doc_type, chunk_index)
                chunk_index += 1
            continue

        # Verifica se o buffer + parágrafo excedem chunk_size
        candidato = buffer + ("\n" if buffer else "") + paragrafo
        if len(candidato) <= chunk_size:
            buffer = candidato
        else:
            # Emite o buffer actual e começa novo com overlap
            if buffer.strip():
                yield _fazer_chunk(buffer, prefixo, nome_ficheiro, doc_type, chunk_index)
                chunk_index += 1
                # Overlap: mantém os últimos N chars do buffer
                buffer = buffer[-overlap:] + "\n" + paragrafo if overlap else paragrafo
            else:
                buffer = paragrafo

    # Emite o buffer final
    if buffer.strip():
        yield _fazer_chunk(buffer, prefixo, nome_ficheiro, doc_type, chunk_index)


def _fazer_chunk(
    texto_puro: str,
    prefixo: str,
    source: str,
    doc_type: str,
    chunk_index: int,
) -> ChunkBruto:
    """Cria um ChunkBruto com o texto final (prefixo + conteúdo)."""
    return ChunkBruto(
        texto_puro=texto_puro.strip(),
        texto_final=prefixo + texto_puro.strip(),
        source=source,
        doc_type=doc_type,
        chunk_index=chunk_index,
        metadata={"titulo_fonte": prefixo.strip("[]").split("|")[0].strip()},
    )


def _dividir_em_paragrafos(texto: str) -> list[str]:
    """Divide por linhas em branco duplas ou por newlines simples."""
    # Tenta divisão por parágrafo real primeiro
    blocos = re.split(r"\n{2,}", texto)
    if len(blocos) > 3:
        return blocos
    # Fallback: divide por newline simples (PDFs com linha por linha)
    return texto.split("\n")


def _dividir_em_sentencas(texto: str, chunk_size: int, overlap: int) -> list[str]:
    """Divide texto longo em partes de chunk_size com overlap."""
    partes = []
    inicio = 0
    while inicio < len(texto):
        fim = inicio + chunk_size
        if fim >= len(texto):
            partes.append(texto[inicio:])
            break
        # Tenta cortar num espaço para não partir palavras
        espaco = texto.rfind(" ", inicio, fim)
        if espaco > inicio:
            fim = espaco
        partes.append(texto[inicio:fim].strip())
        inicio = fim - overlap if overlap else fim
    return [p for p in partes if p.strip()]


# ─────────────────────────────────────────────────────────────────────────────
# Parsers de ficheiro
# ─────────────────────────────────────────────────────────────────────────────

def _parsear_pdf(caminho: str) -> str:
    """
    Extrai texto de PDF usando pymupdf (fitz).
    Local, gratuito, sem API externa.

    ALTERNATIVA: se o PDF for muito complexo (colunas, tabelas aninhadas),
    considera usar camelot-py ou tabula-py especificamente para as tabelas
    de vagas do edital.

    INSTALA: pip install pymupdf
    """
    try:
        import fitz  # pymupdf
        doc = fitz.open(caminho)
        paginas = []
        for i, pagina in enumerate(doc):
            texto = pagina.get_text("text")  # "text" é o mais limpo para tabelas simples
            if texto.strip():
                paginas.append(texto)
        doc.close()
        return "\n\n".join(paginas)
    except ImportError:
        logger.error("❌ pymupdf não instalado. Executa: pip install pymupdf")
        raise
    except Exception as e:
        logger.exception("❌ Falha ao parsear PDF '%s': %s", caminho, e)
        return ""


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
# Limpeza de texto
# ─────────────────────────────────────────────────────────────────────────────

def _limpar_texto(texto: str) -> str:
    """
    Remove ruído comum em PDFs da UEMA preservando estrutura de tabelas.

    O QUE PRESERVAMOS:
      - Pipes (|) → estrutura de tabelas do calendário e edital
      - Maiúsculas → siglas (AC, BR-PPI, CECEN, PROG)
      - Números e datas → críticos para anti-alucinação

    O QUE REMOVEMOS:
      - Cabeçalhos/rodapés repetidos (nome da universidade em cada página)
      - Linhas de separação puras (---|---|---)
      - Espaços e newlines excessivos
    """
    if not texto:
        return ""

    # Remove cabeçalho/rodapé repetido
    texto = re.sub(
        r"UNIVERSIDADE ESTADUAL DO MARANHÃO|www\.uema\.br|UEMA\s*[-–]\s*Campus",
        "", texto, flags=re.IGNORECASE,
    )

    # Remove linhas de separação de tabela sem conteúdo (---|---|---)
    texto = re.sub(r"^[-|=\s]+$", "", texto, flags=re.MULTILINE)

    # Remove linhas com apenas números e pipes (linhas de numeração de tabela)
    texto = re.sub(r"^\s*\|?\s*\d+\s*\|?\s*$", "", texto, flags=re.MULTILINE)

    # Normaliza múltiplas linhas em branco
    texto = re.sub(r"\n{3,}", "\n\n", texto)

    # Remove espaços no início/fim de cada linha mas preserva indentação
    texto = "\n".join(linha.rstrip() for linha in texto.splitlines())

    return texto.strip()


# ─────────────────────────────────────────────────────────────────────────────
# Utilitários
# ─────────────────────────────────────────────────────────────────────────────

def _gerar_chunk_id(source: str, index: int) -> str:
    """
    Gera ID determinístico para o chunk.
    Permite re-ingestão sem criar duplicados (mesma chave = overwrite).
    """
    base = f"{source}:{index}"
    return hashlib.md5(base.encode()).hexdigest()[:16]


def _sources_no_redis() -> set[str]:
    """
    Retorna o conjunto de sources únicos no Redis.

    ESTRATÉGIA (duas abordagens por ordem de preferência):

    1. FT.SEARCH com LIMIT 0 0 + GROUPBY (O(1) — usa o índice já criado)
       Muito mais rápido que SCAN para colecções grandes.

    2. Fallback: SCAN + JSON.get (caso o índice ainda não exista)
       Seguro para a primeira execução antes de inicializar_indices() completar.

    PORQUÊ IMPORTA:
      Com 500 chunks, SCAN faz ~500 round-trips ao Redis.
      FT.SEARCH com GROUPBY faz 1 operação → 500× mais rápido.
      No startup, esta função é chamada 1× por ficheiro no PDF_CONFIG.
    """
    from src.infrastructure.redis_client import IDX_CHUNKS
    from redis.commands.search.aggregation import AggregateRequest
    from redis.commands.search.reducers import count as ft_count

    r = get_redis()

    # ── Tentativa 1: FT.AGGREGATE via índice (rápido) ────────────────────────
    try:
        req = AggregateRequest("*").group_by("@source", ft_count().alias("n"))
        resultado = r.ft(IDX_CHUNKS).aggregate(req)
        sources = set()
        for row in resultado.rows:
            # row é uma lista plana: [campo, valor, campo, valor, ...]
            it = iter(row)
            row_dict = {k: v for k, v in zip(it, it)}
            fonte = row_dict.get(b"source") or row_dict.get("source")
            if fonte:
                sources.add(fonte.decode() if isinstance(fonte, bytes) else str(fonte))
        return sources
    except Exception:
        pass  # índice ainda não existe → usa fallback

    # ── Fallback: SCAN + JSON.get ─────────────────────────────────────────────
    sources: set[str] = set()
    cursor = 0
    while True:
        cursor, keys = r.scan(cursor, match=f"{PREFIX_CHUNKS}*", count=200)
        for key in keys:
            try:
                doc = r.json().get(key, "$.source")
                if doc:
                    fonte = doc[0] if isinstance(doc, list) else doc
                    if fonte:
                        sources.add(str(fonte))
            except Exception:
                pass
        if cursor == 0:
            break

    return sources


# ─────────────────────────────────────────────────────────────────────────────
# Alias de compatibilidade (para código que importa de rag/ingestor.py)
# ─────────────────────────────────────────────────────────────────────────────
# Se ainda tens imports como `from src.rag.ingestor import Ingestor, PDF_CONFIG`
# noutros ficheiros, podes criar um shim em ingestor.py:
#
#   from src.rag.ingestion import Ingestor, PDF_CONFIG  # noqa: F401
#
# Ou simplesmente actualiza os imports em main.py e debug_chainlit.py.