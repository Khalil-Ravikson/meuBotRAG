"""
rag/ingestor.py — Ingestão de PDFs e TXTs no banco vetorial
============================================================
Extraído de rag_service.py (método ingerir_base_conhecimento).

Responsabilidades:
  - Varrer DATA_DIR em busca de PDFs e TXTs
  - Parsear PDFs com LlamaParse (instrução específica por arquivo)
  - Ler TXTs diretamente (sem LlamaParse)
  - Fazer chunking com RecursiveCharacterTextSplitter
  - Salvar chunks no pgvector com metadado 'source' correto
  - Verificar se o banco já está populado (evita re-ingestão)

NOTA SOBRE OS NOMES DE ARQUIVO:
  O metadado 'source' salvo no banco DEVE bater EXATAMENTE com SOURCE_*
  nas tools. Use diagnose_banco() após a ingestão para confirmar.

CORREÇÃO IMPORTANTE vs rag_service.py:
  O código original tinha uma variável 'arquivos_pdf' que incluía TXTs
  mas a variável usada na iteração era 'arquivos_pdf' (apenas PDFs).
  Aqui separamos corretamente: PDFs usam LlamaParse, TXTs são lidos direto.
"""
from __future__ import annotations
import os
import re
import glob
import logging

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from llama_parse import LlamaParse

from src.rag.vector_store import get_vector_store, diagnosticar
from src.infrastructure.settings import settings

logger = logging.getLogger(__name__)

# =============================================================================
# Configuração por arquivo (fonte única da verdade para parsing e chunking)
# =============================================================================
# ATENÇÃO: as chaves aqui são os NOMES EXATOS dos arquivos em DATA_DIR.
# O metadado 'source' salvo no banco será exatamente esse nome.
# Ele deve bater com SOURCE_* nas tools.

PDF_CONFIG: dict[str, dict] = {
    # ── PDFs (processados via LlamaParse) ────────────────────────────────────
    "calendario-academico-2026.pdf": {
        "chunk_size":   400,
        "chunk_overlap": 50,
        "parsing_instruction": (
            "Este PDF é o Calendário Acadêmico da UEMA 2026. "
            "Para CADA linha de evento na tabela, formate assim:\n"
            "EVENTO: [nome do evento] | DATA: [data ou período] | SEM: [semestre]\n"
            "Exemplo: EVENTO: Matrícula de veteranos | DATA: 03/02/2026 a 07/02/2026 | SEM: 2026.1\n"
            "Mantenha TODOS os eventos e datas."
        ),
    },
    "edital_paes_2026.pdf": {
        "chunk_size":   600,
        "chunk_overlap": 80,
        "parsing_instruction": (
            "Este PDF é o Edital do Processo Seletivo PAES 2026 da UEMA. "
            "Para tabelas de vagas, preserve:\n"
            "CURSO: [nome] | TURNO: [turno] | AC: [nº] | PcD: [nº] | TOTAL: [nº]\n"
            "Para categorias de cotas:\n"
            "CATEGORIA: [sigla] | NOME: [nome completo] | PÚBLICO: [descrição]\n"
            "Preserve todos os números de vagas e numeração dos itens."
        ),
    },
    "guia_contatos_2025.pdf": {
        "chunk_size":   300,
        "chunk_overlap": 30,
        "parsing_instruction": (
            "Este PDF é o Guia de Contatos da UEMA 2025. "
            "Para cada linha de contato, formate:\n"
            "CARGO: [cargo] | NOME: [nome completo] | EMAIL: [email] | TEL: [telefone]\n"
            "Exemplo: CARGO: Diretor CECEN | NOME: Regina Célia | EMAIL: cecen@uema.br | TEL: (98) 99232-4837\n"
            "Mantenha o nome do centro/unidade como cabeçalho de cada bloco."
        ),
    },

    # ── TXTs (lidos diretamente, sem LlamaParse) ──────────────────────────────
    "contatos_saoluis.txt": {
        "chunk_size":   300,
        "chunk_overlap": 30,
        "parsing_instruction": None,
    },
    "regras_ru.txt": {
        "chunk_size":   400,
        "chunk_overlap": 50,
        "parsing_instruction": None,
    },
}


class Ingestor:
    """
    Singleton de ingestão.
    Instancie uma vez e reutilize — evita re-carregar o modelo de embedding.
    """

    def __init__(self):
        self._vs = get_vector_store()

    # =========================================================================
    # API pública
    # =========================================================================

    def ingerir_se_necessario(self) -> None:
        """
        Verifica se o banco já está populado.
        Se não estiver, ingere todos os arquivos de DATA_DIR.
        """
        if self._banco_populado():
            logger.info("💾 Banco vetorial já populado. Pulando ingestão.")
            return
        self.ingerir_tudo()

    def ingerir_tudo(self) -> None:
        """
        Força a ingestão de todos os arquivos, mesmo se o banco não estiver vazio.
        Use ao atualizar PDFs.
        """
        data_dir = settings.DATA_DIR
        logger.info("🕵️  Iniciando ingestão em: %s", data_dir)

        arquivos = self._listar_arquivos(data_dir)
        if not arquivos:
            logger.warning("⚠️  Nenhum arquivo encontrado em %s", data_dir)
            return

        logger.info("📁 Arquivos encontrados: %s", [os.path.basename(a) for a in arquivos])

        for arquivo in arquivos:
            self._ingerir_arquivo(arquivo)

        logger.info("✅ Ingestão concluída.")
        self.diagnosticar()

    def diagnosticar(self) -> set[str]:
        """Retorna e loga os sources presentes no banco."""
        sources = diagnosticar()
        print("=" * 60)
        print("🔍 DIAGNÓSTICO DO BANCO VETORIAL")
        print(f"   Sources presentes: {sources}")
        print(f"   Esperados (PDF_CONFIG): {list(PDF_CONFIG.keys())}")
        faltam = set(PDF_CONFIG.keys()) - sources
        if faltam:
            print(f"   ❌ NÃO INGERIDOS: {faltam}")
        else:
            print("   ✅ Todos os arquivos estão no banco.")
        print("=" * 60)
        return sources

    # =========================================================================
    # Internos
    # =========================================================================

    def _banco_populado(self) -> bool:
        try:
            docs = self._vs.similarity_search("UEMA 2026", k=1)
            if docs:
                return True
        except Exception as e:
            logger.warning("⚠️  similarity_search falhou: %s", e)
        try:
            if self._vs._collection.count() > 0:
                return True
        except Exception:
            pass
        return False

    def _listar_arquivos(self, data_dir: str) -> list[str]:
        """Retorna PDFs e TXTs da pasta, ordenados."""
        pdfs = glob.glob(os.path.join(data_dir, "*.[pP][dD][fF]"))
        txts = glob.glob(os.path.join(data_dir, "*.[tT][xX][tT]"))
        return sorted(pdfs + txts)

    def _ingerir_arquivo(self, arquivo: str) -> None:
        nome   = os.path.basename(arquivo)
        config = PDF_CONFIG.get(nome)

        if not config:
            logger.warning("⚠️  '%s' não está no PDF_CONFIG. Pulando.", nome)
            logger.warning("   Esperados: %s", list(PDF_CONFIG.keys()))
            return

        logger.info("📦 Processando '%s'...", nome)
        eh_txt = nome.lower().endswith(".txt")

        try:
            documentos = (
                self._ler_txt(arquivo, nome)
                if eh_txt
                else self._parsear_pdf(arquivo, nome, config["parsing_instruction"])
            )

            if not documentos:
                logger.warning("⚠️  Nenhum conteúdo extraído de '%s'.", nome)
                return

            splitter = RecursiveCharacterTextSplitter(
                chunk_size=config["chunk_size"],
                chunk_overlap=config["chunk_overlap"],
                separators=["\n\n", "\n", " ", ""],
            )
            chunks = splitter.split_documents(documentos)
            self._vs.add_documents(chunks)
            logger.info("✅ '%s': %d chunks salvos.", nome, len(chunks))

        except Exception as e:
            logger.exception("❌ Erro ao ingerir '%s': %s", nome, e)

    def _ler_txt(self, arquivo: str, nome: str) -> list[Document]:
        """Lê TXT diretamente, sem LlamaParse."""
        with open(arquivo, "r", encoding="utf-8") as f:
            texto = _limpar_texto(f.read())
        if not texto:
            return []
        return [Document(page_content=texto, metadata={"source": nome})]

    def _parsear_pdf(
        self, arquivo: str, nome: str, instrucao: str | None
    ) -> list[Document]:
        """Usa LlamaParse para parsear o PDF com instrução específica."""
        parser = LlamaParse(
            api_key=settings.LLAMA_CLOUD_API_KEY,
            result_type="markdown",
            language="pt",
            verbose=False,
            parsing_instruction=instrucao or "",
        )
        llama_docs = parser.load_data(arquivo)
        documentos: list[Document] = []

        for llama_doc in llama_docs:
            texto = _limpar_texto(llama_doc.text)
            if not texto:
                continue
            documentos.append(Document(
                page_content=texto,
                metadata={
                    "source": nome,
                    # Preserva metadados escalares do LlamaParse (page_number, etc.)
                    **{
                        k: v
                        for k, v in (llama_doc.metadata or {}).items()
                        if isinstance(v, (str, int, float, bool))
                    },
                },
            ))

        return documentos


# =============================================================================
# Utilitários
# =============================================================================

def _limpar_texto(text: str) -> str:
    """Remove ruído visual comum em PDFs da UEMA."""
    if not text:
        return ""
    # Remove linhas de tabela com só pipes e números
    text = re.sub(r"^\|[\s\d\|\-:]+\|$", "", text, flags=re.MULTILINE)
    # Remove cabeçalhos repetitivos
    text = re.sub(
        r"UNIVERSIDADE ESTADUAL DO MARANHÃO|www\.uema\.br|UEMA\s*[-–]\s*Campus",
        "", text, flags=re.IGNORECASE,
    )
    # Remove linhas em branco excessivas
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()