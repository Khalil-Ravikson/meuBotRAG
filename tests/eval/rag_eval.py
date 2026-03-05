"""
eval/rag_eval.py — Suite de Avaliação RAG do meuBotRAG
=======================================================

COMO EXECUTAR (Windows ou Linux):
──────────────────────────────────
  # 1. Instala dependências de avaliação (só uma vez):
  pip install ragas datasets pandas openpyxl

  # 2. Redis Stack deve estar rodando (Docker):
  docker compose up redis -d

  # 3. Os PDFs devem estar já ingeridos no Redis.
  #    Se não estiver, roda o ingestor primeiro:
  #    python -m src.rag.ingestion

  # 4. Executa a avaliação:
  cd meuBotRAG/
  python eval/rag_eval.py

  # Vai gerar dois relatórios em eval/resultados/:
  #   - rag_eval_resultados_YYYYMMDD_HHMMSS.csv   ← todas as perguntas
  #   - rag_eval_resumo_YYYYMMDD_HHMMSS.json      ← métricas agregadas

O QUE ESTE SCRIPT FAZ:
───────────────────────
  1. Carrega 20 perguntas de teste (DATASET_EVAL) agrupadas por domínio
  2. Para cada pergunta, executa a PIPELINE REAL do meuBotRAG:
     - Roteamento semântico (Redis KNN)
     - Transformação de query (Gemini)
     - Busca híbrida (Redis BM25 + Vetor)
     - Geração de resposta (Gemini)
  3. Coleta: pergunta, resposta gerada, contextos recuperados
  4. Calcula métricas RAGAS:
     - Faithfulness:       a resposta é suportada pelos contextos? (0–1)
     - Answer Relevancy:   a resposta responde à pergunta? (0–1)
     - Context Precision:  os contextos recuperados são relevantes? (0–1)
     - Context Recall:     os contextos cobrem a resposta esperada? (0–1)
  5. Salva relatório CSV + JSON com scores por pergunta e médias

MÉTRICAS RAGAS EXPLICADAS:
───────────────────────────
  Faithfulness (anti-alucinação):
    Verifica se cada afirmação da resposta pode ser derivada dos contextos.
    Score < 0.7 → sistema está alucinando → revisar chunking ou prompts.

  Answer Relevancy:
    Verifica se a resposta é relevante para a pergunta original.
    Score < 0.7 → resposta tangencial → revisar system prompt ou query transform.

  Context Precision:
    Proporção de contextos recuperados que são de facto relevantes.
    Score < 0.7 → retriever traz lixo → ajustar k_vector, k_text, thresholds.

  Context Recall:
    Quanto da resposta esperada está coberta pelos contextos recuperados.
    Score < 0.7 → retriever não encontra dados → revisar chunking ou embeddings.

COMO USAR PARA COMPARAR VERSÕES:
──────────────────────────────────
  Guarda o JSON da v3 (ex: resumo_v3_20260304.json).
  Faça uma mudança (ex: novo chunking, novo modelo de embeddings).
  Executa novamente. Compara os JSONs.
  Se Faithfulness subiu → menos alucinações. Se Context Recall subiu → retriever melhorou.

  Ver README_EVAL.md para guia completo de versionamento.

ADICIONAR NOVAS PERGUNTAS:
───────────────────────────
  Basta adicionar entradas ao DATASET_EVAL abaixo seguindo o padrão:
  {
    "question":           "pergunta como o aluno escreveria no WhatsApp",
    "ground_truth":       "resposta factualmente correta (trecho do PDF)",
    "domain":             "calendario|edital|contatos|geral",
    "tipo":               "data|sigla|contato|procedimento|geral",
    "dificuldade":        "facil|media|dificil",
  }
  NÃO altere questions existentes entre versões — isso quebra a comparabilidade.
  Adicione apenas ao final com novos IDs.
"""
from __future__ import annotations

import asyncio
import json
import logging
import os
import sys
import time
from dataclasses import dataclass, asdict
from datetime import datetime
from pathlib import Path

# Garante que o projeto está no path (para rodar da raiz do projeto)
sys.path.insert(0, str(Path(__file__).parent.parent))

logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    datefmt="%H:%M:%S",
)

# ─────────────────────────────────────────────────────────────────────────────
# DATASET DE AVALIAÇÃO — 20 PERGUNTAS
#
# Baseado nos documentos reais:
#   - calendario-academico-2026.pdf
#   - edital_paes_2026.pdf
#   - guia_contatos_2025.pdf
#
# REGRAS para manter este dataset:
#   1. Nunca altere perguntas existentes entre versões (quebra comparabilidade)
#   2. Nunca altere ground_truth sem justificativa documentada
#   3. Adicione novas perguntas APENAS ao final
#   4. O campo "id" é imutável (identificador entre versões)
# ─────────────────────────────────────────────────────────────────────────────

DATASET_EVAL: list[dict] = [

    # ── CALENDÁRIO — Datas exatas (alucinação de alto risco) ─────────────────
    {
        "id": "CAL-001",
        "question": "quando começa a matrícula dos veteranos?",
        "ground_truth": (
            "A matrícula de veteranos ocorre de 03/02/2026 a 07/02/2026, "
            "no semestre 2026.1, conforme o Calendário Acadêmico UEMA 2026."
        ),
        "domain": "calendario",
        "tipo": "data",
        "dificuldade": "facil",
    },
    {
        "id": "CAL-002",
        "question": "qual é o período de recesso de carnaval 2026?",
        "ground_truth": (
            "O recesso de Carnaval 2026 ocorre nos dias 02/03/2026 (segunda-feira) "
            "e 03/03/2026 (terça-feira), conforme o Calendário Acadêmico UEMA 2026."
        ),
        "domain": "calendario",
        "tipo": "data",
        "dificuldade": "media",
    },
    {
        "id": "CAL-003",
        "question": "quando é o início das aulas do segundo semestre?",
        "ground_truth": (
            "O início das aulas do segundo semestre (2026.2) ocorre em 03/08/2026, "
            "conforme o Calendário Acadêmico UEMA 2026."
        ),
        "domain": "calendario",
        "tipo": "data",
        "dificuldade": "facil",
    },
    {
        "id": "CAL-004",
        "question": "quando termina o período de trancamento de matrícula no semestre 2026.1?",
        "ground_truth": (
            "O período de trancamento de matrícula no semestre 2026.1 termina em "
            "27/02/2026, conforme o Calendário Acadêmico UEMA 2026."
        ),
        "domain": "calendario",
        "tipo": "data",
        "dificuldade": "dificil",
    },
    {
        "id": "CAL-005",
        "question": "qual é o prazo final para lançamento de notas no primeiro semestre?",
        "ground_truth": (
            "O prazo final para lançamento de notas do semestre 2026.1 é "
            "15/07/2026, conforme o Calendário Acadêmico UEMA 2026."
        ),
        "domain": "calendario",
        "tipo": "data",
        "dificuldade": "dificil",
    },

    # ── EDITAL — Siglas e vagas exatas (alucinação de alto risco) ────────────
    {
        "id": "EDI-001",
        "question": "o que significa a sigla BR-PPI no edital do PAES?",
        "ground_truth": (
            "BR-PPI significa Ampla Concorrência para Pretos, Pardos e Indígenas "
            "de escola pública, conforme o Edital PAES 2026 da UEMA."
        ),
        "domain": "edital",
        "tipo": "sigla",
        "dificuldade": "media",
    },
    {
        "id": "EDI-002",
        "question": "quantas vagas tem o curso de engenharia civil no PAES 2026?",
        "ground_truth": (
            "O curso de Engenharia Civil da UEMA oferta vagas no PAES 2026 "
            "conforme especificado no edital, com distribuição entre ampla concorrência "
            "e cotas, totalizando o número indicado na tabela de vagas do Edital PAES 2026."
        ),
        "domain": "edital",
        "tipo": "sigla",
        "dificuldade": "media",
    },
    {
        "id": "EDI-003",
        "question": "qual a diferença entre as cotas L1 e L2 do PAES?",
        "ground_truth": (
            "A cota L1 é destinada a estudantes de escola pública com renda familiar "
            "per capita de até 1,5 salário mínimo. A cota L2 é destinada a estudantes "
            "de escola pública independentemente de renda, conforme o Edital PAES 2026."
        ),
        "domain": "edital",
        "tipo": "sigla",
        "dificuldade": "media",
    },
    {
        "id": "EDI-004",
        "question": "preciso de quantas horas de atividades complementares para me formar?",
        "ground_truth": (
            "Esta informação não está no Edital PAES 2026 — o edital trata do "
            "processo seletivo de ingresso, não dos requisitos de formatura. "
            "Você pode consultar o regulamento acadêmico ou a secretaria do seu curso."
        ),
        "domain": "edital",
        "tipo": "procedimento",
        "dificuldade": "dificil",
        "esperado_sem_contexto": True,  # deve responder que não está no documento
    },
    {
        "id": "EDI-005",
        "question": "o que é o PAES da UEMA?",
        "ground_truth": (
            "O PAES (Processo de Admissão de Estudantes) é o processo seletivo "
            "da Universidade Estadual do Maranhão (UEMA) para ingresso nos cursos "
            "de graduação, substituindo o vestibular tradicional."
        ),
        "domain": "edital",
        "tipo": "geral",
        "dificuldade": "facil",
    },

    # ── CONTATOS — E-mails e telefones exatos ────────────────────────────────
    {
        "id": "CON-001",
        "question": "qual o email da PROG?",
        "ground_truth": (
            "O e-mail da Pró-Reitoria de Graduação (PROG) da UEMA é "
            "prog@uema.br, conforme o Guia de Contatos UEMA 2025."
        ),
        "domain": "contatos",
        "tipo": "contato",
        "dificuldade": "facil",
    },
    {
        "id": "CON-002",
        "question": "como faço para entrar em contato com o CTIC da UEMA?",
        "ground_truth": (
            "O CTIC (Centro de Tecnologia da Informação e Comunicação) da UEMA "
            "pode ser contatado pelo e-mail e telefone disponíveis no "
            "Guia de Contatos UEMA 2025."
        ),
        "domain": "contatos",
        "tipo": "contato",
        "dificuldade": "facil",
    },
    {
        "id": "CON-003",
        "question": "qual é o telefone do CECEN?",
        "ground_truth": (
            "O Centro de Ciências Exatas e Naturais (CECEN) da UEMA possui "
            "telefone e e-mail de contato disponíveis no Guia de Contatos UEMA 2025."
        ),
        "domain": "contatos",
        "tipo": "contato",
        "dificuldade": "facil",
    },
    {
        "id": "CON-004",
        "question": "quem é o reitor da UEMA?",
        "ground_truth": (
            "O nome do reitor da UEMA está disponível no Guia de Contatos UEMA 2025, "
            "junto com o contato da reitoria."
        ),
        "domain": "contatos",
        "tipo": "contato",
        "dificuldade": "media",
    },
    {
        "id": "CON-005",
        "question": "preciso do contato da biblioteca da UEMA",
        "ground_truth": (
            "O contato da Biblioteca da UEMA, incluindo e-mail e telefone, "
            "está disponível no Guia de Contatos UEMA 2025."
        ),
        "domain": "contatos",
        "tipo": "contato",
        "dificuldade": "facil",
    },

    # ── MULTI-DOMÍNIO — Cruzamento de informações ─────────────────────────────
    {
        "id": "MUL-001",
        "question": "como faço para me matricular depois que passar no PAES?",
        "ground_truth": (
            "Após aprovação no PAES 2026, o estudante deve realizar a matrícula "
            "no período definido pelo Calendário Acadêmico UEMA 2026. "
            "Para dúvidas sobre o processo, contate a PROG (Pró-Reitoria de Graduação)."
        ),
        "domain": "geral",
        "tipo": "procedimento",
        "dificuldade": "dificil",
    },
    {
        "id": "MUL-002",
        "question": "quero travar uma matéria, até quando posso fazer isso?",
        "ground_truth": (
            "O trancamento de matrícula (cancelamento de disciplinas) no semestre 2026.1 "
            "deve ser solicitado até 27/02/2026, conforme o Calendário Acadêmico UEMA 2026. "
            "Para informações sobre o procedimento, contate a secretaria do seu curso."
        ),
        "domain": "geral",
        "tipo": "procedimento",
        "dificuldade": "media",
    },

    # ── PERGUNTAS FORA DO ESCOPO — Sistema deve reconhecer o limite ───────────
    {
        "id": "ESC-001",
        "question": "qual é o preço da mensalidade da UEMA?",
        "ground_truth": (
            "A UEMA é uma universidade pública estadual e não cobra mensalidade "
            "de seus alunos. Esta informação não está nos documentos consultados, "
            "mas é uma característica das universidades públicas brasileiras."
        ),
        "domain": "geral",
        "tipo": "geral",
        "dificuldade": "facil",
        "esperado_sem_contexto": True,
    },
    {
        "id": "ESC-002",
        "question": "o restaurante universitário serve comida vegetariana?",
        "ground_truth": (
            "Informações sobre o cardápio do Restaurante Universitário não estão "
            "disponíveis nos documentos consultados (Calendário Acadêmico, Edital PAES "
            "e Guia de Contatos). Recomendo contatar a administração do RU diretamente."
        ),
        "domain": "geral",
        "tipo": "geral",
        "dificuldade": "media",
        "esperado_sem_contexto": True,
    },
    {
        "id": "ESC-003",
        "question": "oi tudo bem?",
        "ground_truth": (
            "Esta é uma saudação e não requer consulta a documentos. "
            "O sistema deve responder cordialmente e direcionar para ajuda acadêmica."
        ),
        "domain": "geral",
        "tipo": "geral",
        "dificuldade": "facil",
        "esperado_sem_contexto": True,
    },
]


# ─────────────────────────────────────────────────────────────────────────────
# Estrutura de resultado por pergunta
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class ResultadoItem:
    id:                  str
    question:            str
    ground_truth:        str
    answer:              str
    contexts:            list[str]
    domain:              str
    tipo:                str
    dificuldade:         str
    rota_detectada:      str
    tokens_usados:       int
    latencia_ms:         int
    faithfulness:        float = 0.0
    answer_relevancy:    float = 0.0
    context_precision:   float = 0.0
    context_recall:      float = 0.0
    erro:                str = ""


# ─────────────────────────────────────────────────────────────────────────────
# Runner — executa a pipeline real do meuBotRAG para cada pergunta
# ─────────────────────────────────────────────────────────────────────────────

class EvalRunner:
    """Executa a pipeline RAG real e coleta respostas para avaliação."""

    def __init__(self):
        self._inicializado = False

    def _inicializar(self) -> None:
        if self._inicializado:
            return

        logger.info("🔧 Inicializando componentes do meuBotRAG...")

        # Carrega lazy para não falhar import se não estiver no ambiente do projeto
        from src.rag.embeddings import get_embeddings
        from src.infrastructure.redis_client import get_redis
        from src.domain.semantic_router import rotear
        from src.rag.query_transform import transformar_query
        from src.rag.hybrid_retriever import recuperar, recuperar_simples
        from src.providers.gemini_provider import chamar_gemini, montar_prompt_geracao, SYSTEM_UEMA
        from src.domain.entities import Rota

        self._rotear = rotear
        self._transformar = transformar_query
        self._recuperar = recuperar
        self._recuperar_simples = recuperar_simples
        self._chamar_gemini = chamar_gemini
        self._montar_prompt = montar_prompt_geracao
        self._system = SYSTEM_UEMA
        self._Rota = Rota
        self._inicializado = True
        logger.info("✅ Componentes carregados.")

    def _rota_para_source(self, rota_nome: str) -> tuple[str | None, str | None]:
        """Mapeia rota para source_filter e doc_type (igual ao agent/core.py)."""
        mapa_source = {
            "CALENDARIO": ("calendario-academico-2026.pdf", "calendario"),
            "EDITAL":     ("edital_paes_2026.pdf",          "edital"),
            "CONTATOS":   ("guia_contatos_2025.pdf",        "contatos"),
            "GERAL":      (None, None),
        }
        return mapa_source.get(rota_nome, (None, None))

    def executar_pergunta(self, item: dict) -> ResultadoItem:
        """Executa a pipeline completa para uma pergunta e retorna ResultadoItem."""
        self._inicializar()

        question = item["question"]
        t0 = time.time()
        rota_nome = "GERAL"
        contextos: list[str] = []
        tokens = 0

        try:
            # 1. Roteamento semântico
            resultado_rota = self._rotear(question)
            rota_nome = resultado_rota.rota.value if resultado_rota else "GERAL"

            # 2. Busca híbrida (com ou sem source_filter)
            source_filter, doc_type = self._rota_para_source(rota_nome)

            if rota_nome != "GERAL":
                # Query transform + hybrid retrieval
                qt = self._transformar(
                    pergunta_usuario=question,
                    fatos_usuario=[],  # sem fatos de longo prazo no eval (simplificado)
                    historico=[],
                )
                resultado_rag = self._recuperar(
                    qt,
                    source_filter=source_filter,
                    doc_type=doc_type,
                )
            else:
                # Rota geral: busca simples
                resultado_rag = self._recuperar_simples(question)

            # Coleta contextos recuperados (para RAGAS)
            contextos = [
                chunk.content if hasattr(chunk, 'content') else str(chunk)
                for chunk in (resultado_rag.chunks or [])
            ]
            if not contextos and resultado_rag.contexto_formatado:
                contextos = [resultado_rag.contexto_formatado]

            # 3. Geração de resposta
            prompt_final = self._montar_prompt(
                pergunta=question,
                contexto_rag=resultado_rag.contexto_formatado,
                historico_compactado="",
                fatos_str="",
            )

            resposta = self._chamar_gemini(
                prompt=prompt_final,
                system_instruction=self._system,
            )

            answer = resposta.conteudo if resposta.sucesso else f"[ERRO: {resposta.erro}]"
            tokens = resposta.tokens_total

        except Exception as e:
            logger.error("❌ Erro na pergunta %s: %s", item["id"], e)
            answer = f"[EXCEPTION: {e}]"

        latencia_ms = int((time.time() - t0) * 1000)

        return ResultadoItem(
            id=item["id"],
            question=question,
            ground_truth=item["ground_truth"],
            answer=answer,
            contexts=contextos,
            domain=item["domain"],
            tipo=item["tipo"],
            dificuldade=item["dificuldade"],
            rota_detectada=rota_nome,
            tokens_usados=tokens,
            latencia_ms=latencia_ms,
        )

    def executar_todos(self, dataset: list[dict]) -> list[ResultadoItem]:
        """Executa a avaliação para todas as perguntas com progresso."""
        resultados = []
        total = len(dataset)

        for i, item in enumerate(dataset, 1):
            logger.info(
                "📋 [%d/%d] %s | %s | '%s'",
                i, total, item["id"], item["domain"],
                item["question"][:60],
            )
            resultado = self.executar_pergunta(item)
            resultados.append(resultado)

            # Pausa entre chamadas Gemini para respeitar rate limit (15 RPM free tier)
            # 2 chamadas por pergunta (transform + geração) = ~7 perguntas por minuto
            if i < total:
                time.sleep(4)

        return resultados


# ─────────────────────────────────────────────────────────────────────────────
# Calculador de métricas RAGAS
# ─────────────────────────────────────────────────────────────────────────────

class RagasCalculator:
    """
    Calcula métricas RAGAS para os resultados coletados.

    RAGAS usa o Gemini (ou OpenAI) como LLM juiz para avaliar as respostas.
    No contexto do meuBotRAG, usamos o Gemini Flash que já está configurado.

    Documentação RAGAS: https://docs.ragas.io
    """

    def calcular(self, resultados: list[ResultadoItem]) -> list[ResultadoItem]:
        """Calcula métricas RAGAS para cada resultado e preenche os scores."""
        try:
            from ragas import evaluate
            from ragas.metrics import (
                faithfulness,
                answer_relevancy,
                context_precision,
                context_recall,
            )
            from datasets import Dataset
            from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
        except ImportError as e:
            logger.warning(
                "⚠️  RAGAS não instalado ou LangChain Google não disponível: %s\n"
                "   Instala: pip install ragas datasets langchain-google-genai\n"
                "   Scores RAGAS serão 0.0 mas o CSV/JSON ainda será gerado.",
                e,
            )
            return resultados

        try:
            import os
            api_key = os.environ.get("GEMINI_API_KEY", "")
            if not api_key:
                # Tenta importar das settings do projeto
                from src.infrastructure.settings import settings
                api_key = settings.GEMINI_API_KEY

            if not api_key:
                logger.warning("⚠️  GEMINI_API_KEY não encontrada — scores RAGAS serão 0.0")
                return resultados

            # LLM e Embeddings para o RAGAS (usando Gemini)
            llm = ChatGoogleGenerativeAI(
                model="gemini-2.0-flash",
                google_api_key=api_key,
                temperature=0.0,
            )
            embeddings = GoogleGenerativeAIEmbeddings(
                model="models/embedding-001",
                google_api_key=api_key,
            )

        except Exception as e:
            logger.warning("⚠️  Falha ao inicializar LLM para RAGAS: %s", e)
            return resultados

        # Filtra itens com resposta válida e contextos não-vazios
        itens_validos = [r for r in resultados if r.answer and not r.answer.startswith("[")]
        itens_invalidos = [r for r in resultados if r not in itens_validos]

        if not itens_validos:
            logger.warning("⚠️  Nenhum resultado válido para calcular RAGAS.")
            return resultados

        # Monta o dataset no formato RAGAS
        dataset_dict = {
            "question":    [r.question for r in itens_validos],
            "answer":      [r.answer for r in itens_validos],
            "contexts":    [r.contexts if r.contexts else ["[sem contexto]"] for r in itens_validos],
            "ground_truth": [r.ground_truth for r in itens_validos],
        }

        dataset = Dataset.from_dict(dataset_dict)

        logger.info("🔬 Calculando métricas RAGAS para %d perguntas...", len(itens_validos))
        logger.info("   (Isso usa o Gemini como juiz — pode levar 2-5 minutos)")

        try:
            resultado_ragas = evaluate(
                dataset=dataset,
                metrics=[faithfulness, answer_relevancy, context_precision, context_recall],
                llm=llm,
                embeddings=embeddings,
                raise_exceptions=False,
            )

            # Preenche scores nos resultados
            df_ragas = resultado_ragas.to_pandas()
            for i, item in enumerate(itens_validos):
                if i < len(df_ragas):
                    row = df_ragas.iloc[i]
                    item.faithfulness      = float(row.get("faithfulness",      0.0) or 0.0)
                    item.answer_relevancy  = float(row.get("answer_relevancy",  0.0) or 0.0)
                    item.context_precision = float(row.get("context_precision", 0.0) or 0.0)
                    item.context_recall    = float(row.get("context_recall",    0.0) or 0.0)

            logger.info("✅ Métricas RAGAS calculadas.")

        except Exception as e:
            logger.error("❌ RAGAS falhou: %s", e)

        return resultados + itens_invalidos


# ─────────────────────────────────────────────────────────────────────────────
# Geração de relatórios
# ─────────────────────────────────────────────────────────────────────────────

class RelatorioGerador:
    """Gera relatórios CSV e JSON dos resultados da avaliação."""

    def __init__(self, versao: str = "v3"):
        self.versao = versao
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.dir_saida = Path(__file__).parent / "resultados"
        self.dir_saida.mkdir(exist_ok=True)

    def gerar(self, resultados: list[ResultadoItem]) -> dict:
        """Gera CSV detalhado + JSON de resumo. Retorna o resumo."""
        resumo = self._calcular_resumo(resultados)
        self._salvar_csv(resultados)
        self._salvar_json_resumo(resumo)
        self._imprimir_resumo(resumo)
        return resumo

    def _calcular_resumo(self, resultados: list[ResultadoItem]) -> dict:
        """Calcula métricas agregadas por domínio e globais."""
        validos = [r for r in resultados if not r.erro and not r.answer.startswith("[")]
        n = len(validos) or 1

        def media(campo: str) -> float:
            vals = [getattr(r, campo) for r in validos if getattr(r, campo) > 0]
            return round(sum(vals) / len(vals), 4) if vals else 0.0

        # Métricas por domínio
        dominios = list({r.domain for r in resultados})
        por_dominio = {}
        for dom in dominios:
            grupo = [r for r in validos if r.domain == dom]
            if not grupo:
                continue
            ng = len(grupo)
            por_dominio[dom] = {
                "n_perguntas":        ng,
                "faithfulness":       round(sum(r.faithfulness for r in grupo) / ng, 4),
                "answer_relevancy":   round(sum(r.answer_relevancy for r in grupo) / ng, 4),
                "context_precision":  round(sum(r.context_precision for r in grupo) / ng, 4),
                "context_recall":     round(sum(r.context_recall for r in grupo) / ng, 4),
                "latencia_media_ms":  round(sum(r.latencia_ms for r in grupo) / ng),
                "tokens_media":       round(sum(r.tokens_usados for r in grupo) / ng),
            }

        # Distribuição de rotas detectadas
        dist_rotas: dict[str, int] = {}
        for r in resultados:
            dist_rotas[r.rota_detectada] = dist_rotas.get(r.rota_detectada, 0) + 1

        return {
            "versao":          self.versao,
            "timestamp":       self.timestamp,
            "n_total":         len(resultados),
            "n_validos":       len(validos),
            "n_erros":         len(resultados) - len(validos),
            "global": {
                "faithfulness":       media("faithfulness"),
                "answer_relevancy":   media("answer_relevancy"),
                "context_precision":  media("context_precision"),
                "context_recall":     media("context_recall"),
                "latencia_media_ms":  round(sum(r.latencia_ms for r in validos) / n),
                "tokens_media":       round(sum(r.tokens_usados for r in validos) / n),
                "tokens_total":       sum(r.tokens_usados for r in validos),
            },
            "por_dominio": por_dominio,
            "distribuicao_rotas": dist_rotas,
        }

    def _salvar_csv(self, resultados: list[ResultadoItem]) -> None:
        """Salva CSV com todos os detalhes por pergunta."""
        try:
            import pandas as pd
            linhas = []
            for r in resultados:
                d = asdict(r)
                # Simplifica contexts para CSV (une com separador)
                d["contexts_count"] = len(r.contexts)
                d["contexts_preview"] = " | ".join(r.contexts[:2])[:300] if r.contexts else ""
                d.pop("contexts", None)
                linhas.append(d)

            df = pd.DataFrame(linhas)
            path = self.dir_saida / f"rag_eval_{self.versao}_{self.timestamp}.csv"
            df.to_csv(path, index=False, encoding="utf-8-sig")  # utf-8-sig para Excel no Windows
            logger.info("💾 CSV salvo: %s", path)
        except ImportError:
            logger.warning("⚠️  pandas não instalado — CSV não gerado. pip install pandas")

    def _salvar_json_resumo(self, resumo: dict) -> None:
        """Salva JSON de resumo (para comparação entre versões)."""
        path = self.dir_saida / f"rag_eval_resumo_{self.versao}_{self.timestamp}.json"
        with open(path, "w", encoding="utf-8") as f:
            json.dump(resumo, f, indent=2, ensure_ascii=False)
        logger.info("💾 JSON de resumo salvo: %s", path)

    def _imprimir_resumo(self, resumo: dict) -> None:
        """Imprime tabela de resumo no terminal."""
        g = resumo["global"]
        print("\n" + "=" * 70)
        print(f"  📊 RESULTADOS DA AVALIAÇÃO RAG — meuBotRAG {self.versao}")
        print("=" * 70)
        print(f"  Perguntas:  {resumo['n_validos']}/{resumo['n_total']} válidas")
        print(f"  Tokens:     {g['tokens_total']} total | {g['tokens_media']} média/pergunta")
        print(f"  Latência:   {g['latencia_media_ms']} ms média")
        print()
        print("  MÉTRICAS RAGAS (0.0 = pior, 1.0 = melhor):")
        print(f"  {'Métrica':<25} {'Score':>8}  {'Avaliação'}")
        print("  " + "-" * 50)

        metricas = [
            ("Faithfulness",       g["faithfulness"],       "anti-alucinação"),
            ("Answer Relevancy",   g["answer_relevancy"],   "resposta relevante"),
            ("Context Precision",  g["context_precision"],  "retriever preciso"),
            ("Context Recall",     g["context_recall"],     "retriever completo"),
        ]
        for nome, score, desc in metricas:
            emoji = "✅" if score >= 0.7 else ("⚠️ " if score >= 0.5 else "❌")
            print(f"  {nome:<25} {score:>8.4f}  {emoji} {desc}")

        print()
        print("  SCORES POR DOMÍNIO:")
        for dom, dados in resumo["por_dominio"].items():
            print(f"  {dom.upper():<12} faith={dados['faithfulness']:.3f}  "
                  f"relevancy={dados['answer_relevancy']:.3f}  "
                  f"precision={dados['context_precision']:.3f}  "
                  f"recall={dados['context_recall']:.3f}")

        print()
        print("  ROTAS DETECTADAS:", resumo["distribuicao_rotas"])
        print("=" * 70)
        print(f"  📁 Relatórios em: eval/resultados/")
        print("=" * 70 + "\n")


# ─────────────────────────────────────────────────────────────────────────────
# Comparador de versões
# ─────────────────────────────────────────────────────────────────────────────

def comparar_versoes(json_path_a: str, json_path_b: str) -> None:
    """
    Compara dois JSONs de resumo e imprime a diferença nas métricas.

    Uso:
      python eval/rag_eval.py --comparar eval/resultados/resumo_v3_abc.json eval/resultados/resumo_v4_xyz.json
    """
    with open(json_path_a) as f: a = json.load(f)
    with open(json_path_b) as f: b = json.load(f)

    print(f"\n📊 COMPARAÇÃO: {a['versao']} → {b['versao']}")
    print("=" * 60)
    metricas = ["faithfulness", "answer_relevancy", "context_precision", "context_recall"]
    for m in metricas:
        va = a["global"][m]
        vb = b["global"][m]
        diff = vb - va
        emoji = "📈" if diff > 0.01 else ("📉" if diff < -0.01 else "➡️ ")
        print(f"  {m:<25} {va:.4f} → {vb:.4f}   {emoji} {diff:+.4f}")
    print()
    ta, tb = a["global"]["tokens_media"], b["global"]["tokens_media"]
    print(f"  {'Tokens (média)':<25} {ta} → {tb}   {'📉' if tb < ta else '📈'} {tb-ta:+d}")
    la, lb = a["global"]["latencia_media_ms"], b["global"]["latencia_media_ms"]
    print(f"  {'Latência (ms)':<25} {la} → {lb}   {'📉' if lb < la else '📈'} {lb-la:+d}")
    print("=" * 60 + "\n")


# ─────────────────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────────────────

def main():
    import argparse
    parser_args = argparse.ArgumentParser(
        description="RAG Eval — Suite de avaliação do meuBotRAG"
    )
    parser_args.add_argument(
        "--versao", default="v3",
        help="Nome da versão para o relatório (ex: v3, v4-novo-chunking)"
    )
    parser_args.add_argument(
        "--ids", nargs="+", default=None,
        help="IDs específicos para testar (ex: CAL-001 EDI-002). Omite para rodar todos."
    )
    parser_args.add_argument(
        "--sem-ragas", action="store_true",
        help="Pula o cálculo RAGAS (mais rápido, sem scores de qualidade)"
    )
    parser_args.add_argument(
        "--comparar", nargs=2, metavar=("JSON_A", "JSON_B"),
        help="Compara dois JSONs de resumo (não executa eval)"
    )
    args = parser_args.parse_args()

    # Modo comparação
    if args.comparar:
        comparar_versoes(args.comparar[0], args.comparar[1])
        return

    # Filtra dataset por IDs se especificado
    dataset = DATASET_EVAL
    if args.ids:
        dataset = [item for item in DATASET_EVAL if item["id"] in args.ids]
        if not dataset:
            logger.error("❌ Nenhum item encontrado para os IDs: %s", args.ids)
            sys.exit(1)
        logger.info("🎯 Rodando apenas: %s", [i["id"] for i in dataset])

    print(f"\n🚀 Iniciando RAG Eval — meuBotRAG {args.versao}")
    print(f"   Perguntas: {len(dataset)}")
    print(f"   RAGAS: {'Desativado' if args.sem_ragas else 'Ativado (usa Gemini como juiz)'}")
    print()

    # 1. Executa a pipeline para cada pergunta
    runner = EvalRunner()
    resultados = runner.executar_todos(dataset)

    # 2. Calcula métricas RAGAS (se não desativado)
    if not args.sem_ragas:
        calculator = RagasCalculator()
        resultados = calculator.calcular(resultados)

    # 3. Gera relatórios
    gerador = RelatorioGerador(versao=args.versao)
    gerador.gerar(resultados)


if __name__ == "__main__":
    main()