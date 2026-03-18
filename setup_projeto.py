#!/usr/bin/env python3
"""
setup_projeto.py — Configuração inicial completa do meuBotRAG
=============================================================

O QUE FAZ:
  1. Cria toda a estrutura de pastas (dados/, tests/, static/, templates/)
  2. Gera .env.local e .env.test com valores-padrão comentados
  3. Gera PDFs de teste reais (com conteúdo RAG-optimizado) usando reportlab
  4. Gera CSVs de teste (vagas mock e contatos mock)
  5. Actualiza o DOCUMENT_CONFIG automaticamente para os novos ficheiros

COMO USAR:
  python setup_projeto.py

  Ou com opções:
  python setup_projeto.py --so-pastas      # só cria pastas
  python setup_projeto.py --so-envs        # só cria .env.local e .env.test
  python setup_projeto.py --so-dados       # só gera PDFs e CSVs de teste
  python setup_projeto.py --limpar-redis   # aviso + instrução para limpar Redis

ESTRUTURA CRIADA:
  dados/
  ├── PDF/
  │   ├── academicos/   ← PDFs de produção (calendário, edital, contatos)
  │   └── testes/       ← PDFs de teste gerados por este script
  ├── CSV/
  │   ├── academicos/   ← CSVs de produção
  │   └── testes/       ← CSVs mock gerados por este script
  └── uploads/          ← ficheiros ingeridos via !ingerir pelo WhatsApp

  tests/
  ├── unit/
  ├── integration/
  └── data/             ← fixtures dos testes

  static/css/, static/js/
  templates/monitor/
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import textwrap
from pathlib import Path

# ─────────────────────────────────────────────────────────────────────────────
# Raiz do projecto
# ─────────────────────────────────────────────────────────────────────────────

RAIZ = Path(__file__).resolve().parent
print(f"📁 Raiz do projecto: {RAIZ}")


# =============================================================================
# 1. ESTRUTURA DE PASTAS
# =============================================================================

PASTAS = [
    # Dados
    "dados/PDF/academicos",
    "dados/PDF/testes",
    "dados/CSV/academicos",
    "dados/CSV/testes",
    "dados/uploads",

    # Código
    "src/api",
    "src/agent",
    "src/domain",
    "src/infrastructure",
    "src/memory",
    "src/middleware",
    "src/providers",
    "src/rag",
    "src/services",
    "src/tools",
    "src/application",

    # Web
    "static/css",
    "static/js",
    "templates/monitor",

    # Testes
    "tests/unit",
    "tests/integration",
    "tests/data",

    # Debug
    "debug",
]


def criar_pastas() -> None:
    print("\n📂 Criando estrutura de pastas...")
    for pasta in PASTAS:
        caminho = RAIZ / pasta
        caminho.mkdir(parents=True, exist_ok=True)
        gitkeep = caminho / ".gitkeep"
        if not any(caminho.iterdir()):   # pasta vazia → cria .gitkeep
            gitkeep.touch()
    print(f"  ✅ {len(PASTAS)} pastas criadas/verificadas.")


# =============================================================================
# 2. FICHEIROS .ENV
# =============================================================================

_ENV_LOCAL = """\
# =============================================================================
# .env.local — Ambiente de DESENVOLVIMENTO LOCAL (fora do Docker)
# =============================================================================
# Usado por: debug/debug_chainlit.py, tests/test_*.py, python setup_projeto.py
# NÃO usar em produção (Docker usa o .env normal)
# NUNCA commites este ficheiro — está no .gitignore

# ── Gemini ────────────────────────────────────────────────────────────────────
GEMINI_API_KEY=COLOCA_A_TUA_CHAVE_AQUI
GEMINI_MODEL=gemini-2.0-flash
GEMINI_TEMP=0.3
GEMINI_MAX_TOKENS=1024

# ── Redis Stack LOCAL (Docker exposto em localhost) ───────────────────────────
# docker run -p 6379:6379 redis/redis-stack:latest
REDIS_URL=redis://localhost:6379/0

# ── Dados (pasta LOCAL, fora do Docker) ──────────────────────────────────────
DATA_DIR={data_dir_local}

# ── LlamaParse (opcional, para PDFs com tabelas complexas) ───────────────────
PDF_PARSER=pymupdf
LLAMA_CLOUD_API_KEY=

# ── HuggingFace (embedding BAAI/bge-m3) ──────────────────────────────────────
HF_TOKEN=

# ── RBAC — O TEU número como ADMIN (sem + e sem espaço) ──────────────────────
ADMIN_NUMBERS=5598XXXXXXXXX
STUDENT_NUMBERS=
ADMIN_API_KEY=dev_admin_key_local_apenas

# ── Evolution API (não necessário para testes locais) ─────────────────────────
EVOLUTION_BASE_URL=http://localhost:8080
EVOLUTION_API_KEY=
EVOLUTION_INSTANCE_NAME=uema-bot
WHATSAPP_HOOK_URL=http://localhost:9000/webhook
WEBHOOK_SECRET=

# ── Dev ───────────────────────────────────────────────────────────────────────
DEV_MODE=True
DEV_WHITELIST=5598XXXXXXXXX
LOG_LEVEL=DEBUG
"""

_ENV_TEST = """\
# =============================================================================
# .env.test — Ambiente de TESTES ISOLADO
# =============================================================================
# Usa Redis DB 3 (isolado da produção DB 0 e dev DB 0)
# Permite testar sem apagar dados de desenvolvimento
#
# COMO USAR:
#   ENV_FILE_PATH=.env.test python tests/test_pipeline_admin.py
#   ENV_FILE_PATH=.env.test pytest tests/ -v
#
# LIMPAR APENAS OS DADOS DE TESTE (sem tocar na produção):
#   redis-cli -n 3 FLUSHDB

# ── Gemini (mesma chave) ──────────────────────────────────────────────────────
GEMINI_API_KEY=COLOCA_A_TUA_CHAVE_AQUI
GEMINI_MODEL=gemini-2.0-flash
GEMINI_TEMP=0.0
GEMINI_MAX_TOKENS=512

# ── Redis — DB 3 ISOLADO para testes ─────────────────────────────────────────
# DB 0 = produção/dev normal
# DB 3 = dados de teste (pode apagar livremente com: redis-cli -n 3 FLUSHDB)
REDIS_URL=redis://localhost:6379/3

# ── Dados de TESTE (pasta separada) ──────────────────────────────────────────
DATA_DIR={data_dir_testes}

# ── Parser mais rápido para testes (local, sem custo) ────────────────────────
PDF_PARSER=pymupdf
LLAMA_CLOUD_API_KEY=

# ── HuggingFace ───────────────────────────────────────────────────────────────
HF_TOKEN=

# ── RBAC de teste ─────────────────────────────────────────────────────────────
ADMIN_NUMBERS=5598000000001
STUDENT_NUMBERS=5598000000002
ADMIN_API_KEY=test_admin_key_nao_usar_em_prod

# ── Evolution (mock — não precisa funcionar nos testes) ──────────────────────
EVOLUTION_BASE_URL=http://localhost:8080
EVOLUTION_API_KEY=test_key
EVOLUTION_INSTANCE_NAME=test-instance
WHATSAPP_HOOK_URL=http://localhost:9000/webhook
WEBHOOK_SECRET=test_secret

# ── Dev ───────────────────────────────────────────────────────────────────────
DEV_MODE=True
DEV_WHITELIST=5598000000001
LOG_LEVEL=WARNING
"""


def criar_envs() -> None:
    print("\n📝 Criando ficheiros .env...")

    data_dir_local  = str(RAIZ / "dados")
    data_dir_testes = str(RAIZ / "dados" / "PDF" / "testes")

    for nome, conteudo in [
        (".env.local", _ENV_LOCAL.format(data_dir_local=data_dir_local)),
        (".env.test",  _ENV_TEST.format(data_dir_testes=data_dir_testes)),
    ]:
        caminho = RAIZ / nome
        if caminho.exists():
            print(f"  ⏭️  '{nome}' já existe — não sobrescrito.")
        else:
            caminho.write_text(conteudo, encoding="utf-8")
            print(f"  ✅ '{nome}' criado.")

    # Garante que .gitignore tem as entradas correctas
    _actualizar_gitignore()


def _actualizar_gitignore() -> None:
    gitignore = RAIZ / ".gitignore"
    linhas_novas = [
        ".env", ".env.local", ".env.test",
        "__pycache__/", "*.pyc", ".pytest_cache/",
        "dados/uploads/", "dados/PDF/testes/", "dados/CSV/testes/",
        "*.pdf", "*.csv",
    ]
    existentes: set[str] = set()
    if gitignore.exists():
        existentes = set(gitignore.read_text().splitlines())

    novas = [l for l in linhas_novas if l not in existentes]
    if novas:
        with gitignore.open("a", encoding="utf-8") as f:
            f.write("\n# Adicionado pelo setup_projeto.py\n")
            f.write("\n".join(novas) + "\n")
        print(f"  ✅ .gitignore actualizado ({len(novas)} entradas).")


# =============================================================================
# 3. PDF DE TESTE — usando reportlab
# =============================================================================

def gerar_pdf_teste_especificacao() -> Path:
    """
    Gera PDF de especificação técnica do agente com tabelas.
    Optimizado para LlamaParse: usa formatação de tabelas explícita.
    """
    from reportlab.lib import colors
    from reportlab.lib.pagesizes import A4
    from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
    from reportlab.lib.units import cm
    from reportlab.platypus import (
        HRFlowable, Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle,
    )

    caminho = RAIZ / "dados" / "PDF" / "testes" / "agente_rag_uema_spec.pdf"
    doc     = SimpleDocTemplate(str(caminho), pagesize=A4,
                                topMargin=2*cm, bottomMargin=2*cm,
                                leftMargin=2.5*cm, rightMargin=2.5*cm)

    styles  = getSampleStyleSheet()
    titulo_style = ParagraphStyle("titulo", parent=styles["Heading1"],
                                  fontSize=16, spaceAfter=6)
    h2_style     = ParagraphStyle("h2", parent=styles["Heading2"],
                                  fontSize=13, spaceAfter=4)
    normal       = styles["Normal"]
    small        = ParagraphStyle("small", parent=normal, fontSize=9)

    cor_cabecalho = colors.HexColor("#1a1d2e")
    cor_linha_alt = colors.HexColor("#f0f2f5")

    def tabela(dados: list[list], col_widths=None):
        t = Table(dados, colWidths=col_widths)
        t.setStyle(TableStyle([
            ("BACKGROUND", (0, 0), (-1, 0), cor_cabecalho),
            ("TEXTCOLOR",  (0, 0), (-1, 0), colors.white),
            ("FONTNAME",   (0, 0), (-1, 0), "Helvetica-Bold"),
            ("FONTSIZE",   (0, 0), (-1, 0), 9),
            ("FONTSIZE",   (0, 1), (-1, -1), 8),
            ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, cor_linha_alt]),
            ("GRID",       (0, 0), (-1, -1), 0.5, colors.HexColor("#cccccc")),
            ("LEFTPADDING",  (0, 0), (-1, -1), 6),
            ("RIGHTPADDING", (0, 0), (-1, -1), 6),
            ("TOPPADDING",   (0, 0), (-1, -1), 4),
            ("BOTTOMPADDING",(0, 0), (-1, -1), 4),
        ]))
        return t

    story = []

    # Cabeçalho
    story += [
        Paragraph("Especificação Técnica: Agente RAG UEMA v5", titulo_style),
        Paragraph("Documento interno de teste — CTIC/UEMA — Março 2026", small),
        HRFlowable(width="100%", thickness=1, color=colors.HexColor("#6366f1")),
        Spacer(1, 0.3*cm),
    ]

    # Secção 1 — Stack
    story += [
        Paragraph("1. Stack Tecnológico", h2_style),
        tabela([
            ["Componente", "Tecnologia", "Função"],
            ["LLM", "Gemini 2.0 Flash", "Geração de respostas"],
            ["Embedding", "BAAI/bge-m3 (1024 dims)", "Vetorização de texto"],
            ["Vector Store", "Redis Stack (HNSW)", "Busca vectorial"],
            ["BM25", "RediSearch nativo", "Busca por keywords exactas"],
            ["Fusão", "RRF (Reciprocal Rank Fusion)", "Combina vectorial + BM25"],
            ["Queue", "Celery + Redis", "Processamento assíncrono"],
            ["Gateway", "Evolution API v2.3", "WhatsApp integration"],
        ], col_widths=[4*cm, 6*cm, 6*cm]),
        Spacer(1, 0.4*cm),
    ]

    # Secção 2 — Métricas
    story += [
        Paragraph("2. Métricas de Performance por Versão", h2_style),
        tabela([
            ["Versão", "Tokens/msg", "Latência média", "Alucinações"],
            ["v2 (LangChain + Groq)", "~4.300", "500-1500ms", "Alta"],
            ["v3 (Gemini + Redis)", "~1.070", "800-1200ms", "Média"],
            ["v4 (Cache + CRAG)", "~750", "600-1000ms", "Baixa"],
            ["v5 (Guardrails + Self-RAG)", "~520", "400-900ms", "Muito baixa"],
        ], col_widths=[6*cm, 3*cm, 4*cm, 3.5*cm]),
        Spacer(1, 0.4*cm),
    ]

    # Secção 3 — Thresholds CRAG
    story += [
        Paragraph("3. Thresholds do CRAG (Corrective RAG)", h2_style),
        tabela([
            ["Threshold", "Valor", "Acção do Sistema"],
            ["CRAG_THRESHOLD_OK", "0.40", "Contexto bom — gera normalmente"],
            ["CRAG_THRESHOLD_MIN", "0.20", "Contexto fraco — gera com disclaimer"],
            ["Abaixo do mínimo", "< 0.20", "Rejeita contexto — responde sem RAG"],
        ], col_widths=[5*cm, 3*cm, 8.5*cm]),
        Spacer(1, 0.4*cm),
    ]

    # Secção 4 — RBAC
    story += [
        Paragraph("4. Hierarquia de Acesso (RBAC)", h2_style),
        tabela([
            ["Nível", "Código", "Rate Limit", "Tools Disponíveis"],
            ["Visitante", "GUEST", "10/min, 50/hora", "RAG (leitura)"],
            ["Aluno", "STUDENT", "30/min, 200/hora", "RAG + GLPI"],
            ["Administrador", "ADMIN", "Ilimitado", "Todas + Comandos Admin"],
        ], col_widths=[3.5*cm, 3*cm, 4*cm, 6*cm]),
        Spacer(1, 0.4*cm),
    ]

    # Secção 5 — Comandos Admin
    story += [
        Paragraph("5. Comandos Admin via WhatsApp", h2_style),
        tabela([
            ["Comando", "Acção", "Assíncrono"],
            ["!status", "Estado do sistema (Redis, AgentCore, Cache)", "Não"],
            ["!tools", "Lista tools registadas no SemanticRouter", "Não"],
            ["!limpar_cache", "Invalida todo o Semantic Cache", "Sim (Celery)"],
            ["!ingerir", "Ingere ficheiro anexado via WhatsApp", "Sim (Celery)"],
            ["!ragas", "Exporta logs de produção para dataset RAGAS", "Sim (Celery)"],
            ["!fatos [user]", "Lista fatos long-term de um utilizador", "Não"],
            ["!reload", "Reinicia AgentCore com novas tools", "Sim (Celery)"],
        ], col_widths=[4*cm, 9*cm, 3.5*cm]),
        Spacer(1, 0.4*cm),
    ]

    # Secção 6 — Formatos suportados
    story += [
        Paragraph("6. Formatos de Documentos Suportados", h2_style),
        tabela([
            ["Extensão", "Parser", "Custo", "Melhor para"],
            [".pdf", "LlamaParse (cloud)", "$0.003/pág", "Tabelas complexas"],
            [".pdf", "PyMuPDF (local)", "$0.00", "Textos narrativos"],
            [".csv", "pandas", "$0.00", "Tabelas de dados (recomendado)"],
            [".docx", "python-docx", "$0.00", "Manuais e guias"],
            [".xlsx", "openpyxl", "$0.00", "Planilhas de dados"],
            [".txt/.md", "leitura directa", "$0.00", "FAQ e documentação"],
            [".html", "BeautifulSoup", "$0.00", "Páginas Web locais"],
        ], col_widths=[3*cm, 4.5*cm, 3*cm, 6*cm]),
        Spacer(1, 0.4*cm),
    ]

    # Perguntas de referência RAG eval
    story += [
        Paragraph("7. Perguntas de Referência para RAG Eval", h2_style),
        tabela([
            ["ID", "Pergunta", "Resposta Esperada", "Fonte"],
            ["T01", "Quantas dimensões tem o embedding?", "1024", "§1"],
            ["T02", "Threshold CRAG para contexto bom?", "0.40", "§3"],
            ["T03", "Rate limit do nível STUDENT?", "30/min, 200/hora", "§4"],
            ["T04", "Custo do LlamaParse por página?", "$0.003", "§6"],
            ["T05", "Quantos passos tem a pipeline?", "12 passos", "doc"],
            ["T06", "Formato ideal para tabelas de vagas?", "CSV", "§6"],
            ["T07", "Algoritmo de fusão BM25 + vectorial?", "RRF", "§1"],
            ["T08", "Nível para usar !limpar_cache?", "ADMIN", "§5"],
        ], col_widths=[1.5*cm, 6*cm, 4*cm, 2*cm]),
    ]

    doc.build(story)
    print(f"  ✅ PDF gerado: {caminho.relative_to(RAIZ)}")
    return caminho


def gerar_pdf_teste_manual() -> Path:
    """
    Gera PDF do manual de uso com Parte A (bot), Parte B (admin), Parte C (Linux).
    Multi-contexto propositado para testar grounding do LLM.
    """
    from reportlab.lib import colors
    from reportlab.lib.pagesizes import A4
    from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
    from reportlab.lib.units import cm
    from reportlab.platypus import (
        HRFlowable, Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle,
    )

    caminho = RAIZ / "dados" / "PDF" / "testes" / "instrucoes_uso_agente.pdf"
    doc     = SimpleDocTemplate(str(caminho), pagesize=A4,
                                topMargin=2*cm, bottomMargin=2*cm,
                                leftMargin=2.5*cm, rightMargin=2.5*cm)

    styles     = getSampleStyleSheet()
    titulo     = ParagraphStyle("t", parent=styles["Heading1"], fontSize=16)
    h2         = ParagraphStyle("h2", parent=styles["Heading2"], fontSize=13)
    h3         = ParagraphStyle("h3", parent=styles["Heading3"], fontSize=11)
    normal     = styles["Normal"]
    code_style = ParagraphStyle("code", parent=normal, fontName="Courier",
                                fontSize=8, backColor=colors.HexColor("#f5f5f5"),
                                leftIndent=12, rightIndent=12)
    nota       = ParagraphStyle("nota", parent=normal, fontSize=8,
                                textColor=colors.HexColor("#e74c3c"))

    def tabela(dados, widths=None):
        t = Table(dados, colWidths=widths)
        t.setStyle(TableStyle([
            ("BACKGROUND",   (0,0),(-1,0), colors.HexColor("#2c3e50")),
            ("TEXTCOLOR",    (0,0),(-1,0), colors.white),
            ("FONTNAME",     (0,0),(-1,0), "Helvetica-Bold"),
            ("FONTSIZE",     (0,0),(-1,-1), 8),
            ("ROWBACKGROUNDS",(0,1),(-1,-1), [colors.white, colors.HexColor("#ecf0f1")]),
            ("GRID",         (0,0),(-1,-1), 0.4, colors.grey),
            ("LEFTPADDING",  (0,0),(-1,-1), 6),
            ("RIGHTPADDING", (0,0),(-1,-1), 6),
            ("TOPPADDING",   (0,0),(-1,-1), 3),
            ("BOTTOMPADDING",(0,0),(-1,-1), 3),
        ]))
        return t

    story = []
    story += [
        Paragraph("Manual de Uso — Bot UEMA e Referência de Comandos", titulo),
        Paragraph("Versão 1.0 | CTIC/UEMA | Março 2026", styles["Normal"]),
        HRFlowable(width="100%", thickness=1),
        Spacer(1, 0.3*cm),
    ]

    # PARTE A
    story += [
        Paragraph("PARTE A — Como Usar o Assistente Virtual UEMA", h2),
        Paragraph("A.1 Tipos de Perguntas Suportadas", h3),
        tabela([
            ["Categoria", "Exemplos de Perguntas"],
            ["📅 Calendário", "Quando começa o semestre? Data da matrícula de veteranos"],
            ["📋 Edital PAES", "Quantas vagas tem Engenharia Civil? Cota BR-PPI?"],
            ["📞 Contatos", "Email da secretaria de Direito | Telefone da PROG"],
            ["💻 TI e Sistemas", "Como acesso o SIGAA? Esqueci senha do e-mail"],
            ["🎫 Suporte", "Computador do lab sem internet (abre chamado GLPI)"],
        ], widths=[4*cm, 12.5*cm]),
        Spacer(1, 0.3*cm),
        Paragraph("A.2 Rate Limits por Perfil", h3),
        tabela([
            ["Perfil", "Mensagens/Minuto", "Mensagens/Hora"],
            ["Visitante (GUEST)", "10", "50"],
            ["Aluno (STUDENT)", "30", "200"],
            ["Administrador (ADMIN)", "Sem limite", "Sem limite"],
        ], widths=[6*cm, 5*cm, 5.5*cm]),
        Spacer(1, 0.4*cm),
    ]

    # PARTE B
    story += [
        Paragraph("PARTE B — Comandos Avançados para Administradores", h2),
        Paragraph("B.1 Fluxo de Ingestão via WhatsApp", h3),
        Paragraph("1. Admin envia PDF/CSV/DOCX com legenda '!ingerir'", normal),
        Paragraph("2. Bot confirma: '📥 Ficheiro recebido! A ingerir em background...'", normal),
        Paragraph("3. Celery worker baixa o ficheiro via Evolution API", normal),
        Paragraph("4. Detecta extensão pelo mimetype e valida formato esperado", normal),
        Paragraph("5. Salva em /dados/uploads/ e ingere no Redis Stack", normal),
        Paragraph("6. Bot confirma: '✅ Documento ingerido! Chunks: 47'", normal),
        Spacer(1, 0.2*cm),
        Paragraph("B.2 Tempo Estimado de Ingestão por Formato", h3),
        tabela([
            ["Formato", "Tamanho", "Tempo Estimado"],
            ["CSV (vagas)", "< 50 linhas", "~15 segundos"],
            ["PDF (texto)", "~10 páginas", "~45 segundos (PyMuPDF)"],
            ["PDF (tabelas)", "~10 páginas", "~3 minutos (LlamaParse)"],
            ["DOCX (manual)", "~20 páginas", "~30 segundos"],
        ], widths=[4*cm, 4*cm, 8.5*cm]),
        Spacer(1, 0.4*cm),
    ]

    # PARTE C — comandos Linux (multi-contexto para testar grounding)
    story += [
        Paragraph("PARTE C — Referência de Comandos de Sistema", h2),
        Paragraph(
            "NOTA: Esta secção documenta comandos do servidor Linux onde o bot está hospedado.",
            nota,
        ),
        Spacer(1, 0.2*cm),
        Paragraph("C.1 Gestão do Redis", h3),
        tabela([
            ["Comando", "Descrição"],
            ["redis-cli ping", "Verifica se Redis responde"],
            ["redis-cli FLUSHDB", "APAGA TODOS OS DADOS do DB actual"],
            ["redis-cli -n 3 FLUSHDB", "Apaga só o DB 3 (testes isolados)"],
            ["redis-cli DBSIZE", "Número de chaves no DB actual"],
            ["redis-cli keys 'rag:chunk:*'", "Lista chaves de chunks RAG"],
        ], widths=[6*cm, 10.5*cm]),
        Spacer(1, 0.3*cm),
        Paragraph("C.2 Gestão do Docker Compose", h3),
        tabela([
            ["Comando", "Descrição"],
            ["docker-compose up -d", "Inicia todos os serviços"],
            ["docker-compose down", "Para todos os serviços"],
            ["docker-compose logs -f bot", "Logs do bot em tempo real"],
            ["docker-compose logs -f worker", "Logs do Celery worker"],
            ["docker-compose restart bot", "Reinicia apenas o bot"],
        ], widths=[6*cm, 10.5*cm]),
        Spacer(1, 0.4*cm),
    ]

    # Perguntas de teste
    story += [
        Paragraph("D. Perguntas de Teste (RAG Eval)", h2),
        tabela([
            ["ID", "Pergunta", "Resposta Correcta"],
            ["M01", "O bot funciona em grupos de WhatsApp?", "Não, só conversas privadas"],
            ["M02", "Quantas msgs/hora pode um visitante enviar?", "50 mensagens por hora"],
            ["M03", "O bot consegue fazer matrícula?", "Não, usar o SIGAA"],
            ["M04", "Como um admin reinicia o AgentCore?", "Enviando !reload"],
            ["M05", "Quanto demora ingerir PDF 10 págs (PyMuPDF)?", "~45 segundos"],
            ["M06", "Qual comando apaga todos os dados do Redis?", "redis-cli FLUSHDB"],
            ["M07", "Como ver logs do Celery worker?", "docker-compose logs -f worker"],
            ["M08", "Como o admin ingere CSV pelo WhatsApp?", "Anexa com legenda !ingerir"],
        ], widths=[1.5*cm, 8*cm, 7*cm]),
    ]

    doc.build(story)
    print(f"  ✅ PDF gerado: {caminho.relative_to(RAIZ)}")
    return caminho


def gerar_pdfs() -> None:
    print("\n📄 Gerando PDFs de teste...")
    try:
        gerar_pdf_teste_especificacao()
        gerar_pdf_teste_manual()
    except ImportError:
        print("  ⚠️  reportlab não instalado. Instala: pip install reportlab")
        print("  ℹ️  A gerar ficheiros .md como fallback...")
        _gerar_md_fallback()


def _gerar_md_fallback():
    """Fallback se reportlab não estiver disponível: gera .md em vez de .pdf."""
    for nome in ["agente_rag_uema_spec.md", "instrucoes_uso_agente.md"]:
        caminho = RAIZ / "dados" / "PDF" / "testes" / nome
        if not caminho.exists():
            caminho.write_text(
                f"# {nome}\n\nFicheiro de teste. Instala reportlab para gerar PDF.\n",
                encoding="utf-8",
            )
            print(f"  📝 MD fallback criado: {nome}")


# =============================================================================
# 4. CSVs DE TESTE
# =============================================================================

_CSV_VAGAS = [
    ["CURSO", "CAMPUS", "TURNO", "AC", "BR_PPI", "BR_Q", "BR_DC", "PCD", "TOTAL"],
    ["Engenharia Civil", "São Luís", "Noturno", "40", "8", "4", "2", "2", "56"],
    ["Engenharia Elétrica", "São Luís", "Noturno", "35", "7", "3", "2", "1", "48"],
    ["Direito", "São Luís", "Matutino", "50", "10", "5", "3", "2", "70"],
    ["Direito", "São Luís", "Noturno", "50", "10", "5", "3", "2", "70"],
    ["Medicina", "São Luís", "Integral", "30", "6", "3", "1", "1", "41"],
    ["Sistemas de Informação", "São Luís", "Noturno", "40", "8", "4", "2", "2", "56"],
    ["Administração", "São Luís", "Noturno", "50", "10", "5", "3", "2", "70"],
    ["Pedagogia", "São Luís", "Matutino", "60", "12", "6", "3", "3", "84"],
    ["Enfermagem", "Caxias", "Integral", "25", "5", "2", "1", "1", "34"],
    ["Letras Português", "Caxias", "Noturno", "40", "8", "4", "2", "2", "56"],
]

_CSV_CONTATOS = [
    ["SETOR", "CARGO", "NOME", "EMAIL", "TELEFONE"],
    ["PROG", "Pró-Reitor de Graduação", "Prof. Dr. Fulano Silva", "prog@uema.br", "(98) 2016-8100"],
    ["PROEXAE", "Pró-Reitor de Extensão", "Prof. Dra. Beltrana Costa", "proexae@uema.br", "(98) 2016-8200"],
    ["PRPPG", "Pró-Reitor de Pesquisa", "Prof. Dr. Ciclano Pereira", "prppg@uema.br", "(98) 2016-8300"],
    ["CTIC", "Diretor", "Eng. João Técnico", "ctic@uema.br", "(98) 2016-8400"],
    ["CTIC", "Suporte TI", "Equipe Suporte", "suporte.ctic@uema.br", "(98) 2016-8401"],
    ["CECEN", "Diretor", "Prof. Dr. Roberto Melo", "cecen@uema.br", "(98) 2016-8500"],
    ["CESB", "Diretor", "Prof. Dra. Maria Santos", "cesb@uema.br", "(98) 2016-8600"],
    ["Eng. Civil", "Coordenador", "Prof. Dr. Paulo Lima", "coord.civil@uema.br", "(98) 2016-8700"],
    ["Sistemas Info.", "Coordenador", "Prof. Dra. Ana Carvalho", "coord.sis@uema.br", "(98) 2016-8800"],
    ["Medicina", "Coordenador", "Prof. Dr. Carlos Médico", "coord.med@uema.br", "(98) 2016-8900"],
]


def gerar_csvs() -> None:
    print("\n📊 Gerando CSVs de teste...")

    # CSV de vagas mock
    caminho_vagas = RAIZ / "dados" / "CSV" / "testes" / "vagas_mock_2026.csv"
    with caminho_vagas.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerows(_CSV_VAGAS)
    print(f"  ✅ CSV gerado: {caminho_vagas.relative_to(RAIZ)} ({len(_CSV_VAGAS)-1} linhas)")

    # CSV de contatos mock
    caminho_contatos = RAIZ / "dados" / "CSV" / "testes" / "contatos_mock.csv"
    with caminho_contatos.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerows(_CSV_CONTATOS)
    print(f"  ✅ CSV gerado: {caminho_contatos.relative_to(RAIZ)} ({len(_CSV_CONTATOS)-1} linhas)")


# =============================================================================
# 5. ACTUALIZAR DOCUMENT_CONFIG
# =============================================================================

_NOVOS_CONFIGS = """
    # ── PDFs de TESTE (gerados por setup_projeto.py) ──────────────────────────
    "agente_rag_uema_spec.pdf": {{
        "doc_type":   "geral",
        "titulo":     "Especificação Técnica Bot UEMA v5",
        "chunk_size": 400,
        "overlap":    60,
        "label":      "ESPECIFICAÇÃO BOT UEMA v5",
        "parser":     "pymupdf",
    }},
    "instrucoes_uso_agente.pdf": {{
        "doc_type":   "geral",
        "titulo":     "Manual de Uso e Comandos Bot UEMA",
        "chunk_size": 350,
        "overlap":    50,
        "label":      "MANUAL USO BOT UEMA",
        "parser":     "pymupdf",
    }},

    # ── CSVs de TESTE (mock data gerados por setup_projeto.py) ────────────────
    "vagas_mock_2026.csv": {{
        "doc_type":   "edital",
        "titulo":     "Vagas Mock PAES 2026 (Teste)",
        "chunk_size": 300,
        "overlap":    40,
        "label":      "VAGAS MOCK PAES 2026",
    }},
    "contatos_mock.csv": {{
        "doc_type":   "contatos",
        "titulo":     "Contatos Mock UEMA (Teste)",
        "chunk_size": 250,
        "overlap":    30,
        "label":      "CONTATOS MOCK UEMA",
    }},
"""


def mostrar_document_config() -> None:
    """Mostra o snippet a adicionar ao DOCUMENT_CONFIG."""
    print("\n📋 Adiciona ao DOCUMENT_CONFIG em src/rag/ingestion.py:")
    print("─" * 60)
    print(_NOVOS_CONFIGS)
    print("─" * 60)


# =============================================================================
# 6. REDIS — instrução para limpar
# =============================================================================

def instrucao_limpar_redis() -> None:
    print("\n🔴 PARA LIMPAR O REDIS:")
    print("─" * 60)
    print("  # Limpa TODOS os dados (dev + produção):")
    print("  redis-cli FLUSHDB")
    print()
    print("  # Limpa APENAS os dados de teste (DB 3):")
    print("  redis-cli -n 3 FLUSHDB")
    print()
    print("  # Via Docker:")
    print("  docker exec -it redis redis-cli FLUSHDB")
    print()
    print("  # Após limpar, re-ingestão automática no próximo startup:")
    print("  docker-compose restart bot worker")
    print("─" * 60)


# =============================================================================
# Runner
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description="Setup do meuBotRAG")
    parser.add_argument("--so-pastas",    action="store_true")
    parser.add_argument("--so-envs",      action="store_true")
    parser.add_argument("--so-dados",     action="store_true")
    parser.add_argument("--limpar-redis", action="store_true")
    args = parser.parse_args()

    print("\n🚀 meuBotRAG — Setup Inicial")
    print("=" * 60)

    if args.limpar_redis:
        instrucao_limpar_redis()
        return

    tudo = not (args.so_pastas or args.so_envs or args.so_dados)

    if tudo or args.so_pastas:
        criar_pastas()

    if tudo or args.so_envs:
        criar_envs()

    if tudo or args.so_dados:
        gerar_csvs()
        gerar_pdfs()
        mostrar_document_config()

    print("\n" + "=" * 60)
    print("✅ Setup concluído!")
    print()
    print("Próximos passos:")
    print("  1. Edita .env.local com a tua GEMINI_API_KEY e ADMIN_NUMBERS")
    print("  2. Adiciona o snippet do DOCUMENT_CONFIG ao ingestion.py")
    print("  3. Inicia o Redis: docker run -p 6379:6379 redis/redis-stack:latest")
    print("  4. Testa o scraper: python tests/test_wiki_scraper.py")
    print("  5. Testa a pipeline: python tests/test_pipeline_admin.py")
    print("=" * 60)


if __name__ == "__main__":
    main()