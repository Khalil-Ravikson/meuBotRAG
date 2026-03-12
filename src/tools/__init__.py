"""
tools/__init__.py — Registo das tools activas (v4)
====================================================
 
TOOLS DISPONÍVEIS:
───────────────────
  consultar_calendario_academico  → datas, prazos, semestres
  consultar_edital_paes_2026      → vagas, cotas, inscrição
  consultar_contatos_uema         → e-mails, telefones, setores
  consultar_wiki_ctic             → sistemas TI, suporte, SIGAA, redes  ← NOVO v4
 
COMO ADICIONAR NOVA TOOL:
──────────────────────────
  1. Cria src/tools/tool_nome.py com get_tool_nome()
  2. Importa aqui e adiciona à lista em get_tools_ativas()
  3. O SemanticRouter usa a descrição da @tool para routing vectorial —
     escreve uma descrição clara com casos de uso e palavras-chave.
  4. Adiciona Rota.NOME em domain/entities.py
  5. Adiciona mapeamento em _TOOL_PARA_ROTA no semantic_router.py
  6. Adiciona mapeamento em _ROTA_PARA_SOURCE no agent/core.py
"""
from src.tools.calendar_tool    import get_tool_calendario
from src.tools.tool_edital      import get_tool_edital
from src.tools.tool_contatos    import get_tool_contatos
from src.tools.tool_wiki_ctic   import get_tool_wiki_ctic    # ← NOVO v4
 
# Descomenta quando implementares:
# from src.tools.tool_email import get_tool_email
# from src.tools.tool_glpi  import get_tool_glpi
 
 
def get_tools_ativas() -> list:
    """
    Retorna lista de tools instanciadas para o AgentCore / SemanticRouter.
    O SemanticRouter embeda a descrição de cada tool no startup.
    """
    return [
        get_tool_calendario(),
        get_tool_edital(),
        get_tool_contatos(),
        get_tool_wiki_ctic(),   # ← NOVO v4
    ]