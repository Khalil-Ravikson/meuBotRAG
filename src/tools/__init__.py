"""
tools/__init__.py — Registro das tools ativas do agente
========================================================
Para adicionar uma nova tool:
  1. Crie o arquivo src/tools/tool_nome.py
  2. Importe a fábrica aqui
  3. Adicione à lista em get_tools_ativas()

O AgentCore chama get_tools_ativas() no startup.
"""
from src.tools.calendar_tool import get_tool_calendario
from src.tools.tool_edital     import get_tool_edital
from src.tools.tool_contatos   import get_tool_contatos
# from src.tools.tool_email import get_tool_email    # descomente quando implementar
# from src.tools.tool_glpi  import get_tool_glpi     # descomente quando implementar


def get_tools_ativas() -> list:
    """Retorna lista de tools instanciadas para o AgentExecutor."""
    return [
        get_tool_calendario(),
        get_tool_edital(),
        get_tool_contatos(),
    ]