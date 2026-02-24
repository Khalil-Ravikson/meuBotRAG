#!/usr/bin/env python3
"""
setup_estrutura.py — Cria a nova estrutura de pastas
=====================================================
Execute da RAIZ do projeto:
    python setup_estrutura.py

Cria pastas + __init__.py. Não apaga nada existente.
"""
from pathlib import Path

ROOT = Path(__file__).parent

PASTAS = [
    "src/api", "src/application", "src/agent", "src/rag",
    "src/domain", "src/tools", "src/services", "src/providers",
    "src/infrastructure", "src/memory", "src/cache", "src/middleware",
    "debug", "tests/unit", "tests/integration", "tests/e2e",
]

INITS = [
    "src/__init__.py", "src/api/__init__.py", "src/application/__init__.py",
    "src/agent/__init__.py", "src/rag/__init__.py", "src/domain/__init__.py",
    "src/services/__init__.py", "src/providers/__init__.py",
    "src/infrastructure/__init__.py", "src/memory/__init__.py",
    "src/cache/__init__.py", "src/middleware/__init__.py",
    "tests/__init__.py", "tests/unit/__init__.py",
    "tests/integration/__init__.py", "tests/e2e/__init__.py",
]

TOOLS_INIT = """\
\"\"\"
tools/__init__.py — Exporta as tools ativas para o AgentCore.
Para adicionar uma nova tool: importe e adicione à lista get_tools_ativas().
\"\"\"
from src.tools.tool_calendario import get_tool_calendario
from src.tools.tool_edital     import get_tool_edital
from src.tools.tool_contatos   import get_tool_contatos


def get_tools_ativas() -> list:
    return [
        get_tool_calendario(),
        get_tool_edital(),
        get_tool_contatos(),
        # get_tool_email(),   # descomente quando implementar
        # get_tool_glpi(),    # descomente quando implementar
    ]
"""

def main():
    print("🏗  Criando estrutura...\n")
    for pasta in PASTAS:
        p = ROOT / pasta
        p.mkdir(parents=True, exist_ok=True)
        print(f"  📁 {pasta}/")

    print("\n🔧 Criando __init__.py...\n")
    for init in INITS:
        p = ROOT / init
        if not p.exists():
            p.write_text("", encoding="utf-8")
            print(f"  📄 CRIADO: {init}")
        else:
            print(f"  ✅ existe: {init}")

    # tools/__init__.py especial
    tools_init_path = ROOT / "src/tools/__init__.py"
    if not tools_init_path.exists() or tools_init_path.read_text() == "":
        tools_init_path.write_text(TOOLS_INIT, encoding="utf-8")
        print("  📄 CRIADO: src/tools/__init__.py (com get_tools_ativas)")

    print("""
═══════════════════════════════════════════════
✅ Estrutura criada!

PRÓXIMOS PASSOS:
  1. Copie os arquivos gerados para as pastas corretas
  2. Rode os unit tests (sem Docker):
       pytest tests/unit/ -v
  3. Migre rag_service.py → rag/ingestor.py
  4. Adapte dev_guard.py para usar infrastructure/redis_client.py
═══════════════════════════════════════════════
""")

if __name__ == "__main__":
    main()