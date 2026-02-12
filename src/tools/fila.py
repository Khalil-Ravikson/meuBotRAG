from langchain_core.tools import tool
import json

@tool
def consultar_fila():
    """Consulta a fila de chamados pendentes do usuário."""
    try:
        # Mock de dados
        fila = [
            {"id": 101, "titulo": "Sem internet", "status": "Aberto"},
            {"id": 102, "titulo": "Impressora quebrada", "status": "Em andamento"}
        ]

        resultado = {
            "status": "ok",
            "total": len(fila),
            "chamados": fila
        }
        
        # Retorna STRING
        return json.dumps(resultado, ensure_ascii=False)

    except Exception as e:
        return json.dumps({"status": "erro", "error": str(e)})