"""
api/hub.py — Hub Central do Oráculo UEMA
==========================================
Página de entrada que liga todos os painéis do sistema.
Montado em main.py com prefix="" (raiz) ou prefix="/hub".

URL: http://localhost:9000/
"""
from __future__ import annotations

from fastapi import APIRouter, Request
from fastapi.responses import HTMLResponse

router = APIRouter()

# ─────────────────────────────────────────────────────────────────────────────
# HTML inline — sem dependência de templates externos
# ─────────────────────────────────────────────────────────────────────────────

_HUB_HTML = r"""<!DOCTYPE html>
<html lang="pt-BR">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Oráculo UEMA — Sistema Central</title>
<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link href="https://fonts.googleapis.com/css2?family=Share+Tech+Mono&family=Rajdhani:wght@400;500;600;700&display=swap" rel="stylesheet">
<style>
/* ── Reset & Base ──────────────────────────────────────────── */
*, *::before, *::after { box-sizing: border-box; margin: 0; padding: 0; }

:root {
  --bg:         #050a05;
  --bg2:        #080f08;
  --surface:    #0d180d;
  --border:     #1a3a1a;
  --border2:    #2a5a2a;
  --green:      #00ff41;
  --green-dim:  #00c832;
  --green-dark: #007820;
  --green-glow: rgba(0,255,65,0.15);
  --green-faint:rgba(0,255,65,0.04);
  --amber:      #ffb300;
  --red:        #ff3333;
  --blue:       #00aaff;
  --text:       #b8d4b8;
  --text-dim:   #4a6a4a;
  --mono:       'Share Tech Mono', monospace;
  --sans:       'Rajdhani', sans-serif;
}

html, body {
  min-height: 100vh;
  background: var(--bg);
  color: var(--text);
  font-family: var(--mono);
  overflow-x: hidden;
}

/* ── Scanline overlay ────────────────────────────────────── */
body::before {
  content: '';
  position: fixed; inset: 0;
  background: repeating-linear-gradient(
    0deg,
    transparent,
    transparent 2px,
    rgba(0,0,0,0.08) 2px,
    rgba(0,0,0,0.08) 4px
  );
  pointer-events: none;
  z-index: 9999;
}

/* ── Grid background ─────────────────────────────────────── */
body::after {
  content: '';
  position: fixed; inset: 0;
  background-image:
    linear-gradient(var(--green-faint) 1px, transparent 1px),
    linear-gradient(90deg, var(--green-faint) 1px, transparent 1px);
  background-size: 40px 40px;
  pointer-events: none;
  z-index: 0;
}

/* ── Header ──────────────────────────────────────────────── */
header {
  position: relative; z-index: 10;
  border-bottom: 1px solid var(--border);
  padding: 0 40px;
  display: flex; align-items: center; justify-content: space-between;
  height: 64px;
  background: linear-gradient(90deg, var(--surface), var(--bg2));
}

.logo {
  display: flex; align-items: center; gap: 16px;
}

.logo-icon {
  width: 36px; height: 36px;
  border: 1px solid var(--green);
  display: grid; place-items: center;
  font-size: 18px;
  color: var(--green);
  box-shadow: 0 0 10px var(--green-glow), inset 0 0 10px var(--green-glow);
  animation: blink-box 4s ease-in-out infinite;
}
@keyframes blink-box {
  0%,95%  { box-shadow: 0 0 10px var(--green-glow), inset 0 0 10px var(--green-glow); }
  97%     { box-shadow: 0 0 2px  var(--green-glow), inset 0 0 2px  var(--green-glow); }
  100%    { box-shadow: 0 0 10px var(--green-glow), inset 0 0 10px var(--green-glow); }
}

.logo-text { line-height: 1.1; }
.logo-text h1 {
  font-family: var(--sans); font-size: 20px; font-weight: 700;
  color: var(--green); letter-spacing: 2px; text-transform: uppercase;
}
.logo-text span { font-size: 11px; color: var(--text-dim); letter-spacing: 1px; }

.header-right {
  display: flex; align-items: center; gap: 24px;
}

.status-pill {
  display: flex; align-items: center; gap: 8px;
  font-size: 11px; letter-spacing: 1px;
  color: var(--text-dim);
}
.dot {
  width: 8px; height: 8px; border-radius: 50%;
  background: var(--text-dim);
}
.dot.green { background: var(--green); box-shadow: 0 0 6px var(--green); animation: pulse-dot 2s infinite; }
.dot.amber { background: var(--amber); box-shadow: 0 0 6px var(--amber); }
.dot.red   { background: var(--red);   box-shadow: 0 0 6px var(--red);   }
@keyframes pulse-dot { 0%,100% { opacity:1; } 50% { opacity:.4; } }

.clock { font-size: 12px; color: var(--green-dim); letter-spacing: 1px; }

/* ── Boot banner ─────────────────────────────────────────── */
.boot-banner {
  position: relative; z-index: 10;
  padding: 24px 40px 0;
  overflow: hidden;
}
.boot-line {
  font-size: 11px; color: var(--green-dark); letter-spacing: 1px;
  opacity: 0;
  animation: fade-in-line 0.3s forwards;
}
.boot-line:nth-child(1)  { animation-delay: 0.1s; }
.boot-line:nth-child(2)  { animation-delay: 0.4s; }
.boot-line:nth-child(3)  { animation-delay: 0.7s; }
.boot-line:nth-child(4)  { animation-delay: 1.0s; }
@keyframes fade-in-line {
  from { opacity: 0; transform: translateX(-8px); }
  to   { opacity: 1; transform: translateX(0); }
}

/* ── Main content ────────────────────────────────────────── */
main {
  position: relative; z-index: 10;
  max-width: 1200px; margin: 0 auto;
  padding: 48px 40px 80px;
}

/* ── Section title ───────────────────────────────────────── */
.section-label {
  font-size: 10px; letter-spacing: 3px; color: var(--text-dim);
  text-transform: uppercase; margin-bottom: 24px;
  display: flex; align-items: center; gap: 12px;
}
.section-label::after {
  content: '';
  flex: 1; height: 1px;
  background: linear-gradient(90deg, var(--border), transparent);
}

/* ── Primary cards grid ──────────────────────────────────── */
.primary-grid {
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(280px, 1fr));
  gap: 2px;
  margin-bottom: 2px;
  border: 1px solid var(--border);
}

.card {
  background: var(--bg2);
  padding: 32px 28px;
  cursor: pointer;
  transition: background 0.15s, border-color 0.15s;
  position: relative;
  overflow: hidden;
  text-decoration: none;
  display: block;
  border-right: 1px solid var(--border);
  opacity: 0;
  animation: card-appear 0.5s forwards;
}
.card:last-child { border-right: none; }

.card:nth-child(1) { animation-delay: 1.2s; }
.card:nth-child(2) { animation-delay: 1.4s; }
.card:nth-child(3) { animation-delay: 1.6s; }
.card:nth-child(4) { animation-delay: 1.8s; }

@keyframes card-appear {
  from { opacity: 0; transform: translateY(12px); }
  to   { opacity: 1; transform: translateY(0); }
}

.card::before {
  content: '';
  position: absolute; top: 0; left: 0; right: 0; height: 1px;
  background: transparent;
  transition: background 0.2s;
}
.card:hover { background: var(--surface); }
.card:hover::before { background: var(--green); }

.card-corner {
  position: absolute; top: 12px; right: 12px;
  font-size: 10px; color: var(--text-dim); letter-spacing: 1px;
}

.card-number {
  font-size: 10px; color: var(--text-dim); letter-spacing: 2px;
  margin-bottom: 16px;
}

.card-icon {
  font-size: 28px; margin-bottom: 16px; display: block;
  filter: grayscale(0.3);
}

.card-title {
  font-family: var(--sans);
  font-size: 22px; font-weight: 700; letter-spacing: 1px;
  color: var(--green); margin-bottom: 8px;
  text-transform: uppercase;
}

.card-subtitle {
  font-size: 11px; color: var(--text-dim);
  letter-spacing: 0.5px; margin-bottom: 20px;
  line-height: 1.6;
}

.card-tags {
  display: flex; flex-wrap: wrap; gap: 6px;
}
.tag {
  font-size: 9px; letter-spacing: 1.5px; padding: 3px 8px;
  border: 1px solid var(--border2); color: var(--text-dim);
  text-transform: uppercase;
}

.card-arrow {
  position: absolute; bottom: 28px; right: 28px;
  font-size: 20px; color: var(--border2);
  transition: color 0.2s, transform 0.2s;
}
.card:hover .card-arrow {
  color: var(--green); transform: translate(3px, -3px);
}

/* ── Status row ──────────────────────────────────────────── */
.status-grid {
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(220px, 1fr));
  gap: 1px;
  border: 1px solid var(--border);
  margin-top: 2px;
  opacity: 0;
  animation: card-appear 0.5s 2.2s forwards;
}

.status-item {
  background: var(--bg2);
  padding: 20px 24px;
  display: flex; align-items: center; gap: 16px;
  border-right: 1px solid var(--border);
}
.status-item:last-child { border-right: none; }

.status-info { flex: 1; min-width: 0; }
.status-name {
  font-size: 10px; letter-spacing: 2px; color: var(--text-dim);
  text-transform: uppercase; margin-bottom: 4px;
}
.status-value {
  font-size: 13px; color: var(--text);
  white-space: nowrap; overflow: hidden; text-overflow: ellipsis;
}
.status-value.ok    { color: var(--green); }
.status-value.warn  { color: var(--amber); }
.status-value.error { color: var(--red); }

/* ── Quick links row ─────────────────────────────────────── */
.quick-row {
  display: flex; flex-wrap: wrap; gap: 1px;
  border: 1px solid var(--border);
  margin-top: 2px;
  opacity: 0;
  animation: card-appear 0.5s 2.5s forwards;
}

.quick-link {
  flex: 1; min-width: 140px;
  background: var(--bg2);
  padding: 14px 18px;
  text-decoration: none;
  display: flex; align-items: center; gap: 10px;
  font-size: 11px; color: var(--text-dim);
  letter-spacing: 1px;
  border-right: 1px solid var(--border);
  transition: background 0.15s, color 0.15s;
}
.quick-link:last-child { border-right: none; }
.quick-link:hover { background: var(--surface); color: var(--green); }
.quick-link-icon { font-size: 14px; }

/* ── Terminal line at bottom ─────────────────────────────── */
.terminal-bar {
  position: fixed; bottom: 0; left: 0; right: 0; z-index: 100;
  background: var(--surface);
  border-top: 1px solid var(--border);
  padding: 0 40px;
  height: 36px;
  display: flex; align-items: center; gap: 16px;
  font-size: 11px;
}

.term-prompt { color: var(--green); }
.term-text   { color: var(--text-dim); flex: 1; }
.term-blink {
  display: inline-block; width: 8px; height: 14px;
  background: var(--green);
  animation: caret 1s step-end infinite;
  vertical-align: middle; margin-left: 2px;
}
@keyframes caret { 0%,100% { opacity:1; } 50% { opacity:0; } }

.term-version { color: var(--text-dim); font-size: 10px; letter-spacing: 1px; }

/* ── Responsive ──────────────────────────────────────────── */
@media (max-width: 768px) {
  header { padding: 0 20px; }
  main   { padding: 32px 20px 60px; }
  .boot-banner { padding: 16px 20px 0; }
  .terminal-bar { padding: 0 20px; }
  .card  { padding: 24px 20px; }
  .primary-grid { grid-template-columns: 1fr; }
}
</style>
</head>
<body>

<header>
  <div class="logo">
    <div class="logo-icon">⬡</div>
    <div class="logo-text">
      <h1>Oráculo UEMA</h1>
      <span>SISTEMA CENTRAL DE CONTROLO // v6.0</span>
    </div>
  </div>
  <div class="header-right">
    <div class="status-pill">
      <div class="dot green" id="dot-system"></div>
      <span id="sys-status">VERIFICANDO...</span>
    </div>
    <div class="clock" id="clock">--:--:--</div>
  </div>
</header>

<div class="boot-banner">
  <div class="boot-line">[ BOOT ] Oráculo UEMA v6.0 — Assistente Académico Inteligente</div>
  <div class="boot-line">[ INIT ] Redis Stack ▸ PostgreSQL ▸ Gemini 2.0 Flash ▸ Evolution API</div>
  <div class="boot-line">[ RAG  ] Busca híbrida BM25+Vetor ▸ CRAG ▸ Self-RAG ▸ HyDE Routing</div>
  <div class="boot-line">[ REDY ] Todos os subsistemas activos. Seleccione um módulo abaixo.</div>
</div>

<main>

  <div class="section-label">// módulos principais</div>

  <div class="primary-grid">

    <a class="card" href="/monitor/" target="_blank">
      <div class="card-corner">01</div>
      <div class="card-number">// MONITOR</div>
      <span class="card-icon">📊</span>
      <div class="card-title">Dashboard</div>
      <div class="card-subtitle">
        Métricas de conversas em tempo real.<br>
        Tokens, latência, rotas RAG e logs do sistema.
      </div>
      <div class="card-tags">
        <span class="tag">SSE Live</span>
        <span class="tag">Redis</span>
        <span class="tag">Polling 30s</span>
      </div>
      <span class="card-arrow">↗</span>
    </a>

    <a class="card" href="/eval/" target="_blank">
      <div class="card-corner">02</div>
      <div class="card-number">// AVALIAÇÃO RAG</div>
      <span class="card-icon">🔬</span>
      <div class="card-title">RAG Eval</div>
      <div class="card-subtitle">
        Terminal ao vivo da pipeline RAG.<br>
        Chunks recuperados, CRAG score, query transform.
      </div>
      <div class="card-tags">
        <span class="tag">Pipeline</span>
        <span class="tag">Logs SSE</span>
        <span class="tag">Gemini</span>
      </div>
      <span class="card-arrow">↗</span>
    </a>

    <a class="card" href="/admin/" target="_blank">
      <div class="card-corner">03</div>
      <div class="card-number">// ADMINISTRAÇÃO</div>
      <span class="card-icon">⚙️</span>
      <div class="card-title">Admin Portal</div>
      <div class="card-subtitle">
        Gestão completa do sistema.<br>
        Redis, PostgreSQL, ingestão, memória, config.
      </div>
      <div class="card-tags">
        <span class="tag">RBAC</span>
        <span class="tag">30 endpoints</span>
        <span class="tag">Auth</span>
      </div>
      <span class="card-arrow">↗</span>
    </a>

    <a class="card" href="/docs" target="_blank">
      <div class="card-corner">04</div>
      <div class="card-number">// API</div>
      <span class="card-icon">📡</span>
      <div class="card-title">API Docs</div>
      <div class="card-subtitle">
        Swagger UI automático do FastAPI.<br>
        Todos os endpoints REST documentados.
      </div>
      <div class="card-tags">
        <span class="tag">OpenAPI 3</span>
        <span class="tag">Swagger UI</span>
        <span class="tag">FastAPI</span>
      </div>
      <span class="card-arrow">↗</span>
    </a>

  </div>

  <!-- Status row -->
  <div class="section-label" style="margin-top:40px">// estado dos serviços</div>

  <div class="status-grid" id="status-grid">
    <div class="status-item">
      <div class="dot" id="dot-redis"></div>
      <div class="status-info">
        <div class="status-name">Redis Stack</div>
        <div class="status-value" id="val-redis">--</div>
      </div>
    </div>
    <div class="status-item">
      <div class="dot" id="dot-postgres"></div>
      <div class="status-info">
        <div class="status-name">PostgreSQL</div>
        <div class="status-value" id="val-postgres">--</div>
      </div>
    </div>
    <div class="status-item">
      <div class="dot" id="dot-agent"></div>
      <div class="status-info">
        <div class="status-name">Agente RAG</div>
        <div class="status-value" id="val-agent">--</div>
      </div>
    </div>
    <div class="status-item">
      <div class="dot" id="dot-gemini"></div>
      <div class="status-info">
        <div class="status-name">Modelo LLM</div>
        <div class="status-value" id="val-gemini">--</div>
      </div>
    </div>
    <div class="status-item">
      <div class="dot" id="dot-evolution"></div>
      <div class="status-info">
        <div class="status-name">WhatsApp API</div>
        <div class="status-value" id="val-evolution">--</div>
      </div>
    </div>
  </div>

  <!-- Quick links -->
  <div class="section-label" style="margin-top:40px">// acesso rápido</div>

  <div class="quick-row">
    <a class="quick-link" href="/health">
      <span class="quick-link-icon">💚</span> /health — JSON saúde
    </a>
    <a class="quick-link" href="/metrics">
      <span class="quick-link-icon">📈</span> /metrics — métricas
    </a>
    <a class="quick-link" href="/logs">
      <span class="quick-link-icon">📋</span> /logs — erros recentes
    </a>
    <a class="quick-link" href="/banco/sources">
      <span class="quick-link-icon">🗃️</span> /banco/sources — chunks RAG
    </a>
    <a class="quick-link" href="/redoc">
      <span class="quick-link-icon">📖</span> /redoc — API ReDoc
    </a>
    <a class="quick-link" href="/pessoas/">
      <span class="quick-link-icon">👥</span> /pessoas — utilizadores
    </a>
  </div>

</main>

<!-- Terminal bar -->
<div class="terminal-bar">
  <span class="term-prompt">oraculo@uema:~$</span>
  <span class="term-text" id="term-msg">sistema pronto — seleccione um módulo</span>
  <span class="term-blink"></span>
  <span class="term-version">UEMA // CTIC // v6.0</span>
</div>

<script>
// ── Clock ─────────────────────────────────────────────────
function updateClock() {
  const now = new Date();
  document.getElementById('clock').textContent =
    now.toLocaleTimeString('pt-BR', { hour12: false });
}
updateClock();
setInterval(updateClock, 1000);

// ── Health check ──────────────────────────────────────────
const COLOR = { ok:'green', warn:'amber', error:'red' };

function setStatus(dotId, valId, ok, text) {
  const dot = document.getElementById(dotId);
  const val = document.getElementById(valId);
  if (!dot || !val) return;
  dot.className = 'dot ' + (ok ? 'green' : 'red');
  val.textContent = text;
  val.className   = 'status-value ' + (ok ? 'ok' : 'error');
}

async function checkHealth() {
  try {
    const r    = await fetch('/health');
    const data = await r.json();

    setStatus('dot-redis',    'val-redis',    data.redis,    data.redis    ? 'ONLINE' : 'OFFLINE');
    setStatus('dot-postgres', 'val-postgres', data.postgres, data.postgres ? 'ONLINE' : 'OFFLINE');
    setStatus('dot-agent',    'val-agent',    data.agente,   data.agente   ? 'PRONTO' : 'INICIANDO...');
    setStatus('dot-gemini',   'val-gemini',   true,          data.modelo || 'gemini-2.0-flash');
    setStatus('dot-evolution','val-evolution',true,          'Evolution API');

    // Header dot
    const allOk = data.redis && data.postgres && data.agente;
    const sysDot = document.getElementById('dot-system');
    const sysStatus = document.getElementById('sys-status');
    sysDot.className = 'dot ' + (allOk ? 'green' : (data.redis ? 'amber' : 'red'));
    sysStatus.textContent = allOk ? 'SISTEMAS OPERACIONAIS' : (data.agente ? 'DEGRADADO' : 'INICIANDO');

    // Terminal message
    const msgs = {
      true:  'todos os subsistemas operacionais — pronto para receber mensagens WhatsApp',
      false: 'atenção: um ou mais serviços offline — verificar docker-compose logs',
    };
    document.getElementById('term-msg').textContent = msgs[String(allOk)];

  } catch(e) {
    setStatus('dot-redis',    'val-redis',    false, 'ERRO DE CONEXÃO');
    setStatus('dot-postgres', 'val-postgres', false, 'ERRO DE CONEXÃO');
    setStatus('dot-agent',    'val-agent',    false, 'ERRO DE CONEXÃO');
    document.getElementById('dot-system').className = 'dot red';
    document.getElementById('sys-status').textContent = 'OFFLINE';
    document.getElementById('term-msg').textContent = 'erro: servidor não responde — verificar se docker está a correr';
  }
}

// ── Card hover terminal feedback ──────────────────────────
document.querySelectorAll('.card').forEach(card => {
  const title = card.querySelector('.card-title')?.textContent?.toLowerCase() || '';
  const url   = card.getAttribute('href') || '';
  card.addEventListener('mouseenter', () => {
    document.getElementById('term-msg').textContent =
      `navegando para ${url} — ${title}`;
  });
  card.addEventListener('mouseleave', () => {
    document.getElementById('term-msg').textContent =
      'sistema pronto — seleccione um módulo';
  });
});

document.querySelectorAll('.quick-link').forEach(link => {
  const url = link.getAttribute('href') || '';
  link.addEventListener('mouseenter', () => {
    document.getElementById('term-msg').textContent = `GET ${url}`;
  });
  link.addEventListener('mouseleave', () => {
    document.getElementById('term-msg').textContent = 'sistema pronto — seleccione um módulo';
  });
});

// ── Boot ──────────────────────────────────────────────────
setTimeout(checkHealth, 2200);
setInterval(checkHealth, 30000);
</script>
</body>
</html>
"""


@router.get("/hub", response_class=HTMLResponse, include_in_schema=False)
async def hub(request: Request):
    return HTMLResponse(_HUB_HTML)