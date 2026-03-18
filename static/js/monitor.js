/**
 * monitor.js — Dashboard de Monitoramento Bot UEMA v5
 * =====================================================
 * Faz polling a /monitor/data a cada 30s e actualiza o DOM sem reload.
 * Sem dependências externas — vanilla JS puro.
 */

const POLL_INTERVAL = 30_000; // 30 segundos
let   pollTimer     = null;

// ── Utilitários ──────────────────────────────────────────────────────────────

const $ = (sel, ctx = document) => ctx.querySelector(sel);
const $$ = (sel, ctx = document) => [...ctx.querySelectorAll(sel)];

function fmt(n) {
  if (n >= 1_000_000) return (n / 1_000_000).toFixed(1) + "M";
  if (n >= 1_000)     return (n / 1_000).toFixed(1) + "k";
  return String(n);
}

function ts(isoStr) {
  try {
    return new Date(isoStr).toLocaleString("pt-BR", {
      day: "2-digit", month: "2-digit",
      hour: "2-digit", minute: "2-digit", second: "2-digit",
    });
  } catch { return isoStr?.slice(0, 19) ?? "—"; }
}

function showToast(msg = "Actualizado!") {
  const el = $("#toast");
  el.textContent = "✅ " + msg;
  el.classList.add("show");
  setTimeout(() => el.classList.remove("show"), 2500);
}

// ── Fetch de dados ───────────────────────────────────────────────────────────

async function fetchData() {
  try {
    const res  = await fetch("/monitor/data?limit=100");
    if (!res.ok) throw new Error(`HTTP ${res.status}`);
    const data = await res.json();
    renderAll(data);
    showToast("Dados actualizados");
  } catch (err) {
    console.error("Monitor fetch error:", err);
  }
}

// ── Render principal ─────────────────────────────────────────────────────────

function renderAll(data) {
  renderCards(data);
  renderTabela(data.logs ?? []);
  renderBarChart("chart-niveis", data.niveis ?? {}, {
    ADMIN:   "#f97316",
    STUDENT: "#06b6d4",
    GUEST:   "#64748b",
  });
  renderBarChart("chart-rotas", data.rotas ?? {}, {
    CALENDARIO: "#6366f1",
    EDITAL:     "#a855f7",
    CONTATOS:   "#22c55e",
    WIKI:       "#eab308",
    GERAL:      "#64748b",
  });
  renderErros(data.erros ?? []);

  const updEl = $("#updated-at");
  if (updEl) updEl.textContent = ts(data.updated_at);
}

// ── Cards de métricas ─────────────────────────────────────────────────────────

function renderCards(data) {
  const map = {
    "card-msgs":    fmt(data.total_msgs ?? 0),
    "card-tokens":  fmt(data.total_tokens ?? 0),
    "card-lat":     (data.avg_latencia ?? 0) + "ms",
    "card-admin":   fmt(data.niveis?.ADMIN   ?? 0),
    "card-student": fmt(data.niveis?.STUDENT ?? 0),
    "card-guest":   fmt(data.niveis?.GUEST   ?? 0),
  };
  for (const [id, val] of Object.entries(map)) {
    const el = $(`#${id} .value`);
    if (el) el.textContent = val;
  }
}

// ── Tabela de logs ────────────────────────────────────────────────────────────

function renderTabela(logs) {
  const tbody = $("#log-tbody");
  if (!tbody) return;

  if (!logs.length) {
    tbody.innerHTML = `<tr><td colspan="7" class="empty">Sem dados ainda</td></tr>`;
    return;
  }

  tbody.innerHTML = logs.map(l => `
    <tr>
      <td class="mono">${ts(l.ts)}</td>
      <td class="mono" title="${l.user_id}">${l.user_id?.slice(0, 15) ?? "—"}</td>
      <td><span class="badge badge-${l.nivel ?? "GUEST"}">${l.nivel ?? "GUEST"}</span></td>
      <td style="text-align:right">${fmt(l.tokens_total ?? 0)}</td>
      <td style="text-align:right">${l.latencia_ms ?? 0}ms</td>
      <td><span class="rota-badge">${l.rota ?? "—"}</span></td>
      <td style="color:var(--clr-muted);font-size:12px" title="${escapeHtml(l.pergunta ?? '')}">
        ${escapeHtml((l.pergunta ?? "").slice(0, 55))}${(l.pergunta?.length ?? 0) > 55 ? "…" : ""}
      </td>
    </tr>
  `).join("");
}

// ── Bar chart (níveis e rotas) ────────────────────────────────────────────────

function renderBarChart(containerId, data, colorMap = {}) {
  const container = $(`#${containerId}`);
  if (!container) return;

  const entries = Object.entries(data).sort((a, b) => b[1] - a[1]);
  const maxVal  = entries[0]?.[1] ?? 1;

  if (!entries.length) {
    container.innerHTML = `<p class="empty">Sem dados</p>`;
    return;
  }

  container.innerHTML = `<div class="bar-chart">${entries.map(([key, val]) => `
    <div class="bar-row">
      <span class="bar-label">${key}</span>
      <div class="bar-track">
        <div class="bar-fill" style="width:${(val/maxVal*100).toFixed(1)}%;background:${colorMap[key] ?? "var(--clr-primary)"}"></div>
      </div>
      <span class="bar-count">${val}</span>
    </div>
  `).join("")}</div>`;
}

// ── Erros recentes ────────────────────────────────────────────────────────────

function renderErros(erros) {
  const el = $("#erros-tbody");
  if (!el) return;

  if (!erros.length) {
    el.innerHTML = `<tr><td colspan="3" class="empty">Nenhum erro recente ✅</td></tr>`;
    return;
  }

  el.innerHTML = erros.map(e => `
    <tr>
      <td class="mono">${ts(e.ts)}</td>
      <td class="mono">${escapeHtml(e.context ?? "")}</td>
      <td style="font-size:12px;color:var(--clr-red)">${escapeHtml((e.msg ?? "").slice(0, 100))}</td>
    </tr>
  `).join("");
}

// ── XSS guard ────────────────────────────────────────────────────────────────

function escapeHtml(str) {
  return String(str)
    .replace(/&/g, "&amp;")
    .replace(/</g, "&lt;")
    .replace(/>/g, "&gt;")
    .replace(/"/g, "&quot;");
}

// ── Polling ───────────────────────────────────────────────────────────────────

function startPolling() {
  fetchData(); // imediato
  pollTimer = setInterval(fetchData, POLL_INTERVAL);
}

function stopPolling() {
  if (pollTimer) clearInterval(pollTimer);
}

// Actualização manual
document.addEventListener("DOMContentLoaded", () => {
  startPolling();

  const btnRefresh = $("#btn-refresh");
  if (btnRefresh) {
    btnRefresh.addEventListener("click", () => {
      stopPolling();
      startPolling();
    });
  }
});