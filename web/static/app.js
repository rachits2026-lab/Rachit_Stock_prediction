// Shared helpers for every page.

async function api(path, options = {}) {
  const res = await fetch(path, {
    headers: { "Content-Type": "application/json" },
    ...options,
    body: options.body ? JSON.stringify(options.body) : undefined,
  });
  if (res.status === 401 && !location.pathname.startsWith("/login") && !location.pathname.startsWith("/signup")) {
    location.href = "/login";
    throw new Error("Please log in.");
  }
  const data = await res.json().catch(() => ({}));
  if (!res.ok) throw new Error(data.detail?.[0]?.msg || data.detail || "Something went wrong.");
  return data;
}

const byId = (id) => document.getElementById(id);

function esc(s) {
  return String(s ?? "").replace(/[&<>"']/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));
}

const pct = (x, digits = 0) => (x == null ? "—" : `${(x * 100).toFixed(digits)}%`);

function money(x, currency = "USD") {
  if (x == null) return "—";
  return new Intl.NumberFormat("en-US", { style: "currency", currency, maximumFractionDigits: 2 }).format(x);
}

function bigMoney(x, currency = "USD") {
  if (x == null) return "—";
  return new Intl.NumberFormat("en-US", { style: "currency", currency, notation: "compact", maximumFractionDigits: 2 }).format(x);
}

async function renderNav() {
  const el = document.getElementById("nav-user");
  if (!el) return;
  try {
    const me = await api("/api/me");
    el.innerHTML = `<span class="email">${esc(me.email)}</span>
      <a href="/settings">Alerts</a>
      <a href="#" id="logout">Log out</a>`;
    document.getElementById("logout").onclick = async (e) => {
      e.preventDefault();
      await api("/api/auth/logout", { method: "POST" });
      location.href = "/login";
    };
  } catch {}
}

function companyCard(c) {
  return `<a class="company" href="/company/${encodeURIComponent(c.ticker)}">
    <div style="min-width:0"><div class="t">${esc(c.ticker)}</div><div class="n">${esc(c.name)}</div></div>
    <span class="tag">${esc(c.sector || c.country || "")}</span></a>`;
}

// Inline SVG of the white teddy bear mascot.
const BEAR_SVG = `<svg viewBox="0 0 200 220" xmlns="http://www.w3.org/2000/svg" role="img" aria-label="White teddy bear mascot">
  <g stroke="#cfd6e0" stroke-width="3">
    <circle cx="52" cy="48" r="26" fill="#ffffff"/><circle cx="148" cy="48" r="26" fill="#ffffff"/>
    <circle cx="52" cy="48" r="13" fill="#fbd5dc" stroke="none"/><circle cx="148" cy="48" r="13" fill="#fbd5dc" stroke="none"/>
    <ellipse cx="100" cy="170" rx="62" ry="48" fill="#ffffff"/>
    <ellipse cx="38" cy="150" rx="17" ry="28" fill="#ffffff" transform="rotate(25 38 150)"/>
    <ellipse cx="168" cy="118" rx="16" ry="28" fill="#ffffff" transform="rotate(-40 168 118)"/>
    <ellipse cx="66" cy="208" rx="22" ry="12" fill="#ffffff"/><ellipse cx="134" cy="208" rx="22" ry="12" fill="#ffffff"/>
    <circle cx="100" cy="88" r="56" fill="#ffffff"/>
  </g>
  <ellipse cx="100" cy="172" rx="34" ry="28" fill="#f3f5f9"/>
  <ellipse cx="100" cy="106" rx="24" ry="18" fill="#f3f5f9"/>
  <circle cx="78" cy="80" r="6" fill="#1f2937"/><circle cx="122" cy="80" r="6" fill="#1f2937"/>
  <circle cx="80" cy="78" r="2" fill="#fff"/><circle cx="124" cy="78" r="2" fill="#fff"/>
  <ellipse cx="100" cy="98" rx="9" ry="6.5" fill="#1f2937"/>
  <path d="M100 104 v6 M88 110 q12 10 24 0" stroke="#1f2937" stroke-width="3" fill="none" stroke-linecap="round"/>
  <circle cx="64" cy="98" r="7" fill="#fbd5dc" opacity=".8"/><circle cx="136" cy="98" r="7" fill="#fbd5dc" opacity=".8"/>
  <path d="M86 150 l-14 -8 v16 z M114 150 l14 -8 v16 z" fill="#2563eb"/><circle cx="100" cy="150" r="6" fill="#1d4ed8"/>
</svg>`;

document.addEventListener("DOMContentLoaded", () => {
  renderNav();
  document.querySelectorAll("[data-bear]").forEach((el) => (el.innerHTML = BEAR_SVG));
});
