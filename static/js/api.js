/* ============================================================================
   AI TRADER — API layer + shared utilities
   Single place that talks to the backend. Every fetch returns a typed
   result object so pages can render honest loading / error / empty states.
   ========================================================================== */
(function () {
  "use strict";

  /* ---------------- utilities ---------------- */

  const U = {};

  U.esc = function (s) {
    return String(s == null ? "" : s).replace(/[&<>"']/g, function (c) {
      return { "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c];
    });
  };

  U.fmtMoney = function (v, opts) {
    opts = opts || {};
    if (v == null || isNaN(v)) return opts.dash ? "—" : "$0.00";
    const abs = Math.abs(v);
    const s = abs.toLocaleString("en-US", {
      minimumFractionDigits: opts.dec != null ? opts.dec : 2,
      maximumFractionDigits: opts.dec != null ? opts.dec : 2,
    });
    return (v < 0 ? "-$" : "$") + s;
  };

  U.fmtSigned = function (v, opts) {
    opts = opts || {};
    if (v == null || isNaN(v)) return "—";
    const s = U.fmtMoney(Math.abs(v), opts).slice(1);
    return (v > 0 ? "+$" : v < 0 ? "-$" : "$") + s;
  };

  U.fmtPct = function (v, dec) {
    if (v == null || isNaN(v)) return "—";
    dec = dec == null ? 2 : dec;
    const s = Math.abs(v).toFixed(dec);
    return (v > 0 ? "+" : v < 0 ? "-" : "") + s + "%";
  };

  U.fmtNum = function (v, dec) {
    if (v == null || isNaN(v)) return "—";
    return Number(v).toLocaleString("en-US", {
      minimumFractionDigits: dec || 0,
      maximumFractionDigits: dec || 0,
    });
  };

  U.fmtQty = function (v) {
    if (v == null || isNaN(v)) return "—";
    return Number.isInteger(Number(v)) ? String(Math.round(v)) : Number(v).toFixed(4).replace(/0+$/, "").replace(/\.$/, "");
  };

  U.fmtTime = function (iso) {
    if (!iso) return "—";
    const d = U.parseDate(iso);
    if (!d) return "—";
    return d.toLocaleTimeString("en-US", { hour12: false });
  };

  U.fmtDate = function (iso) {
    if (!iso) return "—";
    const d = U.parseDate(iso);
    if (!d) return "—";
    return d.toLocaleDateString("en-US", { month: "short", day: "numeric", year: "numeric" });
  };

  U.fmtDateTime = function (iso) {
    if (!iso) return "—";
    return U.fmtDate(iso) + " " + U.fmtTime(iso);
  };

  // Alpaca history timestamps are epoch seconds without tz marker.
  U.parseDate = function (ts) {
    if (ts instanceof Date) return ts;
    if (typeof ts === "number") return new Date(ts < 1e12 ? ts * 1000 : ts);
    if (!ts) return null;
    const n = Number(ts);
    if (!isNaN(n) && /^\d{10}(\.\d+)?$/.test(String(ts))) return new Date(n * 1000);
    const d = new Date(String(ts).replace(/Z$/, "") + "Z");
    return isNaN(d.getTime()) ? new Date(ts) : d;
  };

  U.ago = function (iso, nowMs) {
    if (!iso) return null;
    const d = U.parseDate(iso);
    if (!d) return null;
    const s = Math.max(0, ((nowMs || Date.now()) - d.getTime()) / 1000);
    if (s < 5) return "just now";
    if (s < 60) return Math.floor(s) + "s ago";
    if (s < 3600) return Math.floor(s / 60) + "m ago";
    if (s < 86400) return Math.floor(s / 3600) + "h ago";
    return Math.floor(s / 86400) + "d ago";
  };

  U.classFor = function (v) {
    if (v == null || isNaN(v) || Number(v) === 0) return "neu";
    return v > 0 ? "pos" : "neg";
  };
  U.signFor = function (v) {
    if (v == null || isNaN(v) || Number(v) === 0) return "neutral";
    return v > 0 ? "positive" : "negative";
  };

  U.clamp = function (v, a, b) { return Math.min(b, Math.max(a, v)); };

  U.debounce = function (fn, ms) {
    let t;
    return function () {
      const args = arguments, self = this;
      clearTimeout(t);
      t = setTimeout(function () { fn.apply(self, args); }, ms);
    };
  };

  // ---- status / signal semantics ----

  U.badgeForDecision = function (d) {
    if (!d) return "badge badge-neutral";
    const k = String(d).toUpperCase();
    if (k === "BUY" || k === "BULLISH" || k === "APPROVED") return "badge badge-buy";
    if (k === "SELL" || k === "BEARISH" || k === "REJECTED") return "badge badge-sell";
    return "badge badge-hold";
  };
  U.classForDecision = function (d) {
    const k = String(d || "").toUpperCase();
    if (k === "BUY" || k === "BULLISH") return "pos";
    if (k === "SELL" || k === "BEARISH") return "neg";
    return "neu";
  };

  U.badgeForOrderStatus = function (s) {
    if (!s) return "badge";
    const k = String(s).toUpperCase().replace(/_/g, " ");
    if (k === "FILLED" || k === "COMPLETE") return "badge badge-filled";
    if (k === "PARTIALLY FILLED") return "badge badge-filled";
    if (k === "NEW" || k === "ACCEPTED" || k === "PENDING NEW" || k === "PENDING" || k === "ACCEPTED FOR BIDDING" || k === "PLAN") return "badge badge-pending";
    if (k === "CANCELED" || k === "CANCELLED" || k === "EXPIRED" || k === "REPLACED") return "badge badge-neutral";
    if (k === "REJECTED" || k === "SUSPENDED" || k === "STOPPED" || k === "CALCULATED") return "badge badge-rejected";
    if (k === "FAILED" || k === "ERROR") return "badge badge-failed";
    return "badge";
  };

  U.badgeForCycle = function (s) {
    if (!s) return "badge";
    const k = String(s).toUpperCase();
    if (k === "OK" || k === "COMPLETED") return "badge badge-ok";
    if (k === "PARTIAL_ERROR" || k === "PARTIAL") return "badge badge-warning";
    if (k === "ERROR" || k === "FAILED") return "badge badge-error";
    if (k === "RUNNING") return "badge badge-info";
    if (k === "NEVER_RUN") return "badge badge-neutral";
    return "badge";
  };

  U.agentMeta = {
    technical:      { label: "Technical Agent",  icon: "chart-candle",  cls: "a-technical" },
    news:           { label: "News Agent",       icon: "news",          cls: "a-news" },
    fundamentals:   { label: "Fundamentals",     icon: "db",            cls: "a-fundamentals" },
    debate:         { label: "Bull/Bear Debate", icon: "scale",         cls: "a-debate" },
    risk:           { label: "Risk Agent",       icon: "shield",        cls: "a-risk" },
    cio:            { label: "CIO",              icon: "gavel",         cls: "a-cio" },
    execution:      { label: "Execution",        icon: "zap",           cls: "a-execution" },
    memory:         { label: "Memory",           icon: "memory",        cls: "a-memory" },
    system:         { label: "System",           icon: "cpu",           cls: "a-system" },
    market:         { label: "Market Data",      icon: "globe",         cls: "a-system" },
  };
  U.agentMeta.tech = U.agentMeta.technical;
  U.agentLabel = function (a) {
    return (U.agentMeta[a] && U.agentMeta[a].label) || (a ? a.charAt(0).toUpperCase() + a.slice(1) : "Unknown");
  };
  U.agentIcon = function (a) {
    return (U.agentMeta[a] && U.agentMeta[a].icon) || "cpu";
  };
  U.agentCls = function (a) {
    return (U.agentMeta[a] && U.agentMeta[a].cls) || "a-system";
  };

  U.levelBadge = function (lvl) {
    const k = String(lvl || "INFO").toUpperCase();
    if (k === "ERROR") return "badge badge-error";
    if (k === "WARNING" || k === "WARN") return "badge badge-warning";
    if (k === "SUCCESS") return "badge badge-ok";
    return ""; // info-level needs no badge (avoids noise)
  };

  // Weight → qualitative track record label (honest, non-predictive)
  U.trackForWeight = function (w, n) {
    if (!n) return null;
    if (w >= 1.25) return { label: "Strong", cls: "pos" };
    if (w >= 1.05) return { label: "Improving", cls: "pos" };
    if (w > 0.95) return { label: "Stable", cls: "neu" };
    if (w > 0.75) return { label: "Softening", cls: "neg" };
    return { label: "Poor", cls: "neg" };
  };

  U.pctToCol = function (pct) {
    if (pct == null || isNaN(pct)) return "neu";
    return pct > 0 ? "pos" : pct < 0 ? "neg" : "neu";
  };

  /* ---------------- fetch layer ---------------- */

  const MAX_AGE = {}; // url -> last success ms
  let online = true;

  async function get(url, timeoutMs) {
    const ctl = new AbortController();
    const t = setTimeout(function () { ctl.abort(); }, timeoutMs || 12000);
    try {
      const r = await fetch(url, { headers: { accept: "application/json" }, signal: ctl.signal, cache: "no-store" });
      if (!r.ok) throw new Error("HTTP " + r.status);
      const j = await r.json();
      MAX_AGE[url] = Date.now();
      online = true;
      return { ok: true, data: j };
    } catch (e) {
      return { ok: false, error: e && e.name === "AbortError" ? "Request timed out" : String((e && e.message) || e) };
    } finally {
      clearTimeout(t);
    }
  }

  async function post(url) {
    try {
      const r = await fetch(url, { method: "POST", headers: { accept: "application/json" } });
      const j = await r.json().catch(function () { return {}; });
      if (!r.ok) throw new Error("HTTP " + r.status);
      MAX_AGE[url] = Date.now();
      online = true;
      return { ok: true, data: j };
    } catch (e) {
      return { ok: false, error: String((e && e.message) || e) };
    }
  }

  const API = {
    portfolio: function () { return get("/api/portfolio"); },
    logs: function () { return get("/api/logs"); },
    accuracy: function () { return get("/api/agent-accuracy"); },
    cycles: function () { return get("/api/cycles"); },
    orders: function (limit) { return get("/api/orders?limit=" + (limit || 100)); },
    history: function (period) { return get("/api/history?period=" + encodeURIComponent(period || "1M")); },
    config: function () { return get("/api/config"); },
    botStart: function () { return post("/api/bot/start"); },
    botStop: function () { return post("/api/bot/stop"); },
    runNow: function () { return post("/api/bot/run-now"); },
    age: function (url) { return MAX_AGE[url] || null; },
    isOnline: function () { return online; },
    wentOffline: function () { online = false; },
  };

  window.U = U;
  window.API = API;
})();
