/* ============================================================================
   AI TRADER — Application core
   Store · polling · router · app shell · drawer / modal / toast / palette ·
   SVG chart engine · demo design-preview
   ========================================================================== */
(function () {
  "use strict";

  var U = window.U, API = window.API, ICON = window.ICON;

  /* ==========================================================================
     Tiny DOM helpers
     ========================================================================== */

  function qs(s, root) { return (root || document).querySelector(s); }
  function qsa(s, root) { return Array.prototype.slice.call((root || document).querySelectorAll(s)); }
  function h(html) {
    var t = document.createElement("template");
    t.innerHTML = html.trim();
    return t.content.firstElementChild;
  }

  /* ==========================================================================
     Store — latest payload of every endpoint + data version counters
     ========================================================================== */

  var store = {
    demo: /[?&]demo=1/.test(location.search),
    ts: {},          // endpoint -> last success ms
    data: {},        // endpoint -> payload
    errors: {},      // endpoint -> last error string
    loading: {},     // endpoint -> bool
    histCache: {},   // period -> {data, ts}
    ui: {
      page: null, params: {},
      lastLogTs: null,      // newest log timestamp seen (for "new" highlight)
      seenCycles: {},       // cycle ids already toasted
      range: "1M",
      cycleRunning: false,
    },
  };

  function bump(ep) { store.ts[ep] = Date.now(); }

  var subs = {};
  function on(ev, fn) { (subs[ev] = subs[ev] || []).push(fn); }
  function emit(ev) { (subs[ev] || []).forEach(function (fn) { try { fn(); } catch (e) { console.error(e); } }); }

  /* ==========================================================================
     Derived data (recomputed whenever logs / portfolio change)
     ========================================================================== */

  var derived = { bySymbol: {}, newestLogTs: null };

  function recomputeDerived() {
    var logs = (store.data.logs && store.data.logs.logs) || [];
    var bySym = {};
    var newest = null;

    logs.forEach(function (l) {
      if (!newest || l.timestamp > newest) newest = l.timestamp;
      var s = l.symbol || "_";
      var b = (bySym[s] = bySym[s] || {});
      if (l.agent === "cio") {
        if (!b.cio || l.timestamp >= b.cio.timestamp) b.cio = l;
      } else if (l.agent === "debate") {
        if (!b.debate || l.timestamp >= b.debate.timestamp) b.debate = l;
      } else if (l.agent === "execution") {
        if (!b.exec || l.timestamp >= b.exec.timestamp) b.exec = l;
      } else if (l.agent === "risk") {
        if (!b.risk || l.timestamp >= b.risk.timestamp) b.risk = l;
      }
      b.signals = b.signals || {};
      if (!b.signals[l.agent] || l.timestamp >= b.signals[l.agent].timestamp) b.signals[l.agent] = l;
    });

    derived.bySymbol = bySym;
    derived.newestLogTs = newest;
  }

  function positionsWithContext() {
    var p = (store.data.portfolio && store.data.portfolio.positions) || [];
    var acct = (store.data.portfolio && store.data.portfolio.account) || {};
    var equity = acct.equity || 0;
    return p.map(function (pos) {
      var d = derived.bySymbol[pos.symbol] || {};
      var w = equity > 0 && pos.market_value ? (pos.market_value / equity) * 100 : null;
      var out = Object.assign({}, pos);
      out.weight_pct = w;
      out.cio = d.cio || null;
      out.bias = d.debate || null;
      out.risk = d.risk || null;
      return out;
    });
  }

  function matchOrdersToCycles() {
    var orders = (store.data.orders && store.data.orders.orders) || [];
    var cycles = (store.data.cycles && store.data.cycles.cycles) || [];
    var map = {};
    orders.forEach(function (o) {
      if (!o.submitted_at) return;
      var t = U.parseDate(o.submitted_at).getTime();
      for (var i = 0; i < cycles.length; i++) {
        var c = cycles[i];
        var a = c.started_at && U.parseDate(c.started_at).getTime();
        var b = c.finished_at && U.parseDate(c.finished_at).getTime();
        if (a && b && t >= a - 60000 && t <= b + 60000) {
          var hit = (c.orders || []).some(function (co) { return co.symbol === o.symbol && String(co.side).toLowerCase() === String(o.side).toLowerCase(); });
          if (hit) { map[o.id] = "CIO Decision · Cycle #" + c.id; return; }
        }
      }
      map[o.id] = "AI Cycle";
    });
    return map;
  }

  /* ==========================================================================
     Polling
     ========================================================================== */

  var POLL_FAST_MS = 10000;
  var POLL_SLOW_MS = 45000;
  var lastPoll = 0;

  async function fetchEndpoint(name, fn) {
    store.loading[name] = true;
    var r = await fn();
    store.loading[name] = false;
    if (r.ok) {
      store.data[name] = r.data;
      delete store.errors[name];
      bump(name);
    } else {
      store.errors[name] = r.error;
      API.wentOffline();
      // First fetch failed and we have nothing: mark the endpoint as
      // unavailable so pages render an explicit error state (not skeletons).
      if (store.data[name] === undefined) store.data[name] = { __unavailable: true, error: r.error };
    }
    return r.ok;
  }

  async function pollLight() {
    if (store.demo) return;
    await Promise.all([
      fetchEndpoint("portfolio", API.portfolio),
      fetchEndpoint("logs", API.logs),
      fetchEndpoint("cycles", API.cycles),
      fetchEndpoint("accuracy", API.accuracy),
    ]);
    afterDataTick();
  }

  async function pollSlow() {
    if (store.demo) return;
    await Promise.all([
      fetchEndpoint("orders", function () { return API.orders(100); }),
      fetchEndpoint("config", API.config),
      fetchEndpoint("health", API.health),
    ]);
    afterDataTick();
  }

  async function pollRealtime() {
    if (store.demo) return;
    await fetchEndpoint("realtime", API.realtime);
    // Real-time ticks only refresh the live strip, never a full re-render
    // (and NEVER an LLM call — the AI cycle stays on its scheduler).
    var el = document.getElementById("live-strip");
    if (el && typeof window.App.updateLiveStrip === "function") {
      try { window.App.updateLiveStrip(el); } catch (e) { /* strip optional */ }
    }
  }

  function afterDataTick() {
    recomputeDerived();
    detectCycleEvents();
    emit("data");
    renderFreshness();
    maybeRerender();
  }

  /* Re-render the current page when fresh data arrives, unless the user is
     interacting with an input. Preserves filter state and scroll position. */
  function maybeRerender() {
    var ae = document.activeElement;
    if (ae && /input|textarea|select/i.test(ae.tagName || "")) return;
    if (typeof App.renderPage === "function" && store.ui.page && !qs("#drawer-root .drawer")) {
      try { App.renderPage(store.ui.page, store.ui.params, true, true); } catch (e) { /* page still rendering */ }
    }
    if (derived.newestLogTs) store.ui.lastLogTs = derived.newestLogTs;
  }

  function detectCycleEvents() {
    var cycles = (store.data.cycles && store.data.cycles.cycles) || [];
    if (!cycles.length) return;
    var top = cycles[0];
    if (store.ui.seenCycles[top.id] && top.status && store.ui.seenCycles[top.id] === top.status) return;
    if (store.ui.seenCycles[top.id] === undefined) {
      // first load — mark everything seen quietly
      cycles.forEach(function (c) { store.ui.seenCycles[c.id] = c.status; });
      return;
    }
    store.ui.seenCycles[top.id] = top.status;
    if (top.status === "OK") toast("Cycle #" + top.id + " completed · " + (top.duration_s != null ? top.duration_s + "s" : "") + " · " + (top.symbols_processed || []).length + " symbols", "ok");
    else if (top.status === "PARTIAL_ERROR") toast("Cycle #" + top.id + " completed with errors — see System Health", "err");
    else if (top.status === "ERROR") toast("Cycle #" + top.id + " failed — see System Health", "err");
    pollLight();
  }

  function startPolling() {
    if (store.demo) { recomputeDerived(); return; }
    pollLight(); pollSlow(); pollRealtime();
    setInterval(function () { pollLight(); }, POLL_FAST_MS);
    setInterval(function () { pollSlow(); }, POLL_SLOW_MS);
    setInterval(function () { pollRealtime(); }, 10000);
    setInterval(tickAgo, 1000);
  }

  /* 1s ticker — refreshes relative timestamps without re-rendering pages */
  function tickAgo() {
    qsa("[data-ago]").forEach(function (el) {
      var v = U.ago(el.getAttribute("data-ago"));
      if (v) el.textContent = v;
    });
    var label;
    if (store.demo) {
      label = "Sample data · demo";
    } else {
      var t = Math.max.apply(null, [0].concat(["portfolio", "logs"].map(function (k) { return store.ts[k] || 0; })));
      if (!t) return;
      var s = Math.floor((Date.now() - t) / 1000);
      label = "Updated " + (s < 60 ? s + "s ago" : Math.floor(s / 60) + "m ago");
      var nf = qs("#freshness");
      if (nf) nf.classList.toggle("stale", s > 45);
    }
    qsa("#freshness, #freshness-m").forEach(function (el) { el.textContent = label; });
    renderClocks();
  }

  function renderClocks() {
    var d = new Date();
    qsa("[data-clock]").forEach(function (el) { el.textContent = d.toLocaleTimeString("en-US", { hour12: false }); });
    var nc = qs("#nextcycle");
    if (nc) updateNextCycle();
  }

  function updateNextCycle() {
    var nc = qs("#nextcycle");
    if (!nc) return;
    var st = store.data.portfolio && store.data.portfolio.bot_state;
    var cfg = store.data.config || {};
    var iv = (cfg.cycle_interval_minutes || 15) * 60000;
    if (!st || !st.running || !st.last_cycle_at) {
      nc.textContent = st && !st.running ? "Paused" : "—";
      return;
    }
    var last = U.parseDate(st.last_cycle_at).getTime();
    var left = iv - (Date.now() - last);
    if (left <= 0) { nc.textContent = "Due now"; nc.className = "num"; return; }
    var m = Math.floor(left / 60000), s = Math.floor((left % 60000) / 1000);
    nc.textContent = (m > 0 ? m + "m " : "") + s + "s";
    nc.classList.toggle("pos", true);
  }

  function renderFreshness() {
    var dot = qs("#freshness-dot");
    if (dot) dot.className = "dot " + (store.errors.portfolio || store.errors.logs ? "dot-err" : "dot-on");
    tickAgo();
  }

  /* ==========================================================================
     Router
     ========================================================================== */

  var ROUTES = ["overview", "portfolio", "positions", "orders", "ai", "agents", "risk", "cycles", "health", "activity", "config"];

  function parseHash() {
    var m = location.hash.match(/^#\/([a-z-]*)(?:\?(.*))?$/);
    var page = m && ROUTES.indexOf(m[1]) >= 0 ? m[1] : "overview";
    var params = {};
    if (m && m[2]) m[2].split("&").forEach(function (kv) {
      var p = kv.split("=");
      params[decodeURIComponent(p[0])] = decodeURIComponent(p[1] || "");
    });
    return { page: page, params: params };
  }

  function navigate(page, params) {
    var q = "";
    if (params) q = "?" + Object.keys(params).map(function (k) { return encodeURIComponent(k) + "=" + encodeURIComponent(params[k]); }).join("&");
    location.hash = "#/" + page + q;
  }

  function renderRoute() {
    var r = parseHash();
    store.ui.page = r.page;
    store.ui.params = r.params;
    window.App.renderPage(r.page, r.params);
    markActiveNav(r.page);
    document.title = "AI Trader — " + (window.App.PAGE_TITLES[r.page] || "Overview");
  }

  function markActiveNav(page) {
    qsa("[data-nav]").forEach(function (a) {
      a.classList.toggle("on", a.getAttribute("data-nav") === page);
      if (a.getAttribute("data-nav") === page) a.setAttribute("aria-current", "page");
      else a.removeAttribute("aria-current");
    });
  }

  /* ==========================================================================
     Toasts
     ========================================================================== */

  function toast(msg, kind) {
    var box = qs("#toasts");
    if (!box) return;
    var el = h('<div class="toast ' + (kind || "info") + '" role="status">' +
      ICON(kind === "ok" ? "check" : kind === "err" ? "alert" : "info") +
      '<div class="t-msg">' + U.esc(msg) + "</div></div>");
    box.appendChild(el);
    setTimeout(function () { el.style.opacity = "0"; el.style.transition = "opacity .25s"; }, 4200);
    setTimeout(function () { el.remove(); }, 4600);
  }

  /* ==========================================================================
     Drawer / modal — with focus trap + escape
     ========================================================================== */

  var focusReturn = null;

  function openDrawer(titleHTML, subHTML, bodyHTML, footHTML, opts) {
    opts = opts || {};
    var root = qs("#drawer-root");
    focusReturn = document.activeElement;
    root.innerHTML =
      '<div class="drawer-veil" data-action="close-overlay"></div>' +
      '<aside class="drawer" role="dialog" aria-modal="true" aria-label="' + U.esc(opts.label || "Details") + '">' +
        '<div class="drawer-hd">' +
          '<div class="col">' +
            '<div class="d-title">' + titleHTML + "</div>" +
            (subHTML ? '<div class="d-sub">' + subHTML + "</div>" : "") +
          "</div>" +
          '<div class="spacer"></div>' +
          '<button class="drawer-close" data-action="close-overlay" aria-label="Close panel">' + ICON("x") + "</button>" +
        "</div>" +
        '<div class="drawer-bd" id="drawer-body">' + bodyHTML + "</div>" +
        (footHTML ? '<div class="drawer-ft">' + footHTML + "</div>" : "") +
      "</aside>";
    document.body.style.overflow = "hidden";
    requestAnimationFrame(function () {
      qs(".drawer-veil", root).classList.add("on");
      qs(".drawer", root).classList.add("on");
    });
    var target = qs("[data-autofocus]", root) || qs(".drawer-close", root);
    target.focus();
  }

  function setDrawerBody(html) {
    var b = qs("#drawer-body");
    if (b) b.innerHTML = html;
    if (b) b.scrollTop = 0;
  }

  function closeOverlay() {
    var root = qs("#drawer-root");
    var d = qs(".drawer", root), v = qs(".drawer-veil", root);
    if (d) { d.classList.remove("on"); v && v.classList.remove("on"); }
    document.body.style.overflow = "";
    setTimeout(function () { root.innerHTML = ""; }, 240);
    if (focusReturn && focusReturn.focus) focusReturn.focus();
    focusReturn = null;
    var m = qs("#modal-root");
    m.innerHTML = "";
  }

  function openModal(html, opts) {
    opts = opts || {};
    var root = qs("#modal-root");
    focusReturn = document.activeElement;
    root.innerHTML =
      '<div class="modal-veil on" data-action="modal-dismiss"><div class="modal" role="alertdialog" aria-modal="true" aria-labelledby="modal-title">' + html + "</div></div>";
    var t = qs("[data-autofocus]", root) || qs(".modal-ft .btn", root);
    if (t) t.focus();
  }

  function closeModal() {
    qs("#modal-root").innerHTML = "";
    if (focusReturn && focusReturn.focus) focusReturn.focus();
    focusReturn = null;
  }

  function confirmModal(cfg) {
    openModal(
      '<div class="modal-hd">' +
        '<div class="m-ico ' + (cfg.tone || "warn") + '">' + ICON(cfg.icon || "alert") + "</div>" +
        '<div><div class="m-title" id="modal-title">' + U.esc(cfg.title) + "</div></div>" +
      "</div>" +
      '<div class="modal-bd">' + cfg.body + "</div>" +
      '<div class="modal-ft">' +
        '<button class="btn" data-action="modal-cancel">' + U.esc(cfg.cancelLabel || "Cancel") + "</button>" +
        '<button class="btn ' + (cfg.danger ? "btn-danger" : "btn-primary") + '" data-action="modal-ok" data-autofocus>' + U.esc(cfg.okLabel || "Confirm") + "</button>" +
      "</div>"
    );
    qs("#modal-root").addEventListener("click", function (e) {
      if (e.target.closest("[data-action='modal-ok']")) { closeModal(); cfg.onOk(); }
      if (e.target.closest("[data-action='modal-cancel'],[data-action='modal-dismiss']")) { closeModal(); }
    });
  }

  /* ==========================================================================
     Bot actions
     ========================================================================== */

  function botAction(action) {
    if (store.demo) return demoBotAction(action);
    if (action === "run") return doRunCycle();
    if (action === "start") return doStartBot();
    if (action === "stop") {
      confirmModal({
        title: "Stop the trading bot?",
        icon: "power",
        tone: "err",
        danger: true,
        okLabel: "Stop Bot",
        body:
          "<p>Scheduled trading cycles will be paused. The bot will not analyze symbols or submit orders until it is started again.</p>" +
          '<div class="note" style="margin-top:8px">' + ICON("info") + "<span>Open positions are left untouched. Orders already submitted to the paper account are not cancelled.</span></div>",
        onOk: doStopBot,
      });
    }
  }

  async function doStartBot() {
    var r = await API.botStart();
    if (r.ok) { toast("Bot started — autonomous cycles are active", "ok"); pollLight(); }
    else toast("Could not start bot: " + r.error, "err");
  }

  async function doStopBot() {
    var r = await API.botStop();
    if (r.ok) { toast("Bot stopped — cycles paused", "info"); pollLight(); }
    else toast("Could not stop bot: " + r.error, "err");
  }

  var runningCycle = false;
  async function doRunCycle() {
    if (runningCycle) { toast("A cycle is already running", "info"); return; }
    runningCycle = true;
    store.ui.cycleRunning = true;
    var btns = qsa("[data-action='run-cycle']");
    btns.forEach(function (b) { b.disabled = true; b.innerHTML = '<span class="btn-spinner"></span>Running…'; });
    toast("Cycle started — running the full agent pipeline", "info");
    var r = await API.runNow();
    runningCycle = false;
    store.ui.cycleRunning = false;
    btns.forEach(function (b) { b.disabled = false; b.innerHTML = ICON("play") + "Run Cycle"; });
    if (r.ok) {
      toast("Cycle finished — refreshing data", "ok");
      await Promise.all([pollLight(), pollSlow()]);
      window.App.renderPage(store.ui.page, store.ui.params, true);
    } else {
      toast("Cycle failed: " + r.error, "err");
    }
  }

  /* ==========================================================================
     Command palette
     ========================================================================== */

  var paletteItems = [];
  function buildPalette() {
    paletteItems = [];
    ROUTES.forEach(function (p) {
      paletteItems.push({ group: "Pages", label: window.App.PAGE_TITLES[p], icon: window.App.PAGE_ICONS[p], run: function () { navigate(p); } });
    });
    paletteItems.push({ group: "Actions", label: "Run cycle now", icon: "play", hint: "POST /api/bot/run-now", run: function () { botAction("run"); } });
    paletteItems.push({ group: "Actions", label: "Start bot", icon: "power", run: function () { botAction("start"); } });
    paletteItems.push({ group: "Actions", label: "Stop bot", icon: "square", run: function () { botAction("stop"); } });
    paletteItems.push({ group: "Actions", label: "Refresh data", icon: "refresh", run: function () { pollLight(); pollSlow(); toast("Refreshing data…", "info"); } });

    var syms = {};
    var pf = store.data.portfolio || {};
    (pf.trade_universe || []).forEach(function (s) { syms[s] = true; });
    ((store.data.orders && store.data.orders.orders) || []).forEach(function (o) { syms[o.symbol] = true; });
    Object.keys(syms).sort().forEach(function (s) {
      paletteItems.push({ group: "Symbols", label: s, icon: "search", hint: "Activity & decisions", run: function () { navigate("activity", { sym: s }); } });
    });
    ["technical", "news", "fundamentals", "debate", "risk", "cio", "execution", "memory", "system"].forEach(function (a) {
      paletteItems.push({ group: "Agents", label: U.agentLabel(a), icon: U.agentIcon(a), run: function () { window.App.openAgentDrawer(a); } });
    });
  }

  function fuzzy(q, s) {
    s = s.toLowerCase(); q = q.toLowerCase();
    var i = 0;
    for (var j = 0; j < s.length && i < q.length; j++) if (s[j] === q[i]) i++;
    return i === q.length;
  }

  function openPalette() {
    buildPalette();
    var root = qs("#palette-root");
    root.innerHTML =
      '<div class="palette-veil on" data-action="palette-close">' +
        '<div class="palette" role="dialog" aria-modal="true" aria-label="Command palette" onclick="event.stopPropagation()">' +
          '<div class="search">' + ICON("search") + '<input id="pal-input" placeholder="Search pages, symbols, agents, actions…" aria-label="Search commands">' +
          '<span class="kbd">esc</span></div>' +
          '<div class="palette-list" id="pal-list" role="listbox"></div>' +
        "</div>" +
      "</div>";
    var input = qs("#pal-input");
    var list = qs("#pal-list");
    var sel = 0;

    function renderList() {
      var q = input.value.trim();
      var items = paletteItems.filter(function (it) { return !q || fuzzy(q, it.label) || it.group.toLowerCase().indexOf(q.toLowerCase()) >= 0; });
      if (!items.length) { list.innerHTML = '<div class="pal-empty">No matches for “' + U.esc(q) + '”</div>'; return; }
      var html = "", lastGroup = "";
      items.forEach(function (it, i) {
        if (it.group !== lastGroup) { html += '<div class="pal-group">' + it.group + "</div>"; lastGroup = it.group; }
        html += '<button class="pal-item' + (i === sel ? " sel" : "") + '" data-pal="' + paletteItems.indexOf(it) + '">' + ICON(it.icon) + "<span>" + U.esc(it.label) + "</span>" + (it.hint ? '<span class="p-hint">' + U.esc(it.hint) + "</span>" : "") + "</button>";
      });
      list.innerHTML = html;
      qsa(".pal-item", list).forEach(function (b) {
        b.addEventListener("click", function () {
          runItem(Number(b.getAttribute("data-pal")));
        });
        b.addEventListener("mousemove", function () {
          sel = items.findIndex(function (it) { return paletteItems.indexOf(it) === Number(b.getAttribute("data-pal")); });
        });
      });
    }

    function runItem(idx) {
      closePalette();
      paletteItems[idx].run();
    }

    function currentIdx() {
      var q = input.value.trim();
      var items = paletteItems.filter(function (it) { return !q || fuzzy(q, it.label) || it.group.toLowerCase().indexOf(q.toLowerCase()) >= 0; });
      return { items: items, item: items[sel] };
    }

    input.addEventListener("input", function () { sel = 0; renderList(); });
    input.addEventListener("keydown", function (e) {
      var cur = currentIdx();
      if (e.key === "ArrowDown") { e.preventDefault(); sel = Math.min(sel + 1, cur.items.length - 1); renderList(); }
      else if (e.key === "ArrowUp") { e.preventDefault(); sel = Math.max(sel - 1, 0); renderList(); }
      else if (e.key === "Enter") { e.preventDefault(); if (cur.item) runItem(paletteItems.indexOf(cur.item)); }
      else if (e.key === "Escape") closePalette();
    });
    renderList();
    input.focus();
  }

  function closePalette() { qs("#palette-root").innerHTML = ""; }

  /* ==========================================================================
     Chart engine (SVG, dependency-free)
     ========================================================================== */

  var charts = [];

  function chart(cfg) {
    var el = cfg.el;
    el.classList.add("chart");
    var obj = { cfg: cfg, el: el, draw: function () { drawChart(obj); } };
    obj.draw();
    charts.push(obj);
    return obj;
  }

  function clearCharts() { charts = []; }

  function redrawCharts() { charts = charts.filter(function (c) { return c.el && c.el.isConnected; }); charts.forEach(function (c) { c.draw(); }); }

  function drawChart(obj) {
    var cfg = obj.cfg;
    var el = obj.el;
    var pts = (cfg.points || []).filter(function (p) { return p.v != null && isFinite(p.v); });
    var W = Math.max(280, el.clientWidth || 800);
    var H = cfg.height || 320;
    var P = { t: 14, r: 54, b: 24, l: 10 };

    if (pts.length < 2) {
      el.innerHTML = window.C ? window.C.stateHTML({
        icon: "trend-up", title: cfg.emptyTitle || "No chart data",
        msg: cfg.emptyMsg || "Portfolio history is not available from the broker for this range yet.",
        compact: true,
      }) : '<div class="state">' + U.esc(cfg.emptyTitle || "No chart data") + "</div>";
      return;
    }

    var min = Math.min.apply(null, pts.map(function (p) { return p.v; }));
    var max = Math.max.apply(null, pts.map(function (p) { return p.v; }));
    if (min === max) { min -= 1; max += 1; }
    var pad = (max - min) * 0.08;
    min -= pad; max += pad;
    var baseline = cfg.baseline != null ? cfg.baseline : pts[0].v;
    var up = pts[pts.length - 1].v >= baseline;
    var color = cfg.color || (up ? getVar("--green") : getVar("--red"));
    var colorDim = up ? getVar("--green-dim") : getVar("--red-dim");

    var plotW = W - P.l - P.r, plotH = H - P.t - P.b;
    function X(i) { return P.l + (i / (pts.length - 1)) * plotW; }
    function Y(v) { return P.t + (1 - (v - min) / (max - min)) * plotH; }

    var line = pts.map(function (p, i) { return (i ? "L" : "M") + X(i).toFixed(1) + " " + Y(p.v).toFixed(1); }).join(" ");
    var area = line + " L" + X(pts.length - 1).toFixed(1) + " " + (P.t + plotH) + " L" + X(0).toFixed(1) + " " + (P.t + plotH) + " Z";

    // gridlines + y labels
    var grid = "", ylabels = "";
    var ticks = 4;
    for (var i = 0; i <= ticks; i++) {
      var v = min + ((max - min) * i) / ticks;
      var y = Y(v);
      grid += '<line class="grid-line" x1="' + P.l + '" x2="' + (W - P.r) + '" y1="' + y + '" y2="' + y + '"/>';
      var lbl = cfg.mode === "delta" ? U.fmtPct(((v - baseline) / baseline) * 100, 1) : U.fmtMoney(v, { dec: v >= 10000 ? 0 : 2 }).replace(".00", "");
      ylabels += '<text class="axis-lbl" x="' + (W - P.r + 8) + '" y="' + (y + 3) + '" text-anchor="start">' + lbl + "</text>";
    }

    // x labels
    var xlabels = "";
    var span = pts[pts.length - 1].t - pts[0].t;
    var intraday = span < 86400000 * 2;
    var nx = Math.min(6, pts.length);
    for (var i = 0; i < nx; i++) {
      var idx = Math.round((i / (nx - 1)) * (pts.length - 1));
      var d = new Date(pts[idx].t);
      var lbl = intraday
        ? d.toLocaleTimeString("en-US", { hour12: false, hour: "2-digit", minute: "2-digit" })
        : d.toLocaleDateString("en-US", { month: "short", day: span > 86400000 * 120 ? undefined : "numeric" });
      var anchor = i === 0 ? "start" : i === nx - 1 ? "end" : "middle";
      xlabels += '<text class="axis-lbl" x="' + X(idx) + '" y="' + (H - 8) + '" text-anchor="' + anchor + '">' + lbl + "</text>";
    }

    var uid = "g" + Math.random().toString(36).slice(2, 8);
    var zeroY = cfg.mode === "delta" ? Y(baseline) : null;
    var lastX = X(pts.length - 1), lastY = Y(pts[pts.length - 1].v);

    el.innerHTML =
      '<svg viewBox="0 0 ' + W + " " + H + '" width="' + W + '" height="' + H + '" role="img" aria-label="' + U.esc(cfg.aria || "Portfolio chart") + '">' +
        "<defs>" +
          '<linearGradient id="' + uid + '" x1="0" y1="0" x2="0" y2="1">' +
            '<stop offset="0%" stop-color="' + color + '" stop-opacity="0.14"/>' +
            '<stop offset="100%" stop-color="' + color + '" stop-opacity="0"/>' +
          "</linearGradient>" +
        "</defs>" +
        grid +
        (zeroY != null ? '<line x1="' + P.l + '" x2="' + (W - P.r) + '" y1="' + zeroY + '" y2="' + zeroY + '" stroke="' + getVar("--ink-faint") + '" stroke-dasharray="2 4" stroke-width="1"/>' : "") +
        '<path d="' + area + '" fill="url(#' + uid + ')"/>' +
        '<path d="' + line + '" fill="none" stroke="' + color + '" stroke-width="1.6"/>' +
        ylabels + xlabels +
        '<g class="cursor" opacity="0">' +
          '<line class="cursor-line" y1="' + P.t + '" y2="' + (P.t + plotH) + '"/>' +
          '<circle r="3.5" fill="' + getVar("--bg-0") + '" stroke="' + color + '" stroke-width="2"/>' +
        "</g>" +
        '<circle cx="' + lastX + '" cy="' + lastY + '" r="3" fill="' + color + '"/>' +
        '<circle cx="' + lastX + '" cy="' + lastY + '" r="3" fill="none" stroke="' + colorDim + '" stroke-width="1" opacity="0.6"/>' +
        '<rect class="hitbox" x="' + P.l + '" y="' + P.t + '" width="' + plotW + '" height="' + plotH + '" tabindex="0" aria-label="Chart data — use arrow keys to inspect values"/>' +
      "</svg>" +
      '<div class="chart-tip" role="status"></div>';

    var svg = el.querySelector("svg");
    var cursor = el.querySelector(".cursor");
    var cline = el.querySelector(".cursor-line");
    var cdot = el.querySelector(".cursor circle");
    var tip = el.querySelector(".chart-tip");
    var hit = el.querySelector(".hitbox");

    function show(i) {
      var p = pts[i];
      cursor.setAttribute("opacity", "1");
      cline.setAttribute("x1", X(i)); cline.setAttribute("x2", X(i));
      cdot.setAttribute("cx", X(i)); cdot.setAttribute("cy", Y(p.v));
      var d = new Date(p.t);
      var dv = p.v - baseline;
      var dp = (dv / baseline) * 100;
      tip.innerHTML =
        '<div class="t-lbl">' + (intraday ? d.toLocaleTimeString("en-US", { hour12: false }) : d.toLocaleDateString("en-US", { month: "short", day: "numeric", year: "numeric" })) + "</div>" +
        '<div class="t-val">' + U.fmtMoney(p.v) + "</div>" +
        '<div class="t-sub">' +
          '<span class="num ' + U.classFor(dv) + '">' + U.fmtSigned(dv) + " (" + U.fmtPct(dp) + ")</span>" +
          (cfg.tipSub ? '<span class="t-faint">' + cfg.tipSub + "</span>" : "") +
        "</div>";
      tip.classList.add("on");
      var tw = tip.offsetWidth || 160;
      var x = X(i) - tw / 2;
      x = Math.max(4, Math.min(W - tw - 4, x));
      tip.style.left = x + "px";
      tip.style.top = Math.max(4, Y(p.v) - 84) + "px";
      hit.setAttribute("aria-valuenow", p.v.toFixed(2));
      hit.setAttribute("aria-valuetext", U.fmtMoney(p.v) + " on " + d.toLocaleString());
    }
    function hide() { cursor.setAttribute("opacity", "0"); tip.classList.remove("on"); }

    hit.addEventListener("mousemove", function (e) {
      var rect = svg.getBoundingClientRect();
      var x = e.clientX - rect.left;
      var i = Math.round(((x - P.l) / plotW) * (pts.length - 1));
      i = U.clamp(i, 0, pts.length - 1);
      show(i);
    });
    hit.addEventListener("mouseleave", hide);
    hit.addEventListener("keydown", function (e) {
      var cur = Number(hit.getAttribute("data-i") || 0);
      if (e.key === "ArrowLeft" || e.key === "ArrowRight") {
        e.preventDefault();
        cur = Number(hit.getAttribute("data-i") || 0);
        var ni = U.clamp(cur + (e.key === "ArrowRight" ? 1 : -1), 0, pts.length - 1);
        hit.setAttribute("data-i", ni); show(ni);
      } else if (e.key === "Home") { hit.setAttribute("data-i", 0); show(0); }
      else if (e.key === "End") { hit.setAttribute("data-i", pts.length - 1); show(pts.length - 1); }
    });
    hit.addEventListener("blur", hide);
    obj.show = show;
  }

  function getVar(name) { return getComputedStyle(document.documentElement).getPropertyValue(name).trim() || "#3fce8f"; }

  function sparkline(el, values, color) {
    if (!values || values.length < 2) { el.innerHTML = ""; return; }
    var W = 100, H = 26;
    var min = Math.min.apply(null, values), max = Math.max.apply(null, values);
    if (min === max) max = min + 1;
    var d = values.map(function (v, i) {
      return (i ? "L" : "M") + ((i / (values.length - 1)) * W).toFixed(1) + " " + (H - 2 - ((v - min) / (max - min)) * (H - 4)).toFixed(1);
    }).join(" ");
    el.innerHTML = '<svg class="spark" viewBox="0 0 ' + W + " " + H + '" preserveAspectRatio="none" aria-hidden="true"><path d="' + d + '" fill="none" stroke="' + (color || getVar("--ink-faint")) + '" stroke-width="1.4"/></svg>';
  }

  /* ==========================================================================
     Demo design-preview — realistic fixtures, clearly labeled DEMO
     Every number below is illustrative sample data for the design preview;
     the live dashboard only ever renders real API responses.
     ========================================================================== */

  var demo = {};

  function rng(seed) {
    var s = seed >>> 0;
    return function () {
      s = (s + 0x6d2b79f5) >>> 0;
      var t = s;
      t = Math.imul(t ^ (t >>> 15), t | 1);
      t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
      return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
    };
  }

  function demoHistory(period, endValue) {
    var n = { "1D": 78, "1W": 130, "1M": 23, "3M": 66, "6M": 130, "1Y": 250, ALL: 300 }[period] || 30;
    var stepMs = { "1D": 5 * 60000, "1W": 15 * 60000, "1M": 86400000, "3M": 86400000, "6M": 43200000, "1Y": 86400000 * 1.5, ALL: 86400000 * 1.8 }[period];
    var r = rng(97 + period.length * 13 + n);
    var walk = [0];
    for (var i = 1; i < n; i++) walk.push(walk[i - 1] + (r() - 0.47) * 1);
    var last = walk[n - 1];
    var pts = [];
    var now = Date.now();
    for (var i = 0; i < n; i++) {
      var drift = (i / (n - 1)) * 0.9;
      var v = endValue + (walk[i] - last * (0.35 + drift * 0.65)) * (endValue * 0.004);
      pts.push({ t: now - (n - 1 - i) * stepMs, v: Math.round(v * 100) / 100 });
    }
    pts[n - 1].v = endValue;
    return pts;
  }
  window.__demoHistory = demoHistory;

  function loadDemo() {
    var now = Date.now();
    var iso = function (minAgo) { return new Date(now - minAgo * 60000).toISOString(); };
    var t = function (minAgo) { return iso(minAgo); };

    var eq = 25482.31;
    var cash = 13430.57;

    var posRows = [
      { symbol: "NVDA", qty: 12, avg: 171.20, cur: 176.42 },
      { symbol: "AAPL", qty: 9, avg: 221.35, cur: 228.90 },
      { symbol: "MSFT", qty: 4, avg: 414.20, cur: 408.10 },
      { symbol: "SPY", qty: 4, avg: 552.10, cur: 556.85 },
      { symbol: "TSLA", qty: 10, avg: 248.50, cur: 236.15 },
      { symbol: "AMD", qty: 6, avg: 152.30, cur: 158.75 },
      { symbol: "GOOGL", qty: 4, avg: 172.40, cur: 175.20 },
    ];
    var positions = posRows.map(function (r) {
      var mv = r.qty * r.cur, cost = r.qty * r.avg;
      return {
        symbol: r.symbol, qty: r.qty, avg_entry_price: r.avg, current_price: r.cur,
        market_value: Math.round(mv * 100) / 100,
        unrealized_pl: Math.round((mv - cost) * 100) / 100,
        unrealized_plpc: Math.round(((mv - cost) / cost) * 10000) / 100,
        side: "long",
      };
    });

    var orders = [
      { id: "ord-9f21", symbol: "NVDA", side: "buy", qty: 12, status: "filled", filled_avg_price: 176.42, submitted_at: t(6) },
      { id: "ord-9f20", symbol: "TSLA", side: "sell", qty: 6, status: "filled", filled_avg_price: 236.40, submitted_at: t(52) },
      { id: "ord-9f19", symbol: "AMD", side: "buy", qty: 6, status: "filled", filled_avg_price: 158.20, submitted_at: t(226) },
      { id: "ord-9f18", symbol: "MSFT", side: "buy", qty: 4, status: "filled", filled_avg_price: 414.10, submitted_at: t(1470) },
      { id: "ord-9f17", symbol: "SPY", side: "buy", qty: 4, status: "filled", filled_avg_price: 551.80, submitted_at: t(1485) },
      { id: "ord-9f16", symbol: "AAPL", side: "buy", qty: 9, status: "filled", filled_avg_price: 221.60, submitted_at: t(1501) },
      { id: "ord-9f15", symbol: "TSLA", side: "buy", qty: 16, status: "filled", filled_avg_price: 249.10, submitted_at: t(2900) },
      { id: "ord-9f14", symbol: "GOOGL", side: "buy", qty: 4, status: "filled", filled_avg_price: 172.80, submitted_at: t(2930) },
      { id: "ord-9f13", symbol: "NVDA", side: "sell", qty: 8, status: "filled", filled_avg_price: 173.05, submitted_at: t(4310) },
      { id: "ord-9f12", symbol: "NVDA", side: "buy", qty: 20, status: "filled", filled_avg_price: 168.90, submitted_at: t(4325) },
      { id: "ord-9f11", symbol: "MSFT", side: "buy", qty: 2, status: "rejected", filled_avg_price: null, submitted_at: t(5760) },
      { id: "ord-9f10", symbol: "SPY", side: "sell", qty: 3, status: "filled", filled_avg_price: 549.20, submitted_at: t(7215) },
      { id: "ord-9f09", symbol: "AMD", side: "sell", qty: 10, status: "filled", filled_avg_price: 149.75, submitted_at: t(7230) },
      { id: "ord-9f08", symbol: "AAPL", side: "sell", qty: 5, status: "filled", filled_avg_price: 224.10, submitted_at: t(10150) },
    ];

    function L(agent, symbol, level, message, minAgo, data) {
      return { agent: agent, symbol: symbol, level: level, message: message, timestamp: t(minAgo), data: data || null };
    }

    var logs = [
      // ---- latest cycle (6-10 min ago) ----
      L("cio", "NVDA", "INFO", "BUY: Technicals show a clean momentum continuation above the 50-day average and news sentiment is strongly positive post-earnings. The bull case clearly outweighs the bear case, and risk has approved a $2,100 position.", 6.1, { agent: "cio", symbol: "NVDA", decision: "BUY", confidence: 0.84, notional_usd: 2100, reasoning: "Technicals show a clean momentum continuation above the 50-day average and news sentiment is strongly positive post-earnings. The bull case clearly outweighs the bear case, and risk has approved a $2,100 position.", error: null }),
      L("execution", "NVDA", "INFO", "Order submitted: BUY NVDA", 6.0, { success: true, order_id: "ord-9f21", symbol: "NVDA", side: "buy", qty: 12, notional_usd: 2100, status: "filled", error: null }),
      L("risk", "NVDA", "INFO", "Approved. Position size within the configured 10% limit; portfolio exposure acceptable after fill.", 6.4, { agent: "risk", symbol: "NVDA", approved: true, max_notional_usd: 2548, risk_level: "LOW", reasoning: "Approved. Position size within the configured 10% limit; portfolio exposure acceptable after fill.", error: null }),
      L("debate", "NVDA", "INFO", "Bull(0.78) vs Bear(0.31), edge=0.47", 6.6, { agent: "debate", symbol: "NVDA", bull_strength: 0.78, bull_summary: "Momentum is firmly bullish: price holds above both the 50-day and 200-day averages, MACD is above its signal line, and post-earnings news flow is the most positive in months. Fundamentals remain solid with accelerating revenue growth.", bear_strength: 0.31, bear_summary: "RSI at 67 is approaching overbought and the stock has run 14% in three weeks, so short-term pullback risk is elevated. Valuation is stretched at 34x forward earnings, leaving little cushion if momentum stalls.", edge: 0.47 }),
      L("fundamentals", "NVDA", "INFO", "NEUTRAL: Valuation rich at 34x forward P/E, but revenue growth (+69% YoY) and margins remain exceptional.", 6.8, { agent: "fundamentals", symbol: "NVDA", signal: "NEUTRAL", confidence: 0.61, summary: "Valuation rich at 34x forward P/E, but revenue growth (+69% YoY) and margins remain exceptional.", error: null }),
      L("news", "NVDA", "INFO", "BULLISH: Strong post-earnings coverage; data-center demand headlines dominate.", 6.9, { agent: "news", symbol: "NVDA", sentiment: "BULLISH", confidence: 0.74, summary: "Strong post-earnings coverage; data-center demand headlines dominate.", error: null }),
      L("technical", "NVDA", "INFO", "BULLISH: Price above SMA50 (168.4) and SMA200 (154.9); MACD above signal; RSI 62.3 — healthy trend, not yet overbought.", 7.1, { agent: "technical", symbol: "NVDA", signal: "BULLISH", confidence: 0.82, summary: "Price above SMA50 (168.4) and SMA200 (154.9); MACD above signal; RSI 62.3 — healthy trend, not yet overbought.", error: null }),
      L("cio", "AAPL", "INFO", "HOLD: Agents are split — constructive news but technicals are flat and the debate edge is near zero. Waiting for a clearer setup.", 7.4, { agent: "cio", symbol: "AAPL", decision: "HOLD", confidence: 0.55, notional_usd: 0, reasoning: "Agents are split — constructive news but technicals are flat and the debate edge is near zero. Waiting for a clearer setup.", error: null }),
      L("risk", "AAPL", "INFO", "Approved. Cash buffer sufficient for a new position within limits.", 7.6, { agent: "risk", symbol: "AAPL", approved: true, max_notional_usd: 2548, risk_level: "LOW", reasoning: "Approved. Cash buffer sufficient for a new position within limits.", error: null }),
      L("debate", "AAPL", "INFO", "Bull(0.52) vs Bear(0.48), edge=0.04", 7.8, { agent: "debate", symbol: "AAPL", bull_strength: 0.52, bull_summary: "Services revenue keeps compounding and news flow is mildly constructive into the product cycle.", bear_summary: "Price is pinned between averages with no momentum; the risk/reward is unremarkable.", bear_strength: 0.48, edge: 0.04 }),
      L("fundamentals", "AAPL", "INFO", "BULLISH: Wide moat, 22% margins, reasonable 29x P/E relative to growth.", 8.0, { agent: "fundamentals", symbol: "AAPL", signal: "BULLISH", confidence: 0.66, summary: "Wide moat, 22% margins, reasonable 29x P/E relative to growth.", error: null }),
      L("news", "AAPL", "INFO", "BULLISH: Positive earnings reaction; supply-chain checks point to stable iPhone demand.", 8.1, { agent: "news", symbol: "AAPL", sentiment: "BULLISH", confidence: 0.58, summary: "Positive earnings reaction; supply-chain checks point to stable iPhone demand.", error: null }),
      L("technical", "AAPL", "INFO", "NEUTRAL: Price consolidating between SMA50 and SMA200; RSI 54.1, MACD flat.", 8.3, { agent: "technical", symbol: "AAPL", signal: "NEUTRAL", confidence: 0.49, summary: "Price consolidating between SMA50 and SMA200; RSI 54.1, MACD flat.", error: null }),
      L("cio", "MSFT", "INFO", "HOLD: Existing position is underwater but fundamentals remain strong; no evidence yet of trend reversal. Hold and re-evaluate next cycle.", 8.6, { agent: "cio", symbol: "MSFT", decision: "HOLD", confidence: 0.58, notional_usd: 0, reasoning: "Existing position is underwater but fundamentals remain strong; no evidence yet of trend reversal. Hold and re-evaluate next cycle.", error: null }),
      L("risk", "MSFT", "INFO", "Approved. Adding to the position would remain within the concentration limit.", 8.8, { agent: "risk", symbol: "MSFT", approved: true, max_notional_usd: 2548, risk_level: "LOW", reasoning: "Approved. Adding to the position would remain within the concentration limit.", error: null }),
      L("debate", "MSFT", "INFO", "Bull(0.55) vs Bear(0.45), edge=0.10", 8.95, { agent: "debate", symbol: "MSFT", bull_summary: "Azure growth and Copilot monetization give the bull case real substance; drawdown looks like sector noise rather than company deterioration.", bull_strength: 0.55, bear_summary: "Cloud spend is cyclical and the multiple leaves little room for a growth miss; technicals have not turned.", bear_strength: 0.45, edge: 0.1 }),
      L("fundamentals", "MSFT", "INFO", "BULLISH: 31% operating margin, net cash balance sheet, double-digit cloud growth.", 9.1, { agent: "fundamentals", symbol: "MSFT", signal: "BULLISH", confidence: 0.71, summary: "31% operating margin, net cash balance sheet, double-digit cloud growth.", error: null }),
      L("news", "MSFT", "INFO", "NEUTRAL: Mixed headlines — strong cloud numbers offset by capex concerns.", 9.2, { agent: "news", symbol: "MSFT", sentiment: "NEUTRAL", confidence: 0.5, summary: "Mixed headlines — strong cloud numbers offset by capex concerns.", error: null }),
      L("technical", "MSFT", "INFO", "BEARISH: Price below SMA50 (417.8), RSI 44.7, MACD below signal — short-term trend is down.", 9.4, { agent: "technical", symbol: "MSFT", signal: "BEARISH", confidence: 0.63, summary: "Price below SMA50 (417.8), RSI 44.7, MACD below signal — short-term trend is down.", error: null }),
      L("cio", "TSLA", "INFO", "HOLD: Bearish technicals argue for trimming, but the risk agent capped further action and the debate is not one-sided enough to force an exit. Position stays open.", 9.7, { agent: "cio", symbol: "TSLA", decision: "HOLD", confidence: 0.51, notional_usd: 0, reasoning: "Bearish technicals argue for trimming, but the risk agent capped further action and the debate is not one-sided enough to force an exit. Position stays open.", error: null }),
      L("risk", "TSLA", "INFO", "Position size within configured limit; concentration approaching the configured threshold — flag for monitoring.", 9.9, "WARNING_DATA"),
      L("debate", "TSLA", "INFO", "Bull(0.38) vs Bear(0.64), edge=-0.26", 10.0, { agent: "debate", symbol: "TSLA", bull_summary: "Energy storage growth and robotaxi optionality keep a floor under the story; sentiment washouts have historically marked bottoms.", bull_strength: 0.38, bear_summary: "Deliveries are falling, margins are compressing, and price has broken below both moving averages with momentum accelerating lower.", bear_strength: 0.64, edge: -0.26 }),
      L("fundamentals", "TSLA", "INFO", "BEARISH: Margin compression two quarters running; valuation still prices flawless execution.", 10.1, { agent: "fundamentals", symbol: "TSLA", signal: "BEARISH", confidence: 0.64, summary: "Margin compression two quarters running; valuation still prices flawless execution.", error: null }),
      L("news", "TSLA", "INFO", "BEARISH: Delivery miss headlines dominate; European registrations down again.", 10.2, { agent: "news", symbol: "TSLA", sentiment: "BEARISH", confidence: 0.69, summary: "Delivery miss headlines dominate; European registrations down again.", error: null }),
      L("technical", "TSLA", "INFO", "BEARISH: Trading below SMA50 (252.3) and SMA200 (245.1); RSI 38.2; MACD rolling over.", 10.4, { agent: "technical", symbol: "TSLA", signal: "BEARISH", confidence: 0.76, summary: "Trading below SMA50 (252.3) and SMA200 (245.1); RSI 38.2; MACD rolling over.", error: null }),
      L("cio", "SPY", "INFO", "HOLD: Index agents neutral across the board; no edge. Cash yield beats a marginal index trade.", 10.7, { agent: "cio", symbol: "SPY", decision: "HOLD", confidence: 0.6, notional_usd: 0, reasoning: "Index agents neutral across the board; no edge. Cash yield beats a marginal index trade.", error: null }),
      L("risk", "SPY", "INFO", "Approved. Exposure acceptable.", 10.85, { agent: "risk", symbol: "SPY", approved: true, max_notional_usd: 2548, risk_level: "LOW", reasoning: "Approved. Exposure acceptable.", error: null }),
      L("debate", "SPY", "INFO", "Bull(0.5) vs Bear(0.5), edge=0.0", 11.0, { agent: "debate", symbol: "SPY", bull_summary: "Earnings breadth is holding and rate-cut odds are rising into year-end.", bull_strength: 0.5, bear_summary: "Valuations are full and positioning is crowded; seasonality is turning.", bear_strength: 0.5, edge: 0 }),
      L("fundamentals", "SPY", "INFO", "NEUTRAL: Aggregate P/E slightly above 5-year average.", 11.15, { agent: "fundamentals", symbol: "SPY", signal: "NEUTRAL", confidence: 0.52, summary: "Aggregate P/E slightly above 5-year average.", error: null }),
      L("news", "SPY", "INFO", "NEUTRAL: Macro headlines balanced.", 11.3, { agent: "news", symbol: "SPY", sentiment: "NEUTRAL", confidence: 0.5, summary: "Macro headlines balanced.", error: null }),
      L("technical", "SPY", "INFO", "NEUTRAL: Above SMA200 but below SMA50; RSI 51 — indecisive.", 11.5, { agent: "technical", symbol: "SPY", signal: "NEUTRAL", confidence: 0.54, summary: "Above SMA200 but below SMA50; RSI 51 — indecisive.", error: null }),
      L("memory", "TSLA", "INFO", "Trade closed. Realized P&L: -1.86%. Recorded for agent accuracy scoring.", 52, { realized_pl_pct: -1.86, symbol: "TSLA" }),
      L("system", null, "WARNING", "TSLA concentration at 9.3% of equity — approaching the configured 10% maximum position size.", 53),
      L("news", "TSLA", "ERROR", "News fetch failed for TSLA: Alpaca News API rate limit (429). Falling back to neutral sentiment.", 54, { error: "Alpaca News API rate limit (429)" }),
      L("execution", "TSLA", "INFO", "Order submitted: SELL TSLA", 52.5, { success: true, order_id: "ord-9f20", symbol: "TSLA", side: "sell", qty: 6, status: "filled", error: null }),
      L("cio", "TSLA", "INFO", "SELL: Bear case decisively stronger; trimming exposure by half while the risk agent still permits the trade.", 53, { agent: "cio", symbol: "TSLA", decision: "SELL", confidence: 0.77, notional_usd: 0, reasoning: "Bear case decisively stronger; trimming exposure by half while the risk agent still permits the trade.", error: null }),
      L("system", null, "INFO", "Bot started. Automated cycle is now active.", 1440 * 3),
      L("memory", "NVDA", "INFO", "Trade closed. Realized P&L: +2.46%. Recorded for agent accuracy scoring.", 1440 * 2.2, { realized_pl_pct: 2.46, symbol: "NVDA" }),
      L("memory", "AMD", "INFO", "Trade closed. Realized P&L: +5.10%. Recorded for agent accuracy scoring.", 1440 * 5, { realized_pl_pct: 5.1, symbol: "AMD" }),
    ];
    // give the TSLA risk warning its structured payload
    logs.forEach(function (l) {
      if (l.message === "Position size within configured limit; concentration approaching the configured threshold — flag for monitoring.") {
        l.data = { agent: "risk", symbol: "TSLA", approved: true, max_notional_usd: 800, risk_level: "MEDIUM", reasoning: "Position size within configured limit; concentration approaching the configured threshold — flag for monitoring.", error: null };
      }
    });

    var cycles = [];
    var cycDefs = [
      { id: 1842, minAgo: 11.8, dur: 18.4, status: "OK", trig: "scheduler", orders: 1 },
      { id: 1841, minAgo: 26.8, dur: 21.2, status: "PARTIAL_ERROR", trig: "scheduler", orders: 1 },
      { id: 1840, minAgo: 41.9, dur: 16.7, status: "OK", trig: "scheduler", orders: 0 },
      { id: 1839, minAgo: 57.1, dur: 19.3, status: "OK", trig: "scheduler", orders: 1 },
      { id: 1838, minAgo: 71.6, dur: 17.9, status: "OK", trig: "scheduler", orders: 0 },
      { id: 1837, minAgo: 87.2, dur: 24.8, status: "PARTIAL_ERROR", trig: "manual", orders: 2 },
      { id: 1836, minAgo: 101.4, dur: 18.1, status: "OK", trig: "scheduler", orders: 0 },
      { id: 1835, minAgo: 116.5, dur: 15.4, status: "OK", trig: "scheduler", orders: 1 },
    ];
    cycDefs.forEach(function (c) {
      cycles.push({
        id: c.id,
        started_at: t(c.minAgo),
        finished_at: t(c.minAgo - c.dur / 60),
        duration_s: c.dur,
        triggered_by: c.trig,
        status: c.status,
        symbols_processed: ["AAPL", "MSFT", "NVDA", "TSLA", "SPY"],
        decisions: [
          { symbol: "NVDA", decision: c.id === 1842 ? "BUY" : "HOLD", confidence: c.id === 1842 ? 0.84 : 0.52, notional_usd: c.id === 1842 ? 2100 : 0 },
          { symbol: "AAPL", decision: "HOLD", confidence: 0.55, notional_usd: 0 },
          { symbol: "MSFT", decision: "HOLD", confidence: 0.58, notional_usd: 0 },
          { symbol: "TSLA", decision: c.id === 1841 ? "SELL" : "HOLD", confidence: 0.77, notional_usd: 0 },
          { symbol: "SPY", decision: "HOLD", confidence: 0.6, notional_usd: 0 },
        ],
        orders: c.id === 1842 ? [{ symbol: "NVDA", side: "buy", order_id: "ord-9f21", status: "filled", success: true, notional_usd: 2100 }]
             : c.id === 1841 ? [{ symbol: "TSLA", side: "sell", order_id: "ord-9f20", status: "filled", success: true, qty: 6 }]
             : [],
        agent_status: {
          AAPL: { market_data: "OK", technical: "OK", news: "OK", fundamentals: "OK", debate: "OK", risk: "OK", cio: "OK", execution: "SKIPPED", memory: "SKIPPED" },
          MSFT: { market_data: "OK", technical: "OK", news: "OK", fundamentals: "OK", debate: "OK", risk: "OK", cio: "OK", execution: "SKIPPED", memory: "SKIPPED" },
          NVDA: { market_data: "OK", technical: "OK", news: "OK", fundamentals: "OK", debate: "OK", risk: "OK", cio: "OK", execution: c.id === 1842 ? "OK" : "SKIPPED", memory: c.id === 1842 ? "OK" : "SKIPPED" },
          TSLA: { market_data: "OK", technical: "OK", news: "ERROR", fundamentals: "UNAVAILABLE", debate: "OK", risk: "OK", cio: "OK", execution: c.id === 1841 ? "OK" : "SKIPPED", memory: "SKIPPED" },
          SPY: { market_data: "OK", technical: "OK", news: "OK", fundamentals: "OK", debate: "OK", risk: "OK", cio: "OK", execution: "SKIPPED", memory: "SKIPPED" },
        },
        llm_usage: {
          groq: { requests: 15, ok: c.status === "PARTIAL_ERROR" ? 14 : 15,
                  errors: c.status === "PARTIAL_ERROR" ? { PROVIDER_ERROR: 1 } : {},
                  by_agent: { technical: { requests: 5, ok: 5, errors: {} }, debate: { requests: 5, ok: 5, errors: {} }, cio: { requests: 5, ok: c.status === "PARTIAL_ERROR" ? 4 : 5, errors: c.status === "PARTIAL_ERROR" ? { PROVIDER_ERROR: 1 } : {} } },
                  by_symbol: {} },
          gemini: { requests: 7, ok: 7, errors: {},
                  by_agent: { news: { requests: 5, ok: 5, errors: {} }, fundamentals: { requests: 2, ok: 2, errors: {} } },
                  by_symbol: {} },
          openrouter: { requests: 5, ok: 5, errors: {},
                  by_agent: { risk: { requests: 5, ok: 5, errors: {} } },
                  by_symbol: {} },
        },
        agent_results: {
          market_data: { ok: 5, unavailable: 0, error: 0, skipped: 0, total: 5 },
          technical: { ok: 5, unavailable: 0, error: 0, skipped: 0, total: 5 },
          news: c.status === "PARTIAL_ERROR" ? { ok: 4, unavailable: 1, error: 0, skipped: 0, total: 5 } : { ok: 5, unavailable: 0, error: 0, skipped: 0, total: 5 },
          fundamentals: { ok: 0, unavailable: 0, error: 0, skipped: 5, total: 5 },
          debate: { ok: 5, unavailable: 0, error: 0, skipped: 0, total: 5 },
          risk: { ok: 5, unavailable: 0, error: 0, skipped: 0, total: 5 },
          cio: { ok: 5, unavailable: 0, error: 0, skipped: 0, total: 5 },
          execution: c.id === 1842 ? { ok: 1, unavailable: 0, error: 0, skipped: 4, total: 5 } : { ok: 0, unavailable: 0, error: 0, skipped: 5, total: 5 },
          memory: { ok: 0, unavailable: 0, error: 0, skipped: 5, total: 5 },
        },
        provider_results: {
          llm_states: {
            groq: { state: "READY", detail: "20 requests in last 24h" },
            gemini: { state: "READY", detail: "5 requests in last 24h" },
            nvidia: { state: "NOT_CONFIGURED", detail: "NVIDIA_API_KEY not set" },
            openrouter: { state: "NOT_CONFIGURED", detail: "OPENROUTER_MODEL not set" },
          },
          market_data: { feed: "IEX", symbols_ok: 5, symbols_unavailable: 0 },
          fundamentals: { provider: "none", symbols_ok: 0, symbols_unavailable: 5 },
        },
        warnings: c.status === "PARTIAL_ERROR" ? 1 : 0,
        errors: c.status === "PARTIAL_ERROR" ? ["TSLA: News fetch failed: Alpaca News API rate limit (429)"] : [],
      });
    });

    store.data = {
      portfolio: {
        account: { cash: cash, equity: eq, buying_power: cash, portfolio_value: eq, day_pl: 428.62, day_pl_pct: 1.71, status: "ACTIVE", error: null },
        positions: positions,
        orders: orders.slice(0, 15),
        history: demoHistory("1M", eq),
        bot_state: { running: true, started_at: t(1440 * 3), last_cycle_at: t(6), last_cycle_status: "OK", uptime_status: "ONLINE" },
        trade_universe: ["AAPL", "MSFT", "NVDA", "TSLA", "SPY"],
      },
      logs: { logs: logs },
      cycles: { cycles: cycles },
      accuracy: {
        enabled: true,
        agents: {
          technical: { agent: "technical", sample_size: 38, hit_rate: 0.684, weight: 1.184 },
          news: { agent: "news", sample_size: 41, hit_rate: 0.612, weight: 1.112 },
          fundamentals: { agent: "fundamentals", sample_size: 29, hit_rate: 0.728, weight: 1.228 },
        },
      },
      orders: { orders: orders, error: null },
      config: {
        trade_universe: ["AAPL", "MSFT", "NVDA", "TSLA", "SPY"],
        cycle_interval_minutes: 15,
        max_position_pct: 0.10,
        enable_fundamentals_agent: true,
        enable_debate: true,
        enable_memory: true,
        trading_mode: "PAPER",
        memory_db_path: "data/memory.db",
        agent_accuracy_lookback: 20,
        llm_routes: {
          cio: { provider: "groq", model: "openai/gpt-oss-120b", fallback_provider: "", fallback_model: "" },
          debate: { provider: "groq", model: "openai/gpt-oss-20b", fallback_provider: "", fallback_model: "" },
          fundamentals: { provider: "gemini", model: "gemini-3.6-flash", fallback_provider: "", fallback_model: "" },
          news: { provider: "gemini", model: "gemini-3.6-flash", fallback_provider: "", fallback_model: "" },
          risk: { provider: "groq", model: "openai/gpt-oss-20b", fallback_provider: "", fallback_model: "" },
          technical: { provider: "groq", model: "openai/gpt-oss-20b", fallback_provider: "", fallback_model: "" },
        },
        llm_model_validation: {
          providers: {
            groq: { ok: true, models_found: 12, checked: { "technical.model": true, "debate.model": true, "risk.model": true, "cio.model": true }, missing: [], error: null },
            gemini: { ok: true, models_found: 21, checked: { "news.model": true, "fundamentals.model": true }, missing: [], error: null },
            nvidia: { ok: false, models_found: 0, checked: {}, missing: [], error: "NVIDIA_API_KEY not configured — catalog not fetched" },
            openrouter: { ok: false, models_found: 0, checked: {}, missing: [], error: "OPENROUTER_MODEL not set — provider not routed" },
          },
          routes: {},
        },
        llm_quota: {
          groq: { daily_request_limit: 0, requests_last_24h: 20, backoff_active: false, backoff_seconds_remaining: 0 },
          gemini: { daily_request_limit: 20, requests_last_24h: 5, backoff_active: false, backoff_seconds_remaining: 0 },
          openrouter: { daily_request_limit: 0, requests_last_24h: 0, backoff_active: false, backoff_seconds_remaining: 0 },
        },
        llm_provider_states: {
          groq: { state: "READY", detail: "20 requests in last 24h", requests_last_24h: 20, daily_request_limit: 0, quota_cooldown_remaining_s: 0, auth_cooldown_remaining_s: 0 },
          gemini: { state: "READY", detail: "5 requests in last 24h", requests_last_24h: 5, daily_request_limit: 20, quota_cooldown_remaining_s: 0, auth_cooldown_remaining_s: 0 },
          nvidia: { state: "NOT_CONFIGURED", detail: "NVIDIA_API_KEY not set", requests_last_24h: 0, daily_request_limit: 0, quota_cooldown_remaining_s: 0, auth_cooldown_remaining_s: 0 },
          openrouter: { state: "NOT_CONFIGURED", detail: "OPENROUTER_MODEL not set", requests_last_24h: 0, daily_request_limit: 0, quota_cooldown_remaining_s: 0, auth_cooldown_remaining_s: 0 },
        },
        fundamentals: { primary: "none", fallback: null, available: ["none"] },
        market_data: { configured_feed: "IEX", note: "IEX is the free/paper-subscription feed" },
        realtime_enabled: true,
        warnings: [],
      },
      health: {
        checked_at: "2026-09-09T18:00:00Z",
        startup: {
          rows: [
            { component: "ALPACA", status: "READY", detail: "paper account OK (equity $25,104.00, status ACTIVE)" },
            { component: "MARKET DATA", status: "READY", detail: "bars OK for SPY on feed=IEX" },
            { component: "FUNDAMENTALS", status: "DATA_UNAVAILABLE", detail: "FUNDAMENTALS_PROVIDER=none — no fundamentals source configured; the bot runs on price/technical/news/risk evidence" },
            { component: "GROQ", status: "READY", detail: "4 model id(s) verified against the live catalog" },
            { component: "NVIDIA", status: "NOT_CONFIGURED", detail: "NVIDIA_API_KEY not set" },
            { component: "GEMINI", status: "READY", detail: "2 model id(s) verified against the live catalog" },
            { component: "OPENROUTER", status: "NOT_CONFIGURED", detail: "OPENROUTER_MODEL not set" },
          ],
        },
        providers: {},
        market_clock: { is_open: true, timestamp: null, next_open: null, next_close: null, error: null },
        realtime: { status: "DISABLED", feed: null, symbols: [], data: {}, news: [], last_event_at: null, last_error: null, reconnects: 0 },
        fundamentals: { primary: "none", fallback: null, available: ["none"] },
        configured_feed: "IEX",
      },
      realtime: {
        status: "CONNECTED",
        feed: "iex",
        symbols: ["AAPL", "MSFT", "NVDA", "TSLA", "SPY"],
        data: {
          AAPL: { last_trade: { price: 175.25, size: 3, timestamp: null }, last_quote: { bid_price: 175.1, ask_price: 175.3, bid_size: 2, ask_size: 2, timestamp: null }, minute_bar: null, updated_at: 1757433600 },
          MSFT: { last_trade: { price: 421.5, size: 2, timestamp: null }, last_quote: { bid_price: 421.4, ask_price: 421.6, bid_size: 1, ask_size: 3, timestamp: null }, minute_bar: null, updated_at: 1757433600 },
          NVDA: { last_trade: { price: 132.8, size: 5, timestamp: null }, last_quote: { bid_price: 132.75, ask_price: 132.85, bid_size: 4, ask_size: 4, timestamp: null }, minute_bar: null, updated_at: 1757433600 },
          TSLA: { last_trade: { price: 248.4, size: 1, timestamp: null }, last_quote: { bid_price: 248.3, ask_price: 248.5, bid_size: 2, ask_size: 1, timestamp: null }, minute_bar: null, updated_at: 1757433600 },
          SPY: { last_trade: { price: 548.9, size: 12, timestamp: null }, last_quote: { bid_price: 548.85, ask_price: 548.95, bid_size: 8, ask_size: 8, timestamp: null }, minute_bar: null, updated_at: 1757433600 },
        },
        news: [{ headline: "Chip demand keeps accelerating into Q4", summary: "", source: "api", symbols: ["NVDA"], created_at: null }],
        last_event_at: 1757433600,
        last_error: null,
        reconnects: 0,
      },
    };
    Object.keys(store.data).forEach(function (k) { store.ts[k] = Date.now(); });
    recomputeDerived();
    store.ui.seenCycles = {};
    cycles.forEach(function (c) { store.ui.seenCycles[c.id] = c.status; });
  }

  function demoBotAction(action) {
    if (action === "start") {
      store.data.portfolio.bot_state.running = true;
      toast("Bot started — autonomous cycles are active", "ok");
      window.App.renderPage(store.ui.page, store.ui.params, true);
    } else if (action === "stop") {
      confirmModal({
        title: "Stop the trading bot?",
        icon: "power", tone: "err", danger: true, okLabel: "Stop Bot",
        body: "<p>Scheduled trading cycles will be paused. The bot will not analyze symbols or submit orders until it is started again.</p>" +
          '<div class="note" style="margin-top:8px">' + ICON("info") + "<span>Open positions are left untouched. Orders already submitted to the paper account are not cancelled.</span></div>",
        onOk: function () {
          store.data.portfolio.bot_state.running = false;
          toast("Bot stopped — cycles paused", "info");
          window.App.renderPage(store.ui.page, store.ui.params, true);
        },
      });
    } else if (action === "run") {
      var btns = qsa("[data-action='run-cycle']");
      btns.forEach(function (b) { b.disabled = true; b.innerHTML = '<span class="btn-spinner"></span>Running…'; });
      toast("Cycle started — running the full agent pipeline", "info");
      var cid = (store.data.cycles.cycles[0] || { id: 1842 }).id + 1;
      var stages = [
        ["technical", "NVDA", "BULLISH momentum intact — running pipeline stage 1 of 9"],
        ["news", "NVDA", "Scanning headlines: 10 articles retrieved"],
        ["fundamentals", "NVDA", "Pulling yfinance fundamentals"],
        ["debate", "NVDA", "Bull and bear researchers arguing the case"],
        ["risk", "NVDA", "Assessing position limits and exposure"],
        ["cio", "NVDA", "Weighing agent consensus and memory"],
        ["execution", "NVDA", "Submitting paper order"],
      ];
      stages.forEach(function (st, i) {
        setTimeout(function () {
          store.data.logs.logs.unshift({
            agent: st[0], symbol: st[1], level: "INFO",
            message: "· DEMO SIMULATION · " + st[2],
            timestamp: new Date().toISOString(),
            data: st[0] === "cio" ? { agent: "cio", symbol: "NVDA", decision: "HOLD", confidence: 0.61, notional_usd: 0, reasoning: "Demo simulation — the live dashboard renders real CIO reasoning here." } : null,
          });
          recomputeDerived();
          emit("data");
        }, 600 * (i + 1));
      });
      setTimeout(function () {
        store.data.cycles.cycles.unshift({
          id: cid, started_at: new Date(Date.now() - 12000).toISOString(), finished_at: new Date().toISOString(),
          duration_s: 12.1, triggered_by: "manual", status: "OK",
          symbols_processed: ["AAPL", "MSFT", "NVDA", "TSLA", "SPY"],
          decisions: [{ symbol: "NVDA", decision: "HOLD", confidence: 0.61, notional_usd: 0 }],
          orders: [], warnings: 0, errors: [],
        });
        store.data.portfolio.bot_state.last_cycle_at = new Date().toISOString();
        btns.forEach(function (b) { b.disabled = false; b.innerHTML = ICON("play") + "Run Cycle"; });
        toast("Cycle #" + cid + " completed · 12.1s · 5 symbols", "ok");
        emit("data");
        window.App.renderPage(store.ui.page, store.ui.params, true);
      }, 600 * 8);
    }
  }

  /* ==========================================================================
     Global event delegation
     ========================================================================== */

  document.addEventListener("click", function (e) {
    var t = e.target.closest("[data-action]");
    if (!t) return;
    var a = t.getAttribute("data-action");

    if (a === "close-overlay") return closeOverlay();
    if (a === "run-cycle") return botAction("run");
    if (a === "start-bot") return botAction("start");
    if (a === "stop-bot") return botAction("stop");
    if (a === "refresh") {
      if (store.demo) { toast("Demo mode — connect the backend for live data", "info"); return; }
      t.disabled = true; t.innerHTML = '<span class="btn-spinner"></span>';
      Promise.all([pollLight(), pollSlow()]).then(function () {
        t.disabled = false; t.innerHTML = ICON("refresh") + "Refresh";
        window.App.renderPage(store.ui.page, store.ui.params, true);
      });
      return;
    }
    if (a === "goto") { navigate(t.getAttribute("data-page"), t.getAttribute("data-sym") ? { sym: t.getAttribute("data-sym") } : null); return; }
    if (a === "pal-open") return openPalette();
    if (a === "palette-close") return closePalette();
    if (a === "open-position") return window.App.openPositionDrawer(t.getAttribute("data-sym"));
    if (a === "open-order") return window.App.openOrderDrawer(t.getAttribute("data-id"));
    if (a === "open-decision") return window.App.openDecisionDrawer(t.getAttribute("data-ts"), t.getAttribute("data-agent"));
    if (a === "open-cycle") return window.App.openCycleDrawer(t.getAttribute("data-id"));
    if (a === "open-agent") return window.App.openAgentDrawer(t.getAttribute("data-agent"));
    if (a === "open-log") return window.App.openLogDrawer(t.getAttribute("data-ts"));
    if (a === "toggle-reason") {
      var r = t.closest(".reason");
      r.setAttribute("data-open", r.getAttribute("data-open") === "true" ? "false" : "true");
      t.setAttribute("aria-expanded", r.getAttribute("data-open") === "true");
      return;
    }
    if (a === "more-open") { qs("#more-root").classList.add("on"); document.body.style.overflow = "hidden"; return; }
    if (a === "more-close") { qs("#more-root").classList.remove("on"); document.body.style.overflow = ""; return; }
  });

  document.addEventListener("keydown", function (e) {
    if (e.key === "Tab") trapFocus(e);
    if ((e.ctrlKey || e.metaKey) && e.key.toLowerCase() === "k") { e.preventDefault(); openPalette(); }
    else if (e.key === "Escape") {
      if (qs("#palette-root").innerHTML) closePalette();
      else if (qs("#modal-root").innerHTML) closeModal();
      else if (qs("#drawer-root").innerHTML) closeOverlay();
      else if (qs("#more-root") && qs("#more-root").classList.contains("on")) { qs("#more-root").classList.remove("on"); document.body.style.overflow = ""; }
    }
    else if (e.key === "/" && !/input|textarea|select/i.test((e.target.tagName || ""))) { e.preventDefault(); openPalette(); }
  });

  /* Focus trap: keeps Tab cycling inside the open dialog (WCAG). */
  function trapFocus(e) {
    var root = qs("#drawer-root .drawer") || qs("#modal-root .modal") || qs("#palette-root .palette");
    if (!root) return;
    var focusables = qsa('a[href], button:not([disabled]), input:not([disabled]), select, textarea, [tabindex]:not([tabindex="-1"])', root)
      .filter(function (el) { return el.offsetParent !== null || el === document.activeElement; });
    if (!focusables.length) return;
    var first = focusables[0], last = focusables[focusables.length - 1];
    if (!root.contains(document.activeElement)) { e.preventDefault(); first.focus(); }
    else if (e.shiftKey && document.activeElement === first) { e.preventDefault(); last.focus(); }
    else if (!e.shiftKey && document.activeElement === last) { e.preventDefault(); first.focus(); }
  }

  window.addEventListener("hashchange", renderRoute);
  var resizeT;
  window.addEventListener("resize", function () { clearTimeout(resizeT); resizeT = setTimeout(redrawCharts, 160); });

  /* ==========================================================================
     Export
     ========================================================================== */

  window.App = {
    U: U, API: API, ICON: ICON,
    qs: qs, qsa: qsa, h: h,
    store: store, derived: derived,
    on: on, emit: emit,
    navigate: navigate,
    startPolling: startPolling,
    pollLight: pollLight, pollSlow: pollSlow,
    toast: toast,
    openDrawer: openDrawer, setDrawerBody: setDrawerBody, closeOverlay: closeOverlay,
    openModal: openModal, closeModal: closeModal, confirmModal: confirmModal,
    botAction: botAction,
    openPalette: openPalette,
    chart: chart, clearCharts: clearCharts, redrawCharts: redrawCharts, sparkline: sparkline,
    positionsWithContext: positionsWithContext,
    matchOrdersToCycles: matchOrdersToCycles,
    loadDemo: loadDemo,

    /* Live market strip (Alpaca WebSockets via /api/realtime). Ticks only
       refresh this strip — they never trigger LLM calls. */
    updateLiveStrip: function (el) {
      var rt = store.data.realtime;
      if (!rt || !rt.data || !Object.keys(rt.data).length) { el.innerHTML = ""; return; }
      var tone = rt.status === "CONNECTED" ? "badge-ok" : rt.status === "DISCONNECTED" ? "badge-error" : "badge-neutral";
      var cells = Object.keys(rt.data).map(function (s) {
        var d = rt.data[s] || {};
        var t = d.last_trade || {}, q = d.last_quote || {};
        var px = t.price != null ? t.price : (q.bid_price != null && q.ask_price != null ? (q.bid_price + q.ask_price) / 2 : null);
        return (
          '<div class="card" style="padding:8px 12px;min-width:120px">' +
            '<div class="row" style="justify-content:space-between"><span class="sym">' + U.esc(s) + '</span>' +
            '<span style="font-size:12px;font-weight:650">' + (px != null ? U.fmtMoney(px) : "—") + "</span></div>" +
            '<div class="t-faint" style="font-size:10.5px">' + (q.bid_price != null ? "bid " + U.fmtMoney(q.bid_price) + " · ask " + U.fmtMoney(q.ask_price) : "no quote yet") + "</div>" +
          "</div>"
        );
      }).join("");
      el.innerHTML =
        '<div class="row" style="justify-content:space-between;margin-bottom:6px">' +
          '<span class="row" style="gap:7px">' + ICON("pulse") + '<span style="font-size:12px;font-weight:650;color:var(--ink-hi)">Live Market</span>' +
          '<span class="badge ' + tone + '">' + U.esc(rt.status) + "</span></span>" +
          '<span class="t-faint" style="font-size:10.5px">Alpaca WebSocket · feed ' + U.esc(rt.feed || "—") + " · ticks never trigger AI</span>" +
        "</div>" +
        '<div class="row" style="gap:8px;overflow-x:auto">' + cells + "</div>";
    },
    PAGES: null, PAGE_TITLES: null, PAGE_ICONS: null,
  };
})();
