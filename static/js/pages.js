/* ============================================================================
   AI TRADER — Pages
   overview · portfolio · positions · orders · ai · agents · risk · cycles ·
   health · activity · config
   ========================================================================== */
(function () {
  "use strict";

  var U = window.U, ICON = window.ICON, App = window.App, C = window.C;
  var store = App.store;
  var qs = App.qs, qsa = App.qsa;

  var P = {};

  /* helpers ---------------------------------------------------------------- */

  function botState() {
    return (store.data.portfolio && store.data.portfolio.bot_state) || {};
  }
  function account() {
    return (store.data.portfolio && store.data.portfolio.account) || {};
  }
  function logs() { return (store.data.logs && store.data.logs.logs) || []; }
  function cycles() { return (store.data.cycles && store.data.cycles.cycles) || []; }
  function config() { return store.data.config || {}; }

  /* Endpoint availability: "loading" (never responded), "down" (failed with
     no data), or "ok". Drives honest error states instead of skeletons. */
  function guard(name) {
    if (store.data[name] === undefined) return "loading";
    if (store.data[name] && store.data[name].__unavailable) return "down";
    return "ok";
  }

  function downStateHTML(err, what) {
    return (
      C.bannerHTML("err", (what || "Live data") + " unavailable.",
        (err ? U.esc(err) : "The backend did not respond.") + " Check that the bot server is running, then retry.",
        '<button class="btn btn-sm" data-action="refresh">' + ICON("refresh") + "Retry</button>") +
      '<section class="card"><div class="card-bd">' +
        C.stateHTML({
          err: true, icon: "offline", title: "API unavailable",
          msg: "This page populates automatically as soon as the backend responds. The dashboard keeps monitoring in the background.",
        }) +
      "</div></section>"
    );
  }

  function headCard(title, icon, inner, foot) {
    return (
      '<section class="card">' +
        '<div class="card-hd">' + ICON(icon) + '<span class="card-title">' + title + "</span>" + (foot || "") + "</div>" +
        inner +
      "</section>"
    );
  }

  function kpiTile(label, valueHTML, subHTML, cls) {
    return (
      '<div class="kpi ' + (cls || "") + '">' +
        '<span class="kpi-lbl">' + label + "</span>" +
        '<span class="kpi-val">' + valueHTML + "</span>" +
        (subHTML ? '<span class="kpi-sub">' + subHTML + "</span>" : "") +
      "</div>"
    );
  }

  function accountErrorBanner(acct) {
    if (!acct.error) return "";
    return C.bannerHTML("err", "Broker API unavailable.",
      U.esc(acct.error) + " Portfolio data cannot be refreshed while the Alpaca API is unreachable.");
  }

  function botOfflineBanner() {
    var st = botState();
    if (st.running) return "";
    return C.bannerHTML("off", "Bot is offline.",
      "The trading engine is currently stopped. No autonomous cycles will run.",
      '<button class="btn btn-sm btn-success" data-action="start-bot">' + ICON("power") + "Start Bot</button>");
  }

  function ringHTML(pct, label, sub, colorVar) {
    var r = 26, c = 2 * Math.PI * r;
    var off = c * (1 - U.clamp(pct, 0, 100) / 100);
    return (
      '<div class="row" style="gap:14px">' +
        '<svg width="68" height="68" viewBox="0 0 68 68" role="img" aria-label="' + U.esc(label) + " " + pct.toFixed(0) + '%">' +
          '<circle cx="34" cy="34" r="' + r + '" fill="none" stroke="var(--bg-inset)" stroke-width="6"/>' +
          '<circle cx="34" cy="34" r="' + r + '" fill="none" stroke="var(' + (colorVar || "--blue") + ')" stroke-width="6" stroke-linecap="round" stroke-dasharray="' + c.toFixed(1) + '" stroke-dashoffset="' + off.toFixed(1) + '" transform="rotate(-90 34 34)"/>' +
          '<text x="34" y="38" text-anchor="middle" fill="var(--ink-hi)" font-size="13" font-weight="600" font-family="var(--font-mono)">' + pct.toFixed(0) + "%</text>" +
        "</svg>" +
        '<div><div style="font-size:12.5px;font-weight:650;color:var(--ink-hi)">' + U.esc(label) + "</div>" +
        (sub ? '<div class="t-faint" style="font-size:11px;margin-top:2px;max-width:200px">' + sub + "</div>" : "") + "</div>" +
      "</div>"
    );
  }

  /* ==========================================================================
     1 · OVERVIEW
     ========================================================================== */

  P.overview = function () {
    var g = guard("portfolio");
    if (g === "loading") return C.pageSkeleton();
    if (g === "down") return downStateHTML(store.errors.portfolio, "Portfolio data");
    var pf = store.data.portfolio;
    var acct = account();
    var st = botState();

    var banners = accountErrorBanner(acct) + botOfflineBanner();

    // KPI strip
    var dayPos = acct.day_pl >= 0;
    var kpis =
      kpiTile("Portfolio Value",
        U.fmtMoney(acct.portfolio_value || acct.equity),
        '<span class="num ' + (dayPos ? "pos" : "neg") + '">' + U.fmtSigned(acct.day_pl) + " (" + U.fmtPct(acct.day_pl_pct) + ")</span> today", "kpi-hero") +
      kpiTile("Equity", acct.equity != null ? U.fmtMoney(acct.equity) : "—",
        acct.equity ? "<span>Long + cash</span>" : "") +
      kpiTile("Cash", acct.cash != null ? U.fmtMoney(acct.cash) : "—",
        '<span>' + U.fmtPct(acct.equity ? (acct.cash / acct.equity) * 100 : null, 1) + " of equity</span>") +
      kpiTile("Buying Power", acct.buying_power != null ? U.fmtMoney(acct.buying_power) : "—",
        '<span class="t-faint">Paper account</span>') +
      kpiTile("Today's P&L",
        '<span class="' + (dayPos ? "pos" : "neg") + '">' + U.fmtSigned(acct.day_pl) + "</span>",
        '<span class="num ' + (dayPos ? "pos" : "neg") + '">' + U.fmtPct(acct.day_pl_pct) + "</span> vs last close") +
      kpiTile("Open Positions", String((pf.positions || []).length),
        (pf.positions || []).length ? '<span class="linklike" data-action="goto" data-page="positions">View positions →</span>' : '<span class="t-faint">Flat</span>');

    // chart card (uses /api/history or portfolio.history)
    var chartCard = P._chartCard({ compactHead: false });

    // latest decision panel
    var latest = C.latestCioLogs(1)[0];
    var decisionPanel;
    if (latest) {
      var win = C.logsForDecision(latest);
      var v = C.verdictOf(latest);
      decisionPanel =
        '<div class="card-hd">' + ICON("gavel") + '<span class="card-title">Latest AI Decision</span>' +
          '<span class="spacer"></span><span class="aux" data-ago="' + U.esc(latest.timestamp) + '">' + U.ago(latest.timestamp) + "</span></div>" +
        '<div class="card-bd" style="padding-top:10px">' +
          '<div class="row row-wrap" style="margin-bottom:10px;gap:10px">' +
            '<span class="sym" style="font-size:15px">' + U.esc(latest.symbol || "") + "</span>" +
            (v ? '<span class="' + v.badge + '" style="font-size:11px">' + v.label + "</span>" : "") +
            (latest.data && latest.data.confidence != null ? '<span class="badge badge-ai">CONF ' + Math.round(latest.data.confidence * 100) + "%</span>" : "") +
            '<span class="feed-time">' + U.fmtTime(latest.timestamp) + "</span>" +
          "</div>" +
          '<div style="font-size:12.5px;color:var(--ink);line-height:1.6;margin-bottom:12px">' + U.esc(latest.data && latest.data.reasoning ? latest.data.reasoning : latest.message) + "</div>" +
          '<div class="row" style="gap:8px">' +
            '<button class="btn btn-sm" data-action="open-decision" data-ts="' + U.esc(latest.timestamp) + '" data-agent="cio">' + ICON("brain") + "Full reasoning</button>" +
            '<button class="btn btn-sm" data-action="goto" data-page="ai">' + ICON("cpu") + "AI Intelligence</button>" +
          "</div>" +
        "</div>";
    } else {
      decisionPanel =
        '<div class="card-hd">' + ICON("gavel") + '<span class="card-title">Latest AI Decision</span></div>' +
        '<div class="card-bd">' + C.stateHTML({ icon: "brain", title: "No decisions yet", msg: "Once the bot completes a cycle, the CIO's latest decision and reasoning will appear here.", compact: true }) + "</div>";
    }

    // pipeline mini + feed
    var feedLogs = logs().slice(0, 9);
    var feedCard =
      '<div class="card-hd">' + ICON("activity") + '<span class="card-title">Agent Activity</span><span class="spacer"></span>' +
        '<button class="btn btn-sm btn-ghost" data-action="goto" data-page="activity">All activity</button></div>' +
      (feedLogs.length
        ? '<div class="feed">' + feedLogs.map(function (l) { return C.feedItemHTML(l, { newAfter: store.ui.lastLogTs }); }).join("") + "</div>"
        : '<div class="card-bd">' + C.stateHTML({ icon: "activity", title: "No activity yet", msg: "Agent events will stream here as cycles run.", compact: true }) + "</div>");

    // latest cycle mini
    var lastC = cycles()[0];
    var cycleCard =
      '<div class="card-hd">' + ICON("repeat") + '<span class="card-title">Latest Cycle</span>' +
        '<span class="spacer"></span>' + (lastC ? '<span class="' + U.badgeForCycle(lastC.status) + '">' + U.esc(String(lastC.status).replace("_", " ")) + "</span>" : "") + "</div>" +
      '<div class="card-bd">' +
        (lastC
          ? C.kvHTML([
              ["Cycle", "#" + lastC.id],
              ["Started", U.fmtTime(lastC.started_at)],
              ["Duration", lastC.duration_s != null ? lastC.duration_s + "s" : "—"],
              ["Symbols", (lastC.symbols_processed || []).length],
              ["Decisions", (lastC.decisions || []).length],
              ["Orders", (lastC.orders || []).length],
            ]) +
            '<button class="btn btn-sm" style="margin-top:10px" data-action="open-cycle" data-id="' + lastC.id + '">' + ICON("eye") + "Cycle detail</button>"
          : C.stateHTML({ icon: "repeat", title: "No cycles recorded", msg: "Cycle history is in-memory and resets when the server restarts.", compact: true })) +
      "</div>";

    var posTable = headCard("Open Positions", "layers",
      '<div class="card-bd-flush">' + C.positionsTableHTML(App.positionsWithContext()) + "</div>",
      '<span class="spacer"></span><button class="btn btn-sm btn-ghost" data-action="goto" data-page="positions">All positions</button>');

    return (
      (store.demo ? window.DemoBanner : "") +
      banners +
      '<div class="kpi-strip mb-12">' + kpis + "</div>" +
      '<div class="grid g-main mb-12">' +
        '<div class="span-main">' + chartCard + "</div>" +
        '<div class="span-side col" style="gap:12px">' +
          '<section class="card">' + decisionPanel + "</section>" +
          '<section class="card">' + cycleCard + "</section>" +
        "</div>" +
      "</div>" +
      posTable +
      '<div class="grid g-main" style="margin-top:12px">' +
        '<div class="span-main">' + feedCard + "</div>" +
        '<div class="span-side">' + headCard("System", "cpu",
          '<div class="card-bd">' + healthMini() + "</div>",
          '<span class="spacer"></span><button class="btn btn-sm btn-ghost" data-action="goto" data-page="health">System health</button>') + "</div>" +
      "</div>"
    );
  };

  function healthMini() {
    var st = botState();
    var cfg = config();
    var errs = logs().filter(function (l) { return l.level === "ERROR"; }).length;
    var warns = logs().filter(function (l) { return l.level === "WARNING"; }).length;
    var agentNames = ["technical", "news", "fundamentals", "debate", "risk", "cio", "execution", "memory", "system"];
    var disabled = C.disabledAgentSet();
    var healthy = agentNames.filter(function (a) { return !disabled[a]; }).length;
    return C.kvHTML([
      ["Bot process", st.running ? '<span class="badge badge-online">ONLINE</span>' : '<span class="badge badge-offline">OFFLINE</span>'],
      ["Broker API", account().error ? '<span class="badge badge-error">UNREACHABLE</span>' : '<span class="badge badge-connected">CONNECTED</span>'],
      ["Scheduler", st.running ? '<span class="badge badge-running">RUNNING</span>' : '<span class="badge badge-neutral">PAUSED</span>'],
      ["Agent services", healthy + "/" + agentNames.length + " enabled"],
      ["Recent errors", errs ? '<span class="neg">' + errs + "</span>" : '<span class="pos">0</span>'],
      ["Recent warnings", warns ? '<span style="color:var(--amber)">' + warns + "</span>" : '<span class="pos">0</span>'],
    ]);
  }

  /* Chart card (shared by overview + portfolio) ------------------------------ */

  P._chartCard = function (opts) {
    opts = opts || {};
    var pf = store.data.portfolio || {};
    var st = botState();
    var ranges = ["1D", "1W", "1M", "3M", "6M", "1Y", "ALL"];
    var cur = store.ui.range || "1M";

    var seg =
      '<div class="seg" role="tablist" aria-label="Chart range">' +
        ranges.map(function (r) {
          return '<button role="tab" aria-selected="' + (r === cur) + '" data-range="' + r + '">' + r + "</button>";
        }).join("") +
      "</div>";

    // data: use history endpoint cache if present, else portfolio.history (1M)
    var hkey = "history:" + cur;
    var pts = [];
    var source = "api";
    if (store.histCache[cur] && store.histCache[cur].points) {
      pts = store.histCache[cur].points;
    } else if (cur === "1M" && pf.history && pf.history.length) {
      pts = pf.history; source = "portfolio";
    }

    var acct = account();
    var last = pts.length ? pts[pts.length - 1].v : (acct.portfolio_value || acct.equity);
    var first = pts.length ? pts[0].v : null;
    var delta = first != null ? last - first : null;
    var deltaPct = first ? (delta / first) * 100 : null;

    var head =
      '<div class="chart-head">' +
        "<div>" +
          '<div class="lbl" style="margin-bottom:3px">Portfolio value</div>' +
          '<span class="chart-big">' + U.fmtMoney(last) + "</span>" +
          '<div class="chart-delta">' +
            (delta != null
              ? '<span class="' + U.classFor(delta) + '">' + U.fmtSigned(delta) + "</span>" +
                '<span class="' + U.classFor(delta) + '">' + U.fmtPct(deltaPct) + "</span>" +
                '<span class="t-faint" style="font-family:var(--font-ui);font-size:11px">· ' + cur + " range</span>"
              : '<span class="t-faint">No history for this range</span>') +
          "</div>" +
        "</div>" +
        '<div class="spacer"></div>' +
        seg +
      "</div>";

    // map to chart points {t, v}
    var cpts = pts.map(function (p) {
      return { t: U.parseDate(p.timestamp != null ? p.timestamp : p.t).getTime(), v: p.equity != null ? p.equity : p.v };
    }).filter(function (p) { return p.t && p.v != null; });

    return (
      '<section class="card">' +
        '<div class="card-bd">' +
          head +
          '<div id="equity-chart" class="chart-wrap" style="min-height:300px" aria-label="Portfolio equity chart"></div>' +
          '<details class="chart-data">' +
            '<summary>' + ICON("download") + "View chart data as table (accessible alternative)</summary>" +
            '<div class="tbl-wrap"><table class="tbl"><thead><tr><th>Time</th><th class="r">Value</th><th class="r">Δ</th></tr></thead><tbody id="chart-fallback"></tbody></table></div>' +
          "</details>" +
        "</div>" +
        '<div class="card-ft">' +
          '<span class="freshness"><span class="dot" id="freshness-dot"></span><span id="freshness">—</span></span>' +
          '<span class="spacer"></span>' +
          '<span class="t-faint">Equity curve · paper account</span>' +
        "</div>" +
      "</section>"
    );
  };

  // range switch behavior is bound after render (bindChart)

  /* ==========================================================================
     2 · PORTFOLIO
     ========================================================================== */

  P.portfolio = function () {
    var g = guard("portfolio");
    if (g === "loading") return C.pageSkeleton();
    if (g === "down") return downStateHTML(store.errors.portfolio, "Portfolio data");
    var pf = store.data.portfolio;
    var acct = account();
    var st = botState();
    var positions = App.positionsWithContext();

    var invested = positions.reduce(function (a, p) { return a + (p.market_value || 0); }, 0);
    var unreal = positions.reduce(function (a, p) { return a + (p.unrealized_pl || 0); }, 0);
    var equity = acct.equity || invested + (acct.cash || 0);

    var alloc =
      '<div class="dsec-hd">' + ICON("scale") + "<span>Allocation</span></div>" +
      (positions.length
        ? positions.slice().sort(function (a, b) { return (b.market_value || 0) - (a.market_value || 0); }).map(function (p) {
            var w = equity ? ((p.market_value || 0) / equity) * 100 : 0;
            return '<div class="bar-row"><span class="b-name">' + U.esc(p.symbol) + "</span>" +
              '<div class="meter' + (p.unrealized_pl >= 0 ? " meter-green" : " meter-red") + '">' + C.meterHTML(w, p.unrealized_pl >= 0 ? "meter-green" : "meter-red").replace(/^<div[^>]*>|<\/div>$/g, "") + "</div>" +
              '<span class="b-val">' + w.toFixed(1) + "%</span></div>";
          }).join("") +
          '<div class="bar-row"><span class="b-name" style="color:var(--ink-lo)">Cash</span>' +
          '<div class="meter"><i style="width:' + (equity ? ((acct.cash || 0) / equity) * 100 : 0) + '%;background:var(--ink-faint)"></i></div>' +
          '<span class="b-val">' + (equity ? ((acct.cash || 0) / equity) * 100 : 0).toFixed(1) + "%</span></div>"
        : C.stateHTML({ icon: "layers", title: "No positions", msg: "Nothing held — the portfolio is fully in cash.", compact: true }));

    var perfStats = C.kvHTML([
      ["Equity", U.fmtMoney(acct.equity)],
      ["Invested", U.fmtMoney(invested)],
      ["Cash", U.fmtMoney(acct.cash)],
      ["Unrealized P&L", '<span class="' + U.classFor(unreal) + '">' + U.fmtSigned(unreal) + "</span>"],
      ["Today's P&L", '<span class="' + U.classFor(acct.day_pl) + '">' + U.fmtSigned(acct.day_pl) + " (" + U.fmtPct(acct.day_pl_pct) + ")</span>"],
      ["Account status", U.esc(acct.status || "—")],
    ]);

    return (
      (store.demo ? window.DemoBanner : "") +
      accountErrorBanner(acct) + botOfflineBanner() +
      '<div class="kpi-strip mb-12">' +
        kpiTile("Equity", U.fmtMoney(acct.equity)) +
        kpiTile("Invested", U.fmtMoney(invested), '<span>' + (positions.length ? positions.length + " positions" : "flat") + "</span>") +
        kpiTile("Cash", U.fmtMoney(acct.cash)) +
        kpiTile("Unrealized P&L", '<span class="' + U.classFor(unreal) + '">' + U.fmtSigned(unreal) + "</span>", '<span class="num ' + U.classFor(unreal) + '">open positions</span>') +
        kpiTile("Today", '<span class="' + U.classFor(acct.day_pl) + '">' + U.fmtSigned(acct.day_pl) + "</span>", '<span class="num ' + U.classFor(acct.day_pl) + '">' + U.fmtPct(acct.day_pl_pct) + "</span>") +
        kpiTile("Buying Power", U.fmtMoney(acct.buying_power)) +
      "</div>" +
      '<div class="grid g-main mb-12">' +
        '<div class="span-main">' + P._chartCard() + "</div>" +
        '<div class="span-side">' + headCard("Performance", "pulse", '<div class="card-bd">' + perfStats + "</div>") + "</div>" +
      "</div>" +
      '<div class="grid g-2">' +
        headCard("Allocation", "scale", '<div class="card-bd">' + alloc + "</div>") +
        headCard("Open Positions", "layers", '<div class="card-bd-flush">' + C.positionsTableHTML(positions, { aiCols: true }) + "</div>") +
      "</div>"
    );
  };

  /* ==========================================================================
     3 · POSITIONS
     ========================================================================== */

  P.positions = function (params, state) {
    var g = guard("portfolio");
    if (g === "loading") return C.pageSkeleton();
    if (g === "down") return downStateHTML(store.errors.portfolio, "Portfolio data");
    var pf = store.data.portfolio;
    state = state || { sort: "market_value", dir: -1, filter: "all", q: "" };
    var positions = App.positionsWithContext();
    var filtered = positions.filter(function (p) {
      if (state.filter === "gainers" && !(p.unrealized_pl > 0)) return false;
      if (state.filter === "losers" && !(p.unrealized_pl < 0)) return false;
      if (state.q && p.symbol.toLowerCase().indexOf(state.q.toLowerCase()) < 0) return false;
      return true;
    });

    // sort
    var key = state.sort, dir = state.dir;
    filtered.sort(function (a, b) {
      var av = a[key], bv = b[key];
      if (typeof av === "string") return av.localeCompare(bv) * dir;
      return ((av || 0) - (bv || 0)) * dir;
    });

    var counts = {
      all: positions.length,
      gainers: positions.filter(function (p) { return p.unrealized_pl > 0; }).length,
      losers: positions.filter(function (p) { return p.unrealized_pl < 0; }).length,
    };

    return (
      (store.demo ? window.DemoBanner : "") +
      accountErrorBanner(account()) +
      '<div class="filterbar mb-12">' +
        '<div class="search">' + ICON("search") + '<input id="pos-q" placeholder="Filter by symbol…" value="' + U.esc(state.q) + '" aria-label="Filter positions by symbol"></div>' +
        '<div class="chips" role="group" aria-label="Position filters">' +
          '<button class="chip" data-posfilter="all" aria-pressed="' + (state.filter === "all") + '">All <span class="cnt">' + counts.all + "</span></button>" +
          '<button class="chip" data-posfilter="gainers" aria-pressed="' + (state.filter === "gainers") + '">Gainers <span class="cnt">' + counts.gainers + "</span></button>" +
          '<button class="chip" data-posfilter="losers" aria-pressed="' + (state.filter === "losers") + '">Losers <span class="cnt">' + counts.losers + "</span></button>" +
        "</div>" +
        '<span class="spacer"></span>' +
        '<span class="t-faint" style="font-size:11px">' + filtered.length + " of " + positions.length + " · click a row for detail</span>" +
      "</div>" +
      headCard("Positions", "layers", '<div class="card-bd-flush">' + C.positionsTableHTML(filtered, { aiCols: true, sort: key, sortDir: dir }) + "</div>")
    );
  };

  P.positions._bind = function (root, rerender) {
    var state = { sort: "market_value", dir: -1, filter: "all", q: "" };
    var q = qs("#pos-q", root);
    if (q) q.addEventListener("input", U.debounce(function () { state.q = q.value; rerender(state); }, 200));
    qsa("[data-posfilter]", root).forEach(function (b) {
      b.addEventListener("click", function () { state.filter = b.getAttribute("data-posfilter"); rerender(state); });
    });
    qsa("th.sortable", root).forEach(function (th) {
      th.addEventListener("click", function () {
        var k = th.getAttribute("data-sort");
        if (state.sort === k) state.dir *= -1; else { state.sort = k; state.dir = -1; }
        rerender(state);
      });
    });
  };

  /* ==========================================================================
     4 · ORDERS
     ========================================================================== */

  P.orders = function (params, state) {
    var gp = guard("portfolio"), go = guard("orders");
    if (gp === "loading" || go === "loading") return C.pageSkeleton();
    if (go === "down") return downStateHTML(store.errors.orders, "Order history");
    if (gp === "down") return downStateHTML(store.errors.portfolio, "Portfolio data");
    state = state || { side: "all", status: "all", q: "" };
    var orders = ((store.data.orders && store.data.orders.orders) || []).slice();

    if (store.errors.orders && !orders.length) {
      return C.bannerHTML("err", "Could not load orders.", U.esc(store.errors.orders)) +
        headCard("Orders", "list-ordered", '<div class="card-bd">' + C.stateHTML({ icon: "alert", title: "Orders unavailable", msg: "The broker API did not return order history.", err: true }) + "</div>");
    }

    var filtered = orders.filter(function (o) {
      if (state.side !== "all" && String(o.side).toLowerCase() !== state.side) return false;
      var st = String(o.status || "").toLowerCase();
      if (state.status === "open" && ["filled", "canceled", "cancelled", "expired", "rejected"].indexOf(st) >= 0) return false;
      if (state.status === "filled" && st !== "filled") return false;
      if (state.status === "rejected" && ["rejected", "failed"].indexOf(st) < 0) return false;
      if (state.q && o.symbol.toLowerCase().indexOf(state.q.toLowerCase()) < 0) return false;
      return true;
    });

    var buys = orders.filter(function (o) { return String(o.side).toLowerCase() === "buy"; }).length;

    return (
      (store.demo ? window.DemoBanner : "") +
      '<div class="filterbar mb-12">' +
        '<div class="search">' + ICON("search") + '<input id="ord-q" placeholder="Filter by symbol…" value="' + U.esc(state.q) + '" aria-label="Filter orders by symbol"></div>' +
        '<select class="sel" id="ord-side" aria-label="Filter by side">' +
          '<option value="all"' + (state.side === "all" ? " selected" : "") + ">All sides</option>" +
          '<option value="buy"' + (state.side === "buy" ? " selected" : "") + ">Buy</option>" +
          '<option value="sell"' + (state.side === "sell" ? " selected" : "") + ">Sell</option>" +
        "</select>" +
        '<select class="sel" id="ord-status" aria-label="Filter by status">' +
          '<option value="all"' + (state.status === "all" ? " selected" : "") + ">All statuses</option>" +
          '<option value="open"' + (state.status === "open" ? " selected" : "") + ">Open / pending</option>" +
          '<option value="filled"' + (state.status === "filled" ? " selected" : "") + ">Filled</option>" +
          '<option value="rejected"' + (state.status === "rejected" ? " selected" : "") + ">Rejected / failed</option>" +
        "</select>" +
        '<span class="spacer"></span>' +
        '<span class="t-faint" style="font-size:11px">' + filtered.length + " of " + orders.length + " orders · paper account</span>" +
      "</div>" +
      headCard("Orders & Executions", "list-ordered",
        '<div class="card-bd-flush">' + C.ordersTableHTML(filtered) + "</div>",
        '<span class="spacer"></span><span class="aux">' + buys + " buys · " + (orders.length - buys) + " sells</span>")
    );
  };

  P.orders._bind = function (root, rerender) {
    var state = { side: "all", status: "all", q: "" };
    var q = qs("#ord-q", root);
    if (q) q.addEventListener("input", U.debounce(function () { state.q = q.value; rerender(state); }, 200));
    var side = qs("#ord-side", root), status = qs("#ord-status", root);
    if (side) side.addEventListener("change", function () { state.side = side.value; rerender(state); });
    if (status) status.addEventListener("change", function () { state.status = status.value; rerender(state); });
  };

  /* ==========================================================================
     5 · AI INTELLIGENCE
     ========================================================================== */

  P.ai = function (params) {
    var g = guard("logs");
    if (g === "loading") return C.pageSkeleton();
    if (g === "down") return downStateHTML(store.errors.logs, "Agent logs");
    var cioLogs = C.latestCioLogs(20);
    var sym = params && params.sym;
    if (sym) cioLogs = cioLogs.filter(function (l) { return l.symbol === sym; });
    var latest = cioLogs[0];

    // symbol selector
    var syms = {};
    logs().forEach(function (l) { if (l.symbol) syms[l.symbol] = true; });
    var symChips =
      '<div class="chips" role="group" aria-label="Symbol filter">' +
        '<button class="chip" data-aisym="" aria-pressed="' + (!sym) + '">All</button>' +
        Object.keys(syms).sort().map(function (s) {
          return '<button class="chip" data-aisym="' + s + '" aria-pressed="' + (sym === s) + '">' + s + "</button>";
        }).join("") +
      "</div>";

    var hero;
    if (latest) {
      var win = C.logsForDecision(latest);
      var v = C.verdictOf(latest);
      var conf = latest.data && latest.data.confidence != null ? Math.round(latest.data.confidence * 100) : null;
      hero =
        '<section class="card mb-12">' +
          '<div class="card-hd">' + ICON("brain") + '<span class="card-title">Decision Pipeline</span>' +
            '<span class="sym" style="font-size:13px">' + U.esc(latest.symbol || "") + "</span>" +
            (v ? '<span class="' + v.badge + '">' + v.label + "</span>" : "") +
            (conf != null ? '<span class="badge badge-ai">CONF ' + conf + "%</span>" : "") +
            '<span class="spacer"></span>' +
            '<span class="aux"><span data-ago="' + U.esc(latest.timestamp) + '">' + U.ago(latest.timestamp) + "</span> · " + U.fmtTime(latest.timestamp) + "</span>" +
            '<button class="btn btn-sm" data-action="open-decision" data-ts="' + U.esc(latest.timestamp) + '" data-agent="cio">' + ICON("gavel") + "Full reasoning</button>" +
          "</div>" +
          '<div class="card-bd" style="padding-top:6px;padding-bottom:4px">' + C.pipelineHTML(latest) + "</div>" +
        "</section>";

      // debate
      var debateCard =
        '<section class="card">' +
          '<div class="card-hd">' + ICON("scale") + '<span class="card-title">Bull vs Bear Debate</span>' +
            '<span class="spacer"></span><span class="aux">' + U.esc(latest.symbol || "") + " · " + U.fmtTime(latest.timestamp) + "</span></div>" +
          '<div class="card-bd">' + C.debateHTML(win.debate) + "</div>" +
        "</section>";

      // decision list
      var listCard =
        '<section class="card">' +
          '<div class="card-hd">' + ICON("gavel") + '<span class="card-title">Recent Decisions</span><span class="spacer"></span><span class="aux">' + cioLogs.length + " in buffer</span></div>" +
          '<div class="feed">' +
            cioLogs.map(function (l) {
              var vv = C.verdictOf(l);
              return (
                '<div class="feed-item rowlink" data-action="open-decision" data-ts="' + U.esc(l.timestamp) + '" data-agent="cio" role="button" tabindex="0">' +
                  '<div class="feed-ico a-cio">' + ICON("gavel") + "</div>" +
                  '<div class="feed-body"><div class="feed-line1">' +
                    '<span class="feed-agent">' + U.esc(l.symbol || "") + "</span>" +
                    (vv ? '<span class="' + vv.badge + '">' + vv.label + "</span>" : "") +
                    (l.data && l.data.confidence != null ? '<span class="badge badge-ai">' + Math.round(l.data.confidence * 100) + "%</span>" : "") +
                  "</div>" +
                  '<div class="feed-msg">' + U.esc((l.data && l.data.reasoning) || l.message) + "</div></div>" +
                  '<div class="feed-side"><span class="feed-time" data-ago="' + U.esc(l.timestamp) + '">' + U.ago(l.timestamp) + "</span><span class=\"feed-time\">" + U.fmtTime(l.timestamp) + "</span></div>" +
                "</div>"
              );
            }).join("") +
          "</div>" +
        "</section>";

      return (
        (store.demo ? window.DemoBanner : "") +
        '<div class="filterbar mb-12">' + symChips + "</div>" +
        hero +
        '<div class="grid g-2">' + debateCard + listCard + "</div>"
      );
    }

    // empty
    return (
      (store.demo ? window.DemoBanner : "") +
      headCard("AI Intelligence", "brain",
        '<div class="card-bd">' + C.stateHTML({
          icon: "brain",
          title: "No AI decisions yet",
          msg: "The CIO agent has not logged any decisions in the current buffer. Run a cycle to see the full decision pipeline, agent consensus, the bull/bear debate and structured reasoning.",
          actions: '<button class="btn btn-primary" data-action="run-cycle">' + ICON("play") + "Run Cycle Now</button>",
        }) + "</div>")
    );
  };

  P.ai._bind = function (root) {
    qsa("[data-aisym]", root).forEach(function (b) {
      b.addEventListener("click", function () {
        var s = b.getAttribute("data-aisym");
        App.navigate("ai", s ? { sym: s } : null);
      });
    });
  };

  /* ==========================================================================
     6 · AGENT PERFORMANCE
     ========================================================================== */

  P.agents = function () {
    var g = guard("accuracy");
    if (g === "loading") return C.pageSkeleton();
    if (g === "down") return downStateHTML(store.errors.accuracy, "Agent accuracy data");
    var acc = store.data.accuracy;

    if (!acc.enabled) {
      return headCard("Agent Performance", "gauge",
        '<div class="card-bd">' + C.stateHTML({
          icon: "memory",
          title: "Learning loop disabled",
          msg: "Agent accuracy tracking is disabled (ENABLE_MEMORY=false). Enable it in the bot configuration to collect hit rates from closed trades.",
        }) + "</div>");
    }

    var meta = [
      { key: "technical", label: "Technical", icon: "chart-candle" },
      { key: "news", label: "News", icon: "news" },
      { key: "fundamentals", label: "Fundamentals", icon: "db" },
    ];

    var rows = "", cards = "";
    meta.forEach(function (m) {
      var a = acc.agents[m.key];
      var has = a && a.sample_size > 0 && a.hit_rate != null;
      var hr = has ? (a.hit_rate * 100).toFixed(1) + "%" : "—";
      var n = a ? a.sample_size : 0;
      var w = a ? a.weight : 1;
      var track = U.trackForWeight(w, n);
      rows +=
        '<tr class="rowlink" data-action="open-agent" data-agent="' + m.key + '" tabindex="0">' +
          "<td>" + U.agentLabel(m.key) + "</td>" +
          "<td>" + (has ? '<span class="num-strong">' + hr + "</span>" : '<span class="t-faint">No data</span>') + "</td>" +
          '<td class="r">' + (n ? n + " calls" : '<span class="t-faint">0</span>') + "</td>" +
          '<td class="r">' + (has ? '<span class="' + (w >= 1 ? "pos" : "neg") + '">' + w.toFixed(2) + "×</span>" : "—") + "</td>" +
          "<td>" + (track ? '<span class="' + track.cls + '" style="font-weight:600">' + track.label + "</span>" : '<span class="t-faint">Awaiting data</span>') + "</td>" +
          "<td>" + (has ? '<span class="badge badge-ok">SCORING</span>' : '<span class="badge badge-neutral">COLLECTING</span>') + "</td>" +
        "</tr>";

      cards +=
        '<div class="mcard" data-action="open-agent" data-agent="' + m.key + '" role="button" tabindex="0">' +
          '<div class="mcard-hd"><span style="font-size:13px;font-weight:650;color:var(--ink-hi)">' + U.agentLabel(m.key) + '</span><span class="spacer"></span>' +
            (has ? '<span class="badge badge-ok">SCORING</span>' : '<span class="badge badge-neutral">COLLECTING</span>') + "</div>" +
          '<div class="mcard-grid">' +
            '<div class="g"><div class="k">Hit rate</div><div class="v">' + (has ? hr : "—") + "</div></div>" +
            '<div class="g"><div class="k">Sample</div><div class="v">' + n + "</div></div>" +
            '<div class="g"><div class="k">Weight</div><div class="v">' + (has ? w.toFixed(2) + "×" : "—") + "</div></div>" +
            '<div class="g"><div class="k">Recent</div><div class="v">' + (track ? track.label : "Awaiting data") + "</div></div>" +
          "</div>" +
        "</div>";
    });

    var table =
      '<div class="tbl-wrap responsive"><table class="tbl"><thead><tr>' +
        "<th>Agent</th><th>Hit Rate</th><th class='r'>Sample Size</th><th class='r'>Current Weight</th><th>Recent Performance</th><th>Status</th>" +
      "</tr></thead><tbody>" + rows + "</tbody></table>" +
      '<div class="mcards">' + cards + "</div></div>";

    // comparison bars
    var bars = meta.map(function (m) {
      var a = acc.agents[m.key];
      var has = a && a.sample_size > 0 && a.hit_rate != null;
      var pct = has ? a.hit_rate * 100 : 0;
      return '<div class="bar-row"><span class="b-name">' + m.label + "</span>" +
        '<div class="meter' + (pct >= 55 ? " meter-green" : pct ? " meter-amber" : "") + '">' + (has ? C.meterHTML(pct, pct >= 55 ? "meter-green" : "meter-amber").replace(/^<div[^>]*>|<\/div>$/g, "") : "") + "</div>" +
        '<span class="b-val">' + (has ? pct.toFixed(1) + "%" : "no data") + "</span></div>";
    }).join("");

    return (
      (store.demo ? window.DemoBanner : "") +
      headCard("Agent Performance", "gauge",
        '<div class="card-bd-flush">' + table + "</div>" +
        '<div class="tbl-note">' + ICON("info") + "<span>Accuracy is measured from closed trades in the rolling lookback (last " + (config().agent_accuracy_lookback || 20) + " closed calls). It reflects <b>historical performance only</b> — not a guarantee of future results.</span></div>") +
      '<div class="grid g-2 mt-12">' +
        headCard("Hit Rate Comparison", "chart-candle", '<div class="card-bd"><div class="bars">' + bars + "</div>" +
          '<div class="meter-scale" style="margin-top:10px"><span>0%</span><span>50%</span><span>100%</span></div></div>') +
        headCard("How Scoring Works", "memory", '<div class="card-bd">' +
          '<div style="font-size:12px;color:var(--ink-lo);line-height:1.65">' +
            "<p style='margin-bottom:8px'>Every BUY decision is recorded to persistent memory with each agent's directional call. When the position later closes, realized P&L labels the call: a hit means the agent's direction matched the outcome.</p>" +
            "<p style='margin-bottom:8px'>The rolling hit rate becomes a weight (0.5–1.5×) that the CIO uses to lean on reliable agents and discount unreliable ones.</p>" +
            "<p>Agents with zero closed samples show no percentage — we never invent accuracy numbers.</p>" +
          "</div></div>") +
      "</div>"
    );
  };

  /* ==========================================================================
     7 · RISK
     ========================================================================== */

  P.risk = function () {
    var g = guard("portfolio");
    if (g === "loading") return C.pageSkeleton();
    if (g === "down") return downStateHTML(store.errors.portfolio, "Portfolio data");
    var pf = store.data.portfolio;
    var acct = account();
    var positions = App.positionsWithContext();
    var cfg = config();

    var equity = acct.equity || 0;
    var invested = positions.reduce(function (a, p) { return a + (p.market_value || 0); }, 0);
    var cashPct = equity ? ((acct.cash || 0) / equity) * 100 : null;
    var maxPos = positions.length ? positions.reduce(function (a, b) { return (a.market_value || 0) >= (b.market_value || 0) ? a : b; }) : null;
    var maxConc = maxPos && equity ? ((maxPos.market_value || 0) / equity) * 100 : null;
    var limit = (cfg.max_position_pct || 0.10) * 100;

    // live risk events from the log
    var riskLogs = logs().filter(function (l) { return l.agent === "risk" || (l.level === "WARNING" && /concentration|exposure|limit/i.test(l.message)); });
    var latestRiskBySym = {};
    riskLogs.forEach(function (l) {
      if (!l.symbol) return;
      if (!latestRiskBySym[l.symbol] || l.timestamp > latestRiskBySym[l.symbol].timestamp) latestRiskBySym[l.symbol] = l;
    });

    var monitorChecks = "";
    Object.keys(latestRiskBySym).sort().forEach(function (s) {
      var l = latestRiskBySym[s];
      var d = l.data || {};
      var approved = d.approved === true;
      monitorChecks +=
        '<div class="check ' + (l.level === "WARNING" ? "warn" : approved ? "ok" : "err") + '">' +
          '<div class="c-ico">' + ICON(l.level === "WARNING" ? "alert" : approved ? "check" : "x") + "</div>" +
          '<div class="c-body">' + U.esc(d.reasoning || l.message) +
            '<div class="c-sub">' + U.esc(s) + " · " + U.fmtDateTime(l.timestamp) + (d.risk_level ? " · risk level " + d.risk_level : "") + "</div>" +
          "</div>" +
        "</div>";
    });

    var concentrationChecks = positions.map(function (p) {
      var w = p.weight_pct || 0;
      var state = w >= limit ? "err" : w >= limit * 0.85 ? "warn" : "ok";
      return (
        '<div class="check ' + state + '">' +
          '<div class="c-ico">' + ICON(state === "ok" ? "check" : "alert") + "</div>" +
          '<div class="c-body">' + U.esc(p.symbol) + " at " + w.toFixed(1) + "% of equity" +
            '<div class="c-sub">limit ' + limit.toFixed(0) + "%" + (state === "warn" ? " — approaching threshold" : state === "err" ? " — above configured maximum" : "") + "</div>" +
          "</div>" +
        "</div>"
      );
    }).join("");

    return (
      (store.demo ? window.DemoBanner : "") +
      accountErrorBanner(acct) +
      '<div class="kpi-strip mb-12" style="grid-template-columns:repeat(4,minmax(0,1fr))">' +
        kpiTile("Portfolio Exposure", equity ? (invested / equity * 100).toFixed(1) + "%" : "—", '<span>' + U.fmtMoney(invested, { dec: 0 }) + " invested</span>") +
        kpiTile("Cash Exposure", cashPct != null ? cashPct.toFixed(1) + "%" : "—", '<span>' + U.fmtMoney(acct.cash || 0, { dec: 0 }) + " unallocated</span>") +
        kpiTile("Largest Position", maxPos ? U.esc(maxPos.symbol) : "—", maxPos ? U.fmtMoney(maxPos.market_value, { dec: 0 }) : '<span class="t-faint">Flat</span>') +
        kpiTile("Position Concentration", maxConc != null ? maxConc.toFixed(1) + "%" : "—", '<span class="' + (maxConc >= limit ? "neg" : maxConc >= limit * 0.85 ? "" : "pos") + '">limit ' + limit.toFixed(0) + "%</span>") +
      "</div>" +
      '<div class="grid g-2 mb-12">' +
        headCard("Risk Monitor", "shield",
          '<div class="card-bd">' +
            (monitorChecks
              ? monitorChecks
              : C.stateHTML({ icon: "shield", title: "No risk events in buffer", msg: "The risk agent logs an assessment for every symbol each cycle. Events will appear here after the next cycle.", compact: true })) +
          "</div>") +
        headCard("Position Concentration", "scale",
          '<div class="card-bd">' +
            (concentrationChecks || C.stateHTML({ icon: "layers", title: "No open positions", msg: "With no positions, concentration risk is zero.", compact: true })) +
          "</div>",
          '<span class="spacer"></span><span class="aux">max ' + limit.toFixed(0) + "% / position</span>") +
      "</div>" +
      headCard("Historical Risk Events", "scroll",
        riskLogs.length
          ? '<div class="feed">' + riskLogs.slice(0, 20).map(function (l) { return C.feedItemHTML(l); }).join("") + "</div>"
          : '<div class="card-bd">' + C.stateHTML({ icon: "scroll", title: "No risk events recorded", msg: "Risk warnings and assessments from past cycles will accumulate here.", compact: true }) + "</div>")
    );
  };

  /* ==========================================================================
     8 · CYCLES
     ========================================================================== */

  P.cycles = function () {
    var g = guard("cycles");
    if (g === "loading") return C.pageSkeleton();
    if (g === "down") return downStateHTML(store.errors.cycles, "Cycle history");
    var cs = cycles();
    var st = botState();
    var cfg = config();
    var iv = cfg.cycle_interval_minutes || 15;

    var nextC =
      st.running && st.last_cycle_at
        ? '<span class="num" id="nextcycle">—</span>'
        : '<span class="t-faint">' + (st.running ? "—" : "Paused — bot offline") + "</span>";

    var ok = cs.filter(function (c) { return c.status === "OK"; }).length;
    var partial = cs.filter(function (c) { return c.status === "PARTIAL_ERROR"; }).length;
    var err = cs.filter(function (c) { return c.status === "ERROR"; }).length;

    var rows = cs.map(function (c) {
      var badge = '<span class="' + U.badgeForCycle(c.status) + '">' + U.esc(String(c.status).replace("_", " ")) + "</span>";
      return (
        '<tr class="rowlink" data-action="open-cycle" data-id="' + c.id + '" tabindex="0">' +
          '<td class="num-strong">#' + c.id + "</td>" +
          "<td>" + U.fmtDateTime(c.started_at) + '<div class="td-sub" data-ago="' + U.esc(c.started_at) + '">' + U.ago(c.started_at) + "</div></td>" +
          "<td>" + badge + "</td>" +
          '<td class="r">' + (c.duration_s != null ? c.duration_s + "s" : "—") + "</td>" +
          '<td class="r">' + (c.symbols_processed || []).length + "</td>" +
          '<td class="r">' + (c.decisions || []).length + "</td>" +
          '<td class="r">' + ((c.orders || []).length ? '<span class="num pos">' + c.orders.length + "</span>" : "0") + "</td>" +
          '<td class="r">' + (c.warnings ? '<span style="color:var(--amber)">' + c.warnings + "</span>" : "0") + "</td>" +
          '<td class="r">' + (c.errors && c.errors.length ? '<span class="neg">' + c.errors.length + "</span>" : "0") + "</td>" +
          '<td class="t-lo" style="font-size:11px">' + U.esc(String(c.triggered_by || "scheduler").toUpperCase()) + "</td>" +
        "</tr>"
      );
    }).join("");

    var cards = cs.map(function (c) {
      return (
        '<div class="mcard" data-action="open-cycle" data-id="' + c.id + '" role="button" tabindex="0">' +
          '<div class="mcard-hd"><span class="num-strong">#' + c.id + '</span><span class="' + U.badgeForCycle(c.status) + '">' + U.esc(String(c.status).replace("_", " ")) + "</span>" +
            '<span class="spacer"></span><span class="feed-time">' + U.fmtDateTime(c.started_at) + "</span></div>" +
          '<div class="mcard-grid">' +
            '<div class="g"><div class="k">Duration</div><div class="v">' + (c.duration_s != null ? c.duration_s + "s" : "—") + "</div></div>" +
            '<div class="g"><div class="k">Symbols</div><div class="v">' + (c.symbols_processed || []).length + "</div></div>" +
            '<div class="g"><div class="k">Decisions</div><div class="v">' + (c.decisions || []).length + "</div></div>" +
            '<div class="g"><div class="k">Orders</div><div class="v">' + (c.orders || []).length + "</div></div>" +
            '<div class="g"><div class="k">Warnings</div><div class="v">' + (c.warnings || 0) + "</div></div>" +
            '<div class="g"><div class="k">Trigger</div><div class="v">' + U.esc(String(c.triggered_by || "scheduler").toUpperCase()) + "</div></div>" +
          "</div>" +
        "</div>"
      );
    }).join("");

    return (
      (store.demo ? window.DemoBanner : "") +
      botOfflineBanner() +
      '<div class="kpi-strip mb-12" style="grid-template-columns:repeat(4,minmax(0,1fr))">' +
        kpiTile("Last Cycle", cs[0] ? "#" + cs[0].id : "—", cs[0] ? '<span data-ago="' + U.esc(cs[0].started_at) + '">' + U.ago(cs[0].started_at) + "</span> · " + U.esc(String(cs[0].status).replace("_", " ")) : '<span class="t-faint">Never run</span>') +
        kpiTile("Next Scheduled", nextC, '<span>interval ' + iv + "m</span>") +
        kpiTile("Cycle Interval", iv + "m", '<span>scheduler ' + (st.running ? "running" : "paused") + "</span>") +
        kpiTile("Recent Outcomes", ok + " ok", '<span>' + partial + " partial · " + err + " failed</span>") +
      "</div>" +
      headCard("Trading Cycles", "repeat",
        cs.length
          ? '<div class="tbl-wrap responsive"><table class="tbl"><thead><tr>' +
              "<th>Cycle</th><th>Started</th><th>Status</th><th class='r'>Duration</th><th class='r'>Symbols</th><th class='r'>Decisions</th><th class='r'>Orders</th><th class='r'>Warn</th><th class='r'>Err</th><th>Trigger</th>" +
            "</tr></thead><tbody>" + rows + "</tbody></table>" +
            '<div class="mcards">' + cards + "</div></div>" +
          '<div class="tbl-note">' + ICON("info") + "<span>Cycle history is kept in memory and resets when the server restarts. " + cs.length + " cycles recorded this session.</span></div>"
          : '<div class="card-bd">' + C.stateHTML({
              icon: "repeat",
              title: st.last_cycle_status === "NEVER_RUN" ? "No cycles yet" : "No cycle history this session",
              msg: st.running
                ? "The scheduler is running — the first cycle record will appear after the next cycle completes (every " + iv + " minutes, or run one now)."
                : "Start the bot to begin autonomous cycles.",
              actions: st.running ? "" : '<button class="btn btn-success" data-action="start-bot">' + ICON("power") + "Start Bot</button>",
            }) + "</div>")
    );
  };

  /* ==========================================================================
     9 · SYSTEM HEALTH
     ========================================================================== */

  P.health = function () {
    var g = guard("portfolio");
    if (g === "loading") return C.pageSkeleton();
    if (g === "down") return downStateHTML(store.errors.portfolio, "Health data");
    var st = botState();
    var acct = account();
    var cfg = config();

    var agentNames = ["technical", "news", "fundamentals", "debate", "risk", "cio", "execution", "memory", "system"];
    var disabled = C.disabledAgentSet();
    var enabled = agentNames.filter(function (a) { return !disabled[a]; });
    var agentErrs = {};
    logs().forEach(function (l) { if (l.level === "ERROR" && l.agent) agentErrs[l.agent] = (agentErrs[l.agent] || 0) + 1; });

    var cards =
      healthCard("Bot Process", st.running ? "ONLINE" : "OFFLINE", st.running ? "ok" : "err", "power",
        st.running ? "Running since " + U.fmtDateTime(st.started_at) : "The trading engine is stopped") +
      healthCard("Alpaca API", acct.error ? "UNREACHABLE" : "CONNECTED", acct.error ? "err" : "ok", "link",
        acct.error ? U.esc(acct.error) : "Paper trading endpoint") +
      healthCard("AI Providers", cfg.warnings && cfg.warnings.length ? "DEGRADED" : "CONNECTED", cfg.warnings && cfg.warnings.length ? "warn" : "ok", "brain",
        cfg.warnings ? U.esc(cfg.warnings.length + " configuration warning" + (cfg.warnings.length > 1 ? "s" : "")) : "Groq · Gemini · OpenRouter") +
      healthCard("Database", cfg.enable_memory ? "CONNECTED" : "DISABLED", cfg.enable_memory ? "ok" : "warn", "db",
        cfg.enable_memory ? "SQLite · " + U.esc(cfg.memory_db_path || "data/memory.db") : "Memory disabled via config") +
      healthCard("Scheduler", st.running ? "RUNNING" : "PAUSED", st.running ? "ok" : "warn", "clock",
        st.running ? "Every " + (cfg.cycle_interval_minutes || 15) + " minutes" : "Bot is offline") +
      healthCard("Agent Services", enabled.length + "/" + agentNames.length + " ENABLED", enabled.length === agentNames.length ? "ok" : "warn", "cpu",
        disabled.fundamentals || disabled.debate ? "Optional agents disabled via config" : "All pipeline agents enabled");

    function healthCard(name, val, tone, icon, sub) {
      var badge = tone === "ok" ? "badge-ok" : tone === "warn" ? "badge-warning" : "badge-error";
      var dot = tone === "ok" ? "dot-on" : tone === "warn" ? "dot-warn" : "dot-err";
      return (
        '<div class="card" style="padding:12px 14px;gap:4px">' +
          '<div class="row" style="justify-content:space-between">' +
            '<span class="row" style="gap:7px">' + ICON(icon) + '<span style="font-size:12px;font-weight:650;color:var(--ink-hi)">' + name + "</span></span>" +
            '<span class="badge ' + badge + '"><span class="dot ' + dot + '"></span>' + val + "</span>" +
          "</div>" +
          '<div class="t-faint" style="font-size:11px;min-height:15px;white-space:nowrap;overflow:hidden;text-overflow:ellipsis" title="' + U.esc(sub) + '">' + sub + "</div>" +
        "</div>"
      );
    }

    var errs = logs().filter(function (l) { return l.level === "ERROR"; });
    var warns = logs().filter(function (l) { return l.level === "WARNING"; });

    var cfgWarns = (cfg.warnings || []);

    return (
      (store.demo ? window.DemoBanner : "") +
      botOfflineBanner() +
      (acct.error ? C.bannerHTML("err", "Broker API unreachable.", U.esc(acct.error)) : "") +
      '<div class="grid g-3 mb-12">' + cards + "</div>" +
      '<div class="grid g-2">' +
        headCard("Recent Errors", "alert",
          errs.length
            ? '<div class="feed">' + errs.slice(0, 10).map(function (l) { return C.feedItemHTML(l); }).join("") + "</div>"
            : '<div class="card-bd">' + C.stateHTML({ icon: "check", title: "No errors in buffer", msg: "No agent or API errors have been logged this session.", compact: true }) + "</div>",
          '<span class="spacer"></span>' + (errs.length ? '<span class="badge badge-error">' + errs.length + "</span>" : "")) +
        headCard("Recent Warnings", "alert",
          warns.length
            ? '<div class="feed">' + warns.slice(0, 10).map(function (l) { return C.feedItemHTML(l); }).join("") + "</div>"
            : '<div class="card-bd">' + C.stateHTML({ icon: "check", title: "No warnings in buffer", msg: "Nothing requires attention right now.", compact: true }) + "</div>",
          '<span class="spacer"></span>' + (warns.length ? '<span class="badge badge-warning">' + warns.length + "</span>" : "")) +
      "</div>" +
      (cfgWarns.length
        ? '<div class="mt-12">' + headCard("Configuration Warnings", "settings",
            '<div class="card-bd">' + cfgWarns.map(function (w) {
              return '<div class="check warn"><div class="c-ico">' + ICON("alert") + '</div><div class="c-body">' + U.esc(w) + "</div></div>";
            }).join("") + "</div>") + "</div>"
        : "") +
      headCard("Agent Service Status", "cpu",
        '<div class="card-bd-flush"><div class="tbl-wrap"><table class="tbl"><thead><tr><th>Agent</th><th>Config</th><th>Recent Errors</th><th>Status</th></tr></thead><tbody>' +
          agentNames.map(function (a) {
            var off = disabled[a];
            var e = agentErrs[a] || 0;
            return "<tr>" +
              "<td>" + U.agentLabel(a) + "</td>" +
              "<td>" + (off ? '<span class="badge badge-neutral">DISABLED</span>' : '<span class="badge badge-ok">ENABLED</span>') + "</td>" +
              '<td class="r">' + (e ? '<span class="neg">' + e + "</span>" : "0") + "</td>" +
              "<td>" + (off ? '<span class="t-faint">Off</span>' : e ? '<span class="badge badge-warning">ERRORS</span>' : '<span class="badge badge-ok">HEALTHY</span>') + "</td>" +
            "</tr>";
          }).join("") +
        "</tbody></table></div></div>", "")
    );
  };

  /* ==========================================================================
     10 · ACTIVITY
     ========================================================================== */

  P.activity = function (params, state) {
    var g = guard("logs");
    if (g === "loading") return C.pageSkeleton();
    if (g === "down") return downStateHTML(store.errors.logs, "Activity logs");
    state = state || { cat: "all", agent: "all", level: "all", q: "", sym: (params && params.sym) || "" };
    var all = logs();

    var syms = {};
    all.forEach(function (l) { if (l.symbol) syms[l.symbol] = true; });

    var filtered = all.filter(function (l) {
      if (state.sym && l.symbol !== state.sym) return false;
      if (state.agent !== "all" && l.agent !== state.agent) return false;
      if (state.level !== "all" && String(l.level).toUpperCase() !== state.level) return false;
      if (state.q) {
        var q = state.q.toLowerCase();
        var hay = (l.message + " " + (l.symbol || "") + " " + U.agentLabel(l.agent) + " " + l.timestamp).toLowerCase();
        if (hay.indexOf(q) < 0) return false;
      }
      if (state.cat === "trades" && !["execution", "memory"].includes(l.agent)) return false;
      if (state.cat === "decisions" && l.agent !== "cio") return false;
      if (state.cat === "agents" && ["cio", "execution", "system"].includes(l.agent)) return false;
      if (state.cat === "risk" && l.agent !== "risk") return false;
      if (state.cat === "system" && l.agent !== "system") return false;
      if (state.cat === "errors" && l.level !== "ERROR") return false;
      if (state.cat === "warnings" && l.level !== "WARNING") return false;
      return true;
    });

    var cats = [
      ["all", "All"],
      ["trades", "Trades"],
      ["decisions", "AI Decisions"],
      ["agents", "Agent Events"],
      ["risk", "Risk"],
      ["system", "System"],
      ["errors", "Errors"],
      ["warnings", "Warnings"],
    ];

    var agents = ["technical", "news", "fundamentals", "debate", "risk", "cio", "execution", "memory", "system"];

    return (
      (store.demo ? window.DemoBanner : "") +
      '<div class="filterbar mb-12">' +
        '<div class="search">' + ICON("search") + '<input id="act-q" placeholder="Search events, symbols, agents…" value="' + U.esc(state.q) + '" aria-label="Search activity"></div>' +
        '<select class="sel" id="act-agent" aria-label="Filter by agent">' +
          '<option value="all">All agents</option>' +
          agents.map(function (a) { return '<option value="' + a + '"' + (state.agent === a ? " selected" : "") + ">" + U.agentLabel(a) + "</option>"; }).join("") +
        "</select>" +
        '<select class="sel" id="act-level" aria-label="Filter by severity">' +
          '<option value="all">All severities</option>' +
          ["INFO", "WARNING", "ERROR"].map(function (l) { return '<option value="' + l + '"' + (state.level === l ? " selected" : "") + ">" + l + "</option>"; }).join("") +
        "</select>" +
        (state.sym ? '<button class="chip on" id="act-sym-chip">' + U.esc(state.sym) + " ✕</button>" : "") +
      "</div>" +
      '<div class="chips mb-12" role="group" aria-label="Event category">' +
        cats.map(function (c) { return '<button class="chip" data-actcat="' + c[0] + '" aria-pressed="' + (state.cat === c[0]) + '">' + c[1] + "</button>"; }).join("") +
        '<span class="spacer"></span><span class="t-faint" style="font-size:11px">' + filtered.length + " events · buffer holds " + all.length + "</span>" +
      "</div>" +
      headCard("Activity & Audit Log", "scroll",
        filtered.length
          ? '<div class="feed">' + filtered.slice(0, 120).map(function (l) { return C.feedItemHTML(l); }).join("") + "</div>" +
            (filtered.length > 120 ? '<div class="tbl-note">Showing the 120 most recent matching events.</div>' : "")
          : '<div class="card-bd">' + C.stateHTML({ icon: "search", title: "No matching events", msg: "No events match the current filters. Adjust the filters or run a cycle to generate activity." }) + "</div>",
        '<span class="spacer"></span><span class="aux">Live buffer · ' + U.fmtTime(derived_newest()) + "</span>")
    );

    function derived_newest() {
      return App.derived.newestLogTs || "";
    }
  };

  P.activity._bind = function (root, rerender) {
    var params = App.store.ui.params;
    var state = { cat: "all", agent: "all", level: "all", q: "", sym: (params && params.sym) || "" };
    var q = qs("#act-q", root);
    if (q) q.addEventListener("input", U.debounce(function () { state.q = q.value; rerender(state); }, 200));
    var ag = qs("#act-agent", root), lv = qs("#act-level", root);
    if (ag) ag.addEventListener("change", function () { state.agent = ag.value; rerender(state); });
    if (lv) lv.addEventListener("change", function () { state.level = lv.value; rerender(state); });
    qsa("[data-actcat]", root).forEach(function (b) {
      b.addEventListener("click", function () { state.cat = b.getAttribute("data-actcat"); rerender(state); });
    });
    var chip = qs("#act-sym-chip", root);
    if (chip) chip.addEventListener("click", function () { state.sym = ""; rerender(state); });
  };

  /* ==========================================================================
     11 · CONFIGURATION
     ========================================================================== */

  P.config = function () {
    var g = guard("config");
    if (g === "loading") return C.pageSkeleton();
    if (g === "down") return downStateHTML(store.errors.config, "Configuration data");
    var cfg = store.data.config;

    var rot = function (on) {
      return '<span class="rotoggle' + (on ? " on" : "") + '" role="img" aria-label="' + (on ? "Enabled" : "Disabled") + '"><span class="track"></span>' + (on ? "ON" : "OFF") + "</span>";
    };

    var rows =
      cfgRow("Trading Universe", "Symbols the bot analyzes every cycle", (cfg.trade_universe || []).map(function (s) { return '<span class="tag">' + U.esc(s) + "</span>"; }).join(" "), "") +
      cfgRow("Cycle Interval", "Autonomous cycle cadence", '<span class="num">' + U.esc(cfg.cycle_interval_minutes || "—") + " min</span>", "") +
      cfgRow("Maximum Position %", "Per-position concentration cap used by the risk agent", '<span class="num">' + ((cfg.max_position_pct || 0) * 100).toFixed(0) + "%</span>", "") +
      cfgRow("Trading Mode", "This system is hardcoded to paper trading", '<span class="badge badge-info" style="font-size:11px">PAPER TRADING</span>', "") +
      cfgRow("Fundamentals Agent", "Optional — yfinance valuation & growth analysis", rot(cfg.enable_fundamentals_agent), "") +
      cfgRow("Bull/Bear Debate", "Optional — opposing researcher personas", rot(cfg.enable_debate), "") +
      cfgRow("Memory & Learning", "Persistent decision log + agent accuracy scoring", rot(cfg.enable_memory), "") +
      cfgRow("Accuracy Lookback", "Closed trades used for rolling hit rates", '<span class="num">' + U.esc(cfg.agent_accuracy_lookback || "—") + "</span>", "") +
      cfgRow("Memory Database", "SQLite path (persistent across restarts)", '<span class="num" style="font-size:11.5px">' + U.esc(cfg.memory_db_path || "—") + "</span>", "");

    function cfgRow(k, d, v) {
      return '<div class="cfg-row"><div><div class="c-k">' + k + '</div><div class="c-d">' + d + '</div></div><div class="c-v">' + v + "</div></div>";
    }

    return (
      (store.demo ? window.DemoBanner : "") +
      ((cfg.warnings || []).length
        ? C.bannerHTML("warn", "Configuration needs attention.", (cfg.warnings || []).length + " warning" + ((cfg.warnings || []).length > 1 ? "s" : "") + " — see below.") 
        : C.bannerHTML("info", "All required configuration is set.", "No missing API keys or invalid settings detected.")) +
      headCard("Active Configuration", "settings", '<div class="card-bd-flush">' + rows + "</div>",
        '<span class="spacer"></span><span class="aux">' + ICON("eye") + "Read-only</span>") +
      ((cfg.warnings || []).length
        ? headCard("Configuration Warnings", "alert",
            '<div class="card-bd">' + (cfg.warnings || []).map(function (w) {
              return '<div class="check warn"><div class="c-ico">' + ICON("alert") + '</div><div class="c-body">' + U.esc(w) + "</div></div>";
            }).join("") +
            '<div class="note" style="margin-top:8px">' + ICON("info") + "<span>Configuration is managed via environment variables on the server. Fix warnings by setting the required keys in <span class='num'>.env</span> and restarting the bot.</span></div></div>")
        : "") +
      headCard("Environment", "cpu",
        '<div class="card-bd">' + C.kvHTML([
          ["LLM providers", "Groq · Gemini · OpenRouter"],
          ["Broker", "Alpaca (paper)"],
          ["Dashboard", "FastAPI static app"],
          ["Data sources", "Alpaca bars & news · yfinance"],
        ]) + "</div>", "")
    );
  };

  /* ==========================================================================
     Page registry / render
     ========================================================================== */

  var PAGE_TITLES = {
    overview: "Overview",
    portfolio: "Portfolio",
    positions: "Positions",
    orders: "Orders & Executions",
    ai: "AI Intelligence",
    agents: "Agent Performance",
    risk: "Risk",
    cycles: "Trading Cycles",
    health: "System Health",
    activity: "Activity & Audit Log",
    config: "Configuration",
  };
  var PAGE_ICONS = {
    overview: "grid", portfolio: "chart-candle", positions: "layers", orders: "list-ordered",
    ai: "brain", agents: "gauge", risk: "shield", cycles: "repeat",
    health: "pulse", activity: "scroll", config: "settings",
  };
  var PAGE_DESCS = {
    overview: "Portfolio, latest AI decisions and live agent activity at a glance.",
    portfolio: "Equity curve, allocation and account performance.",
    positions: "Open positions with AI bias and latest decisions.",
    orders: "Execution history from the paper trading account.",
    ai: "Decision pipeline, agent consensus, bull/bear debate and reasoning.",
    agents: "Historical accuracy and weighting of the AI agents.",
    risk: "Exposure, concentration and the risk agent's decisions.",
    cycles: "Every autonomous trading cycle, start to finish.",
    health: "Process, API and agent service monitoring.",
    activity: "Searchable audit timeline of everything the bot has done.",
    config: "Read-only view of the active bot configuration.",
  };

  var lastPageState = null;

  App.renderPage = function (page, params, keepState, keepScroll) {
    App.clearCharts();
    if (!keepState || !lastPageState || lastPageState.page !== page) lastPageState = null;
    var root = qs("#page");
    var html;
    try {
      html = P[page](params, lastPageState);
    } catch (e) {
      console.error(e);
      html = C.bannerHTML("err", "Something went wrong rendering this page.", U.esc(String(e && e.message)));
    }
    root.innerHTML = '<div class="pageload">' + html + "</div>";
    if (!keepScroll) window.scrollTo(0, 0);

    // bind page-level behaviors (filters, sorting) with state preservation
    if (P[page] && P[page]._bind) {
      var rerender = function (state) {
        lastPageState = state;
        root.innerHTML = '<div class="pageload">' + P[page](params, state) + "</div>";
        P[page]._bind(root, rerender, state);
        bindChart(root);
      };
      P[page]._bind(root, rerender, lastPageState);
    }

    bindChart(root);
    renderHeaderTimes();
  };

  function bindChart(root) {
    var wrap = qs("#equity-chart", root);
    if (!wrap) return;

    // lazy-load the selected range if we don't have it yet (live mode)
    var range = store.ui.range || "1M";
    if (!store.demo && !store.histCache[range] && range !== "1M") {
      API.history(range).then(function (r) {
        store.histCache[range] = r.ok ? r.data : { points: [] };
        drawEquityChart();
        fillFallback();
      });
    }
    drawEquityChart();
    fillFallback();

    qsa("[data-range]", root).forEach(function (b) {
      b.addEventListener("click", async function () {
        store.ui.range = b.getAttribute("data-range");
        qsa("[data-range]", root).forEach(function (x) { x.setAttribute("aria-selected", String(x === b)); });
        if (!store.demo) {
          if (!store.histCache[store.ui.range]) {
            var r = await API.history(store.ui.range);
            store.histCache[store.ui.range] = r.ok ? r.data : { points: [] };
          }
        } else {
          store.histCache[store.ui.range] = { points: demoPts(store.ui.range) };
        }
        // refresh header numbers too
        App.renderPage(store.ui.page, store.ui.params, true, true);
      });
    });
  }

  function demoPts(range) {
    var pf = store.data.portfolio || {};
    var eq = (pf.account && pf.account.portfolio_value) || 0;
    return window.__demoHistory(range, eq);
  }

  function drawEquityChart() {
    var wrap = qs("#equity-chart");
    if (!wrap) return;
    var range = store.ui.range || "1M";
    var pts;
    if (store.demo) {
      pts = demoPts(range);
    } else if (store.histCache[range] && store.histCache[range].points) {
      pts = store.histCache[range].points;
    } else if (range === "1M") {
      pts = (store.data.portfolio && store.data.portfolio.history) || [];
    } else {
      pts = [];
    }
    var mapped = (pts || []).map(function (p) {
      return { t: U.parseDate(p.timestamp != null ? p.timestamp : p.t).getTime(), v: p.equity != null ? p.equity : p.v };
    }).filter(function (p) { return p.t && p.v != null; });

    App.chart({
      el: wrap,
      points: mapped,
      height: 300,
      aria: "Portfolio equity over " + range,
      emptyMsg: "No equity history available for this range from the broker yet.",
    });
  }

  function fillFallback() {
    var fb = qs("#chart-fallback");
    if (!fb) return;
    var range = store.ui.range || "1M";
    var pts = store.demo ? demoPts(range) : (store.histCache[range] && store.histCache[range].points) || (range === "1M" ? ((store.data.portfolio && store.data.portfolio.history) || []) : []);
    var mapped = (pts || []).map(function (p) {
      return { t: U.parseDate(p.timestamp != null ? p.timestamp : p.t).getTime(), v: p.equity != null ? p.equity : p.v };
    }).filter(function (p) { return p.t && p.v != null; });
    fb.innerHTML = mapped.slice(-40).map(function (p) {
      var d = new Date(p.t);
      return "<tr><td>" + d.toLocaleString() + '</td><td class="r num-strong">' + U.fmtMoney(p.v) + "</td></tr>";
    }).join("") || '<tr><td colspan="3" class="t-faint">No data</td></tr>';
  }

  function renderHeaderTimes() {
    var st = botState();
    var el = qs("#lastcycle-val");
    if (el) el.textContent = st.last_cycle_at ? U.ago(st.last_cycle_at) : "never";
  }

  App.PAGES = P;
  App.PAGE_TITLES = PAGE_TITLES;
  App.PAGE_ICONS = PAGE_ICONS;
  App.PAGE_DESCS = PAGE_DESCS;

  // expose demo history generator for app.js demo mode
  App._demoHistoryRef = function () {};
})();
