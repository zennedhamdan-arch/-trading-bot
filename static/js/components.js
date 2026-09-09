/* ============================================================================
   AI TRADER — Shared UI components & drawer contents
   ========================================================================== */
(function () {
  "use strict";

  var U = window.U, ICON = window.ICON, App = window.App;
  var qs = App.qs, qsa = App.qsa, h = App.h;
  var store = App.store;

  /* ==========================================================================
     States
     ========================================================================== */

  function stateHTML(cfg) {
    return (
      '<div class="state ' + (cfg.err ? "state-err" : cfg.warn ? "state-warn" : "") + " " + (cfg.off ? "state-off" : "") + '">' +
        ICON(cfg.icon || "inbox") +
        '<div class="state-title">' + U.esc(cfg.title) + "</div>" +
        (cfg.msg ? '<div class="state-msg">' + cfg.msg + "</div>" : "") +
        (cfg.actions ? '<div class="state-actions">' + cfg.actions + "</div>" : "") +
      "</div>"
    );
  }

  function skeletonRows(n, hgt) {
    var s = "";
    for (var i = 0; i < (n || 6); i++) s += '<div class="sk sk-row" style="height:' + (hgt || 34) + 'px"></div>';
    return s;
  }

  function pageSkeleton(kind) {
    if (kind === "table") {
      return '<div class="card"><div class="card-bd">' + skeletonRows(8, 38) + "</div></div>";
    }
    return (
      '<div class="kpi-strip mb-12">' + Array.from({ length: 6 }).map(function () { return '<div class="sk sk-kpi"></div>'; }).join("") + "</div>" +
      '<div class="card mb-12"><div class="card-bd"><div class="sk" style="height:280px"></div></div></div>' +
      '<div class="card"><div class="card-bd">' + skeletonRows(6) + "</div></div>"
    );
  }

  function bannerHTML(kind, title, body, actions) {
    return (
      '<div class="banner banner-' + kind + '" role="' + (kind === "err" ? "alert" : "status") + '">' +
        ICON(kind === "err" ? "alert" : kind === "warn" ? "alert" : kind === "off" ? "offline" : "info") +
        '<div class="grow b-body"><span class="b-title">' + title + "</span>" + (body ? " " + body : "") + "</div>" +
        (actions ? '<div class="b-actions">' + actions + "</div>" : "") +
      "</div>"
    );
  }

  /* ==========================================================================
     Small pieces
     ========================================================================== */

  function meterHTML(pct, cls, attrs) {
    return '<div class="meter ' + (cls || "") + '" ' + (attrs || "") + ' role="presentation"><i style="width:' + U.clamp(pct || 0, 0, 100) + '%"></i></div>';
  }

  function confHTML(conf) {
    if (conf == null) return '<span class="t-faint">—</span>';
    var pct = Math.round(conf * 100);
    return '<span class="conf" title="Confidence ' + pct + '%">' + meterHTML(pct, "meter-violet") + '<span class="conf-v num">' + pct + "%</span></span>";
  }

  function kvHTML(rows) {
    return (
      '<div class="kv">' + rows.map(function (r) {
        return "<div>" + '<div class="k">' + U.esc(r[0]) + "</div>" + '<div class="v ' + (r[2] || "") + '">' + r[1] + "</div></div>";
      }).join("") + "</div>"
    );
  }

  function sectionTitle(icon, title, aux) {
    return (
      '<div class="dsec-hd">' + ICON(icon) + "<span>" + U.esc(title) + "</span></div>"
    );
  }

  /* ==========================================================================
     Verdicts — map any agent log to a directional verdict
     ========================================================================== */

  function verdictOf(log) {
    if (!log) return null;
    var d = log.data || {};
    var a = log.agent;
    function v(label, dir, conf) { return { label: label, dir: dir, conf: conf, badge: badgeForVerdict(label) }; }
    if (a === "technical" || a === "fundamentals") {
      var s = (d.signal || "").toUpperCase();
      if (s === "BULLISH") return v("BULLISH", "bull", d.confidence);
      if (s === "BEARISH") return v("BEARISH", "bear", d.confidence);
      return v("NEUTRAL", "neutral", d.confidence);
    }
    if (a === "news") {
      var s2 = (d.sentiment || d.signal || "").toUpperCase();
      if (s2 === "BULLISH" || s2 === "POSITIVE") return v("BULLISH", "bull", d.confidence);
      if (s2 === "BEARISH" || s2 === "NEGATIVE") return v("BEARISH", "bear", d.confidence);
      return v("NEUTRAL", "neutral", d.confidence);
    }
    if (a === "cio") {
      var k = (d.decision || "").toUpperCase();
      return v(k || "HOLD", k === "BUY" ? "bull" : k === "SELL" ? "bear" : "neutral", d.confidence);
    }
    if (a === "risk") {
      if (log.level === "WARNING") return v("REVIEW", "neutral", null);
      return d.approced ? v("APPROVED", "bull", null) : d.approved === false ? v("REJECTED", "bear", null) : v(d.risk_level || "—", "neutral", null);
    }
    if (a === "debate") {
      var e = Number(d.edge);
      if (isNaN(e)) return v("—", "neutral", null);
      if (e > 0.15) return v("BULLISH", "bull", Math.max(d.bull_strength || 0, 0.5));
      if (e < -0.15) return v("BEARISH", "bear", Math.max(d.bear_strength || 0, 0.5));
      return v("NEUTRAL", "neutral", 0.5);
    }
    if (a === "execution") {
      return d.success ? v("SUBMITTED", "bull", null) : v("FAILED", "bear", null);
    }
    return null;
  }

  function badgeForVerdict(label) {
    var k = String(label || "").toUpperCase();
    if (["BUY", "BULLISH", "APPROVED", "SUBMITTED", "FILLED"].indexOf(k) >= 0) return "badge badge-buy";
    if (["SELL", "BEARISH", "REJECTED", "FAILED"].indexOf(k) >= 0) return "badge badge-sell";
    if (["HOLD", "NEUTRAL", "REVIEW"].indexOf(k) >= 0) return "badge badge-hold";
    return "badge";
  }

  function verdictHTML(log) {
    var v = verdictOf(log);
    if (!v) return "";
    return '<span class="' + v.badge + '">' + U.esc(v.label) + "</span>";
  }

  /* Directional agents' opinion vs the CIO decision */
  var DIRECTIONAL = ["technical", "news", "fundamentals", "debate"];

  function decisionWindow(cioLog) {
    var t = U.parseDate(cioLog.timestamp).getTime();
    return { symbol: cioLog.symbol, from: t - 26 * 60000, to: t + 6 * 60000, cioAt: t };
  }

  function logsForDecision(cioLog) {
    var w = decisionWindow(cioLog);
    var logs = (store.data.logs && store.data.logs.logs) || [];
    var out = {};
    logs.forEach(function (l) {
      if (l.symbol !== w.symbol) return;
      var t = U.parseDate(l.timestamp).getTime();
      if (t < w.from || t > w.to) return;
      if (!out[l.agent] || l.timestamp > out[l.agent].timestamp) out[l.agent] = l;
    });
    return out;
  }

  function latestCioLogs(limit) {
    var logs = (store.data.logs && store.data.logs.logs) || [];
    return logs.filter(function (l) { return l.agent === "cio"; }).slice(0, limit || 20);
  }

  /* ==========================================================================
     Consensus
     ========================================================================== */

  function consensusHTML(win) {
    var cioV = verdictOf(win.cio);
    var rows = "", agree = 0, total = 0;
    DIRECTIONAL.forEach(function (a) {
      var l = win[a];
      if (!l) return;
      var v = verdictOf(l);
      if (!v) return;
      total++;
      if (v.dir === cioV.dir) agree++;
      var pct = Math.round((v.conf || 0) * 100);
      var bar;
      if (v.dir === "bull") bar = '<div class="divmeter"><span class="mid"></span><span class="fill" style="left:50%;width:' + (pct / 2) + '%;background:var(--green)"></span></div>';
      else if (v.dir === "bear") bar = '<div class="divmeter"><span class="mid"></span><span class="fill" style="right:50%;width:' + (pct / 2) + '%;background:var(--red)"></span></div>';
      else bar = '<div class="divmeter"><span class="mid"></span><span class="fill" style="left:' + (50 - pct / 4) + '%;width:' + (pct / 2) + '%;background:var(--amber);opacity:.75"></span></div>';
      rows +=
        '<div class="cons-row">' +
          '<span class="cons-agent">' + U.agentLabel(a).replace(" Agent", "") + "</span>" +
          bar +
          '<span class="cons-verdict ' + U.classForDecision(v.label) + '">' + U.esc(v.label) + (v.conf != null ? " · " + Math.round(v.conf * 100) + "%" : "") + "</span>" +
        "</div>";
    });
    if (win.risk) {
      var rv = verdictOf(win.risk);
      rows += '<div class="cons-row"><span class="cons-agent">Risk</span>' +
        '<div class="divmeter"><span class="mid"></span><span class="fill" style="left:50%;width:' + (win.risk.data && win.risk.data.approved ? "40%" : "0") + ';background:var(--blue)"></span></div>' +
        '<span class="cons-verdict ' + (rv.dir === "bull" ? "pos" : rv.dir === "bear" ? "neg" : "neu") + '">' + U.esc(rv.label) + "</span></div>";
    }
    if (!rows) return stateHTML({ icon: "brain", title: "No agent data", msg: "Agent reports for this decision window are no longer in the live log buffer." });
    var agreement = total ? Math.round((agree / total) * 100) : null;
    return (
      '<div class="consensus">' + rows + "</div>" +
      (agreement != null
        ? '<div class="row" style="margin-top:10px;justify-content:space-between">' +
            '<span class="t-faint" style="font-size:11px">DIRECTIONAL AGREEMENT</span>' +
            '<span class="num ' + (agreement >= 67 ? "pos" : agreement >= 34 ? "neu" : "neg") + '" style="font-size:12px">' + agree + "/" + total + " agents · " + agreement + "%</span>" +
          "</div>"
        : "")
    );
  }

  /* ==========================================================================
     Pipeline
     ========================================================================== */

  var PIPELINE_STAGES = [
    { agent: "market", name: "Market Data", icon: "globe" },
    { agent: "technical", name: "Technical Agent", icon: "chart-candle" },
    { agent: "news", name: "News Agent", icon: "news" },
    { agent: "fundamentals", name: "Fundamentals", icon: "db" },
    { agent: "debate", name: "Bull / Bear Debate", icon: "scale" },
    { agent: "risk", name: "Risk Agent", icon: "shield" },
    { agent: "cio", name: "CIO / Final Decision", icon: "gavel" },
    { agent: "execution", name: "Execution", icon: "zap" },
  ];

  function pipelineStageHTML(stage, log, opts) {
    opts = opts || {};
    var v = log ? verdictOf(log) : null;
    var nodeCls = "pipe-node", statusTxt = "", sub = "";
    var disabledAgents = disabledAgentSet();

    if (log) {
      if (log.level === "ERROR") { nodeCls += " err"; statusTxt = '<span class="badge badge-error">ERROR</span>'; }
      else nodeCls += " done";
      if (stage.agent === "execution" && (!v || v.label !== "SUBMITTED")) nodeCls = "pipe-node skipped";
    } else {
      if (stage.agent === "market") {
        nodeCls += " done";
        sub = "Indicators & headlines fetched";
      } else if (stage.agent === "execution") {
        nodeCls += " skipped";
        sub = "No order required (HOLD)";
      } else if (disabledAgents[stage.agent]) {
        nodeCls += " skipped";
        sub = "Disabled via configuration";
        statusTxt = '<span class="badge badge-neutral">OFF</span>';
      } else {
        nodeCls += " skipped";
        sub = "No record in live buffer";
      }
    }

    var meta = [];
    if (log) {
      meta.push('<span class="m">' + ICON("clock") + U.fmtTime(log.timestamp) + "</span>");
      if (v && v.conf != null) meta.push('<span class="m">CONF <span class="num">' + Math.round(v.conf * 100) + "%</span></span>");
    }

    return (
      '<div class="pipe-stage" ' + (log ? 'data-action="open-log" data-ts="' + U.esc(log.timestamp) + '" role="button" tabindex="0"' : "") + ">" +
        '<div class="pipe-rail"><div class="' + nodeCls + '">' + ICON(stage.icon) + '</div><div class="pipe-line' + (log ? " done" : "") + '"></div></div>' +
        '<div class="pipe-body">' +
          '<div class="pipe-hd"><span class="pipe-name">' + stage.name + "</span>" + (v ? verdictHTML(log) : "") + statusTxt + "</div>" +
          '<div class="pipe-sub">' + (sub || (log ? U.esc(log.message) : "")) + "</div>" +
          (meta.length ? '<div class="pipe-meta">' + meta.join("") + "</div>" : "") +
        "</div>" +
        '<div class="feed-side"></div>' +
      "</div>"
    );
  }

  function disabledAgentSet() {
    var cfg = store.data.config || {};
    var d = {};
    if (cfg.enable_fundamentals_agent === false) d.fundamentals = true;
    if (cfg.enable_debate === false) d.debate = true;
    if (cfg.enable_memory === false) d.memory = true;
    return d;
  }

  function pipelineHTML(cioLog, opts) {
    var win = logsForDecision(cioLog);
    var stages = PIPELINE_STAGES.map(function (s) {
      return pipelineStageHTML(s, win[s.agent], { symbol: cioLog.symbol });
    }).join("");
    return '<div class="pipe">' + stages + "</div>";
  }

  /* ==========================================================================
     Debate
     ========================================================================== */

  function debateHTML(debateLog) {
    if (!debateLog || !debateLog.data) {
      return stateHTML({ icon: "scale", title: "No debate recorded", msg: "The bull/bear debate either did not run for this decision or is disabled in configuration." });
    }
    var d = debateLog.data;
    var bull = Math.round((d.bull_strength || 0) * 100);
    var bear = Math.round((d.bear_strength || 0) * 100);
    var edge = d.edge || 0;
    var bias = edge > 0.15 ? ["BULLISH", "badge-buy", "pos"] : edge < -0.15 ? ["BEARISH", "badge-sell", "neg"] : ["NEUTRAL", "badge-hold", "neu"];
    var edgePct = Math.round(Math.abs(edge) * 100);
    var edgeLeft = edge < 0 ? 50 - Math.min(50, edgePct / 2) : null;
    var edgeWidth = Math.min(50, edgePct / 2);

    return (
      '<div class="debate">' +
        '<div class="debate-side debate-bull">' +
          '<div class="debate-hd">' + ICON("trend-up") + '<span class="d-name">Bull Case</span><span class="spacer"></span><span class="num pos">' + bull + "%</span></div>" +
          '<div class="debate-bd">' + U.esc(d.bull_summary || "—") + "</div>" +
          '<div class="debate-ft">' + meterHTML(bull, "meter-green", 'aria-label="Bull strength ' + bull + '%"') + "</div>" +
        "</div>" +
        '<div class="debate-side debate-bear">' +
          '<div class="debate-hd">' + ICON("trend-down") + '<span class="d-name">Bear Case</span><span class="spacer"></span><span class="num neg">' + bear + "%</span></div>" +
          '<div class="debate-bd">' + U.esc(d.bear_summary || "—") + "</div>" +
          '<div class="debate-ft">' + meterHTML(bear, "meter-red", 'aria-label="Bear strength ' + bear + '%"') + "</div>" +
        "</div>" +
      "</div>" +
      '<div class="mt-12">' +
        '<div class="row" style="justify-content:space-between;margin-bottom:5px">' +
          '<span class="lbl">Debate Edge</span>' +
          '<span class="num t-lo" style="font-size:11px">bull − bear = ' + (edge >= 0 ? "+" : "") + edge.toFixed(2) + "</span>" +
        "</div>" +
        '<div class="divmeter" style="height:8px" role="img" aria-label="Debate edge ' + bias[0] + '">' +
          "<span class=\"mid\"></span>" +
          (edge >= 0
            ? '<span class="fill" style="left:50%;width:' + edgeWidth + '%;background:var(--green)"></span>'
            : '<span class="fill" style="right:50%;width:' + edgeWidth + '%;background:var(--red)"></span>') +
        "</div>" +
        '<div class="meter-scale"><span class="neg">Bearish</span><span>Neutral</span><span class="pos">Bullish</span></div>' +
        '<div class="row" style="justify-content:space-between;margin-top:10px">' +
          '<span class="lbl">Final Bias</span>' +
          '<span class="' + bias[1] + '">' + bias[0] + "</span>" +
        "</div>" +
      "</div>"
    );
  }

  /* ==========================================================================
     Feed item
     ========================================================================== */

  function feedItemHTML(l, opts) {
    opts = opts || {};
    var isNew = opts.newAfter && l.timestamp > opts.newAfter;
    var badge = U.levelBadge(l.level);
    var v = l.agent === "cio" || l.agent === "execution" ? verdictHTML(l) : "";
    var actionable = l.agent === "cio" || l.agent === "execution" || l.agent === "risk";
    return (
      '<div class="feed-item' + (actionable ? " rowlink" : "") + (isNew ? " new" : "") + '" ' +
        (actionable ? 'data-action="open-log" data-ts="' + U.esc(l.timestamp) + '" role="button" tabindex="0"' : "") + ">" +
        '<div class="feed-ico ' + U.agentCls(l.agent) + " " + (l.level === "ERROR" || l.level === "WARNING" ? "l-" + l.level : "") + '">' + ICON(U.agentIcon(l.agent)) + "</div>" +
        '<div class="feed-body">' +
          '<div class="feed-line1">' +
            '<span class="feed-agent">' + U.agentLabel(l.agent) + "</span>" +
            (l.symbol ? '<a class="sym sym-sm" data-action="goto" data-page="activity" data-sym="' + U.esc(l.symbol) + '" href="#/activity?sym=' + U.esc(l.symbol) + '">' + U.esc(l.symbol) + "</a>" : "") +
            badge +
          "</div>" +
          '<div class="feed-msg">' + U.esc(l.message) + "</div>" +
        "</div>" +
        '<div class="feed-side">' +
          '<span class="feed-time" data-ago="' + U.esc(l.timestamp) + '">' + U.ago(l.timestamp) + "</span>" +
          '<span class="feed-time">' + U.fmtTime(l.timestamp) + "</span>" +
          v +
        "</div>" +
      "</div>"
    );
  }

  /* ==========================================================================
     Positions table (responsive: table on desktop, cards on mobile)
     ========================================================================== */

  function positionsTableHTML(positions, opts) {
    opts = opts || {};
    if (!positions.length) {
      return stateHTML({
        icon: "layers", title: "No open positions",
        msg: "The bot has not opened any positions yet. When the CIO agent issues a BUY decision and the order fills, positions will appear here.",
      });
    }
    var totalMV = positions.reduce(function (a, p) { return a + (p.market_value || 0); }, 0);
    var totalPL = positions.reduce(function (a, p) { return a + (p.unrealized_pl || 0); }, 0);

    var head =
      "<tr>" +
        '<th class="sortable" data-sort="symbol">Symbol' + sortInd("symbol", opts) + "</th>" +
        '<th class="sortable r" data-sort="qty">Position' + sortInd("qty", opts) + "</th>" +
        '<th class="sortable r" data-sort="avg_entry_price">Avg Entry</th>' +
        '<th class="sortable r" data-sort="current_price">Current</th>' +
        '<th class="sortable r" data-sort="market_value">Market Value</th>' +
        '<th class="sortable r" data-sort="unrealized_pl">Unrealized P&L</th>' +
        '<th class="sortable r" data-sort="unrealized_plpc">P&L %</th>' +
        '<th class="sortable r" data-sort="weight_pct">Weight</th>' +
        (opts.aiCols ? "<th>AI Bias</th><th>Last Decision</th>" : "") +
        "<th>Status</th>" +
      "</tr>";

    var rows = positions.map(function (p) {
      var dec = p.cio && p.cio.data ? p.cio.data.decision : null;
      var bias = biasOf(p.bias);
      return (
        '<tr class="rowlink" data-action="open-position" data-sym="' + U.esc(p.symbol) + '" tabindex="0" aria-label="Position details for ' + U.esc(p.symbol) + '">' +
          '<td><span class="sym">' + U.esc(p.symbol) + '</span><div class="td-sub">Long</div></td>' +
          '<td class="r num-strong">' + U.fmtQty(p.qty) + ' <span class="t-faint">sh</span></td>' +
          '<td class="r">' + (p.avg_entry_price != null ? U.fmtMoney(p.avg_entry_price) : "—") + "</td>" +
          '<td class="r num-strong">' + (p.current_price != null ? U.fmtMoney(p.current_price) : "—") + "</td>" +
          '<td class="r">' + U.fmtMoney(p.market_value) + "</td>" +
          '<td class="r ' + U.classFor(p.unrealized_pl) + '">' + U.fmtSigned(p.unrealized_pl) + "</td>" +
          '<td class="r ' + U.classFor(p.unrealized_plpc) + '">' + U.fmtPct(p.unrealized_plpc) + "</td>" +
          '<td class="r">' + (p.weight_pct != null ? p.weight_pct.toFixed(1) + "%" : "—") + meterHTML(p.weight_pct, "meter-thin", 'style="min-width:46px;display:inline-block;vertical-align:middle;margin-left:6px"') + "</td>" +
          (opts.aiCols
            ? "<td>" + (bias ? '<span class="' + bias[1] + '">' + bias[0] + "</span>" : '<span class="t-faint">—</span>') + "</td>" +
              "<td>" + (dec ? '<span class="' + U.badgeForDecision(dec) + '">' + dec + "</span>" : '<span class="t-faint">—</span>') + "</td>"
            : "") +
          '<td><span class="badge badge-filled">OPEN</span></td>' +
        "</tr>"
      );
    }).join("");

    var foot =
      '<tr style="background:var(--bg-inset)">' +
        '<td class="lbl" style="color:var(--ink-lo)">' + positions.length + " positions</td><td></td><td></td>" +
        '<td class="lbl" style="text-align:right;color:var(--ink-lo)">Total</td>' +
        '<td class="r num-strong">' + U.fmtMoney(totalMV) + "</td>" +
        '<td class="r num-strong ' + U.classFor(totalPL) + '">' + U.fmtSigned(totalPL) + "</td>" +
        '<td class="r ' + U.classFor(totalPL) + '">' + U.fmtPct(totalMV ? (totalPL / (totalMV - totalPL)) * 100 : 0) + "</td>" +
        "<td></td>" + (opts.aiCols ? "<td></td><td></td>" : "") + "<td></td>" +
      "</tr>";

    var table =
      '<div class="tbl-wrap responsive"><table class="tbl"><thead>' + head + '</thead><tbody>' + rows + foot + "</tbody></table>";

    // mobile cards
    var cards = positions.map(function (p) {
      var dec = p.cio && p.cio.data ? p.cio.data.decision : null;
      var bias = biasOf(p.bias);
      return (
        '<div class="mcard" data-action="open-position" data-sym="' + U.esc(p.symbol) + '" role="button" tabindex="0">' +
          '<div class="mcard-hd"><span class="sym">' + U.esc(p.symbol) + '</span><span class="badge badge-filled">OPEN</span><span class="spacer"></span>' +
            (dec ? '<span class="' + U.badgeForDecision(dec) + '">' + dec + "</span>" : "") + "</div>" +
          '<div class="mcard-main">' +
            '<div><div class="lbl">Market value</div><div class="big">' + U.fmtMoney(p.market_value) + "</div></div>" +
            '<div style="text-align:right"><div class="lbl">Unrealized P&L</div><div class="big ' + U.classFor(p.unrealized_pl) + '">' + U.fmtSigned(p.unrealized_pl) + "</div>" +
            '<div class="num ' + U.classFor(p.unrealized_plpc) + '" style="font-size:12px">' + U.fmtPct(p.unrealized_plpc) + "</div></div>" +
          "</div>" +
          '<div class="mcard-grid">' +
            '<div class="g"><div class="k">Shares</div><div class="v">' + U.fmtQty(p.qty) + "</div></div>" +
            '<div class="g"><div class="k">Avg entry</div><div class="v">' + U.fmtMoney(p.avg_entry_price) + "</div></div>" +
            '<div class="g"><div class="k">Current</div><div class="v">' + U.fmtMoney(p.current_price) + "</div></div>" +
            '<div class="g"><div class="k">Weight</div><div class="v">' + (p.weight_pct != null ? p.weight_pct.toFixed(1) + "%" : "—") + "</div></div>" +
            '<div class="g"><div class="k">AI bias</div><div class="v">' + (bias ? bias[0] : "—") + "</div></div>" +
            '<div class="g"><div class="k">Last decision</div><div class="v">' + (dec || "—") + "</div></div>" +
          "</div>" +
        "</div>"
      );
    }).join("");
    table += '<div class="mcards">' + cards + "</div></div>";
    return table;
  }

  function biasOf(debateLog) {
    if (!debateLog || !debateLog.data) return null;
    var e = Number(debateLog.data.edge);
    if (isNaN(e)) return null;
    if (e > 0.15) return ["BULLISH", "badge badge-buy", "pos"];
    if (e < -0.15) return ["BEARISH", "badge badge-sell", "neg"];
    return ["NEUTRAL", "badge badge-hold", "neu"];
  }

  function sortInd(key, opts) {
    if (!opts || opts.sort !== key) return "";
    return '<span class="sort-ind">' + (opts.sortDir === -1 ? "▼" : "▲") + "</span>";
  }

  /* ==========================================================================
     Orders table
     ========================================================================== */

  function orderValue(o) {
    if (o.filled_avg_price != null && o.qty != null) return o.filled_avg_price * o.qty;
    return null;
  }

  function ordersTableHTML(orders, opts) {
    opts = opts || {};
    if (!orders.length) {
      return stateHTML({
        icon: "list-ordered", title: "No orders yet",
        msg: "When the execution agent submits paper orders, they will appear here with full fill details.",
      });
    }
    var trig = App.matchOrdersToCycles();
    var rows = orders.map(function (o) {
      var val = orderValue(o);
      return (
        '<tr class="rowlink" data-action="open-order" data-id="' + U.esc(o.id) + '" tabindex="0">' +
          "<td>" + U.fmtDateTime(o.submitted_at) + '<div class="td-sub" data-ago="' + U.esc(o.submitted_at || "") + '">' + U.ago(o.submitted_at) + "</div></td>" +
          '<td><span class="sym">' + U.esc(o.symbol) + "</span></td>" +
          '<td><span class="badge badge-side ' + (String(o.side).toLowerCase() === "buy" ? "badge-buy" : "badge-sell") + '">' + U.esc(String(o.side || "").toUpperCase()) + "</span></td>" +
          '<td class="r">' + U.fmtQty(o.qty) + "</td>" +
          "<td>" + '<span class="' + U.badgeForOrderStatus(o.status) + '">' + U.esc(String(o.status || "—").toUpperCase().replace(/_/g, " ")) + "</span></td>" +
          '<td class="r">' + (o.filled_avg_price != null ? U.fmtMoney(o.filled_avg_price) : "—") + "</td>" +
          '<td class="r">' + (val != null ? U.fmtMoney(val, { dec: 0 }) : "—") + "</td>" +
          '<td class="t-lo" style="font-size:11.5px">' + U.esc(trig[o.id] || "AI Cycle") + "</td>" +
        "</tr>"
      );
    }).join("");

    var cards = orders.map(function (o) {
      var val = orderValue(o);
      return (
        '<div class="mcard" data-action="open-order" data-id="' + U.esc(o.id) + '" role="button" tabindex="0">' +
          '<div class="mcard-hd"><span class="sym">' + U.esc(o.symbol) + "</span>" +
            '<span class="badge badge-side ' + (String(o.side).toLowerCase() === "buy" ? "badge-buy" : "badge-sell") + '">' + U.esc(String(o.side || "").toUpperCase()) + "</span>" +
            '<span class="' + U.badgeForOrderStatus(o.status) + '">' + U.esc(String(o.status || "").toUpperCase().replace(/_/g, " ")) + "</span>" +
            '<span class="spacer"></span><span class="feed-time">' + U.fmtDateTime(o.submitted_at) + "</span></div>" +
          '<div class="mcard-grid">' +
            '<div class="g"><div class="k">Quantity</div><div class="v">' + U.fmtQty(o.qty) + "</div></div>" +
            '<div class="g"><div class="k">Fill price</div><div class="v">' + (o.filled_avg_price != null ? U.fmtMoney(o.filled_avg_price) : "—") + "</div></div>" +
            '<div class="g"><div class="k">Value</div><div class="v">' + (val != null ? U.fmtMoney(val, { dec: 0 }) : "—") + "</div></div>" +
            '<div class="g"><div class="k">Triggered by</div><div class="v" style="font-family:var(--font-ui);font-size:11px">' + U.esc(trig[o.id] || "AI Cycle") + "</div></div>" +
          "</div>" +
        "</div>"
      );
    }).join("");

    return (
      '<div class="tbl-wrap responsive"><table class="tbl"><thead><tr>' +
        "<th>Time</th><th>Symbol</th><th>Side</th><th class='r'>Qty</th><th>Status</th><th class='r'>Price</th><th class='r'>Value</th><th>Triggered By</th>" +
      "</tr></thead><tbody>" + rows + "</tbody></table>" +
      '<div class="mcards">' + cards + "</div></div>"
    );
  }

  /* ==========================================================================
     Reasoning section (expandable)
     ========================================================================== */

  function reasonHTML(icon, title, badgeLog, bodyHTML, open) {
    var v = badgeLog ? verdictOf(badgeLog) : null;
    return (
      '<div class="reason" data-open="' + (open ? "true" : "false") + '">' +
        '<button class="r-hd" data-action="toggle-reason" aria-expanded="' + (open ? "true" : "false") + '">' +
          ICON(icon) + "<span>" + U.esc(title) + "</span>" +
          (v ? '<span style="margin-left:2px">' + '<span class="' + v.badge + '">' + U.esc(v.label) + "</span></span>" : "") +
          '<span class="chev">' + ICON("chev-r") + "</span>" +
        "</button>" +
        '<div class="r-bd">' + bodyHTML + "</div>" +
      "</div>"
    );
  }

  function statlinesHTML(rows) {
    return rows.map(function (r) {
      return '<div class="statline"><span class="sn">' + U.esc(r[0]) + '</span><span class="sv ' + (r[2] || "") + '">' + r[1] + "</span></div>";
    }).join("");
  }

  /* ==========================================================================
     Export
     ========================================================================== */

  window.C = {
    stateHTML: stateHTML,
    skeletonRows: skeletonRows,
    pageSkeleton: pageSkeleton,
    bannerHTML: bannerHTML,
    meterHTML: meterHTML,
    confHTML: confHTML,
    kvHTML: kvHTML,
    sectionTitle: sectionTitle,
    verdictOf: verdictOf,
    verdictHTML: verdictHTML,
    badgeForVerdict: badgeForVerdict,
    decisionWindow: decisionWindow,
    logsForDecision: logsForDecision,
    latestCioLogs: latestCioLogs,
    consensusHTML: consensusHTML,
    pipelineHTML: pipelineHTML,
    pipelineStages: PIPELINE_STAGES,
    debateHTML: debateHTML,
    feedItemHTML: feedItemHTML,
    positionsTableHTML: positionsTableHTML,
    ordersTableHTML: ordersTableHTML,
    orderValue: orderValue,
    reasonHTML: reasonHTML,
    statlinesHTML: statlinesHTML,
    biasOf: biasOf,
    sortInd: sortInd,
    disabledAgentSet: disabledAgentSet,
  };
})();
