/* ============================================================================
   AI TRADER — Detail drawers
   Position · Order · AI Decision · Cycle · Agent · Log
   ========================================================================== */
(function () {
  "use strict";

  var U = window.U, ICON = window.ICON, App = window.App, C = window.C;
  var store = App.store;

  function D(agent, symbol) {
    var logs = (store.data.logs && store.data.logs.logs) || [];
    return logs;
  }

  /* ==========================================================================
     Position drawer
     ========================================================================== */

  function openPositionDrawer(sym) {
    var pf = store.data.portfolio || {};
    var positions = pf.positions || [];
    var p = null;
    for (var i = 0; i < positions.length; i++) if (positions[i].symbol === sym) p = positions[i];
    if (!p) { App.toast("No open position for " + sym, "info"); return; }

    var ctx = App.positionsWithContext().filter(function (x) { return x.symbol === sym; })[0];
    var logs = D().filter(function (l) { return l.symbol === sym; });
    var orders = ((store.data.orders && store.data.orders.orders) || []).filter(function (o) { return o.symbol === sym; });
    var acct = pf.account || {};

    var cost = p.qty * p.avg_entry_price;
    var pl = p.unrealized_pl || 0;
    var plPct = cost > 0 ? (pl / cost) * 100 : 0;
    var bias = C.biasOf(ctx && ctx.bias);
    var dec = ctx && ctx.cio ? C.verdictOf(ctx.cio) : null;

    // agent opinions: latest log per agent
    var opinions = {};
    logs.forEach(function (l) {
      if (!opinions[l.agent] || l.timestamp > opinions[l.agent].timestamp) opinions[l.agent] = l;
    });

    var body =
      // ---- position information
      '<div class="dsec">' + C.sectionTitle("layers", "Position") +
        C.kvHTML([
          ["Quantity", U.fmtQty(p.qty) + " shares"],
          ["Side", "Long"],
          ["Avg entry", U.fmtMoney(p.avg_entry_price)],
          ["Current price", U.fmtMoney(p.current_price)],
          ["Market value", U.fmtMoney(p.market_value)],
          ["Cost basis", U.fmtMoney(cost)],
          ["Portfolio weight", ctx && ctx.weight_pct != null ? ctx.weight_pct.toFixed(1) + "%" : "—"],
          ["Status", '<span class="badge badge-filled">OPEN</span>'],
        ]) +
      "</div>" +
      // ---- P&L
      '<div class="dsec">' + C.sectionTitle("scale", "Unrealized P&L") +
        '<div class="row" style="justify-content:space-between;margin-bottom:6px">' +
          '<span class="num ' + U.classFor(pl) + '" style="font-size:20px;font-weight:600">' + U.fmtSigned(pl) + "</span>" +
          '<span class="num ' + U.classFor(pl) + '" style="font-size:14px">' + U.fmtPct(plPct) + "</span>" +
        "</div>" +
        C.meterHTML(Math.min(100, Math.abs(plPct) * 10), pl >= 0 ? "meter-green" : "meter-red", 'role="img" aria-label="' + U.fmtSigned(pl) + " " + U.fmtPct(plPct) + ' unrealized"') +
        '<div class="meter-scale"><span>cost ' + U.fmtMoney(cost, { dec: 0 }) + "</span><span>current " + U.fmtMoney(p.market_value, { dec: 0 }) + "</span></div>" +
      "</div>" +
      // ---- AI context
      '<div class="dsec">' + C.sectionTitle("brain", "AI Context") +
        C.kvHTML([
          ["Last CIO decision", dec ? '<span class="' + dec.badge + '">' + dec.label + "</span>" : "—"],
          ["Decision time", ctx && ctx.cio ? U.fmtDateTime(ctx.cio.timestamp) : "—"],
          ["Debate bias", bias ? '<span class="' + bias[1] + '">' + bias[0] + "</span>" : "—"],
          ["Risk status", ctx && ctx.risk && ctx.risk.data ? (ctx.risk.data.approved ? '<span class="badge badge-approved">APPROVED</span>' : '<span class="badge badge-rejected">REJECTED</span>') : "—"],
        ]) +
        (ctx && ctx.cio
          ? '<div class="note" style="margin-top:8px">' + ICON("info") + "<span>" + U.esc(ctx.cio.message) + "</span></div>" +
            '<button class="btn btn-sm" style="margin-top:8px" data-action="open-decision" data-ts="' + U.esc(ctx.cio.timestamp) + '" data-agent="cio">' + ICON("gavel") + "View full AI reasoning</button>"
          : "") +
      "</div>" +
      // ---- agent opinions
      (Object.keys(opinions).length
        ? '<div class="dsec">' + C.sectionTitle("cpu", "Latest Agent Opinions") +
            ["technical", "news", "fundamentals", "debate", "risk", "cio"].map(function (a) {
              var l = opinions[a];
              if (!l) return "";
              var v = C.verdictOf(l);
              return (
                '<div class="row" style="justify-content:space-between;padding:6px 0;border-bottom:1px dashed var(--line)">' +
                  "<span>" + '<span style="font-size:12px;font-weight:600;color:var(--ink-hi)">' + U.agentLabel(a) + "</span>" +
                  '<div class="feed-time" style="margin-top:1px">' + U.fmtDateTime(l.timestamp) + "</div></span>" +
                  "<span class='row'>" + (v && v.conf != null ? C.confHTML(v.conf) : "") + (v ? '<span class="' + v.badge + '">' + v.label + "</span>" : "") + "</span>" +
                "</div>"
              );
            }).join("") +
          "</div>"
        : "") +
      // ---- order history
      (orders.length
        ? '<div class="dsec">' + C.sectionTitle("list-ordered", "Order History") +
            C.ordersTableHTML(orders.slice(0, 6)) +
          "</div>"
        : "") +
      // ---- activity
      (logs.length
        ? '<div class="dsec">' + C.sectionTitle("activity", "Recent Activity") +
            '<div class="card" style="box-shadow:none">' + logs.slice(0, 8).map(function (l) { return C.feedItemHTML(l); }).join("") + "</div>" +
          "</div>"
        : '<div class="dsec">' + C.sectionTitle("activity", "Recent Activity") +
            C.stateHTML({ icon: "activity", title: "No activity recorded", msg: "No live log entries for this symbol in the current buffer.", compact: true }) +
          "</div>");

    App.openDrawer(
      '<span class="sym" style="font-size:16px">' + U.esc(sym) + "</span>" +
        '<span class="badge">LONG</span>' +
        '<span class="' + (pl >= 0 ? "pos" : "neg") + '" style="font-size:14px;font-weight:600;font-family:var(--font-mono)">' + U.fmtSigned(pl) + "</span>",
      "Position detail · Paper account",
      body,
      '<button class="btn btn-sm" data-action="goto" data-page="activity" data-sym="' + U.esc(sym) + '">' + ICON("activity") + "Symbol activity</button>" +
      '<span class="spacer"></span><span class="t-faint" style="font-size:11px">' + U.esc(sym) + " · PAPER</span>",
      { label: "Position detail " + sym }
    );
  }

  /* ==========================================================================
     Order drawer
     ========================================================================== */

  function openOrderDrawer(id) {
    var orders = ((store.data.orders && store.data.orders.orders) || []).concat(((store.data.portfolio && store.data.portfolio.orders) || []));
    var o = null;
    for (var i = 0; i < orders.length; i++) if (String(orders[i].id) === String(id)) o = orders[i];
    if (!o) { App.toast("Order not found in recent history", "info"); return; }

    var val = C.orderValue(o);
    var trig = App.matchOrdersToCycles()[o.id] || "AI Cycle";
    var logs = D().filter(function (l) { return l.symbol === o.symbol && l.agent === "cio"; });
    var relDecision = null;
    var t = U.parseDate(o.submitted_at).getTime();
    logs.forEach(function (l) {
      var lt = U.parseDate(l.timestamp).getTime();
      if (lt <= t + 60000 && (!relDecision || lt > U.parseDate(relDecision.timestamp).getTime())) relDecision = l;
    });

    var body =
      '<div class="dsec">' + C.sectionTitle("list-ordered", "Order") +
        C.kvHTML([
          ["Order ID", U.esc(o.id)],
          ["Status", '<span class="' + U.badgeForOrderStatus(o.status) + '">' + U.esc(String(o.status || "—").toUpperCase().replace(/_/g, " ")) + "</span>"],
          ["Symbol", '<span class="sym">' + U.esc(o.symbol) + "</span>"],
          ["Side", '<span class="' + (String(o.side).toLowerCase() === "buy" ? "pos" : "neg") + '">' + U.esc(String(o.side || "").toUpperCase()) + "</span>"],
          ["Quantity", U.fmtQty(o.qty)],
          ["Filled price", o.filled_avg_price != null ? U.fmtMoney(o.filled_avg_price) : "—"],
          ["Value", val != null ? U.fmtMoney(val) : "—"],
          ["Submitted", U.fmtDateTime(o.submitted_at)],
          ["Triggered by", U.esc(trig)],
          ["Account", "Paper trading"],
        ]) +
      "</div>" +
      (relDecision
        ? '<div class="dsec">' + C.sectionTitle("gavel", "Related AI Decision") +
            C.verdictHTML(relDecision) +
            '<div class="note" style="margin-top:8px">' + ICON("info") + "<span>" + U.esc(relDecision.message) + "</span></div>" +
            '<button class="btn btn-sm" style="margin-top:8px" data-action="open-decision" data-ts="' + U.esc(relDecision.timestamp) + '" data-agent="cio">' + ICON("gavel") + "View full reasoning</button>" +
          "</div>"
        : '<div class="dsec">' + C.sectionTitle("gavel", "Related AI Decision") +
            C.stateHTML({ icon: "brain", title: "No linked decision", msg: "No CIO decision for " + U.esc(o.symbol) + " was found shortly before this order in the live log buffer.", compact: true }) +
          "</div>") +
      '<div class="dsec">' + C.sectionTitle("info", "Note") +
        '<div class="note">' + ICON("info") + "<span>Order type and exchange routing are not exposed by the current API. Market orders are used by the execution agent.</span></div>" +
      "</div>";

    App.openDrawer(
      '<span class="sym" style="font-size:15px">' + U.esc(o.symbol) + "</span>" +
        '<span class="badge badge-side ' + (String(o.side).toLowerCase() === "buy" ? "badge-buy" : "badge-sell") + '">' + U.esc(String(o.side || "").toUpperCase()) + "</span>" +
        '<span class="' + U.badgeForOrderStatus(o.status) + '">' + U.esc(String(o.status || "").toUpperCase().replace(/_/g, " ")) + "</span>",
      "Order " + U.esc(o.id) + " · " + U.fmtDateTime(o.submitted_at),
      body,
      '<button class="btn btn-sm" data-action="goto" data-page="activity" data-sym="' + U.esc(o.symbol) + '">' + ICON("activity") + "Symbol activity</button>",
      { label: "Order detail " + o.id }
    );
  }

  /* ==========================================================================
     AI decision drawer — the full reasoning panel
     ========================================================================== */

  function openDecisionDrawer(ts, agent) {
    var logs = (store.data.logs && store.data.logs.logs) || [];
    var log = null;
    logs.forEach(function (l) { if (l.timestamp === ts && l.agent === (agent || "cio")) log = l; });
    if (!log) { App.toast("Decision not found in the live buffer", "info"); return; }

    var win = C.logsForDecision(log);
    var d = log.data || {};
    var v = C.verdictOf(log);
    var conf = d.confidence != null ? Math.round(d.confidence * 100) : null;
    var risk = win.risk, tech = win.technical, news = win.news, fund = win.fundamentals, debate = win.debate, exec = win.execution;
    var memory = win.memory;

    App.openDrawer(
      '<span class="sym" style="font-size:16px">' + U.esc(log.symbol || "") + "</span>" +
        (v ? '<span class="' + v.badge + '">' + v.label + "</span>" : "") +
        (conf != null ? '<span class="badge badge-ai">CONF ' + conf + "%</span>" : ""),
      "AI decision · " + U.fmtDateTime(log.timestamp),
      // ------- final decision
      '<div class="dsec">' + C.sectionTitle("gavel", "Final Decision") +
        C.kvHTML([
          ["Decision", v ? '<span class="' + v.badge + '" style="font-size:13px">' + v.label + "</span>" : "—"],
          ["Confidence", conf != null ? conf + "%" : "—"],
          ["Position recommendation", d.notional_usd ? U.fmtMoney(d.notional_usd, { dec: 0 }) + " notional" : "No new position"],
          ["Risk assessment", risk && risk.data ? (risk.data.approved ? '<span class="badge badge-approved">APPROVED</span>' : '<span class="badge badge-rejected">REJECTED</span>') + " · " + U.esc(risk.data.risk_level || "") + (risk.data.max_notional_usd ? " · max " + U.fmtMoney(risk.data.max_notional_usd, { dec: 0 }) : "") : "Not in buffer"],
          ["Execution result", exec ? (exec.data && exec.data.success ? '<span class="badge badge-filled">ORDER SUBMITTED</span>' : '<span class="badge badge-failed">FAILED</span>') : "No order required"],
        ]) +
      "</div>" +
      // ------- consensus
      '<div class="dsec">' + C.sectionTitle("scale", "Agent Consensus") + C.consensusHTML(win) + "</div>" +
      // ------- reasoning sections
      '<div class="dsec">' + C.sectionTitle("brain", "Reasoning") +
        '<div class="col" style="gap:8px">' +
          C.reasonHTML("chart-candle", "Technical Analysis", tech, tech
            ? U.esc(tech.message) + (tech.data && tech.data.signal ? statlinesHTMLSafe([["Signal", tech.data.signal, U.classForDecision(tech.data.signal)]]) : "")
            : "No technical report in the live buffer for this decision window.", true) +
          C.reasonHTML("news", "News Analysis", news, news ? U.esc(news.message) : "No news report in the live buffer for this decision window.") +
          C.reasonHTML("db", "Fundamental Analysis", fund, fund ? U.esc(fund.message) : "No fundamentals report in the live buffer for this decision window.") +
          C.reasonHTML("trend-up", "Bull Case", debate ? { data: null, agent: "x" } : null, debate && debate.data ? '<div class="pos" style="font-weight:600;margin-bottom:4px">Strength ' + Math.round((debate.data.bull_strength || 0) * 100) + "%</div>" + U.esc(debate.data.bull_summary || "—") : "No debate record for this decision window.") +
          C.reasonHTML("trend-down", "Bear Case", debate ? { data: null, agent: "x" } : null, debate && debate.data ? '<div class="neg" style="font-weight:600;margin-bottom:4px">Strength ' + Math.round((debate.data.bear_strength || 0) * 100) + "%</div>" + U.esc(debate.data.bear_summary || "—") : "No debate record for this decision window.") +
          C.reasonHTML("shield", "Risk Assessment", risk, risk ? U.esc(risk.message) + (risk.data ? statlinesHTMLSafe([["Risk level", U.esc(risk.data.risk_level || "—")], ["Max notional", risk.data.max_notional_usd ? U.fmtMoney(risk.data.max_notional_usd, { dec: 0 }) : "—"]]) : "") : "No risk report in the live buffer for this decision window.") +
          C.reasonHTML("gavel", "Final CIO Reasoning", log, U.esc(d.reasoning || log.message), true) +
          C.reasonHTML("zap", "Execution Result", exec, exec ? U.esc(exec.message) + (exec.data && exec.data.order_id ? statlinesHTMLSafe([["Order ID", U.esc(exec.data.order_id)]]) : "") : d.notional_usd > 0 ? "The CIO recommended a trade but no execution record is present in the live buffer." : "No order was required for this decision (HOLD).") +
        "</div>" +
      "</div>" +
      // ------- memory
      (memory
        ? '<div class="dsec">' + C.sectionTitle("memory", "Memory / Learning Loop") +
            '<div class="note">' + ICON("memory") + "<span>" + U.esc(memory.message) + "</span></div>" +
          "</div>"
        : "") +
      '<div class="note" style="margin-top:4px">' + ICON("info") + "<span>Reasoning is rendered from the agent pipeline. Raw JSON payloads are available per event in the activity log.</span></div>",
      null,
      { label: "AI decision " + (log.symbol || "") }
    );

    function statlinesHTMLSafe(rows) { return '<div style="margin-top:8px">' + C.statlinesHTML(rows) + "</div>"; }
  }

  /* ==========================================================================
     Cycle drawer
     ========================================================================== */

  function openCycleDrawer(id) {
    var cycles = (store.data.cycles && store.data.cycles.cycles) || [];
    var c = null;
    for (var i = 0; i < cycles.length; i++) if (String(cycles[i].id) === String(id)) c = cycles[i];
    if (!c) { App.toast("Cycle not found in history", "info"); return; }

    // Find logs inside the cycle window to reconstruct the pipeline
    var a = c.started_at && U.parseDate(c.started_at).getTime();
    var b = c.finished_at && U.parseDate(c.finished_at).getTime();
    var logs = ((store.data.logs && store.data.logs.logs) || []).filter(function (l) {
      if (!a || !b) return false;
      var t = U.parseDate(l.timestamp).getTime();
      return t >= a - 30000 && t <= b + 30000;
    });

    function agentStatusMatrix(c) {
      var as = c.agent_status;
      if (!as || !Object.keys(as).length) return "";
      var symbols = Object.keys(as);
      var stageNames = {
        market_data: "Market Data", technical: "Technical", news: "News",
        fundamentals: "Fundamentals", debate: "Debate", risk: "Risk",
        cio: "CIO", execution: "Execution", memory: "Memory",
      };
      var stages = Object.keys(stageNames);
      function cell(status) {
        if (status === "OK") return '<span class="badge badge-ok">' + ICON("check") + "OK</span>";
        if (status === "ERROR") return '<span class="badge badge-error">' + ICON("x") + "ERROR</span>";
        if (status === "UNAVAILABLE") return '<span class="badge badge-warning">' + ICON("alert") + "N/A</span>";
        return '<span class="badge badge-neutral">—</span>';
      }
      var head = "<tr><th>Symbol</th>" + stages.map(function (s) { return "<th class='c'>" + stageNames[s] + "</th>"; }).join("") + "</tr>";
      var rows = symbols.map(function (sym) {
        return "<tr><td><span class='sym'>" + U.esc(sym) + "</span></td>" +
          stages.map(function (s) {
            var st = (as[sym] || {})[s];
            return "<td class='c'>" + cell(st || "SKIPPED") + "</td>";
          }).join("") + "</tr>";
      }).join("");
      return (
        '<div class="dsec">' + C.sectionTitle("cpu", "Agent Execution (per symbol)") +
          '<div class="tbl-wrap"><table class="tbl"><thead>' + head + "</thead><tbody>" + rows + "</tbody></table></div>" +
          '<div class="note" style="margin-top:8px">' + ICON("info") + "<span>OK = stage ran cleanly · N/A = data source or LLM provider unavailable · — = stage not required or disabled · ERROR = stage failed. A cycle is only OK when every enabled stage succeeded.</span></div>" +
        "</div>"
      );
    }

    function agentResultsLine(c) {
      var ar = c.agent_results;
      if (!ar) return "";
      var names = {
        market_data: "Market Data", technical: "Technical", news: "News",
        fundamentals: "Fundamentals", debate: "Debate", risk: "Risk",
        cio: "CIO", execution: "Execution", memory: "Memory",
      };
      var chips = Object.keys(names).map(function (k) {
        var d = ar[k];
        if (!d) return "";
        var cls = d.ok === d.total ? "badge-ok"
          : d.ok > 0 ? "badge-warning" : (d.skipped === d.total ? "badge-neutral" : "badge-error");
        return '<span class="badge ' + cls + '">' + names[k] + " " + d.ok + "/" + d.total + "</span>";
      }).join(" ");
      if (!chips) return "";
      return (
        '<div class="dsec">' + C.sectionTitle("gauge", "Agent Results (ok / attempted)") +
          '<div class="row" style="gap:6px;flex-wrap:wrap">' + chips + "</div>" +
        "</div>"
      );
    }

    function llmUsageSection(c) {
      var u = c.llm_usage;
      if (!u || !Object.keys(u).length) return "";
      var provNames = { groq: "Groq", openrouter: "OpenRouter", gemini: "Gemini" };
      var agentNames = { technical: "Technical", debate: "Debate", cio: "CIO", risk: "Risk", news: "News", fundamentals: "Fundamentals" };
      var rows = Object.keys(u).map(function (p) {
        var d = u[p];
        var agents = Object.keys(d.by_agent || {}).map(function (a) {
          var ad = d.by_agent[a];
          return U.esc(agentNames[a] || a) + " " + ad.ok + "/" + ad.requests;
        }).join(", ");
        var errs = Object.keys(d.errors || {}).map(function (s) { return U.esc(s); }).join(", ");
        return (
          "<tr><td><span class='sym'>" + U.esc(provNames[p] || p) + "</span></td>" +
          '<td class="r">' + d.requests + "</td>" +
          '<td class="r" style="color:var(--green)">' + d.ok + "</td>" +
          (errs
            ? '<td class="r" style="color:var(--red)">' + (d.requests - d.ok) + "</td>"
            : '<td class="r" style="color:var(--green)">0</td>') +
          "<td>" + (agents || "\u2014") + "</td>" +
          "<td>" + (errs ? '<span style="color:var(--red)">' + errs + "</span>" : '<span class="pos">clean</span>') + "</td>" +
          "</tr>"
        );
      }).join("");
      return (
        '<div class="dsec">' + C.sectionTitle("cpu", "LLM Calls (per provider)") +
          '<div class="tbl-wrap"><table class="tbl"><thead><tr><th>Provider</th><th class="r">Requests</th><th class="r">OK</th><th class="r">Failed</th><th>Agents (ok/req)</th><th>Errors</th></tr></thead><tbody>' + rows + "</tbody></table></div>" +
          (c.provider_results && c.provider_results.llm_states
            ? '<div class="row" style="gap:6px;flex-wrap:wrap;margin-top:8px">' + Object.keys(c.provider_results.llm_states).map(function (p) {
                var s = c.provider_results.llm_states[p] || {};
                var cls = s.state === "READY" ? "badge-ok" : s.state === "NOT_CONFIGURED" ? "badge-neutral" : s.state === "DEGRADED" ? "badge-warning" : "badge-error";
                return '<span class="badge ' + cls + '">' + p + ": " + U.esc(s.state) + "</span>";
              }).join(" ") + "</div>"
            : "") +
          '<div class="note" style="margin-top:8px">' + ICON("info") + "<span>Live LLM requests this cycle. Re-used analyses (unchanged headlines/fundamentals) are not re-sent; quota-blocked requests appear as errors. Provider circuits are shown below the table.</span>" +
        "</div></div>"
      );
    }

    var stages = [
      { name: "Market Data", icon: "globe", ok: c.status !== "ERROR" && logs.length > 0 },
      { name: "Technical Agent", icon: "chart-candle", ok: has("technical") },
      { name: "News Agent", icon: "news", ok: has("news") },
      { name: "Fundamentals", icon: "db", ok: has("fundamentals") },
      { name: "Debate", icon: "scale", ok: has("debate") },
      { name: "Risk", icon: "shield", ok: has("risk") },
      { name: "CIO", icon: "gavel", ok: has("cio") },
      { name: "Execution", icon: "zap", ok: (c.orders || []).length > 0 },
      { name: "Memory", icon: "memory", ok: has("memory") },
    ];
    function has(agent) { return logs.some(function (l) { return l.agent === agent; }) || (agent === "execution" && (c.orders || []).length > 0); }

    var disabled = C.disabledAgentSet();
    var cfg = store.data.config || {};

    var decisionRows = (c.decisions || []).map(function (d) {
      var qBadge = { SUFFICIENT: "badge-ok", DEGRADED: "badge-warning", INSUFFICIENT: "badge-error" };
      return (
        '<tr class="rowlink" data-action="open-decision" data-ts="' + findCioTs(d.symbol) + '" data-agent="cio" tabindex="0">' +
          '<td><span class="sym">' + U.esc(d.symbol) + "</span></td>" +
          "<td>" + '<span class="' + U.badgeForDecision(d.decision) + '">' + U.esc(d.decision) + "</span>" +
            (d.blocked_reason ? ' <span class="badge badge-warning" title="' + U.esc(d.blocked_reason) + '">' + U.esc(d.blocked_reason) + "</span>" : "") + "</td>" +
          '<td class="r">' + (d.confidence != null ? Math.round(d.confidence * 100) + "%" : "—") + "</td>" +
          '<td class="r">' + (d.notional_usd ? U.fmtMoney(d.notional_usd, { dec: 0 }) : "—") + "</td>" +
          '<td>' + (d.evidence_quality ? '<span class="badge ' + (qBadge[d.evidence_quality] || "") + '">' + U.esc(d.evidence_quality) + "</span>" : '<span class="t-faint">—</span>') + "</td>" +
        "</tr>"
      );
    }).join("");

    function findCioTs(sym) {
      var logs = (store.data.logs && store.data.logs.logs) || [];
      for (var i = 0; i < logs.length; i++) {
        var l = logs[i];
        if (l.agent === "cio" && l.symbol === sym && l.timestamp <= (c.finished_at || c.started_at) && l.timestamp >= c.started_at) return l.timestamp;
      }
      return "";
    }

    var orderRows = (c.orders || []).map(function (o) {
      return (
        '<div class="check ' + (o.success ? "ok" : "err") + '">' +
          '<div class="c-ico">' + ICON(o.success ? "check" : "x") + "</div>" +
          '<div class="c-body">' + U.esc(String(o.side || "").toUpperCase()) + " " + U.esc(o.symbol) +
            (o.notional_usd ? " · " + U.fmtMoney(o.notional_usd, { dec: 0 }) : "") + (o.qty ? " · " + U.fmtQty(o.qty) + " sh" : "") +
            '<div class="c-sub">' + (o.order_id ? "Order " + U.esc(o.order_id) + " · " : "") + (o.success ? U.esc(o.status || "submitted") : "Error: " + U.esc(o.error || "unknown")) + "</div>" +
          "</div>" +
        "</div>"
      );
    }).join("");

    var body =
      '<div class="dsec">' + C.sectionTitle("repeat", "Cycle") +
        C.kvHTML([
          ["Cycle", "#" + c.id],
          ["Status", '<span class="' + U.badgeForCycle(c.status) + '">' + U.esc(String(c.status || "—").replace("_", " ")) + "</span>"],
          ["Started", U.fmtDateTime(c.started_at)],
          ["Duration", c.duration_s != null ? c.duration_s + "s" : "—"],
          ["Trigger", U.esc((c.triggered_by || "scheduler").toUpperCase())],
          ["Symbols", (c.symbols_processed || []).length],
        ]) +
      "</div>" +
      '<div class="dsec">' + C.sectionTitle("activity", "Pipeline") +
        stages.map(function (s) {
          var off = (s.name === "Fundamentals" && disabled.fundamentals) || (s.name === "Debate" && disabled.debate) || (s.name === "Memory" && disabled.memory);
          return (
            '<div class="row" style="justify-content:space-between;padding:5px 0;border-bottom:1px dashed var(--line)">' +
              "<span class='row'>" + ICON(s.icon) + '<span style="font-size:12px;color:var(--ink-hi)">' + s.name + "</span></span>" +
              (off
                ? '<span class="badge badge-neutral">DISABLED</span>'
                : s.ok
                  ? '<span class="row" style="color:var(--green);font-size:11px;font-weight:600">' + ICON("check") + "RAN</span>"
                  : c.status === "ERROR"
                    ? '<span class="row" style="color:var(--red);font-size:11px;font-weight:600">' + ICON("x") + "FAILED</span>"
                    : '<span class="t-faint" style="font-size:11px">Not recorded</span>') +
            "</div>"
          );
        }).join("") +
      "</div>" +
      agentStatusMatrix(c) +
      agentResultsLine(c) +
      llmUsageSection(c) +
      (decisionRows
        ? '<div class="dsec">' + C.sectionTitle("gavel", "Decisions") +
            '<div class="tbl-wrap"><table class="tbl"><thead><tr><th>Symbol</th><th>Decision</th><th class="r">Confidence</th><th class="r">Notional</th><th>Evidence</th></tr></thead><tbody>' + decisionRows + "</tbody></table></div>" +
          '<div class="note" style="margin-top:8px">' + ICON("info") + "<span>Evidence quality: SUFFICIENT (all analyst evidence available) · DEGRADED (some evidence missing — CIO informed) · INSUFFICIENT (critical evidence missing — BUY blocked; a risk-cap limit alone never justifies a trade).</span></div>" +
          "</div>"
        : "") +
      (orderRows
        ? '<div class="dsec">' + C.sectionTitle("zap", "Orders") + orderRows + "</div>"
        : "") +
      ((c.warnings || 0) > 0 || (c.errors || []).length
        ? '<div class="dsec">' + C.sectionTitle("alert", "Warnings & Errors") +
            (c.errors || []).map(function (e) {
              // Structured errors: {provider, type, agent, symbol, message}
              if (e && typeof e === "object") {
                var chips =
                  '<span class="badge badge-error">' + U.esc(e.type || "ERROR") + "</span> " +
                  '<span class="t-faint">' + U.esc(e.provider || "?") +
                  (e.agent ? " · " + U.esc(e.agent) : "") +
                  (e.symbol ? " · " + U.esc(e.symbol) : "") + "</span>";
                return '<div class="check err"><div class="c-ico">' + ICON("x") + '</div><div class="c-body">' +
                  chips + "<div>" + U.esc(e.message || "") + "</div></div></div>";
              }
              return '<div class="check err"><div class="c-ico">' + ICON("x") + '</div><div class="c-body">' + U.esc(e) + "</div></div>";
            }).join("") +
            ((c.warnings || 0) > 0 && !(c.errors || []).some(function (e) { return (typeof e === "string" ? e : (e && e.message) || "").indexOf("Warning") >= 0; })
              ? '<div class="check warn"><div class="c-ico">' + ICON("alert") + '</div><div class="c-body">' + c.warnings + " warning" + (c.warnings > 1 ? "s" : "") + " raised during this cycle</div></div>"
              : "") +
          "</div>"
        : "") +
      (logs.length
        ? '<div class="dsec">' + C.sectionTitle("scroll", "Cycle Events") +
            logs.slice(0, 14).map(function (l) { return C.feedItemHTML(l); }).join("") +
          "</div>"
        : "");

    App.openDrawer(
      "Cycle #" + c.id + ' <span class="' + U.badgeForCycle(c.status) + '">' + U.esc(String(c.status || "").replace("_", " ")) + "</span>",
      U.fmtDateTime(c.started_at) + " · " + (c.duration_s != null ? c.duration_s + "s · " : "") + U.esc((c.triggered_by || "").toUpperCase()),
      body,
      null,
      { label: "Cycle " + c.id }
    );
  }

  /* ==========================================================================
     Agent drawer
     ========================================================================== */

  var AGENT_ROLES = {
    technical: "Interprets RSI, 50/200-day moving averages and MACD into a BULLISH / BEARISH / NEUTRAL signal (Groq; model set via GROQ_TECH_MODEL).",
    news: "Scans recent headlines via the Alpaca News API and scores sentiment (Gemini; model set via GEMINI_MODEL).",
    fundamentals: "Reads valuation, growth, margins and balance-sheet health from yfinance data (Gemini; model set via GEMINI_MODEL).",
    debate: "Two opposing researchers argue the strongest bull and bear cases from the same data in one structured call (Groq; model set via GROQ_DEBATE_MODEL).",
    risk: "Hard constraint layer: approves/blocks trades and caps notional by exposure and concentration rules (OpenRouter; model set via OPENROUTER_RISK_MODEL).",
    cio: "Chief Investment Officer: weighs every agent report, the debate edge, agent accuracy weights and memory, then issues the final BUY / SELL / HOLD (Groq; model set via GROQ_CIO_MODEL).",
    execution: "Submits paper market orders to Alpaca when the CIO issues an actionable decision.",
    memory: "SQLite-backed decision log; computes realized P&L on closed trades and feeds agent accuracy weights back into the CIO.",
    system: "Bot process, scheduler and configuration events.",
    market: "Market data ingestion: bars, indicators and headlines from Alpaca.",
  };

  function openAgentDrawer(agent) {
    var logs = (store.data.logs && store.data.logs.logs) || [];
    var mine = logs.filter(function (l) { return l.agent === agent; });
    var acc = store.data.accuracy && store.data.accuracy.agents ? store.data.accuracy.agents[agent] : null;
    var errors = mine.filter(function (l) { return l.level === "ERROR"; }).length;
    var disabled = C.disabledAgentSet()[agent];

    var body =
      '<div class="dsec">' + C.sectionTitle("cpu", "Agent") +
        C.kvHTML([
          ["Role", '<span style="font-family:var(--font-ui)">' + U.esc(AGENT_ROLES[agent] || "—") + "</span>"],
          ["Status", disabled ? '<span class="badge badge-neutral">DISABLED</span>' : errors ? '<span class="badge badge-warning">RECENT ERRORS</span>' : mine.length ? '<span class="badge badge-ok">ACTIVE</span>' : '<span class="t-faint">No recent events</span>'],
          acc ? ["Hit rate (rolling " + ((store.data.config || {}).agent_accuracy_lookback || 20) + ")", acc.hit_rate != null ? (acc.hit_rate * 100).toFixed(1) + "%" : "No data"] : ["Learning loop", "Not scored for this agent"],
          acc ? ["Sample size", acc.sample_size + " closed calls"] : null,
          acc ? ["Current weight", acc.weight.toFixed(2) + "×"] : null,
        ].filter(Boolean)) +
      "</div>" +
      '<div class="dsec">' + C.sectionTitle("scroll", "Recent Events") +
        (mine.length
          ? '<div class="card" style="box-shadow:none">' + mine.slice(0, 12).map(function (l) { return C.feedItemHTML(l); }).join("") + "</div>"
          : C.stateHTML({ icon: "activity", title: "No recent events", msg: "This agent has not logged anything in the current buffer." })) +
      "</div>";

    App.openDrawer(
      U.agentLabel(agent),
      "Agent detail",
      body,
      '<button class="btn btn-sm" data-action="goto" data-page="activity">' + ICON("activity") + "All activity</button>",
      { label: "Agent " + agent }
    );
  }

  /* ==========================================================================
     Generic log drawer (from feeds)
     ========================================================================== */

  function openLogDrawer(ts) {
    var logs = (store.data.logs && store.data.logs.logs) || [];
    var log = null;
    logs.forEach(function (l) { if (l.timestamp === ts) log = l; });
    if (!log) return;

    if (log.agent === "cio" && log.data) return openDecisionDrawer(ts, "cio");

    var d = log.data || {};
    var kvRows = [];
    if (log.symbol) kvRows.push(["Symbol", U.esc(log.symbol)]);
    kvRows.push(["Agent", U.agentLabel(log.agent)]);
    kvRows.push(["Level", U.esc(log.level)]);
    kvRows.push(["Time", U.fmtDateTime(log.timestamp)]);
    if (d.decision) kvRows.push(["Decision", U.esc(d.decision)]);
    if (d.confidence != null) kvRows.push(["Confidence", Math.round(d.confidence * 100) + "%"]);
    if (d.signal) kvRows.push(["Signal", U.esc(d.signal)]);
    if (d.sentiment) kvRows.push(["Sentiment", U.esc(d.sentiment)]);
    if (d.approved != null) kvRows.push(["Risk verdict", d.approved ? "APPROVED" : "REJECTED"]);
    if (d.risk_level) kvRows.push(["Risk level", U.esc(d.risk_level)]);
    if (d.max_notional_usd != null) kvRows.push(["Max notional", U.fmtMoney(d.max_notional_usd, { dec: 0 })]);
    if (d.notional_usd != null) kvRows.push(["Notional", U.fmtMoney(d.notional_usd, { dec: 0 })]);
    if (d.order_id) kvRows.push(["Order ID", U.esc(d.order_id)]);
    if (d.qty != null) kvRows.push(["Quantity", U.fmtQty(d.qty)]);
    if (d.success != null) kvRows.push(["Result", d.success ? "SUBMITTED" : "FAILED"]);
    if (d.bull_strength != null) kvRows.push(["Bull / Bear", Math.round(d.bull_strength * 100) + "% / " + Math.round(d.bear_strength * 100) + "%"]);
    if (d.edge != null) kvRows.push(["Debate edge", (d.edge >= 0 ? "+" : "") + d.edge]);
    if (d.realized_pl_pct != null) kvRows.push(["Realized P&L", U.fmtPct(d.realized_pl_pct)]);
    if (d.error) kvRows.push(["Error", U.esc(d.error)]);

    var isOrder = log.agent === "execution" && d.order_id;

    App.openDrawer(
      U.agentLabel(log.agent) + (log.symbol ? ' <span class="sym">' + U.esc(log.symbol) + "</span>" : ""),
      "Event · " + U.fmtDateTime(log.timestamp),
      '<div class="dsec">' + C.sectionTitle("scroll", "Event") +
        '<div class="reason" data-open="true"><div class="r-bd" style="display:block;padding:2px 0 0;color:var(--ink);font-size:12.5px;line-height:1.6">' + U.esc(log.message) + "</div></div>" +
      "</div>" +
      (kvRows.length ? '<div class="dsec">' + C.sectionTitle("info", "Details") + C.kvHTML(kvRows) + "</div>" : "") +
      (isOrder
        ? '<button class="btn btn-sm" data-action="open-order" data-id="' + U.esc(d.order_id) + '">' + ICON("list-ordered") + "Open order detail</button>"
        : ""),
      null,
      { label: "Event detail" }
    );
  }

  /* export */
  App.openPositionDrawer = openPositionDrawer;
  App.openOrderDrawer = openOrderDrawer;
  App.openDecisionDrawer = openDecisionDrawer;
  App.openCycleDrawer = openCycleDrawer;
  App.openAgentDrawer = openAgentDrawer;
  App.openLogDrawer = openLogDrawer;
})();
