/**
 * tests/ui_regression.js — jsdom UI regression for the trading-bot dashboard.
 * Keeps a copy in the repo because /tmp is wiped between sessions.
 *
 * Usage:
 *   cd tests && npm install jsdom   (or install jsdom anywhere reachable)
 *   node ui_regression.js
 *
 * Verifies: all 11 pages render in demo mode; the Overview live-market strip;
 * the Health page startup-health table + LLM circuit table + market/realtime
 * cards; the cycle drawer's agent-execution matrix, agent-results chips, LLM
 * calls table and provider-state chips; read-only config.
 */
const fs = require("fs");
const path = require("path");
const { JSDOM } = require("jsdom");

const ROOT = process.env.REPO_ROOT
  || (fs.existsSync(path.join(__dirname, "..", "static", "index.html"))
      ? path.join(__dirname, "..")
      : "/home/user/-trading-bot");
const html = fs.readFileSync(path.join(ROOT, "static/index.html"), "utf8");

function boot(url) {
  const dom = new JSDOM(html, { url, runScripts: "outside-only", pretendToBeVisual: true });
  const w = dom.window;
  w.matchMedia = w.matchMedia || (() => ({ matches: false, addListener() {}, removeListener() {} }));
  w.requestAnimationFrame = (cb) => setTimeout(cb, 0);
  w.ResizeObserver = w.ResizeObserver || class { observe() {} unobserve() {} disconnect() {} };
  w.scrollTo = () => {};
  for (const b of ["icons.js", "api.js", "app.js", "components.js", "drawers.js", "pages.js"]) {
    w.eval(fs.readFileSync(path.join(ROOT, "static/js", b), "utf8"));
  }
  w.App.loadDemo();
  w.eval(`
    (function () {
      App.store.ui.page = "overview";
      App.renderPage("overview", {});
    })();
  `);
  return w;
}

let failures = [];
function check(name, cond) {
  if (cond) console.log("  ok  " + name);
  else { failures.push(name); console.log("  FAIL " + name); }
}

const w = boot("http://localhost:8000/?demo=1#/overview");
const doc = w.document;

check("demo mode active", w.App.store.demo === true);
check("boot rendered overview into #page", doc.querySelector("#page").innerHTML.length > 200);

// --- all 11 pages -----------------------------------------------------------
const pages = ["overview", "portfolio", "positions", "orders", "ai", "agents", "risk", "cycles", "health", "activity", "config"];
for (const p of pages) {
  let ok = false;
  try {
    w.App.renderPage(p, {});
    ok = doc.querySelector("#page").innerHTML.length > 200;
  } catch (e) {
    console.log("      render error (" + p + "):", e.message);
  }
  check(`page renders: ${p}`, ok);
}

// --- overview: live market strip --------------------------------------------
w.App.renderPage("overview", {});
const overviewHTML = doc.querySelector("#page").innerHTML;
check("overview: live market strip present", /Live Market/.test(overviewHTML));
check("overview: live prices rendered (AAPL)", /175\.25/.test(overviewHTML));
check("overview: ticks-never-trigger-AI note", /never trigger AI/.test(overviewHTML));

// --- health page: startup table + circuits + cards ---------------------------
w.App.renderPage("health", {});
const healthHTML = doc.querySelector("#page").innerHTML;
check("health: startup health table", /Startup Health Check/.test(healthHTML));
check("health: ALPACA row READY", /ALPACA<\/span><\/td><td>.*READY/.test(healthHTML.replace(/\s+/g, " ")) || /ALPACA/.test(healthHTML) && /READY/.test(healthHTML));
check("health: FUNDAMENTALS DATA_UNAVAILABLE", /FUNDAMENTALS/.test(healthHTML) && /DATA_UNAVAILABLE/.test(healthHTML));
check("health: NVIDIA + OPENROUTER NOT_CONFIGURED", /NVIDIA/.test(healthHTML) && /OPENROUTER/.test(healthHTML) && /NOT_CONFIGURED/.test(healthHTML));
check("health: LLM circuit table", /LLM Provider Circuits/.test(healthHTML));
check("health: circuit states rendered (READY)", /badge-ok">\s*READY|READY<\/span>/.test(healthHTML.replace(/\s+/g, " ")));
check("health: market card (OPEN/CLOSED)", /Market/.test(healthHTML) && /OPEN|CLOSED|UNKNOWN/.test(healthHTML));
check("health: realtime feed card", /Realtime Feed/.test(healthHTML));
check("health: data feed card names IEX", /Data Feed/.test(healthHTML) && /IEX/.test(healthHTML));
check("health: no secrets in output", !/test-secret|sk-|Bearer /.test(healthHTML));

// --- cycle drawer: matrix + agent results + LLM table + provider states ------
w.App.renderPage("cycles", {});
w.App.openCycleDrawer(1842);
const drawerHTML = doc.querySelector("#drawer-root").innerHTML;
check("cycle drawer opens", drawerHTML.length > 500);
check("drawer: agent execution matrix", /Agent Execution \(per symbol\)/.test(drawerHTML));
check("drawer: agent results chips (ok/attempted)", /Agent Results \(ok \/ attempted\)/.test(drawerHTML));
check("drawer: agent chip values (Technical 5/5)", /Technical 5\/5/.test(drawerHTML));
check("drawer: fundamentals chip 0/5 (no provider)", /Fundamentals 0\/5/.test(drawerHTML));
check("drawer: LLM calls table", /LLM Calls \(per provider\)/.test(drawerHTML));
check("drawer: provider state chips", /groq: READY/.test(drawerHTML) && /nvidia: NOT_CONFIGURED/.test(drawerHTML));
check("drawer: structured errors shown for PARTIAL cycles",
      (function () {
        w.App.openCycleDrawer(1841); // PARTIAL_ERROR fixture
        const d = doc.querySelector("#drawer-root").innerHTML;
        return /Warnings &amp; Errors|Warnings & Errors/.test(d);
      })());

// --- config page still read-only ---------------------------------------------
w.App.renderPage("config", {});
const configHTML = doc.querySelector("#page").innerHTML;
check("config page renders (read-only)", configHTML.length > 200 && !/<input[^>]*value=/.test(configHTML));

// --- static source assertions -------------------------------------------------
const drawersSrc = fs.readFileSync(path.join(ROOT, "static/js/drawers.js"), "utf8");
check("drawers render agentResultsLine + matrix + llmUsage",
  /agentResultsLine\(c\)/.test(drawersSrc) && /agentStatusMatrix\(c\)/.test(drawersSrc) && /llmUsageSection\(c\)/.test(drawersSrc));
const appSrc = fs.readFileSync(path.join(ROOT, "static/js/app.js"), "utf8");
check("app polls /api/health and /api/realtime",
  /API\.health/.test(appSrc) && /API\.realtime/.test(appSrc));
check("updateLiveStrip exists (in-place tick refresh)", /updateLiveStrip/.test(appSrc));
const apiSrc = fs.readFileSync(path.join(ROOT, "static/js/api.js"), "utf8");
check("api.js exposes health + realtime", /\/api\/health/.test(apiSrc) && /\/api\/realtime/.test(apiSrc));

console.log();
if (failures.length) {
  console.log(failures.length + " FAILURE(S):");
  failures.forEach((f) => console.log("  - " + f));
  process.exit(1);
}
console.log("UI REGRESSION PASSED — 11 pages, live strip, health tables, cycle drawer");
