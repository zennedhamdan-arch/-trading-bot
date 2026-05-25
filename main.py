import os
import logging
from flask import Flask, request, jsonify
import alpaca_trade_api as tradeapi
from groq import Groq

# ── Logging ───────────────────────────────────────────────────────────────────
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)

app = Flask(__name__)

# ── Clients (loaded from Railway environment variables) ───────────────────────
ALPACA_KEY    = os.environ["ALPACA_KEY"]
ALPACA_SECRET = os.environ["ALPACA_SECRET"]
ALPACA_URL    = os.environ.get("ALPACA_URL", "https://paper-api.alpaca.markets")
GROQ_KEY      = os.environ["GROQ_KEY"]

alpaca = tradeapi.REST(ALPACA_KEY, ALPACA_SECRET, ALPACA_URL, api_version="v2")
groq   = Groq(api_key=GROQ_KEY)

# ── Risk limits ───────────────────────────────────────────────────────────────
MAX_QTY          = 10   # max shares/units per order
MAX_DAILY_TRADES = 5    # stop after this many trades per day
daily_trade_count = 0

# ── Health check ──────────────────────────────────────────────────────────────
@app.route("/", methods=["GET"])
def health():
    return jsonify({"status": "AI Trading Bot is running ✓", "ai": "Groq Llama 3.3 70B (free)", "mode": "paper trading"})

# ── Webhook endpoint ──────────────────────────────────────────────────────────
@app.route("/webhook", methods=["POST"])
def webhook():
    global daily_trade_count

    # 1. Parse incoming alert from TradingView
    try:
        data = request.get_json(force=True)
    except Exception:
        log.warning("Bad JSON received")
        return jsonify({"status": "error", "reason": "invalid JSON"}), 400

    symbol = str(data.get("symbol", "")).upper().strip()
    action = str(data.get("action", "")).upper().strip()
    qty    = int(data.get("qty", 1))

    log.info(f"Signal received: {action} {qty} x {symbol}")

    # 2. Basic validation
    if not symbol or action not in ("BUY", "SELL"):
        return jsonify({"status": "ignored", "reason": "missing symbol or invalid action"}), 200

    if qty > MAX_QTY:
        return jsonify({"status": "rejected", "reason": f"qty {qty} exceeds max allowed {MAX_QTY}"}), 200

    if daily_trade_count >= MAX_DAILY_TRADES:
        return jsonify({"status": "rejected", "reason": "daily trade limit reached — resuming tomorrow"}), 200

    # 3. Ask Groq AI (Llama 3.3 70B) to confirm the trade
    try:
        account = alpaca.get_account()
        cash    = float(account.cash)
        equity  = float(account.portfolio_value)

        prompt = (
            f"Trade signal:\n"
            f"  Action : {action}\n"
            f"  Symbol : {symbol}\n"
            f"  Qty    : {qty} shares\n"
            f"  Cash   : ${cash:.2f}\n"
            f"  Equity : ${equity:.2f}\n\n"
            f"You are a strict risk manager for a beginner with under $1,000.\n"
            f"Rules: max 2% risk per trade, reject if position size is too large for the account, "
            f"reject if this looks like a revenge trade or overtrading.\n"
            f"Reply with APPROVE or REJECT followed by one short reason. Max 15 words total."
        )

        response = groq.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[
                {"role": "system", "content": "You are a professional forex and stock risk manager. Be strict and brief."},
                {"role": "user",   "content": prompt}
            ],
            max_tokens=50,
            temperature=0.1   # low temperature = consistent, reliable decisions
        )
        decision = response.choices[0].message.content.strip()
        log.info(f"Groq AI decision: {decision}")

    except Exception as e:
        log.error(f"Groq AI error: {e}")
        return jsonify({"status": "error", "reason": "AI check failed — order blocked for safety"}), 500

    # 4. Execute only if approved
    if decision.upper().startswith("APPROVE"):
        try:
            side  = "buy" if action == "BUY" else "sell"
            order = alpaca.submit_order(
                symbol=symbol,
                qty=qty,
                side=side,
                type="market",
                time_in_force="gtc"
            )
            daily_trade_count += 1
            log.info(f"Order placed: {order.id}")
            return jsonify({
                "status"   : "order placed ✓",
                "order_id" : order.id,
                "action"   : action,
                "symbol"   : symbol,
                "qty"      : qty,
                "ai_engine": "Groq Llama 3.3 70B",
                "ai_reason": decision,
                "trades_today": daily_trade_count
            }), 200

        except Exception as e:
            log.error(f"Alpaca order error: {e}")
            return jsonify({"status": "error", "reason": str(e)}), 500

    # 5. Rejected by AI
    return jsonify({
        "status"   : "rejected by AI",
        "action"   : action,
        "symbol"   : symbol,
        "ai_engine": "Groq Llama 3.3 70B",
        "ai_reason": decision
    }), 200


# ── Account status endpoint ───────────────────────────────────────────────────
@app.route("/status", methods=["GET"])
def status():
    try:
        account = alpaca.get_account()
        orders  = alpaca.list_orders(status="all", limit=5)
        return jsonify({
            "account_status" : account.status,
            "cash"           : float(account.cash),
            "portfolio_value": float(account.portfolio_value),
            "trades_today"   : daily_trade_count,
            "ai_engine"      : "Groq Llama 3.3 70B (FREE)",
            "recent_orders"  : [
                {"id": o.id, "symbol": o.symbol, "side": o.side,
                 "qty": o.qty, "status": o.status}
                for o in orders
            ]
        })
    except Exception as e:
        return jsonify({"status": "error", "reason": str(e)}), 500


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8080))
    app.run(host="0.0.0.0", port=port)
