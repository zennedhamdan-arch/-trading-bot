import os
import logging
from flask import Flask, request, jsonify
from alpaca.trading.client import TradingClient
from alpaca.trading.requests import MarketOrderRequest
from alpaca.trading.enums import OrderSide, TimeInForce
from groq import Groq

# ── Logging ───────────────────────────────────────────────────────────────────
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)

app = Flask(__name__)

# ── Clients ───────────────────────────────────────────────────────────────────
ALPACA_KEY    = os.environ["ALPACA_KEY"]
ALPACA_SECRET = os.environ["ALPACA_SECRET"]
GROQ_KEY      = os.environ["GROQ_KEY"]

# paper=True means paper trading — no real money
alpaca = TradingClient(ALPACA_KEY, ALPACA_SECRET, paper=True)
groq   = Groq(api_key=GROQ_KEY)

# ── Risk limits ───────────────────────────────────────────────────────────────
MAX_QTY           = 10
MAX_DAILY_TRADES  = 5
daily_trade_count = 0

# ── Health check ──────────────────────────────────────────────────────────────
@app.route("/", methods=["GET"])
def health():
    return jsonify({
        "status" : "AI Trading Bot is running",
        "ai"     : "Groq Llama 3.3 70B (free)",
        "mode"   : "paper trading — no real money"
    })

# ── Webhook endpoint ──────────────────────────────────────────────────────────
@app.route("/webhook", methods=["POST"])
def webhook():
    global daily_trade_count

    # 1. Parse TradingView alert
    try:
        data = request.get_json(force=True)
    except Exception:
        return jsonify({"status": "error", "reason": "invalid JSON"}), 400

    symbol = str(data.get("symbol", "")).upper().strip()
    action = str(data.get("action", "")).upper().strip()
    qty    = int(data.get("qty", 1))

    log.info(f"Signal: {action} {qty} x {symbol}")

    # 2. Validate
    if not symbol or action not in ("BUY", "SELL"):
        return jsonify({"status": "ignored", "reason": "invalid symbol or action"}), 200
    if qty > MAX_QTY:
        return jsonify({"status": "rejected", "reason": f"qty {qty} exceeds max {MAX_QTY}"}), 200
    if daily_trade_count >= MAX_DAILY_TRADES:
        return jsonify({"status": "rejected", "reason": "daily trade limit reached"}), 200

    # 3. Get account info
    try:
        account = alpaca.get_account()
        cash    = float(account.cash)
        equity  = float(account.equity)
    except Exception as e:
        return jsonify({"status": "error", "reason": f"Alpaca account error: {e}"}), 500

    # 4. Ask Groq AI to confirm
    try:
        prompt = (
            f"Trade signal:\n"
            f"  Action : {action}\n"
            f"  Symbol : {symbol}\n"
            f"  Qty    : {qty} shares\n"
            f"  Cash   : ${cash:.2f}\n"
            f"  Equity : ${equity:.2f}\n\n"
            f"You are a strict risk manager for a beginner trader with under $1,000.\n"
            f"Rules: max 2% risk per trade, reject if position too large for the account.\n"
            f"Reply with APPROVE or REJECT and one short reason. Max 15 words total."
        )
        response = groq.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[
                {"role": "system", "content": "You are a professional trading risk manager. Be strict and brief."},
                {"role": "user",   "content": prompt}
            ],
            max_tokens=50,
            temperature=0.1
        )
        decision = response.choices[0].message.content.strip()
        log.info(f"Groq decision: {decision}")
    except Exception as e:
        log.error(f"Groq error: {e}")
        return jsonify({"status": "error", "reason": "AI check failed — order blocked"}), 500

    # 5. Execute if approved
    if decision.upper().startswith("APPROVE"):
        try:
            side  = OrderSide.BUY if action == "BUY" else OrderSide.SELL
            order = alpaca.submit_order(
                MarketOrderRequest(
                    symbol=symbol,
                    qty=qty,
                    side=side,
                    time_in_force=TimeInForce.GTC
                )
            )
            daily_trade_count += 1
            log.info(f"Order placed: {order.id}")
            return jsonify({
                "status"      : "order placed",
                "order_id"    : str(order.id),
                "action"      : action,
                "symbol"      : symbol,
                "qty"         : qty,
                "ai_reason"   : decision,
                "trades_today": daily_trade_count
            }), 200
        except Exception as e:
            return jsonify({"status": "error", "reason": str(e)}), 500

    # 6. Rejected
    return jsonify({
        "status"   : "rejected by AI",
        "symbol"   : symbol,
        "action"   : action,
        "ai_reason": decision
    }), 200


# ── Status endpoint ───────────────────────────────────────────────────────────
@app.route("/status", methods=["GET"])
def status():
    try:
        account = alpaca.get_account()
        orders  = alpaca.get_orders()
        return jsonify({
            "account_status" : account.status,
            "cash"           : float(account.cash),
            "equity"         : float(account.equity),
            "trades_today"   : daily_trade_count,
            "ai_engine"      : "Groq Llama 3.3 70B (FREE)",
            "recent_orders"  : [
                {"id": str(o.id), "symbol": o.symbol,
                 "side": str(o.side), "qty": o.qty, "status": str(o.status)}
                for o in (orders[:5] if orders else [])
            ]
        })
    except Exception as e:
        return jsonify({"status": "error", "reason": str(e)}), 500


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8080))
    app.run(host="0.0.0.0", port=port)
