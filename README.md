# AI Trading Bot — Railway Deployment Guide

## Files in this folder
| File            | Purpose                                      |
|-----------------|----------------------------------------------|
| main.py         | The bot server (Flask + Alpaca + Groq AI)    |
| requirements.txt| Python packages Railway will install         |
| Procfile        | Tells Railway how to start the server        |

---

## AI Engine: Groq Llama 3.3 70B
- 100% FREE — no credit card, no expiry
- 200-500 tokens/second (3-10x faster than ChatGPT)
- 1,000 requests/day free — your bot uses max 5/day
- Sign up at: console.groq.com

---

## Step 1 — Get your FREE Groq API key
1. Go to console.groq.com
2. Sign up with email or Google (no card needed)
3. Click "API Keys" → "Create API Key"
4. Copy it — you'll need it in Step 3

## Step 2 — Push files to GitHub (free)
1. Go to github.com → sign up free
2. Click "New repository" → name it "trading-bot" → Create
3. Upload all 4 files: main.py, requirements.txt, Procfile, README.md

## Step 3 — Deploy on Railway
1. Go to railway.app → New Project
2. "Deploy from GitHub repo" → select "trading-bot"
3. Railway auto-detects Python and deploys

## Step 4 — Add environment variables in Railway
Variables tab → add these 3:

| Variable      | Value                              | Where to get it          |
|---------------|------------------------------------|--------------------------|
| ALPACA_KEY    | your Alpaca paper API key ID       | app.alpaca.markets       |
| ALPACA_SECRET | your Alpaca paper secret key       | app.alpaca.markets       |
| GROQ_KEY      | your Groq API key                  | console.groq.com         |

## Step 5 — Get your webhook URL
Railway gives you a URL like:
  https://trading-bot-xxxx.railway.app

Your webhook is:
  https://trading-bot-xxxx.railway.app/webhook

## Step 6 — Test it
Visit in browser:
  https://trading-bot-xxxx.railway.app/        ← health check
  https://trading-bot-xxxx.railway.app/status  ← account + recent orders

## Step 7 — TradingView alert message
In TradingView → Alert → Message box:
  {"symbol": "AAPL", "action": "BUY", "qty": 1}

---

## Safety limits built into the bot
- Max 10 shares per order
- Max 5 trades per day
- Groq Llama 3.3 70B approves/rejects every trade
- Paper trading mode by default (no real money at risk)

## Cost breakdown
- Groq AI: FREE forever
- Railway hosting: FREE (hobby plan)
- Alpaca paper trading: FREE
- TradingView: FREE (1 webhook alert)
- Total monthly cost: $0
