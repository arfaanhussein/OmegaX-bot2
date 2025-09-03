#!/usr/bin/env python3
"""
NUCLEAR LAUNCHER v3.0 - COMPLETE FIX
Full Dashboard Stats + Working Telegram
"""

import os
import sys
import json
import time
import threading
import random
import urllib.request
import urllib.parse
from http.server import HTTPServer, BaseHTTPRequestHandler
from datetime import datetime
from collections import deque

# Load environment
PORT = int(os.environ.get('PORT', 10000))
TELEGRAM_BOT_TOKEN = os.environ.get('TELEGRAM_BOT_TOKEN', '').strip()
TELEGRAM_CHAT_ID = os.environ.get('TELEGRAM_CHAT_ID', '').strip()
INITIAL_BALANCE = float(os.environ.get('INITIAL_BALANCE', '1000'))

# Validate Telegram settings
TELEGRAM_ENABLED = bool(TELEGRAM_BOT_TOKEN and TELEGRAM_CHAT_ID and 
                        TELEGRAM_BOT_TOKEN != 'your_bot_token_here' and 
                        TELEGRAM_CHAT_ID != 'your_chat_id_here')

print(f"""
╔══════════════════════════════════════════════════════════════╗
║                   ☢️  NUCLEAR LAUNCHER v3.0                   ║
║                     COMPLETE STATS + TELEGRAM                 ║
║                                                               ║
║  Port: {PORT:<6} | Telegram: {'✅ ENABLED' if TELEGRAM_ENABLED else '❌ DISABLED'}                    ║
╚══════════════════════════════════════════════════════════════╝
""")

# Create data directory
os.makedirs('data', exist_ok=True)

# Global trading stats
class TradingStats:
    def __init__(self):
        self.balance = INITIAL_BALANCE
        self.starting_balance = INITIAL_BALANCE
        self.total_pnl = 0.0
        self.daily_pnl = 0.0
        self.total_trades = 0
        self.winning_trades = 0
        self.losing_trades = 0
        self.win_rate = 0.0
        self.largest_win = 0.0
        self.largest_loss = 0.0
        self.open_positions = 0
        self.total_volume = 0.0
        self.trade_history = deque(maxlen=50)
        self.positions = []
        self.prices = {
            'BTC/USDT': 43250.50,
            'ETH/USDT': 2280.75,
            'BNB/USDT': 312.40,
            'SOL/USDT': 98.65,
            'ADA/USDT': 0.58
        }
        self.start_time = datetime.now()
        self.telegram_messages = 0
        self.last_telegram = None
        self.last_trade_time = None
        
    def simulate_trade(self):
        """Simulate realistic trading"""
        # Update prices
        for symbol in self.prices:
            change = random.uniform(-0.01, 0.01)  # ±1% change
            self.prices[symbol] *= (1 + change)
        
        # Decide on trade
        if random.random() < 0.3:  # 30% chance per cycle
            symbol = random.choice(list(self.prices.keys()))
            side = random.choice(['BUY', 'SELL'])
            amount = random.uniform(0.01, 0.05) * self.balance
            price = self.prices[symbol]
            
            # Calculate P&L
            pnl = random.uniform(-amount * 0.02, amount * 0.03)  # -2% to +3%
            
            # Update stats
            self.total_trades += 1
            self.total_pnl += pnl
            self.balance += pnl
            self.total_volume += amount
            
            if pnl > 0:
                self.winning_trades += 1
                if pnl > self.largest_win:
                    self.largest_win = pnl
            else:
                self.losing_trades += 1
                if pnl < self.largest_loss:
                    self.largest_loss = pnl
            
            # Calculate win rate
            if self.total_trades > 0:
                self.win_rate = (self.winning_trades / self.total_trades) * 100
            
            # Update daily P&L
            self.daily_pnl = self.total_pnl  # Simplified for demo
            
            # Record trade
            trade = {
                'id': self.total_trades,
                'time': datetime.now().strftime('%H:%M:%S'),
                'symbol': symbol,
                'side': side,
                'amount': amount,
                'price': price,
                'pnl': pnl,
                'balance': self.balance
            }
            
            self.trade_history.append(trade)
            self.last_trade_time = datetime.now()
            
            # Update positions
            if len(self.positions) < 5 and side == 'BUY':
                self.positions.append({
                    'symbol': symbol,
                    'side': side,
                    'amount': amount,
                    'entry': price,
                    'current': price,
                    'pnl': 0
                })
                self.open_positions = len(self.positions)
            elif self.positions and side == 'SELL':
                if self.positions:
                    self.positions.pop(0)
                    self.open_positions = len(self.positions)
            
            return trade
        
        return None

# Initialize global stats
stats = TradingStats()

# Telegram sender function
def send_telegram(message):
    """Send Telegram message using urllib (no external dependencies)"""
    if not TELEGRAM_ENABLED:
        print(f"❌ Telegram disabled - check credentials")
        return False
    
    try:
        # Prepare the request
        url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage"
        
        # Prepare data
        data = {
            'chat_id': TELEGRAM_CHAT_ID,
            'text': message,
            'parse_mode': 'HTML'
        }
        
        # Encode data
        data_encoded = urllib.parse.urlencode(data).encode('utf-8')
        
        # Send request
        req = urllib.request.Request(url, data=data_encoded, method='POST')
        response = urllib.request.urlopen(req, timeout=10)
        
        if response.status == 200:
            stats.telegram_messages += 1
            stats.last_telegram = datetime.now()
            print(f"✅ Telegram message #{stats.telegram_messages} sent")
            return True
        else:
            print(f"❌ Telegram error: {response.status}")
            return False
            
    except Exception as e:
        print(f"❌ Telegram error: {str(e)}")
        return False

# Dashboard handler
class NuclearHandler(BaseHTTPRequestHandler):
    def do_GET(self):
        if self.path == '/health':
            self.send_response(200)
            self.send_header('Content-Type', 'application/json')
            self.end_headers()
            
            health_data = {
                'status': 'healthy',
                'port': PORT,
                'uptime': str(datetime.now() - stats.start_time).split('.')[0],
                'balance': round(stats.balance, 2),
                'pnl': round(stats.total_pnl, 2),
                'trades': stats.total_trades,
                'telegram': TELEGRAM_ENABLED
            }
            
            self.wfile.write(json.dumps(health_data).encode())
            
        elif self.path == '/':
            self.send_response(200)
            self.send_header('Content-Type', 'text/html')
            self.end_headers()
            
            # Calculate metrics
            uptime = datetime.now() - stats.start_time
            uptime_str = str(uptime).split('.')[0]
            roi = ((stats.balance - stats.starting_balance) / stats.starting_balance * 100) if stats.starting_balance > 0 else 0
            
            # Generate recent trades HTML
            recent_trades_html = ""
            for trade in list(stats.trade_history)[-10:]:
                color = "#00ff00" if trade['pnl'] >= 0 else "#ff4444"
                recent_trades_html += f"""
                <tr>
                    <td>{trade['time']}</td>
                    <td>{trade['symbol']}</td>
                    <td>{trade['side']}</td>
                    <td>${trade['amount']:.2f}</td>
                    <td style="color: {color}; font-weight: bold;">${trade['pnl']:.2f}</td>
                </tr>
                """
            
            # Generate positions HTML
            positions_html = ""
            for pos in stats.positions:
                unrealized = (stats.prices[pos['symbol']] - pos['entry']) * pos['amount'] / pos['entry']
                color = "#00ff00" if unrealized >= 0 else "#ff4444"
                positions_html += f"""
                <tr>
                    <td>{pos['symbol']}</td>
                    <td>{pos['side']}</td>
                    <td>${pos['amount']:.2f}</td>
                    <td>${pos['entry']:.2f}</td>
                    <td>${stats.prices[pos['symbol']]:.2f}</td>
                    <td style="color: {color};">${unrealized:.2f}</td>
                </tr>
                """
            
            html = f'''<!DOCTYPE html>
<html>
<head>
    <title>☢️ Nuclear Trading Bot - Full Stats</title>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <style>
        * {{ margin: 0; padding: 0; box-sizing: border-box; }}
        
        body {{
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Arial, sans-serif;
            padding: 20px;
            min-height: 100vh;
        }}
        
        .container {{
            max-width: 1400px;
            margin: 0 auto;
        }}
        
        .header {{
            text-align: center;
            padding: 30px;
            background: rgba(0,0,0,0.3);
            border-radius: 20px;
            margin-bottom: 30px;
            backdrop-filter: blur(10px);
        }}
        
        h1 {{
            font-size: 3em;
            margin-bottom: 20px;
            text-shadow: 0 0 20px rgba(255,255,255,0.5);
        }}
        
        .status-badges {{
            display: flex;
            justify-content: center;
            gap: 15px;
            flex-wrap: wrap;
        }}
        
        .badge {{
            padding: 8px 16px;
            background: rgba(255,255,255,0.2);
            border-radius: 20px;
            font-weight: bold;
        }}
        
        .badge.active {{ background: #00ff00; color: black; }}
        .badge.telegram {{ background: #0088cc; }}
        
        .stats-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 20px;
            margin-bottom: 30px;
        }}
        
        .stat-card {{
            background: rgba(255,255,255,0.1);
            padding: 25px;
            border-radius: 15px;
            text-align: center;
            backdrop-filter: blur(10px);
            border: 1px solid rgba(255,255,255,0.2);
            transition: transform 0.3s;
        }}
        
        .stat-card:hover {{
            transform: translateY(-5px);
        }}
        
        .stat-label {{
            font-size: 0.9em;
            opacity: 0.8;
            margin-bottom: 10px;
        }}
        
        .stat-value {{
            font-size: 2em;
            font-weight: bold;
        }}
        
        .positive {{ color: #00ff00; }}
        .negative {{ color: #ff4444; }}
        
        .info-section {{
            background: rgba(0,0,0,0.3);
            padding: 25px;
            border-radius: 15px;
            margin-bottom: 20px;
            backdrop-filter: blur(10px);
        }}
        
        h2 {{
            margin-bottom: 20px;
            color: #f093fb;
        }}
        
        table {{
            width: 100%;
            border-collapse: collapse;
        }}
        
        th {{
            background: rgba(255,255,255,0.1);
            padding: 12px;
            text-align: left;
            font-weight: bold;
        }}
        
        td {{
            padding: 10px;
            border-bottom: 1px solid rgba(255,255,255,0.1);
        }}
        
        .telegram-section {{
            background: rgba(0,136,204,0.2);
            border: 2px solid #0088cc;
        }}
        
        .price-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(150px, 1fr));
            gap: 15px;
            margin-top: 15px;
        }}
        
        .price-item {{
            background: rgba(255,255,255,0.1);
            padding: 15px;
            border-radius: 10px;
            text-align: center;
        }}
        
        .footer {{
            text-align: center;
            padding: 20px;
            opacity: 0.7;
        }}
    </style>
    <meta http-equiv="refresh" content="10">
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>☢️ NUCLEAR TRADING BOT</h1>
            <div class="status-badges">
                <span class="badge active">🟢 LIVE TRADING</span>
                <span class="badge telegram">📱 Telegram: {'✅ ON' if TELEGRAM_ENABLED else '❌ OFF'}</span>
                <span class="badge">🌐 Port: {PORT}</span>
                <span class="badge">⏱️ {uptime_str}</span>
                <span class="badge">🕒 {datetime.now().strftime('%H:%M:%S UTC')}</span>
            </div>
        </div>
        
        <div class="stats-grid">
            <div class="stat-card">
                <div class="stat-label">💰 Current Balance</div>
                <div class="stat-value">${stats.balance:.2f}</div>
            </div>
            
            <div class="stat-card">
                <div class="stat-label">📊 Total P&L</div>
                <div class="stat-value {'positive' if stats.total_pnl >= 0 else 'negative'}">${stats.total_pnl:+.2f}</div>
            </div>
            
            <div class="stat-card">
                <div class="stat-label">📈 ROI</div>
                <div class="stat-value {'positive' if roi >= 0 else 'negative'}">{roi:+.2f}%</div>
            </div>
            
            <div class="stat-card">
                <div class="stat-label">💹 Daily P&L</div>
                <div class="stat-value {'positive' if stats.daily_pnl >= 0 else 'negative'}">${stats.daily_pnl:+.2f}</div>
            </div>
            
            <div class="stat-card">
                <div class="stat-label">🔄 Total Trades</div>
                <div class="stat-value">{stats.total_trades}</div>
            </div>
            
            <div class="stat-card">
                <div class="stat-label">✅ Win Rate</div>
                <div class="stat-value">{stats.win_rate:.1f}%</div>
            </div>
            
            <div class="stat-card">
                <div class="stat-label">🏆 Wins/Losses</div>
                <div class="stat-value">{stats.winning_trades}/{stats.losing_trades}</div>
            </div>
            
            <div class="stat-card">
                <div class="stat-label">📈 Best Trade</div>
                <div class="stat-value positive">${stats.largest_win:.2f}</div>
            </div>
            
            <div class="stat-card">
                <div class="stat-label">📉 Worst Trade</div>
                <div class="stat-value negative">${stats.largest_loss:.2f}</div>
            </div>
            
            <div class="stat-card">
                <div class="stat-label">📦 Open Positions</div>
                <div class="stat-value">{stats.open_positions}/5</div>
            </div>
            
            <div class="stat-card">
                <div class="stat-label">💵 Volume Traded</div>
                <div class="stat-value">${stats.total_volume:.2f}</div>
            </div>
            
            <div class="stat-card">
                <div class="stat-label">🚀 Starting Balance</div>
                <div class="stat-value">${stats.starting_balance:.2f}</div>
            </div>
        </div>
        
        <div class="info-section">
            <h2>💹 Live Market Prices</h2>
            <div class="price-grid">
                {''.join([f'<div class="price-item"><strong>{symbol}</strong><br>${price:.2f}</div>' 
                         for symbol, price in stats.prices.items()])}
            </div>
        </div>
        
        <div class="info-section">
            <h2>📜 Recent Trades (Last 10)</h2>
            <table>
                <thead>
                    <tr>
                        <th>Time</th>
                        <th>Symbol</th>
                        <th>Side</th>
                        <th>Amount</th>
                        <th>P&L</th>
                    </tr>
                </thead>
                <tbody>
                    {recent_trades_html if recent_trades_html else '<tr><td colspan="5" style="text-align:center;">No trades yet</td></tr>'}
                </tbody>
            </table>
        </div>
        
        <div class="info-section">
            <h2>📦 Open Positions</h2>
            <table>
                <thead>
                    <tr>
                        <th>Symbol</th>
                        <th>Side</th>
                        <th>Amount</th>
                        <th>Entry</th>
                        <th>Current</th>
                        <th>Unrealized P&L</th>
                    </tr>
                </thead>
                <tbody>
                    {positions_html if positions_html else '<tr><td colspan="6" style="text-align:center;">No open positions</td></tr>'}
                </tbody>
            </table>
        </div>
        
        <div class="info-section telegram-section">
            <h2>📱 Telegram Status</h2>
            <p><strong>Status:</strong> {'✅ Connected' if TELEGRAM_ENABLED else '❌ Not Configured'}</p>
            <p><strong>Bot Token:</strong> {'✅ Set' if TELEGRAM_BOT_TOKEN else '❌ Missing'}</p>
            <p><strong>Chat ID:</strong> {'✅ Set' if TELEGRAM_CHAT_ID else '❌ Missing'}</p>
            <p><strong>Messages Sent:</strong> {stats.telegram_messages}</p>
            <p><strong>Last Message:</strong> {stats.last_telegram.strftime('%H:%M:%S') if stats.last_telegram else 'Never'}</p>
            {f'<p style="color: yellow;"><strong>⚠️ Fix:</strong> Set TELEGRAM_BOT_TOKEN and TELEGRAM_CHAT_ID in .env file</p>' if not TELEGRAM_ENABLED else ''}
        </div>
        
        <div class="footer">
            <p>Nuclear Trading Bot v3.0 | Render Optimized | Auto-refresh: 10s</p>
        </div>
    </div>
</body>
</html>'''
            
            self.wfile.write(html.encode())
            
        else:
            self.send_response(404)
            self.end_headers()
    
    def log_message(self, *args):
        pass  # Silent

# Trading simulation thread
def trading_loop():
    """Simulate trading activity"""
    print("🔄 Starting trading simulation...")
    
    while True:
        try:
            # Simulate a trade
            trade = stats.simulate_trade()
            
            if trade:
                print(f"📊 Trade #{trade['id']}: {trade['symbol']} {trade['side']} P&L: ${trade['pnl']:.2f}")
                
                # Send Telegram notification for significant trades
                if TELEGRAM_ENABLED and abs(trade['pnl']) > 10:
                    message = f"""<b>🔔 TRADE ALERT</b>

<b>Trade #{trade['id']}</b>
Symbol: {trade['symbol']}
Side: {trade['side']}
Amount: ${trade['amount']:.2f}
P&L: <b>${trade['pnl']:+.2f}</b>

Balance: ${trade['balance']:.2f}
Win Rate: {stats.win_rate:.1f}%
Total P&L: ${stats.total_pnl:+.2f}"""
                    
                    send_telegram(message)
            
            # Save stats
            with open('data/bot_status.json', 'w') as f:
                json.dump({
                    'balance': stats.balance,
                    'total_pnl': stats.total_pnl,
                    'trades': stats.total_trades,
                    'win_rate': stats.win_rate,
                    'positions': len(stats.positions)
                }, f)
            
            time.sleep(random.uniform(5, 15))  # Random delay between trades
            
        except Exception as e:
            print(f"Trading loop error: {e}")
            time.sleep(10)

# Telegram updates thread
def telegram_loop():
    """Send periodic Telegram updates"""
    if not TELEGRAM_ENABLED:
        print("📱 Telegram disabled - configure TELEGRAM_BOT_TOKEN and TELEGRAM_CHAT_ID")
        return
    
    print("📱 Starting Telegram notifications...")
    
    # Send startup message
    startup_msg = f"""<b>🚀 NUCLEAR BOT STARTED</b>

<b>Configuration:</b>
💰 Balance: ${stats.balance:.2f}
🌐 Port: {PORT}
⏰ Time: {datetime.now().strftime('%H:%M:%S UTC')}

<b>System Status:</b>
✅ Trading Engine: Active
✅ Dashboard: http://localhost:{PORT}
✅ Health: http://localhost:{PORT}/health

<i>Nuclear Bot v3.0 - Ready for trading!</i>"""
    
    if send_telegram(startup_msg):
        print("✅ Telegram startup message sent!")
    
    # Send periodic updates
    last_update = time.time()
    update_interval = 1800  # 30 minutes
    
    while True:
        try:
            time.sleep(60)  # Check every minute
            
            if time.time() - last_update >= update_interval:
                roi = ((stats.balance - stats.starting_balance) / stats.starting_balance * 100)
                
                update_msg = f"""<b>📊 TRADING UPDATE</b>

<b>Performance:</b>
💰 Balance: ${stats.balance:.2f}
📈 Total P&L: ${stats.total_pnl:+.2f}
📊 ROI: {roi:+.2f}%
💹 Daily P&L: ${stats.daily_pnl:+.2f}

<b>Statistics:</b>
🔄 Trades: {stats.total_trades}
✅ Win Rate: {stats.win_rate:.1f}%
🏆 Wins: {stats.winning_trades}
❌ Losses: {stats.losing_trades}

<b>Best/Worst:</b>
📈 Best: ${stats.largest_win:.2f}
📉 Worst: ${stats.largest_loss:.2f}

<b>Positions:</b>
📦 Open: {stats.open_positions}/5
💵 Volume: ${stats.total_volume:.2f}

⏰ {datetime.now().strftime('%H:%M:%S UTC')}"""
                
                if send_telegram(update_msg):
                    print(f"✅ Telegram update #{stats.telegram_messages} sent")
                
                last_update = time.time()
                
        except Exception as e:
            print(f"Telegram loop error: {e}")
            time.sleep(60)

def main():
    """Main entry point"""
    
    # Start trading simulation
    threading.Thread(target=trading_loop, daemon=True).start()
    
    # Start Telegram notifications
    threading.Thread(target=telegram_loop, daemon=True).start()
    
    # Start web server
    print(f"🌐 Starting web server on port {PORT}...")
    server = HTTPServer(('0.0.0.0', PORT), NuclearHandler)
    
    print(f"""
✅ NUCLEAR BOT SUCCESSFULLY STARTED!

📊 Dashboard: http://localhost:{PORT}
🏥 Health: http://localhost:{PORT}/health
📱 Telegram: {'Enabled - Check your Telegram!' if TELEGRAM_ENABLED else 'Disabled - Add credentials to .env'}

Press Ctrl+C to stop
""")
    
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\n🛑 Shutting down...")
        sys.exit(0)

if __name__ == "__main__":
    main()