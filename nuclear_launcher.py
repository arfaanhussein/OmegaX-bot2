#!/usr/bin/env python3
"""
MINIMAL NUCLEAR LAUNCHER - Render Compatible
Starts a web server immediately on PORT
"""

import os
import sys
import json
import time
import threading
from http.server import HTTPServer, BaseHTTPRequestHandler
from datetime import datetime

# Get PORT from environment - CRITICAL for Render
PORT = int(os.environ.get('PORT', 10000))

print(f"""
╔══════════════════════════════════════════════════════════════╗
║                   ☢️  NUCLEAR LAUNCHER v2.0                   ║
║                  Starting on PORT {PORT}                      ║
╚══════════════════════════════════════════════════════════════╝
""")

# Create data directory
os.makedirs('data', exist_ok=True)

# Global stats
stats = {
    'status': 'Running',
    'balance': float(os.environ.get('INITIAL_BALANCE', '1000')),
    'port': PORT,
    'start_time': datetime.now().isoformat()
}

class NuclearHandler(BaseHTTPRequestHandler):
    """Main HTTP handler"""
    
    def do_GET(self):
        if self.path == '/health':
            self.send_response(200)
            self.send_header('Content-Type', 'application/json')
            self.end_headers()
            response = json.dumps({'status': 'healthy', 'port': PORT})
            self.wfile.write(response.encode())
            
        elif self.path == '/':
            self.send_response(200)
            self.send_header('Content-Type', 'text/html')
            self.end_headers()
            
            html = f'''<!DOCTYPE html>
<html>
<head>
    <title>Nuclear Bot - Render</title>
    <meta charset="UTF-8">
    <style>
        body {{
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            font-family: Arial, sans-serif;
            padding: 20px;
            text-align: center;
        }}
        .container {{
            max-width: 800px;
            margin: 0 auto;
            background: rgba(0,0,0,0.3);
            padding: 30px;
            border-radius: 20px;
        }}
        .status {{
            background: #00ff00;
            color: black;
            padding: 5px 15px;
            border-radius: 20px;
            display: inline-block;
            font-weight: bold;
        }}
        .metric {{
            background: rgba(255,255,255,0.1);
            padding: 20px;
            margin: 10px;
            border-radius: 10px;
            display: inline-block;
            min-width: 150px;
        }}
    </style>
    <meta http-equiv="refresh" content="30">
</head>
<body>
    <div class="container">
        <h1>☢️ Nuclear Trading Bot</h1>
        <div class="status">ONLINE</div>
        
        <h2>System Status</h2>
        <div class="metric">
            <h3>Port</h3>
            <p>{PORT}</p>
        </div>
        <div class="metric">
            <h3>Balance</h3>
            <p>${stats['balance']:.2f}</p>
        </div>
        <div class="metric">
            <h3>Status</h3>
            <p>{stats['status']}</p>
        </div>
        <div class="metric">
            <h3>Time</h3>
            <p>{datetime.now().strftime('%H:%M:%S UTC')}</p>
        </div>
        
        <h2>Render Deployment</h2>
        <p>✅ Server is running on port {PORT}</p>
        <p>✅ Health endpoint: <a href="/health" style="color: yellow;">/health</a></p>
        <p>✅ Started: {stats['start_time']}</p>
        
        <div style="margin-top: 30px; opacity: 0.7;">
            <p>Nuclear Launcher v2.0 - Optimized for Render Free Tier</p>
        </div>
    </div>
</body>
</html>'''
            
            self.wfile.write(html.encode())
            
        else:
            self.send_response(404)
            self.end_headers()
            self.wfile.write(b'Not Found')
    
    def log_message(self, *args):
        # Suppress default logging
        pass

def status_updater():
    """Update status file periodically"""
    while True:
        try:
            with open('data/bot_status.json', 'w') as f:
                json.dump(stats, f)
            time.sleep(30)
        except Exception as e:
            print(f"Status update error: {e}")
            time.sleep(30)

def run_bot_simulator():
    """Simulate bot activity"""
    while True:
        try:
            # Simulate trading
            stats['balance'] = stats['balance'] * (1 + (time.time() % 10 - 5) / 1000)
            stats['status'] = 'Trading Active'
            time.sleep(60)
        except Exception as e:
            print(f"Bot simulator error: {e}")
            time.sleep(60)

def main():
    """Main entry point"""
    
    # Start background threads
    print("🚀 Starting background services...")
    threading.Thread(target=status_updater, daemon=True).start()
    threading.Thread(target=run_bot_simulator, daemon=True).start()
    
    # START THE WEB SERVER IMMEDIATELY - This is what Render needs!
    print(f"🌐 Starting web server on 0.0.0.0:{PORT}")
    server = HTTPServer(('0.0.0.0', PORT), NuclearHandler)
    
    print(f"✅ Server is running at http://0.0.0.0:{PORT}")
    print(f"✅ Health check endpoint: http://0.0.0.0:{PORT}/health")
    print(f"🔥 Nuclear bot is LIVE!")
    
    # This will run forever and keep the port open
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\n🛑 Shutdown requested")
        server.shutdown()
        sys.exit(0)

if __name__ == "__main__":
    main()