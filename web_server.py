#!/usr/bin/env python3
import os
import json
import time
from http.server import HTTPServer, BaseHTTPRequestHandler
import threading

# Ensure data directory exists
os.makedirs('data', exist_ok=True)

class HealthHandler(BaseHTTPRequestHandler):
    def do_GET(self):
        if self.path == '/health':
            self.send_response(200)
            self.send_header('Content-Type', 'application/json')
            self.end_headers()
            self.wfile.write(b'{"status": "healthy"}')
        elif self.path == '/':
            self.send_response(200)
            self.send_header('Content-Type', 'text/html')
            self.end_headers()
            html = f"""
            <!DOCTYPE html>
            <html><head><title>Quantum Bot</title></head>
            <body style="font-family: Arial; background: #1a1a2e; color: white; text-align: center; padding: 50px;">
                <h1>🚀 Quantum Trading Bot</h1>
                <p>Status: <span style="color: #00ff00;">RUNNING</span></p>
                <p>Port: {os.environ.get('PORT', '10000')}</p>
                <p>Time: {time.strftime('%Y-%m-%d %H:%M:%S UTC')}</p>
                <div style="margin: 20px; padding: 20px; background: rgba(255,255,255,0.1); border-radius: 10px;">
                    <h3>✅ System Status</h3>
                    <p>Server: Online</p>
                    <p>Memory: Optimized</p>
                    <p>Render: Compatible</p>
                </div>
            </body></html>
            """
            self.wfile.write(html.encode())
        else:
            self.send_response(404)
            self.end_headers()
    
    def log_message(self, *args):
        pass  # Silent logging

def start_server():
    port = int(os.environ.get('PORT', 10000))
    server = HTTPServer(('0.0.0.0', port), HealthHandler)
    print(f"🌐 Server started on http://0.0.0.0:{port}")
    print(f"🔗 Health check: http://0.0.0.0:{port}/health")
    server.serve_forever()

def update_status():
    """Keep status file updated"""
    while True:
        try:
            with open('data/bot_status.json', 'w') as f:
                json.dump({
                    'status': 'running',
                    'timestamp': time.time(),
                    'port': os.environ.get('PORT', '10000'),
                    'balance': float(os.environ.get('INITIAL_BALANCE', '1000'))
                }, f)
        except Exception as e:
            print(f"Status update error: {e}")
        time.sleep(30)

if __name__ == "__main__":
    print("🚀 Starting Quantum Bot Web Server...")
    
    # Start status updater in background
    threading.Thread(target=update_status, daemon=True).start()
    
    # Start main server (this will block and keep the process alive)
    start_server()