#!/usr/bin/env python3
"""
NUCLEAR LAUNCHER v2.0 - Render Free Tier Optimized
Python 3.13.4 Compatible - No Workers Required
"""

import os
import sys
import asyncio
import threading
import subprocess
import time
import json
import signal
from datetime import datetime, timedelta
import aiohttp
import psutil
from apscheduler.schedulers.background import BackgroundScheduler
from apscheduler.triggers.interval import IntervalTrigger

# ======================== NUCLEAR CONFIGURATION ========================

# Force environment optimization
os.environ['PYTHONOPTIMIZE'] = '2'
os.environ['PYTHONUNBUFFERED'] = '1'
os.environ['ASYNCIO_LOOP'] = 'uvloop'

# Memory management for free tier (512MB limit)
MEMORY_LIMIT_MB = 450  # Leave some headroom
CHECK_INTERVAL = 60  # Check every minute

print("""
╔══════════════════════════════════════════════════════════════╗
║                   ☢️  NUCLEAR LAUNCHER v2.0                   ║
║                  Python 3.13.4 - Render Optimized            ║
╚══════════════════════════════════════════════════════════════╝
""")

class NuclearKeepalive:
    """Advanced keep-alive system for Render free tier"""
    
    def __init__(self):
        self.scheduler = BackgroundScheduler()
        self.last_ping = datetime.now()
        self.ping_count = 0
        self.service_url = os.environ.get('RENDER_EXTERNAL_URL', 'http://localhost:10000')
        
    async def health_ping(self):
        """Send health check ping"""
        try:
            async with aiohttp.ClientSession() as session:
                await session.get(f"{self.service_url}/health", timeout=5)
                self.ping_count += 1
                self.last_ping = datetime.now()
                print(f"💗 Heartbeat #{self.ping_count} sent at {self.last_ping.strftime('%H:%M:%S')}")
        except Exception as e:
            print(f"⚠️ Heartbeat failed: {e}")
            
    async def aggressive_keepalive(self):
        """Aggressive keep-alive for Render's 15-minute timeout"""
        while True:
            # Ping every 10 minutes (well before 15-minute timeout)
            await self.health_ping()
            await asyncio.sleep(600)  # 10 minutes
            
            # Extra ping if approaching timeout
            time_since_ping = datetime.now() - self.last_ping
            if time_since_ping > timedelta(minutes=13):
                print("🚨 Emergency ping - approaching timeout!")
                await self.health_ping()
                
    def start(self):
        """Start the keep-alive system"""
        # Schedule regular pings
        self.scheduler.add_job(
            lambda: asyncio.run(self.health_ping()),
            IntervalTrigger(minutes=10),
            id='keepalive',
            name='Keep service alive',
            replace_existing=True
        )
        self.scheduler.start()
        
        # Start aggressive keep-alive in background
        threading.Thread(
            target=lambda: asyncio.run(self.aggressive_keepalive()),
            daemon=True
        ).start()

class MemoryGuardian:
    """Prevent OOM kills on free tier"""
    
    def __init__(self, limit_mb=MEMORY_LIMIT_MB):
        self.limit_mb = limit_mb
        self.process = psutil.Process()
        
    def check_memory(self):
        """Check and manage memory usage"""
        memory_mb = self.process.memory_info().rss / 1024 / 1024
        
        if memory_mb > self.limit_mb:
            print(f"⚠️ Memory critical: {memory_mb:.1f}MB / {self.limit_mb}MB")
            self.cleanup_memory()
            
        return memory_mb
        
    def cleanup_memory(self):
        """Emergency memory cleanup"""
        import gc
        gc.collect()
        
        # Clear caches
        if hasattr(sys, 'intern'):
            sys.intern.clear()
            
        print("🧹 Memory cleaned")
        
    async def monitor_loop(self):
        """Continuous memory monitoring"""
        while True:
            memory_mb = self.check_memory()
            if memory_mb > self.limit_mb * 0.9:  # 90% threshold
                print(f"📊 Memory usage: {memory_mb:.1f}MB - approaching limit")
                
            await asyncio.sleep(CHECK_INTERVAL)

class RenderOptimizer:
    """Render-specific optimizations"""
    
    @staticmethod
    def configure_environment():
        """Configure environment for Render free tier"""
        # Render-specific optimizations
        if 'RENDER' in os.environ:
            print("🔧 Detected Render environment - applying optimizations")
            
            # Reduce thread pool sizes
            os.environ['OMP_NUM_THREADS'] = '1'
            os.environ['OPENBLAS_NUM_THREADS'] = '1'
            os.environ['MKL_NUM_THREADS'] = '1'
            os.environ['VECLIB_MAXIMUM_THREADS'] = '1'
            os.environ['NUMEXPR_NUM_THREADS'] = '1'
            
            # Optimize pandas/numpy
            os.environ['PANDAS_COPY_ON_WRITE'] = '1'
            
            # Network optimizations
            os.environ['ASYNC_TIMEOUT'] = '30'
            
    @staticmethod
    def setup_crash_recovery():
        """Setup automatic crash recovery"""
        def signal_handler(signum, frame):
            print(f"🚨 Received signal {signum} - attempting graceful shutdown")
            # Save state before exit
            try:
                with open('data/crash_state.json', 'w') as f:
                    json.dump({
                        'timestamp': datetime.now().isoformat(),
                        'signal': signum,
                        'recovered': False
                    }, f)
            except:
                pass
            sys.exit(0)
            
        signal.signal(signal.SIGTERM, signal_handler)
        signal.signal(signal.SIGINT, signal_handler)

async def run_quantum_bot():
    """Run the main quantum trading bot"""
    try:
        # Check if the quantum bot exists
        if not os.path.exists('quantum_trading_bot.py'):
            print("⚠️ quantum_trading_bot.py not found - creating mock")
            
            # Create a minimal mock for testing
            with open('quantum_trading_bot.py', 'w') as f:
                f.write('''
import asyncio
import json
import os
from http.server import HTTPServer, BaseHTTPRequestHandler
import threading

os.makedirs('data', exist_ok=True)

class Handler(BaseHTTPRequestHandler):
    def do_GET(self):
        if self.path == '/health':
            self.send_response(200)
            self.end_headers()
            self.wfile.write(b'OK')
        else:
            self.send_response(200)
            self.send_header('Content-type', 'text/html')
            self.end_headers()
            self.wfile.write(b'<h1>Quantum Bot Running</h1>')
    def log_message(self, *args): pass

def serve():
    port = int(os.environ.get('PORT', 10000))
    server = HTTPServer(('0.0.0.0', port), Handler)
    print(f"Server on port {port}")
    server.serve_forever()

threading.Thread(target=serve, daemon=True).start()

while True:
    with open('data/bot_status.json', 'w') as f:
        json.dump({'status': 'Running', 'balance': 1000}, f)
    asyncio.run(asyncio.sleep(10))
''')
        
        # Import and run the quantum bot
        # Using exec to run in same process (more efficient for free tier)
        with open('quantum_trading_bot.py', 'r') as f:
            bot_code = f.read()
            
        # Execute in global namespace
        exec(bot_code, globals())
        
    except subprocess.CalledProcessError as e:
        print(f"❌ Bot process failed: {e}")
        # Auto-restart after failure
        await asyncio.sleep(10)
        await run_quantum_bot()
        
    except Exception as e:
        print(f"❌ Error running bot: {e}")
        # Keep the service alive even if bot fails
        while True:
            await asyncio.sleep(60)

async def main():
    """Main nuclear launcher"""
    print("🚀 Starting Nuclear Launcher...")
    
    # Configure Render optimizations
    RenderOptimizer.configure_environment()
    RenderOptimizer.setup_crash_recovery()
    
    # Start keep-alive system
    keepalive = NuclearKeepalive()
    keepalive.start()
    print("✅ Keep-alive system activated")
    
    # Start memory guardian
    guardian = MemoryGuardian()
    asyncio.create_task(guardian.monitor_loop())
    print("✅ Memory guardian activated")
    
    # Check for crash recovery
    if os.path.exists('data/crash_state.json'):
        print("🔄 Detected previous crash - recovering...")
        with open('data/crash_state.json', 'r') as f:
            crash_data = json.load(f)
            print(f"  Previous crash at: {crash_data['timestamp']}")
        
    # Run the quantum bot
    print("⚛️ Launching Quantum Trading Bot...")
    await run_quantum_bot()

if __name__ == "__main__":
    try:
        # Use uvloop if available for maximum performance
        try:
            import uvloop
            uvloop.install()
            print("⚡ uvloop installed - maximum performance mode")
        except ImportError:
            print("📌 Using standard asyncio")
            
        # Run the launcher
        asyncio.run(main())
        
    except KeyboardInterrupt:
        print("\n🛑 Shutdown requested")
        sys.exit(0)
    except Exception as e:
        print(f"💥 Fatal error: {e}")
        # Keep container alive to prevent restart loop
        while True:
            time.sleep(60)