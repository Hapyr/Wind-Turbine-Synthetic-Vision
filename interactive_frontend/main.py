#!/usr/bin/env python3
"""
Simple HTTP server for the Wind Turbine Detection web app.

Usage:
    python main.py [port]
    
Default port: 8000
"""

import http.server
import socketserver
import sys
import os

def main():
    # Get port from command line argument or use default
    PORT = int(sys.argv[1]) if len(sys.argv) > 1 else 8000
    
    # Change to the directory containing this script
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    
    Handler = http.server.SimpleHTTPRequestHandler
    
    # Add CORS headers for development
    class CORSRequestHandler(Handler):
        def end_headers(self):
            self.send_header('Access-Control-Allow-Origin', '*')
            self.send_header('Cross-Origin-Embedder-Policy', 'require-corp')
            self.send_header('Cross-Origin-Opener-Policy', 'same-origin')
            super().end_headers()
    
    with socketserver.TCPServer(("", PORT), CORSRequestHandler) as httpd:
        print(f"🌐 Server running at http://localhost:{PORT}")
        print(f"📁 Serving files from: {os.getcwd()}")
        
        try:
            httpd.serve_forever()
        except KeyboardInterrupt:
            print("\n\n👋 Server stopped")
            sys.exit(0)

if __name__ == "__main__":
    main()
