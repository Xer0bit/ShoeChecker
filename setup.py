import uvicorn
from src.api.main import app
import socket

def is_port_available(port):
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        try:
            s.bind(('localhost', port))
            return True
        except:
            return False

def get_valid_port():
    while True:
        try:
            port = input("Enter port number (default: 8000): ").strip()
            if not port:
                port = 8000
            port = int(port)
            if 1024 <= port <= 65535:
                if is_port_available(port):
                    return port
                else:
                    print(f"Port {port} is already in use. Please choose another port.")
            else:
                print("Port must be between 1024 and 65535")
        except ValueError:
            print("Please enter a valid number")

def main():
    print("=== Shoe Damage Analysis API Setup ===")
    port = get_valid_port()
    print(f"\nStarting server on port {port}...")
    print(f"API Documentation will be available at: http://localhost:{port}/docs")
    print("Press CTRL+C to stop the server")
    
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=port,
        log_level="info"
    )

if __name__ == "__main__":
    main()
