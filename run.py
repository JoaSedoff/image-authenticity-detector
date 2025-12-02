#!/usr/bin/env python
"""
Script principal para ejecutar la aplicación Flask
"""

from app import create_app
import socket

def get_local_ip():
    """Obtiene la IP local de la máquina"""
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect(("8.8.8.8", 80))
        ip = s.getsockname()[0]
        s.close()
        return ip
    except Exception:
        return "127.0.0.1"

if __name__ == '__main__':
    app = create_app()
    
    local_ip = get_local_ip()
    port = 5000
    
    print("\n" + "="*70)
    print("🚀 Servidor iniciado correctamente")
    print("="*70)
    print(f"\n📱 Accede desde tu navegador en:")
    print(f"   • Local:  http://localhost:{port}")
    print(f"   • Red:    http://{local_ip}:{port}")
    print("\n💡 Comparte la URL de red con otros dispositivos en la misma WiFi")
    print("⚠️  Las imágenes grandes (>10MB) pueden tardar unos segundos en procesarse")
    print("🛑 Presiona CTRL+C para detener el servidor\n")
    print("="*70 + "\n")
    
    # host='0.0.0.0' permite conexiones desde otros dispositivos en la red
    # threaded=True permite múltiples peticiones simultáneas
    app.run(host='0.0.0.0', port=port, debug=True, threaded=True)
