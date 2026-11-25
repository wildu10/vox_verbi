import os
import sys
import paramiko

if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
if hasattr(sys.stderr, 'reconfigure'):
    sys.stderr.reconfigure(encoding='utf-8', errors='replace')

HOSTNAME = "192.168.1.7"
USERNAME = "wildu"
PASSWORD = "4123"

def main():
    print("=" * 75)
    print("EJECUTANDO BENCHMARK FORENSE REMOTO (4 CONSULTAS EN STREAMING REAL)")
    print("=" * 75)
    
    c = paramiko.SSHClient()
    c.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    c.connect(HOSTNAME, username=USERNAME, password=PASSWORD, timeout=10)
    
    cmd = "/home/wildu/llama.cpp/venv/bin/python /home/wildu/llama.cpp/run_live_benchmark.py"
    stdin, stdout, stderr = c.exec_command(cmd)
    
    while True:
        line = stdout.readline()
        if not line:
            break
        print(line, end='', flush=True)
        
    err = stderr.read().decode('utf-8', errors='replace')
    if err.strip():
        print("\nSTDERR DE EJECUCIÓN:\n", err.strip())
        
    c.close()
    print("\n" + "=" * 75)
    print("EJECUCIÓN DEL BENCHMARK COMPLETADA")
    print("=" * 75)

if __name__ == "__main__":
    main()
