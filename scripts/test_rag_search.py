import paramiko

c = paramiko.SSHClient()
c.set_missing_host_key_policy(paramiko.AutoAddPolicy())
c.connect('192.168.1.7', username='wildu', password='4123')

script = """
import sys
sys.path.insert(0, '/home/wildu/llama.cpp')
from vox_verbi.rag_engine import RAGEngine

rag = RAGEngine()
print(f'Total records in RAG: {len(rag.metadata)}, FAISS ntotal: {rag.index.ntotal}')

queries = [
    'Vox Verbi, ¿qué significa Bautismo?',
    'Vox Verbi, háblame durante un minuto sobre la Trinidad.'
]

for q in queries:
    res = rag.search(q, top_k=2)
    print('\\nQUERY:', q)
    print('CONFIDENCE:', res['confidence'], 'MIN DIST:', res['min_distance'])
    for idx, r in enumerate(res['results'], 1):
        print(f'  Fuente {idx} [{r.get(\"columna\")}]: {r.get(\"cita_corta\")} | {r.get(\"titulo\")}')
        print(f'    Texto: {r.get(\"texto\")[:100]}...')
"""

sftp = c.open_sftp()
with sftp.file('/home/wildu/llama.cpp/check_rag.py', 'wb') as f:
    f.write(script.encode('utf-8'))
sftp.close()

stdin, stdout, stderr = c.exec_command('/home/wildu/llama.cpp/venv/bin/python /home/wildu/llama.cpp/check_rag.py')
print("STDOUT:\n", stdout.read().decode('utf-8', errors='replace'))
print("STDERR:\n", stderr.read().decode('utf-8', errors='replace'))
c.close()
