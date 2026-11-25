import paramiko

c = paramiko.SSHClient()
c.set_missing_host_key_policy(paramiko.AutoAddPolicy())
c.connect('192.168.1.7', username='wildu', password='4123')

script = """
import sys
sys.path.insert(0, '/home/wildu/llama.cpp')
from vox_verbi.rag_engine import RAGEngine
import faiss

rag = RAGEngine()
q = 'Vox Verbi, ¿qué enseña la Iglesia sobre la Eucaristía?'
q_expanded = q + ' Eucaristía transustanciación presencia real cuerpo y sangre de Cristo pan y vino este es mi cuerpo haced esto en memoria mía Juan 6 1 Corintios 11'
q_vec = rag._get_embedder().encode([q_expanded], convert_to_numpy=True)
faiss.normalize_L2(q_vec)
scores, indices = rag.index.search(q_vec, 1200)

found = []
for s, idx in zip(scores[0], indices[0]):
    m = rag.metadata[idx]
    cita = m.get('cita_corta', '')
    if '1 Corintios 11' in cita or 'Juan 6' in cita:
        found.append((cita, s, m.get('texto')[:60]))

print(f'Found {len(found)} matches in top 1200!')
for f in found[:5]:
    print(f)
"""

sftp = c.open_sftp()
with sftp.file('/home/wildu/llama.cpp/find_euc.py', 'wb') as f:
    f.write(script.encode('utf-8'))
sftp.close()

stdin, stdout, stderr = c.exec_command('/home/wildu/llama.cpp/venv/bin/python /home/wildu/llama.cpp/find_euc.py')
print(stdout.read().decode('utf-8', errors='replace'))
c.close()
