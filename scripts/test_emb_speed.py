import paramiko

c = paramiko.SSHClient()
c.set_missing_host_key_policy(paramiko.AutoAddPolicy())
c.connect('192.168.1.7', username='wildu', password='4123')

script = """
import time
from sentence_transformers import SentenceTransformer

t0 = time.time()
model = SentenceTransformer('all-MiniLM-L6-v2')
print(f'Model loaded in {time.time() - t0:.2f}s')

texts = ["En el principio existía la Palabra y la Palabra estaba con Dios."] * 256
t1 = time.time()
emb = model.encode(texts, batch_size=128, show_progress_bar=False)
t2 = time.time()
rate = len(texts) / (t2 - t1)
print(f'Encoded {len(texts)} chunks in {t2 - t1:.2f}s ({rate:.0f} chunks/sec)')
"""

sftp = c.open_sftp()
with sftp.file('/home/wildu/llama.cpp/test_emb_speed.py', 'wb') as f:
    f.write(script.encode('utf-8'))
sftp.close()

stdin, stdout, stderr = c.exec_command('/home/wildu/llama.cpp/venv/bin/python /home/wildu/llama.cpp/test_emb_speed.py')
print("STDOUT:\n", stdout.read().decode('utf-8', errors='replace'))
print("STDERR:\n", stderr.read().decode('utf-8', errors='replace'))
c.close()
