import sys
import json
import requests
import paramiko

c = paramiko.SSHClient()
c.set_missing_host_key_policy(paramiko.AutoAddPolicy())
c.connect('192.168.1.7', username='wildu', password='4123')

script = """
import sys
sys.path.insert(0, '/home/wildu/llama.cpp')
import json
import requests
from vox_verbi.rag_engine import RAGEngine
from vox_verbi.llm_catequist import LLMCatequist

rag = RAGEngine()
llm = LLMCatequist()

res = rag.search('Vox Verbi, ¿qué significa Bautismo?', top_k=2)
records = res.get('results', [])

sys_p, usr_p, max_tok = llm.build_prompt('Vox Verbi, ¿qué significa Bautismo?', records, duration_sec=30)

print('--- SYSTEM PROMPT (chars=' + str(len(sys_p)) + ') ---')
r1 = requests.post('http://127.0.0.1:8080/tokenize', json={'content': sys_p})
tokens_sys = len(r1.json().get('tokens', []))
print('Tokens in system prompt:', tokens_sys)

print('\\n--- USER PROMPT (chars=' + str(len(usr_p)) + ') ---')
print(usr_p)
r2 = requests.post('http://127.0.0.1:8080/tokenize', json={'content': usr_p})
tokens_usr = len(r2.json().get('tokens', []))
print('Tokens in user prompt:', tokens_usr)
print('Total prompt tokens:', tokens_sys + tokens_usr)
"""

sftp = c.open_sftp()
with sftp.file('/home/wildu/llama.cpp/check_tokens.py', 'wb') as f:
    f.write(script.encode('utf-8'))
sftp.close()

stdin, stdout, stderr = c.exec_command('/home/wildu/llama.cpp/venv/bin/python /home/wildu/llama.cpp/check_tokens.py')
print("STDOUT:\n", stdout.read().decode('utf-8', errors='replace'))
print("STDERR:\n", stderr.read().decode('utf-8', errors='replace'))
c.close()
