import paramiko

c = paramiko.SSHClient()
c.set_missing_host_key_policy(paramiko.AutoAddPolicy())
c.connect('192.168.1.7', username='wildu', password='4123')

script = """
import os, re
from bs4 import BeautifulSoup

base_path = '/home/wildu/llama.cpp/La-Biblia-de-Jerusalen-Project/La Biblia'
html_files = []
for root, dirs, files in os.walk(base_path):
    if '!Contenido' in root or '00- Indice' in root or 'bible-front-end' in root or 'PowerShell' in root:
        continue
    for f in sorted(files):
        if f.endswith('.html') and not f.startswith('search') and not f.startswith('index') and not f.startswith('Work'):
            html_files.append(os.path.join(root, f))

print(f'Total Bible chapter files: {len(html_files)}')

bible_chunks = []
for fpath in html_files:
    fname = os.path.basename(fpath).replace('.html', '')
    parent = os.path.basename(os.path.dirname(fpath))
    clean_book = re.sub(r'^\d+-\s*', '', parent).strip()
    
    with open(fpath, 'r', encoding='utf-8', errors='ignore') as f:
        soup = BeautifulSoup(f.read(), 'html.parser')
    h1 = soup.find('h1')
    ch_title = h1.get_text().strip() if h1 else fname
    
    # Extract verses
    v_list = []
    for p in soup.find_all('p'):
        a = p.find('a')
        v_num = a.get_text().strip() if a else ''
        ptxt = p.get_text().strip()
        if v_num and ptxt.startswith(v_num):
            ptxt = ptxt[len(v_num):].strip()
        ptxt = re.sub(r'\s+', ' ', ptxt)
        if len(ptxt) > 5:
            v_list.append((v_num, ptxt))
            
    # Group verses into ~60-80 word chunks
    cur_chunk = []
    cur_words = 0
    cur_v_start = None
    cur_v_end = None
    
    for v_num, v_text in v_list:
        w_cnt = len(v_text.split())
        if not cur_v_start:
            cur_v_start = v_num
        cur_v_end = v_num
        cur_chunk.append((v_num, v_text))
        cur_words += w_cnt
        
        if cur_words >= 55:
            # Emit chunk
            joined_text = " ".join([vt[1] for vt in cur_chunk])
            cita = f"{ch_title}, {cur_v_start}-{cur_v_end}" if cur_v_start != cur_v_end else f"{ch_title}, {cur_v_start}"
            bible_chunks.append({
                "cita": cita,
                "libro": clean_book,
                "texto": joined_text
            })
            cur_chunk = []
            cur_words = 0
            cur_v_start = None
            
    if cur_chunk:
        joined_text = " ".join([vt[1] for vt in cur_chunk])
        cita = f"{ch_title}, {cur_v_start}-{cur_v_end}" if cur_v_start != cur_v_end else f"{ch_title}, {cur_v_start}"
        bible_chunks.append({
            "cita": cita,
            "libro": clean_book,
            "texto": joined_text
        })

print(f'Total Bible chunks produced: {len(bible_chunks)}')
print('Sample chunk 0:', bible_chunks[0])
print('Sample chunk 100:', bible_chunks[100])
"""

stdin, stdout, stderr = c.exec_command('/home/wildu/llama.cpp/venv/bin/python -c "' + script.replace('"', '\\"').replace('\n', ';') + '"')
# Better: write script to remote /tmp/test_parse.py
sftp = c.open_sftp()
with sftp.file('/home/wildu/llama.cpp/test_parse.py', 'w') as f:
    f.write(script)
sftp.close()

stdin, stdout, stderr = c.exec_command('/home/wildu/llama.cpp/venv/bin/python /home/wildu/llama.cpp/test_parse.py')
print("STDOUT:\n", stdout.read().decode('utf-8', errors='replace'))
print("STDERR:\n", stderr.read().decode('utf-8', errors='replace'))
c.close()
