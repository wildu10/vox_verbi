import paramiko

c = paramiko.SSHClient()
c.set_missing_host_key_policy(paramiko.AutoAddPolicy())
c.connect('192.168.1.7', username='wildu', password='4123')

script = """
import pickle
with open('/home/wildu/llama.cpp/vox_verbi/data/textos_teologicos_meta.pkl', 'rb') as f:
    meta = pickle.load(f)

print('--- BAUTISMO IN COMPENDIO ---')
for idx, m in enumerate(meta):
    cita = m.get('cita_corta', '')
    tit = m.get('titulo', '')
    if 'Compendio' in cita:
        if any(f'n. {n}' in cita for n in range(250, 265)):
            print(f'Idx {idx}: {cita} | {tit}')

print('\\n--- BAUTISMO IN ESCRITURA ---')
for idx, m in enumerate(meta):
    cita = m.get('cita_corta', '')
    txt = m.get('texto', '')
    if ('Romanos 6' in cita) or ('Mateo 28' in cita and '19' in cita) or ('Juan 3' in cita and 'agua' in txt.lower()):
        print(f'Idx {idx}: {cita} | {m.get(\"texto\")[:70]}...')
"""

sftp = c.open_sftp()
with sftp.file('/home/wildu/llama.cpp/find_bautismo.py', 'wb') as f:
    f.write(script.encode('utf-8'))
sftp.close()

stdin, stdout, stderr = c.exec_command('/home/wildu/llama.cpp/venv/bin/python /home/wildu/llama.cpp/find_bautismo.py')
print(stdout.read().decode('utf-8', errors='replace'))
c.close()
