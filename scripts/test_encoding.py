import paramiko

c = paramiko.SSHClient()
c.set_missing_host_key_policy(paramiko.AutoAddPolicy())
c.connect('192.168.1.7', username='wildu', password='4123')

script = """
import re
with open('/home/wildu/llama.cpp/catecismo.txt', 'r', encoding='utf-8', errors='ignore') as f:
    text = f.read()

questions = re.findall(r'(?m)^(\d{1,3})\.\s+(\u00bf[^\n]+)$', text)
print(f'Numbered questions with inverted question mark: {len(questions)}')
for q in questions[:5]:
    print('Q:', q)
for q in questions[-5:]:
    print('Q:', q)
"""

sftp = c.open_sftp()
with sftp.file('/home/wildu/llama.cpp/check_enc.py', 'wb') as f:
    f.write(script.encode('utf-8'))
sftp.close()

stdin, stdout, stderr = c.exec_command('/home/wildu/llama.cpp/venv/bin/python /home/wildu/llama.cpp/check_enc.py')
out = stdout.read().decode('utf-8', errors='replace')
err = stderr.read().decode('utf-8', errors='replace')
print("STDOUT:\n", out)
print("STDERR:\n", err)
c.close()
