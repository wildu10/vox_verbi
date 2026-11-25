import paramiko

def main():
    c = paramiko.SSHClient()
    c.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    c.connect('192.168.1.7', username='wildu', password='4123')
    
    remote_code = '''
import pickle
meta = pickle.load(open("/home/wildu/llama.cpp/vox_verbi/data/textos_teologicos_meta.pkl", "rb"))

print("--- HEBREOS 1 ---")
for m in meta:
    if "hebreos 1, 1" in m.get("cita_corta", "").lower():
        print(m.get("cita_corta"), "|", m.get("titulo"), "|", m.get("texto")[:100])
        break

print("--- DEI VERBUM ---")
for m in meta:
    if "dei verbum" in (m.get("cita_corta", "") + " " + m.get("titulo", "")).lower():
        print(m.get("cita_corta"), "|", m.get("titulo"), "|", m.get("texto")[:100])

print("--- COMPENDIO REVELACION (6-9) ---")
for m in meta:
    cita = m.get("cita_corta", "").lower()
    if "compendio" in cita and any(f"n. {i}" in cita for i in [6, 7, 8, 9]):
        print(m.get("cita_corta"), "|", m.get("titulo"), "|", m.get("texto")[:100])
'''
    sftp = c.open_sftp()
    with sftp.open("/home/wildu/llama.cpp/remote_inspect.py", "w") as f:
        f.write(remote_code)
    sftp.close()
    stdin, stdout, stderr = c.exec_command("python3 /home/wildu/llama.cpp/remote_inspect.py")
    out = stdout.read().decode('utf-8', errors='replace')
    err = stderr.read().decode('utf-8', errors='replace')
    print("STDOUT:\n", out)
    if err:
        print("STDERR:\n", err)
    c.close()

if __name__ == "__main__":
    main()
