import os
import sys
import paramiko

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

HOSTNAME = os.environ.get("VOX_VERBI_HOST", "192.168.1.9")
USERNAME = os.environ.get("VOX_VERBI_USER", "wildu")
PASSWORD = os.environ.get("VOX_VERBI_PASS", "4123")

BASE_LOCAL = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
llama_service = open(os.path.join(BASE_LOCAL, "systemd", "llama-server.service"), encoding="utf-8").read()
asistente_service = open(os.path.join(BASE_LOCAL, "systemd", "asistente-vivo.service"), encoding="utf-8").read()

client = paramiko.SSHClient()
client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
client.connect(HOSTNAME, username=USERNAME, password=PASSWORD, timeout=15)

try:
    sftp = client.open_sftp()
    with sftp.file("/home/wildu/llama-server.service", "w") as f:
        f.write(llama_service.replace("\r\n", "\n"))
    with sftp.file("/home/wildu/asistente-vivo.service", "w") as f:
        f.write(asistente_service.replace("\r\n", "\n"))
    sftp.close()

    cmd = (
        f"echo '{PASSWORD}' | sudo -S bash -c '"
        "mv /home/wildu/llama-server.service /etc/systemd/system/llama-server.service && "
        "mv /home/wildu/asistente-vivo.service /etc/systemd/system/asistente-vivo.service && "
        "systemctl daemon-reload && "
        "systemctl enable llama-server.service asistente-vivo.service && "
        "systemctl restart llama-server.service"
        "'"
    )
    _, stdout, stderr = client.exec_command(cmd)
    out = stdout.read().decode("utf-8", errors="replace")
    err = stderr.read().decode("utf-8", errors="replace")
    if out:
        print("STDOUT:\n" + out)
    if err:
        print("STDERR:\n" + err)
finally:
    client.close()
