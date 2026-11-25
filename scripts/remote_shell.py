import os
import sys
import paramiko

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

HOST = os.environ.get("VOX_VERBI_HOST", "192.168.1.11")
USER = os.environ.get("VOX_VERBI_USER", "wildu")
PASS = os.environ.get("VOX_VERBI_PASS", "4123")


def exec_remote(cmd):
    c = paramiko.SSHClient()
    c.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    c.connect(HOST, username=USER, password=PASS, timeout=15)
    if cmd.strip().startswith("sudo"):
        full_cmd = f"echo '{PASS}' | sudo -S {cmd[5:]}"
    else:
        full_cmd = cmd
    _, stdout, stderr = c.exec_command(full_cmd)
    out = stdout.read().decode("utf-8", errors="replace")
    err = stderr.read().decode("utf-8", errors="replace")
    c.close()
    return out, err


if __name__ == "__main__":
    if len(sys.argv) > 1:
        out, err = exec_remote(sys.argv[1])
        if out:
            print("STDOUT:\n", out)
        if err:
            print("STDERR:\n", err)
