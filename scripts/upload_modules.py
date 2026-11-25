import os
import paramiko

HOSTNAME = os.environ.get("VOX_VERBI_HOST", "192.168.1.9")
USERNAME = os.environ.get("VOX_VERBI_USER", "wildu")
PASSWORD = os.environ.get("VOX_VERBI_PASS", "4123")

SCRATCH_DIR = r"C:\Users\wildu\.gemini\antigravity-ide\scratch\vox_verbi"
REMOTE_PKG_DIR = "/home/wildu/llama.cpp/vox_verbi"

files_to_upload = [
    ("__init__.py", f"{REMOTE_PKG_DIR}/__init__.py"),
    ("config.py", f"{REMOTE_PKG_DIR}/config.py"),
    ("hardware_hooks.py", f"{REMOTE_PKG_DIR}/hardware_hooks.py"),
    ("tts_engine.py", f"{REMOTE_PKG_DIR}/tts_engine.py"),
    ("audio_mixer.py", f"{REMOTE_PKG_DIR}/audio_mixer.py"),
    ("state_machine.py", f"{REMOTE_PKG_DIR}/state_machine.py"),
    ("command_router.py", f"{REMOTE_PKG_DIR}/command_router.py"),
    ("pre_filter.py", f"{REMOTE_PKG_DIR}/pre_filter.py"),
    ("rag_engine.py", f"{REMOTE_PKG_DIR}/rag_engine.py"),
    ("llm_catequist.py", f"{REMOTE_PKG_DIR}/llm_catequist.py"),
    ("main.py", f"{REMOTE_PKG_DIR}/main.py"),
    ("build_whitelist_rag.py", f"{REMOTE_PKG_DIR}/scripts/build_whitelist_rag.py"),
    ("test_vox_verbi_suite.py", f"{REMOTE_PKG_DIR}/scripts/test_vox_verbi_suite.py"),
    ("test_wake_and_stt.py", f"{REMOTE_PKG_DIR}/scripts/test_wake_and_stt.py"),
    ("run_live_benchmark.py", f"{REMOTE_PKG_DIR}/scripts/run_live_benchmark.py"),
]

client = paramiko.SSHClient()
client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
client.connect(HOSTNAME, username=USERNAME, password=PASSWORD, timeout=15)

try:
    sftp = client.open_sftp()
    for local_name, remote_dest in files_to_upload:
        local_p = os.path.join(SCRATCH_DIR, local_name)
        if not os.path.exists(local_p):
            local_p = os.path.join(SCRATCH_DIR, "scripts", local_name)
        sftp.put(local_p, remote_dest)
        print(f"Uploaded {local_name} -> {remote_dest}")
    sftp.close()
finally:
    client.close()
print("All module files successfully deployed!")
