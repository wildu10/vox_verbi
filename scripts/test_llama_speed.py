import time
import requests

t0 = time.time()
payload = {
    "messages": [
        {"role": "user", "content": "Explica brevemente la gracia santificante."}
    ],
    "max_tokens": 80
}

res = requests.post("http://127.0.0.1:8080/v1/chat/completions", json=payload, timeout=10)
t1 = time.time()

print(f"Status: {res.status_code}")
print(f"Total time: {t1 - t0:.3f} s")
data = res.json()
print("Response:\n", data["choices"][0]["message"]["content"])
print("Usage:", data.get("usage"))
