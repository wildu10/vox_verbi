import sys
sys.path.insert(0, "/home/wildu/llama.cpp")
from vox_verbi.llm_catequist import LLMCatequist
from vox_verbi.rag_engine import RAGEngine

rag = RAGEngine()
llm = LLMCatequist()

q = "Vox Verbi, háblame durante un minuto sobre la Trinidad."
res = rag.search(q, top_k=2)
cands = res.get("results", [])
sys_c, user_c, max_tok = llm.build_prompt(q, cands, 60)
print("=== USER CONTENT ===")
print(user_c)
print(f"Max tokens configured: {max_tok}")
print("=== GENERATING ===")
text = llm.generate(q, cands, 60)
print("=== GENERATED TEXT ===")
print(text)
print("Word count:", len(text.split()))
print("Ends with period:", text.endswith(('.', '!', '?')))
print("Has 'En definitiva,':", "En definitiva," in text or "en definitiva," in text)
