import sys
sys.path.insert(0, "/home/wildu/llama.cpp")

from vox_verbi.pre_filter import PreFilter
from vox_verbi.rag_engine import RAGEngine
from vox_verbi.command_router import normalize_text, match_and_strip_wake_word

pf = PreFilter()
rag = RAGEngine()

queries = [
    "Vox Verbi, quiero confesar mis pecados contigo.",
    "Vox Verbi, quien ganó el partido de fútbol ayer.",
    "Busverby, quien ganó el partido de fútbol ayer.",
    "Vos ver mi, ¿Quién es Jesús?",
    "Vox Verbi, ¿qué enseña de i verbum sobre la revelación?",
    "Busverby, al paraíso",
    "BusBerby. Llevamos..."
]

for q in queries:
    print("=" * 60)
    print("QUERY:", q)
    norm = normalize_text(q)
    matched_wake, rest = match_and_strip_wake_word(q)
    print(f"normalize_text: '{norm}'")
    print(f"match_wake: {matched_wake}, rest: '{rest}'")
    
    rag_res = rag.search(q)
    print(f"RAG conf: {rag_res.get('confidence')}, min_dist: {rag_res.get('min_distance'):.4f}")
    cands = rag_res.get("results", [])
    for i, c in enumerate(cands):
        print(f"  F{i+1}: [{c.get('columna')}] {c.get('cita_corta')} - {c.get('titulo')}")
        
    allowed, resp = pf.evaluate(q, rag_res)
    print(f"PreFilter: allowed={allowed}, resp='{resp}'")
