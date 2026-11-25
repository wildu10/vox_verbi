import os
import sys
import time
import subprocess

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

BASE_DIR = "/home/wildu/llama.cpp"
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

from vox_verbi import config
from vox_verbi.rag_engine import RAGEngine
from vox_verbi.llm_catequist import LLMCatequist
from vox_verbi.audio_mixer import AudioMixer
from vox_verbi.tts_engine import VoiceEngine


def run_benchmark():
    print("=" * 75)
    print("BENCHMARK EN VIVO - VOX VERBI v3.0 (GEMMA 4 E4B VULKAN + SUPERTONIC M5)")
    print("=" * 75)

    silent = os.environ.get("VOX_LIVE_AUDIO", "0") != "1"
    rag = RAGEngine()
    llm = LLMCatequist()
    mixer = AudioMixer(output_alsa=None if silent else config.OUTPUT_ALSA)
    tts = VoiceEngine()
    llm.warmup()

    print(f"\nTotal registros en base_vox_verbi.index: {rag.index.ntotal} | Voz: {tts.backend}:{tts.voice_name} ({tts.sample_rate}Hz)")

    queries = [
        "Vox Verbi, ¿qué enseña la Iglesia sobre la Eucaristía?",
        "Vox Verbi, ¿quién es el Espíritu Santo?",
        "Vox Verbi, ¿qué es la misericordia?",
        "Vox Verbi, ¿qué enseña Lumen Gentium sobre el Pueblo de Dios?",
    ]

    for i, q in enumerate(queries, 1):
        print(f"\n--- CONSULTA {i}: «{q}» ---")
        t0 = time.time()
        rag_res = rag.search(q, top_k=3)
        rag_ms = (time.time() - t0) * 1000
        records = rag_res.get("results", [])
        for r in records:
            print(f"  * [{r.get('columna')}] {r.get('cita_corta')} (score={r.get('score'):.3f})")

        final_text, spoken_sec, ttfa_ms = llm.stream_and_speak(q, records, mixer, tts, duration_sec=30)
        print(f"  RAG: {rag_ms:.1f}ms | TTFA: {ttfa_ms:.0f}ms | Audio: {spoken_sec:.2f}s | Palabras: {len(final_text.split())}")
        print(f"  Respuesta: «{final_text}»")


if __name__ == "__main__":
    run_benchmark()
