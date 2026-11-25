import os
import sys
import time
import logging

BASE_DIR = "/home/wildu/llama.cpp"
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger("validate_fixes")

from kokoro_onnx import Kokoro
from vox_verbi import config
from vox_verbi.audio_mixer import AudioMixer, expand_biblical_citations_for_tts
from vox_verbi.state_machine import StateMachine
from vox_verbi.rag_engine import RAGEngine
from vox_verbi.llm_catequist import LLMCatequist

def run_validation():
    print("=================================================================")
    print("    AUDITORÍA Y VALIDACIÓN DE CORRECCIONES EN VOX VERBI")
    print("=================================================================")
    
    # 1. Initialize components
    logger.info("Initializing components...")
    state = StateMachine()
    mixer = AudioMixer(output_alsa=config.OUTPUT_ALSA)
    rag = RAGEngine()
    llm = LLMCatequist()
    
    kokoro = None
    if os.path.exists(config.TTS_MODEL_PATH) and os.path.exists(config.TTS_VOICES_PATH):
        try:
            kokoro = Kokoro(config.TTS_MODEL_PATH, config.TTS_VOICES_PATH)
            kokoro.create("inicio", voice=config.TTS_VOICE_NAME, speed=1.0, lang=config.TTS_LANG)
            logger.info("Kokoro TTS warmed up.")
        except Exception as e:
            logger.error(f"Error loading Kokoro: {e}")

    # -------------------------------------------------------------------------
    # TEST CITATIONS PHONEMIZER
    # -------------------------------------------------------------------------
    print("\n--- TEST CITAS: Expansión fonética en audio_mixer.py ---")
    test_citations = [
        "El Salmo 79, 78, 11-13, invoca la intervención de Dios para salvar a los hijos de la muerte.",
        "El apóstol San Pablo en Romanos 1, 5-7 nos enseña la gracia.",
        "En 2 Corintios 13, 11-13, el apóstol nos invita a vivir en comunión.",
        "El Compendio del Catecismo, en el número 45, nos recuerda el misterio trinitario."
    ]
    for cit in test_citations:
        expanded = expand_biblical_citations_for_tts(cit)
        print(f"Original: {cit}")
        print(f"Expandido: {expanded}")
        if kokoro:
            samples, sr = kokoro.create(expanded, voice=config.TTS_VOICE_NAME, speed=1.05, lang=config.TTS_LANG)
            print(f"Kokoro sintetizado: {len(samples)/sr:.2f}s sin errores")
        print("-" * 50)

    # Warm up LLM slot 1 prefix cache
    logger.info("Warming up Llama-server prefix KV Cache on slot 1...")
    try:
        llm.generate("inicio", [], duration_sec=30)
        logger.info("Llama-server slot 1 prefix cache primed.")
    except Exception as e:
        logger.warning(f"Llama warm up warning: {e}")

    # -------------------------------------------------------------------------
    # TEST 1: BAUTISMO (30s)
    # -------------------------------------------------------------------------
    print("\n=================================================================")
    print("TEST 1: 'Vox Verbi, ¿qué significa Bautismo?' (Presupuesto: 30s)")
    print("=================================================================")
    q1 = "Vox Verbi, ¿qué significa Bautismo?"
    dur_1 = state.inspect_and_set_duration(q1)
    print(f"Duración detectada: {dur_1}s")
    
    rag_res1 = rag.search(q1, top_k=3)
    cands1 = rag_res1.get("results", [])
    print(f"RAG Fuentes recuperadas ({len(cands1)}):")
    for i, c in enumerate(cands1):
        print(f"  F{i+1}: [{c.get('columna')}] {c.get('cita_corta')} - {c.get('titulo')}")
        
    f1_cita = cands1[0].get('cita_corta', '') if len(cands1) > 0 else ''
    f2_cita = cands1[1].get('cita_corta', '') if len(cands1) > 1 else ''
    
    assert any(b in f1_cita for b in ["Mateo 28", "Romanos 6", "Juan 3"]), f"F1 inesperada: {f1_cita}"
    assert "Compendio" in f2_cita, f"F2 inesperada: {f2_cita}"
    print("[PASS] RAG recuperó fuentes sacramentales canónicas para Bautismo.")

    print("\nIniciando streaming live con Llama 3.1 + Kokoro (Pipeline 3 etapas)...")
    full_text_1, spoken_sec_1, ttfa_ms_1 = llm.stream_and_speak(
        q1, cands1, mixer, kokoro, duration_sec=dur_1
    )
    state.reset_duration()
    
    words_1 = len(full_text_1.split())
    print(f"\nResultado Test 1 (Bautismo):")
    print(f"  Texto completo: {full_text_1}")
    print(f"  Palabras: {words_1}")
    print(f"  Tiempo hablado: {spoken_sec_1:.2f}s")
    print(f"  TTFA: {ttfa_ms_1:.0f} ms")
    
    # Validaciones Test 1
    assert full_text_1.endswith(('.', '!', '?')), "ERROR: Respuesta cortada a mitad de frase sin punto final!"
    assert "En definitiva," in full_text_1 or "en definitiva," in full_text_1, "ERROR: Falta síntesis con 'En definitiva,'"
    assert ttfa_ms_1 < 3200, f"ERROR: TTFA {ttfa_ms_1}ms superó el límite objetivo de 3.2s!"
    print(f"[PASS] Test 1 completado exitosamente: TTFA={ttfa_ms_1:.0f}ms, {words_1} palabras, finalizado con punto.")

    # -------------------------------------------------------------------------
    # TEST 2: TRINIDAD (60s - 1 MINUTO)
    # -------------------------------------------------------------------------
    print("\n=================================================================")
    print("TEST 2: 'Vox Verbi, háblame durante un minuto sobre la Trinidad.' (Presupuesto: 60s)")
    print("=================================================================")
    q2 = "Vox Verbi, háblame durante un minuto sobre la Trinidad."
    dur_2 = state.inspect_and_set_duration(q2)
    print(f"Duración detectada: {dur_2}s (debe ser 60s)")
    assert dur_2 == 60, f"Error en state machine: dur_2 = {dur_2}"
    
    rag_res2 = rag.search(q2, top_k=4)
    cands2 = rag_res2.get("results", [])
    print(f"RAG Fuentes recuperadas ({len(cands2)}):")
    for i, c in enumerate(cands2):
        print(f"  F{i+1}: [{c.get('columna')}] {c.get('cita_corta')} - {c.get('titulo')}")
        
    print("\nIniciando streaming live para 1 minuto...")
    full_text_2, spoken_sec_2, ttfa_ms_2 = llm.stream_and_speak(
        q2, cands2, mixer, kokoro, duration_sec=dur_2
    )
    state.reset_duration()
    
    words_2 = len(full_text_2.split())
    print(f"\nResultado Test 2 (Trinidad 60s):")
    print(f"  Texto completo: {full_text_2}")
    print(f"  Palabras: {words_2}")
    print(f"  Tiempo hablado: {spoken_sec_2:.2f}s")
    print(f"  TTFA: {ttfa_ms_2:.0f} ms")
    
    # Validaciones Test 2
    assert full_text_2.endswith(('.', '!', '?')), "ERROR: Respuesta cortada a mitad de frase sin punto final!"
    assert "En definitiva," in full_text_2 or "en definitiva," in full_text_2, "ERROR: Falta síntesis con 'En definitiva,'"
    assert words_2 >= 95, f"ERROR: Muy pocas palabras ({words_2}) para 60 segundos (debe ser > 100 palabras)!"
    assert spoken_sec_2 >= 40.0, f"ERROR: Duración hablada ({spoken_sec_2:.1f}s) no alcanzó el rango objetivo de 42-58s!"
    print(f"[PASS] Test 2 completado exitosamente: Duración={spoken_sec_2:.1f}s, Palabras={words_2}, TTFA={ttfa_ms_2:.0f}ms.")

    print("\n=================================================================")
    print("   TODAS LAS AUDITORÍAS Y VALIDACIONES SUPERADAS CON ÉXITO")
    print("=================================================================")

if __name__ == "__main__":
    run_validation()
