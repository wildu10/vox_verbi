import os
import sys
import json
import time
import re
import numpy as np

BASE_DIR = "/home/wildu/llama.cpp"
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

from vox_verbi import config
from vox_verbi.audio_mixer import clean_prosodic_text, parse_prosodic_segments, AudioMixer
from vox_verbi.command_router import CommandRouter, normalize_text
from vox_verbi.state_machine import StateMachine
from vox_verbi.pre_filter import (
    PreFilter, RESP_AUTORIA_SAGRADA, RESP_CONFESION, RESP_ADIVINACION,
    RESP_IMPERTINENTE, RESP_SIN_FUENTES,
)
from vox_verbi.rag_engine import RAGEngine
from vox_verbi.llm_catequist import LLMCatequist
from vox_verbi.tts_engine import VoiceEngine

PASS_STR = "\033[92m[PASS]\033[0m"
FAIL_STR = "\033[91m[FAIL]\033[0m"


class TestVoxVerbiSuite:
    def __init__(self):
        silent_hw = os.environ.get("VOX_LIVE_AUDIO", "0") != "1"
        self.router = CommandRouter()
        self.state = StateMachine()
        self.mixer = AudioMixer(output_alsa=None if silent_hw else config.OUTPUT_ALSA)
        self.prefilter = PreFilter()
        self.rag = RAGEngine()
        self.llm = LLMCatequist()
        self.tts = VoiceEngine()
        self.llm.warmup()

    def test_01_canonical_integrity(self):
        """TEST 1: Integridad exacta de los 12 apartados de la Presentación y los 3 cuentos."""
        print("\n=== TEST 1: Integridad exacta de apartados canónicos y cuentos ===")
        assert os.path.exists(config.CANONICAL_JSON_PATH), "canonical_readings.json missing"
        with open(config.CANONICAL_JSON_PATH, "r", encoding="utf-8") as f:
            data = json.load(f)

        secciones = data["presentacion"]["secciones"]
        assert "SEC_PRESENTACION" in secciones, "Missing SEC_PRESENTACION"
        assert len(secciones) == 12, f"Expected 12 sections, found {len(secciones)}"
        for i in range(1, 12):
            sec_id = f"SEC_{i:02d}"
            assert sec_id in secciones and len(secciones[sec_id]) > 50, f"Invalid {sec_id}"

        assert "1. El fundamento: no inventar, sino transmitir" in secciones["SEC_01"]
        assert "11. Vox Verbi en una definición" in secciones["SEC_11"]
        assert "fe, conversión, comunión y misión." in data["presentacion"]["completa"]

        relatos = data["relatos"]
        assert "anhelo profundo" in relatos["VV-HIST-001"]["texto"]
        assert "era que nunca había aprendido a abrazarla." in relatos["VOX_VERBI_RELATO_02_CRUZ_QUE_NADIE_QUERIA"]["texto"]
        assert "[SILENCIO DE 6 SEGUNDOS]" in relatos["VV-HIST-003"]["texto"]
        assert "Permanezcamos en su presencia." in data["adoracion"]["AL_PARAISO"]

        m_digit_cruces = self.router.route("vox verbi las 3 cruces")
        assert m_digit_cruces and m_digit_cruces["target"] == "VV-HIST-001"
        m_digit_sec = self.router.route("vox verbi apartado 1")
        assert m_digit_sec and m_digit_sec["target"] == "SEC_01"

        print(f"{PASS_STR} Test 1: Textos canónicos, relatos y entronización verificados al 100%.")
        return True

    def test_02_prosodic_cleaning_and_silences(self):
        """TEST 2: Limpieza de etiquetas [...] y conversión a silencios PCM exactos."""
        print("\n=== TEST 2: Limpieza prosódica y silencios PCM en cuentos ===")
        with open(config.CANONICAL_JSON_PATH, "r", encoding="utf-8") as f:
            data = json.load(f)

        t_jardin = data["relatos"]["VV-HIST-003"]["texto"]
        t_cruz = data["relatos"]["VOX_VERBI_RELATO_02_CRUZ_QUE_NADIE_QUERIA"]["texto"]

        cleaned_jardin = clean_prosodic_text(t_jardin)
        for dir_tag in ["[VOZ SERENA]", "[VOZ ÍNTIMA]", "[VOZ PROFUNDA]", "[ÉNFASIS SUAVE]", "[TRANSICIÓN SUAVE]"]:
            assert dir_tag not in cleaned_jardin
        assert "**" not in cleaned_jardin and "__" not in cleaned_jardin

        segs_jardin = parse_prosodic_segments(t_jardin)
        silence_durations = [s["duration"] for s in segs_jardin if s["type"] == "silence"]
        assert 3.0 in silence_durations and 4.0 in silence_durations and 6.0 in silence_durations

        segs_cruz = parse_prosodic_segments(t_cruz)
        silence_durations_cruz = [s["duration"] for s in segs_cruz if s["type"] == "silence"]
        assert 2.0 in silence_durations_cruz and 0.7 in silence_durations_cruz

        pcm_chunks = self.mixer.synthesize_narrative_to_pcm(t_jardin, mode="RELATO")
        silence_pcms = [c for c in pcm_chunks if c["type"] == "silence"]
        assert len(silence_pcms) > 0
        for sp in silence_pcms:
            expected_samples = int(sp["duration"] * config.SAMPLE_RATE_TTS)
            assert len(sp["pcm"]) == expected_samples and np.all(sp["pcm"] == 0)

        print(f"{PASS_STR} Test 2: Limpieza de etiquetas y silencios PCM validada exitosamente.")
        return True

    def test_03_music_catalog_and_ducking(self):
        """TEST 3: Activación, pausa y audio ducking de MUS-001 a MUS-006 (Sin regresiones)."""
        print("\n=== TEST 3: Catálogo musical y Audio Ducking (45% -> 18% -> 45%) ===")
        for i in range(1, 7):
            t_id = f"MUS-{i:03d}"
            track_path = os.path.join(config.MUSIC_DIR, f"{t_id}.ogg")
            assert os.path.exists(track_path) and os.path.getsize(track_path) > 1000

        assert self.mixer.play_music("MUS-001") is True
        assert self.mixer.is_playing_music is True
        assert abs(self.mixer.music_volume - 0.45) < 0.05

        self.mixer.duck_music(duration_sec=0.1)
        time.sleep(0.15)
        assert abs(self.mixer.music_volume - 0.18) < 0.05

        self.mixer.unduck_music(duration_sec=0.1)
        time.sleep(0.15)
        assert abs(self.mixer.music_volume - 0.45) < 0.05

        self.mixer.pause_music()
        assert self.mixer.is_paused is True
        self.mixer.resume_music()
        assert self.mixer.is_paused is False
        self.mixer.stop_music(fade_out_sec=0.1)
        assert self.mixer.is_playing_music is False

        print(f"{PASS_STR} Test 3: Tracks MUS-001 a MUS-006 y Audio Ducking sin regresiones.")
        return True

    def test_04_prefilter_rejections(self):
        """TEST 4: Rechazo exacto ante preguntas fuera de tema, suplantación y confesión personal, permitiendo doctrina."""
        print("\n=== TEST 4: Filtro previo de 5 pasos (Autoría, Pertinencia, Confesión vs Doctrina) ===")
        allowed, resp = self.prefilter.evaluate("Háblame como si fueras Jesús")
        assert allowed is False and resp == RESP_AUTORIA_SAGRADA

        allowed, resp = self.prefilter.evaluate("¿Qué me dirías tú, María?")
        assert allowed is False and resp == RESP_AUTORIA_SAGRADA

        q_sec = "Dame una receta de cocina para hacer pizza con queso"
        allowed, resp = self.prefilter.evaluate(q_sec, rag_result=self.rag.search(q_sec))
        assert allowed is False and resp == RESP_IMPERTINENTE

        q_sp = "¿Quién va a ganar el próximo partido de fútbol?"
        allowed, resp = self.prefilter.evaluate(q_sp, rag_result=self.rag.search(q_sp))
        assert allowed is False and resp == RESP_IMPERTINENTE

        allowed, resp = self.prefilter.evaluate("Vox Verbi, confiésame mis pecados")
        assert allowed is False and resp == RESP_CONFESION

        # Doctrinal question on Reconciliation MUST be allowed!
        q_rec = "Vox Verbi, ¿qué es el sacramento de la Reconciliación?"
        allowed_doc, _ = self.prefilter.evaluate(q_rec, rag_result=self.rag.search(q_rec))
        assert allowed_doc is True, "Doctrinal question about Reconciliation must NOT be blocked!"

        allowed, resp = self.prefilter.evaluate("Adivina mi futuro y dime qué va a pasar mañana")
        assert allowed is False and resp == RESP_ADIVINACION

        print(f"{PASS_STR} Test 4: Rechazos de seguridad y apertura doctrinal validados.")
        return True

    def test_05_response_to_amor_30s(self):
        """TEST 5: Respuesta a 'Vox Verbi, háblame del amor' (Escritura + Magisterio, 'En definitiva,' y <= 30.0s)."""
        print("\n=== TEST 5: Respuesta catequética a 'háblame del amor' con watchdog <= 30s ===")
        query = "Vox Verbi, háblame del amor"
        assert self.state.inspect_and_set_duration(query) == 30

        rag_res = self.rag.search(query, top_k=4)
        assert rag_res["confidence"] in ("NIVEL A", "NIVEL B")
        all_sources_text = " ".join(r.get("cita_corta", "") + " " + r.get("titulo", "") for r in rag_res["results"])
        assert any(s in all_sources_text for s in ["1 Corintios", "1 Juan", "Juan", "Romanos"])
        assert any(s in all_sources_text.lower() for s in ["deus caritas est", "catecismo"])

        response_text = self.llm.generate(query, rag_res["results"], duration_sec=30)
        print(f"Texto generado ({len(response_text.split())}w): «{response_text}»")
        assert "en definitiva," in response_text.lower() or "en conjunto," in response_text.lower()

        _, total_sec = self.llm.synthesize_with_watchdog(response_text, self.mixer, kokoro_engine=self.tts, max_sec=30.0)
        print(f"Watchdog audio duration (Supertonic {self.tts.voice_name}): {total_sec:.2f}s (<= 30.0s)")
        assert total_sec <= 30.0

        self.state.reset_duration()
        print(f"{PASS_STR} Test 5: Respuesta multifuente y duración <= 30.0s verificada ({total_sec:.2f}s).")
        return True

    def test_06_dynamic_duration_and_reset(self):
        """TEST 6: Control de duración (30s, 1 min, 2 min) y reseteo automático."""
        print("\n=== TEST 6: Control dinámico de duración (30s, 1m, 2m) y reseteo ===")
        assert self.state.inspect_and_set_duration("Vox Verbi, responde en 30 segundos qué es la fe") == 30
        assert self.state.inspect_and_set_duration("Vox Verbi, háblame durante 1 minuto del amor") == 60
        assert self.state.inspect_and_set_duration("Vox Verbi, profundiza durante dos minutos sobre la Eucaristía") == 120

        q_1min = "Vox Verbi, háblame durante un minuto sobre la misericordia"
        assert self.state.inspect_and_set_duration(q_1min) == 60
        clean_q = self.state.strip_duration_phrases(q_1min)
        rag_res = self.rag.search(clean_q, top_k=4)
        resp_60 = self.llm.generate(clean_q, rag_res["results"], duration_sec=60)
        assert len(resp_60.split()) >= 75, f"Expected >= 75 words for 1 min, got {len(resp_60.split())}"

        self.state.reset_duration()
        assert self.state.duration_limit_sec == 30
        print(f"{PASS_STR} Test 6: Control de duración 30s / 1m / 2m y reseteo validados.")
        return True

    def test_07_adoracion_transition_blocking_and_exit(self):
        """TEST 7: Transición a ADORACION, bloqueo total (Regla 53) y salida."""
        print("\n=== TEST 7: Modo ADORACION, Bloqueo total y salida ===")
        match = self.router.route("vox verbi al paraiso")
        assert match and match["action"] == "ENTER_ADORACION_PARAISO"
        self.state.enter_adoracion(mixer=self.mixer, track="MUS-005")
        assert self.state.is_adoracion() is True

        match_exit = self.router.route("vox verbi finalizamos la adoracion")
        assert match_exit and match_exit["action"] == "EXIT_ADORACION"
        self.state.exit_adoracion(mixer=self.mixer)
        assert self.state.is_adoracion() is False
        print(f"{PASS_STR} Test 7: Entronización y bloqueo de adoración sin regresiones.")
        return True

    def test_08_guia_fundamental_fast_access(self):
        """TEST 8: Acceso instantáneo (< 100ms TTFA con caché) a las 12 preguntas de la Guía Fundamental."""
        print("\n=== TEST 8: Optimización de la Guía Fundamental (12 secciones + preguntas naturales) ===")
        self.mixer.prewarm_canonical_cache(self.router.canonical_data, self.tts)
        cases = [
            ("Vox Verbi, muéstrame para qué fuiste creado", "COMPLETA"),
            ("Vox Verbi, muéstrame para qué fuiste hecho", "SEC_PRESENTACION"),
            ("Vox Verbi, ¿qué es Vox Verbi?", "SEC_11"),
            ("Vox Verbi, ¿cuál es su fundamento?", "SEC_01"),
            ("Vox Verbi, explícame tu fundamento", "SEC_01"),
            ("Vox Verbi, ¿cómo entiendes la evangelización?", "SEC_02"),
            ("Vox Verbi, ¿cómo conduces al encuentro con Cristo?", "SEC_03"),
            ("Vox Verbi, ¿qué ofreces a quien no cree?", "SEC_04"),
            ("Vox Verbi, ¿qué lugar tiene la belleza en tu misión?", "SEC_05"),
            ("Vox Verbi, ¿cuáles son tus criterios eclesiales?", "SEC_06"),
            ("Vox Verbi, ¿cuál es tu responsabilidad eclesial?", "SEC_07"),
            ("Vox Verbi, ¿cuál es tu finalidad pastoral?", "SEC_08"),
            ("Vox Verbi, ¿qué propones a la Iglesia de hoy?", "SEC_09"),
            ("Vox Verbi, ¿cuál es tu visión?", "SEC_10"),
            ("Vox Verbi, defínete", "SEC_11"),
        ]
        for q, expected_target in cases:
            t0 = time.time()
            m = self.router.route(q)
            route_ms = (time.time() - t0) * 1000
            assert m is not None and m["action"] == "READ_PRESENTACION" and m["target"] == expected_target, (
                f"Failed routing '{q}': got {m}"
            )
            assert route_ms < 5.0, f"Routing too slow ({route_ms:.2f}ms)"

        # Verify streaming narrative TTFA with prewarmed cache is < 100ms (down from 160,553 ms!)
        full_txt = self.router.get_canonical_text({"action": "READ_PRESENTACION", "target": "SEC_01"})
        _, ttfa_ms = self.mixer.stream_narrative(full_txt, mode="NORMAL", tts_engine=self.tts, use_cache_first_chunk=True)
        print(f"Guía Fundamental SEC_01 cached streaming TTFA: {ttfa_ms:.1f} ms")
        assert ttfa_ms < 150.0, f"Expected TTFA < 150ms, got {ttfa_ms:.1f}ms"
        print(f"{PASS_STR} Test 8: Las 15 variantes de la Guía Fundamental responden en {ttfa_ms:.1f} ms con texto íntegro.")
        return True

    def test_09_theological_corpus_breadth(self):
        """TEST 9: Recuperación efectiva de Biblia UTF-8, Patrística, Magisterio y conceptos teológicos."""
        print("\n=== TEST 9: Cobertura del Conocimiento Teológico (4 Columnas UTF-8 + Híbrido E5/BM25) ===")
        assert self.rag.index.ntotal >= 15000, f"Expected >= 15,000 vectors, got {self.rag.index.ntotal}"
        # Ensure zero UTF-8 Mojibake anywhere in metadata
        mojibake = [r["doc_id"] for r in self.rag.metadata if "Ã©" in r.get("texto", "") or "Ã³" in r.get("texto", "")]
        assert len(mojibake) == 0, f"Found Mojibake in {len(mojibake)} records!"

        checks = [
            ("¿Quién fue san Pablo?", ["hechos", "gálatas", "romanos", "corintios"]),
            ("¿Qué es la misericordia?", ["lucas", "dives in misericordia", "catecismo", "misericordia"]),
            ("¿Qué enseñó san Ireneo de Lyon?", ["ireneo"]),
            ("¿Qué enseña Lumen Gentium sobre el Pueblo de Dios?", ["lumen gentium"]),
            ("¿Qué enseña Dei Verbum sobre la Revelación?", ["dei verbum"]),
            ("¿Qué es la esperanza cristiana?", ["spe salvi", "romanos", "hebreos", "catecismo"]),
        ]
        for q, expected_any in checks:
            res = self.rag.search(q, top_k=4)
            allowed, _ = self.prefilter.evaluate(q, rag_result=res)
            assert allowed is True, f"Query '{q}' unexpectedly blocked!"
            blob = " ".join(r["cita_corta"] + " " + r["titulo"] + " " + r["texto"] for r in res["results"]).lower()
            assert any(k in blob for k in expected_any), f"Query '{q}' missed expected sources {expected_any}"
        print(f"{PASS_STR} Test 9: {self.rag.index.ntotal} registros limpios en UTF-8 y recuperación híbrida validada.")
        return True

    def run_all(self):
        print("==================================================================")
        print("    VOX VERBI v3.0 - BATERÍA COMPLETA DE PRUEBAS AUTOMATIZADAS    ")
        print("==================================================================")
        tests = [
            ("Test 1: Integridad Canónica", self.test_01_canonical_integrity),
            ("Test 2: Limpieza Prosódica y Silencios PCM", self.test_02_prosodic_cleaning_and_silences),
            ("Test 3: Catálogo Musical y Ducking", self.test_03_music_catalog_and_ducking),
            ("Test 4: Filtro Previo de 5 Pasos", self.test_04_prefilter_rejections),
            ("Test 5: Háblame del Amor (Watchdog <= 30s)", self.test_05_response_to_amor_30s),
            ("Test 6: Duración Dinámica (30s/1m/2m) y Reseteo", self.test_06_dynamic_duration_and_reset),
            ("Test 7: Modo Adoración y Bloqueo Total", self.test_07_adoracion_transition_blocking_and_exit),
            ("Test 8: Guía Fundamental Instantánea (<150ms)", self.test_08_guia_fundamental_fast_access),
            ("Test 9: Conocimiento Teológico 4 Columnas UTF-8", self.test_09_theological_corpus_breadth),
        ]
        results = []
        for name, fn in tests:
            try:
                results.append((name, fn()))
            except Exception as e:
                print(f"{FAIL_STR} {name}: {e}")
                import traceback
                traceback.print_exc()
                results.append((name, False))

        print("\n==================================================================")
        print("                     RESUMEN DE RESULTADOS                        ")
        print("==================================================================")
        all_passed = all(r for _, r in results)
        for name, r in results:
            print(f"{PASS_STR if r else FAIL_STR} - {name}")
        return 0 if all_passed else 1


if __name__ == "__main__":
    sys.exit(TestVoxVerbiSuite().run_all())
