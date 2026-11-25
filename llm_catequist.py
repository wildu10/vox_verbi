import json
import logging
import re
import time
import queue
import threading
import requests
import numpy as np
from typing import List, Dict, Any

from . import config
from .audio_mixer import int_to_spanish, expand_biblical_citations_for_tts

logger = logging.getLogger("vox_verbi.llm_catequist")

# -------------------------------------------------------------------------
# STATIC & INVARIABLE SYSTEM PROMPT (KV Cache Prefix Persistent in Slot 0)
# -------------------------------------------------------------------------
STATIC_SYSTEM_PROMPT = """Eres Vox Verbi («Voz de la Palabra»), herramienta de apoyo catequético católico al servicio del predicador, fundamentada exclusivamente en las fuentes autorizadas de la Sagrada Escritura, el Magisterio de la Iglesia, el Catecismo y la Tradición Patrística.
Hablas con serenidad contemplativa, fidelidad doctrinal, caridad pastoral y claridad pedagógica.
Reglas obligatorias de formulación oral:
1. Discurso hablado limpio: responde directamente sin saludos ni preámbulos, en prosa continua y fluida. Prohibido usar corchetes, paréntesis, viñetas, asteriscos o abreviaturas como n., cap. o 1 Jn. Escribe siempre las referencias en lenguaje hablado natural («En el Evangelio según San Lucas, capítulo quince...», «En la encíclica Deus Caritas Est, numeral uno...», «El Catecismo de la Iglesia Católica, en el numeral mil ochocientos veintidós...»).
2. Prohibición de suplantación sagrada: habla siempre como testigo y catequista en tercera persona («Jesús enseña», «San Pablo afirma», «Benedicto Dieciséis explica»). Jamás hables en primera persona como Dios, Jesucristo, la Virgen María ni los santos.
3. Distinción fiel de autoridad y no invención: utiliza únicamente las FUENTES suministradas sin inventar citas ni doctrinas. Distingue la Palabra de Dios y la enseñanza del Magisterio frente a la interpretación patrística («San Ireneo interpreta...», «San Agustín reflexiona...»). Si la pregunta atribuye por error una idea falsa a una fuente, corrígelo con respeto.
4. Enseñanza e integración: no recites citas aisladas; explica con hondura qué significa cada fuente para la pregunta planteada y cómo se iluminan mutuamente. Si el usuario pregunta explícitamente por un documento magisterial o conciliar, inicia la primera oración citando y explicando ese documento.
5. Síntesis pedagógica final y control del tiempo: cierra SIEMPRE la última oración con «En definitiva,» reuniendo la enseñanza central, respeta estrictamente el presupuesto de palabras indicado y guarda silencio al concluir."""


def format_spoken_source(rec: Dict[str, Any]) -> str:
    cita = rec.get("cita_corta", "").strip()
    c_lower = cita.lower()

    mag_docs = [
        ("dei verbum", "la constitución Dei Verbum"),
        ("sacrosanctum", "la constitución Sacrosanctum Concilium"),
        ("lumen gentium", "la constitución Lumen Gentium"),
        ("gaudium", "la constitución pastoral Gaudium et Spes"),
        ("deus caritas", "la encíclica Deus Caritas Est de Benedicto Dieciséis"),
        ("spe salvi", "la encíclica Spe Salvi de Benedicto Dieciséis"),
        ("lumen fidei", "la encíclica Lumen Fidei"),
        ("dives in misericordia", "la encíclica Dives in Misericordia de San Juan Pablo Segundo"),
        ("ecclesia de eucharistia", "la encíclica Ecclesia de Eucharistia"),
        ("fides et ratio", "la encíclica Fides et Ratio"),
        ("evangelii gaudium", "la exhortación Evangelii Gaudium"),
        ("mysterium fidei", "la encíclica Mysterium Fidei de San Pablo Sexto"),
    ]
    for key, label in mag_docs:
        if key in c_lower:
            m = re.search(r"n\.\s*(\d+)", cita)
            num_str = f", numeral {int_to_spanish(int(m.group(1)))}" if m else ""
            return f"{label}{num_str}"

    if "compendio" in c_lower:
        m = re.search(r"(?:n\.|numeral|#)?\s*(\d+)", cita)
        num_str = f", numeral {int_to_spanish(int(m.group(1)))}" if m else ""
        return f"el Compendio del Catecismo{num_str}"

    if "catecismo" in c_lower:
        m = re.search(r"(?:n\.|numeral|#)?\s*(\d+)", cita)
        num_str = f", numeral {int_to_spanish(int(m.group(1)))}" if m else ""
        return f"el Catecismo de la Iglesia Católica{num_str}"

    if int(rec.get("nivel_autoridad", 1)) >= 4:
        return expand_biblical_citations_for_tts(cita)

    m_cap = re.search(r"(\d+)\s*[,:]", cita)
    cap_str = f", capítulo {int_to_spanish(int(m_cap.group(1)))}" if m_cap else ""

    nt_ot_books = [
        ("hebreos", "la Carta a los Hebreos"),
        ("hechos", "el libro de los Hechos de los Apóstoles"),
        ("romanos", "la Carta de San Pablo a los Romanos"),
        ("1 corintios", "la Primera Carta de San Pablo a los Corintios"),
        ("2 corintios", "la Segunda Carta de San Pablo a los Corintios"),
        ("corintios", "la Carta de San Pablo a los Corintios"),
        ("gálatas", "la Carta de San Pablo a los Gálatas"),
        ("galatas", "la Carta de San Pablo a los Gálatas"),
        ("efesios", "la Carta de San Pablo a los Efesios"),
        ("filipenses", "la Carta de San Pablo a los Filipenses"),
        ("colosenses", "la Carta de San Pablo a los Colosenses"),
        ("tesalonicenses", "la Carta a los Tesalonicenses"),
        ("timoteo", "la Carta a Timoteo"),
        ("tito", "la Carta a Tito"),
        ("filemón", "la Carta a Filemón"),
        ("santiago", "la Carta de Santiago"),
        ("1 pedro", "la Primera Carta de San Pedro"),
        ("2 pedro", "la Segunda Carta de San Pedro"),
        ("pedro", "la Carta de San Pedro"),
        ("1 juan", "la Primera Carta de San Juan"),
        ("2 juan", "la Segunda Carta de San Juan"),
        ("3 juan", "la Tercera Carta de San Juan"),
        ("juan", "el Evangelio según San Juan"),
        ("mateo", "el Evangelio según San Mateo"),
        ("marcos", "el Evangelio según San Marcos"),
        ("lucas", "el Evangelio según San Lucas"),
        ("apocalipsis", "el libro del Apocalipsis"),
        ("génesis", "el libro del Génesis"),
        ("genesis", "el libro del Génesis"),
        ("éxodo", "el libro del Éxodo"),
        ("exodo", "el libro del Éxodo"),
        ("salmo", "el libro de los Salmos"),
        ("isaías", "el profeta Isaías"),
        ("isaias", "el profeta Isaías"),
        ("jeremías", "el profeta Jeremías"),
        ("ezequiel", "el profeta Ezequiel"),
    ]
    for bkey, blabel in nt_ot_books:
        if bkey in c_lower:
            return f"{blabel}{cap_str}"

    return expand_biblical_citations_for_tts(cita) or "la Sagrada Escritura"


def get_tokens_and_words_budget(duration_sec: int) -> tuple[int, str, float]:
    """
    Returns (max_tokens, prompt_instruction, max_audio_sec) calibrated for the ~2.20 w/s contemplative baritone voice.
    """
    if duration_sec <= 30:
        return (
            180,
            "Extensión estricta: responde en EXACTAMENTE 3 oraciones breves (de 35 a 45 palabras en total; máximo 15 palabras por oración). Primera oración cita y explica la fuente 1; segunda oración integra la fuente 2; tercera oración cierra obligatoriamente iniciando con «En definitiva,».",
            30.0,
        )
    elif duration_sec <= 60:
        return (
            270,
            "Extensión obligatoria: entre 110 y 128 palabras en discurso continuo (~50 a 58 segundos hablados), explicando e integrando las fuentes con profundidad pastoral y cerrando la última oración con «En definitiva,».",
            62.0,
        )
    elif duration_sec <= 120:
        return (
            500,
            "Extensión obligatoria: entre 225 y 255 palabras en desarrollo teológico profundo (~100 a 116 segundos hablados para dos minutos), exponiendo el contexto bíblico, la doctrina magisterial, la riqueza espiritual y cerrando la última oración con «En definitiva,».",
            124.0,
        )
    else:
        budget = config.DURATION_BUDGETS.get(duration_sec, config.DURATION_BUDGETS[180])
        tw = budget["max_words"]
        return (
            budget["max_tokens"],
            f"Extensión obligatoria: alrededor de {tw - 20} a {tw} palabras habladas para {duration_sec} segundos, integrando todas las fuentes y cerrando con «En definitiva,».",
            budget["max_sec"],
        )


class LLMCatequist:
    """Client for Gemma 4 E4B Instruct on llama-server Vulkan with 3-stage streaming & audio duration watchdog."""

    def __init__(self, server_url=config.LLAMA_SERVER_URL):
        self.server_url = server_url
        self.model_name = getattr(config, "LLAMA_MODEL_NAME", "gemma-4-E4B-it-Q4_K_M.gguf")
        self._active_stop_event = None
        self._active_http_resp = None

    def abort_stream(self):
        """Immediately aborts any active LLM streaming & TTS synthesis pipeline."""
        ev = self._active_stop_event
        if ev is not None:
            ev.set()
        resp = self._active_http_resp
        if resp is not None:
            try:
                resp.close()
            except Exception:
                pass

    def warmup(self) -> float:
        logger.info("Initializing silent KV Cache warmup on llama-server (slot 0)...")
        t0 = time.time()
        payload = {
            "model": self.model_name,
            "messages": [
                {"role": "system", "content": STATIC_SYSTEM_PROMPT},
                {"role": "user", "content": "FUENTES:\n1. "},
            ],
            "max_tokens": 1,
            "temperature": 0.1,
            "cache_prompt": True,
            "id_slot": 0,
            "stream": False,
            "chat_template_kwargs": {"enable_thinking": False},
        }
        elapsed_ms = 0.0
        try:
            resp = requests.post(f"{self.server_url}/v1/chat/completions", json=payload, timeout=20)
            elapsed_ms = (time.time() - t0) * 1000
            if resp.status_code == 200:
                logger.info(f"KV Cache warmup completed in {elapsed_ms:.1f} ms (slot 0 cached in Vulkan VRAM).")
            else:
                logger.warning(f"KV Cache warmup status {resp.status_code}: {resp.text}")
        except Exception as e:
            logger.warning(f"Could not complete KV Cache warmup: {e}")
        return elapsed_ms

    def build_prompt(self, user_query: str, rag_records: List[Dict[str, Any]], duration_sec: int = 30) -> tuple[str, str, int]:
        q_lower = user_query.lower()
        mag_cues = (
            "dei verbum", "dei verb", "de verb", "berbun", "lumen gentium", "lumen gencium", "lumen yentium",
            "sacrosanctum", "sacro sanctum", "concilium", "gaudium et spes", "gaudium",
            "deus caritas", "uscaritas", "spe salvi", "espe salvi",
            "lumen fidei", "dives in misericordia", "ecclesia de eucharistia", "ecclesia de eucaristia",
            "fides et ratio", "fides et racio", "evangelii gaudium", "mysterium fidei", "catecismo", "compendio",
        )
        asks_magisterium = any(cue in q_lower for cue in mag_cues)

        records = list(rag_records)
        if asks_magisterium and len(records) >= 2:
            if records[0].get("nivel_autoridad", 1) == 1 and records[1].get("nivel_autoridad", 2) in (2, 3):
                records[0], records[1] = records[1], records[0]

        # Scale number of sources and snippet richness with requested duration!
        max_sources = 2 if duration_sec <= 30 else (3 if duration_sec <= 60 else min(5, len(records)))
        words_per_source = 45 if duration_sec <= 30 else (75 if duration_sec <= 60 else 95)

        source_lines = []
        for idx, rec in enumerate(records[:max_sources], 1):
            col = rec.get("columna", "Fuente")
            if int(rec.get("nivel_autoridad", 1)) >= 4:
                col_tag = "Tradición Patrística - interpretación patrística"
            else:
                col_tag = col
            spoken_ref = format_spoken_source(rec)
            raw_txt = re.sub(r"^\d+[\.\s]+", "", rec.get("texto", "")).strip()
            snip = " ".join(raw_txt.split()[:words_per_source])
            source_lines.append(f"{idx}. [{col_tag}] {spoken_ref}: {snip}")

        max_tokens, instruction, _ = get_tokens_and_words_budget(duration_sec)
        if asks_magisterium and source_lines:
            first_ref = format_spoken_source(records[0])
            instruction = f"Inicia la primera oración citando y explicando {first_ref}. " + instruction

        user_content = (
            "FUENTES:\n"
            + "\n".join(source_lines)
            + f"\n\nPREGUNTA: {user_query}\n"
            + instruction
        )
        return STATIC_SYSTEM_PROMPT, user_content, max_tokens

    def generate(self, user_query: str, rag_records: List[Dict[str, Any]], duration_sec: int = 30) -> str:
        system_content, user_content, max_tokens = self.build_prompt(user_query, rag_records, duration_sec)
        payload = {
            "model": self.model_name,
            "messages": [
                {"role": "system", "content": system_content},
                {"role": "user", "content": user_content},
            ],
            "max_tokens": max_tokens,
            "temperature": 0.2,
            "top_p": 0.9,
            "cache_prompt": True,
            "id_slot": 0,
            "stream": True,
            "chat_template_kwargs": {"enable_thinking": False},
        }
        full_text = ""
        try:
            resp = requests.post(f"{self.server_url}/v1/chat/completions", json=payload, stream=True, timeout=30)
            if resp.status_code == 200:
                for line in resp.iter_lines():
                    if not line:
                        continue
                    decoded = line.decode("utf-8", errors="replace")
                    if decoded.startswith("data: "):
                        data_str = decoded[6:]
                        if data_str == "[DONE]":
                            break
                        try:
                            delta = json.loads(data_str)["choices"][0].get("delta", {})
                            if delta.get("content"):
                                full_text += delta["content"]
                        except Exception:
                            pass
        except Exception as e:
            logger.error(f"Error connecting to llama-server: {e}")
            raise RuntimeError(f"Llama-server live generation failed: {e}")

        return self._clean_text(full_text)

    def stream_and_speak(self, user_query: str, rag_records: List[Dict[str, Any]], audio_mixer, kokoro_engine, duration_sec: int = 30) -> tuple[str, float, float]:
        """
        3-stage asynchronous pipeline with real-time oral duration watchdog:
          Stage 1 (LLM SSE Streamer): Streams tokens from Gemma 4 E4B on Vulkan, early-cuts clause 0 for fast TTFA.
          Stage 2 (TTS Synth Worker): Synthesizes clauses with Supertonic M5 and enforces max_audio_sec watchdog.
          Stage 3 (Audio Player Worker): Plays 44.1 kHz PCM seamlessly with ducking.
        """
        speed = config.SPEED_NORMAL
        system_content, user_content, max_tokens = self.build_prompt(user_query, rag_records, duration_sec)
        _, _, max_audio_sec = get_tokens_and_words_budget(duration_sec)

        payload = {
            "model": self.model_name,
            "messages": [
                {"role": "system", "content": system_content},
                {"role": "user", "content": user_content},
            ],
            "max_tokens": max_tokens,
            "temperature": 0.2,
            "top_p": 0.9,
            "cache_prompt": True,
            "id_slot": 0,
            "stream": True,
            "chat_template_kwargs": {"enable_thinking": False},
        }

        text_queue = queue.Queue()
        pcm_audio_queue = queue.Queue()
        stop_pipeline_event = threading.Event()
        self._active_stop_event = stop_pipeline_event

        total_spoken_sec = [0.0]
        planned_sec = [0.0]
        spoken_sentences = []
        first_audio_started = [False]
        t0 = time.time()
        ttfa_ms = [0.0]

        if audio_mixer:
            audio_mixer._stop_speech_event.clear()
            audio_mixer._narrative_active = True
            audio_mixer.duck_music()

        def audio_player_worker():
            while True:
                item = pcm_audio_queue.get()
                if item is None:
                    pcm_audio_queue.task_done()
                    break
                if stop_pipeline_event.is_set():
                    pcm_audio_queue.task_done()
                    continue
                samples, sr, frase = item
                dur = len(samples) / float(sr)
                if not first_audio_started[0]:
                    first_audio_started[0] = True
                    ttfa_ms[0] = (time.time() - t0) * 1000
                    logger.info(f"[TTFA - Latencia percibida hasta el primer audio]: {ttfa_ms[0]:.0f} ms")
                try:
                    if audio_mixer:
                        audio_mixer.play_speech(samples, sr)
                    total_spoken_sec[0] += dur
                    spoken_sentences.append(frase)
                except Exception as ex:
                    logger.error(f"Audio playback error: {ex}")
                finally:
                    pcm_audio_queue.task_done()

        def tts_synth_worker():
            budget_reached = False
            try:
                while True:
                    item = text_queue.get()
                    try:
                        if item is None:
                            break
                        if stop_pipeline_event.is_set():
                            continue
                        frase = item.strip()
                        if not frase:
                            continue
                        is_closing = ("en definitiva" in frase.lower() or "en conjunto" in frase.lower())
                        if budget_reached and not is_closing:
                            continue
                        tts_text = expand_biblical_citations_for_tts(frase)
                        if kokoro_engine is not None:
                            samples, sr = kokoro_engine.create(tts_text, voice=config.TTS_VOICE_NAME, speed=speed, lang=config.TTS_LANG)
                        else:
                            sr = config.SAMPLE_RATE_TTS
                            dur_est = max(0.5, len(frase.split()) / 2.2)
                            samples = np.zeros(int(dur_est * sr), dtype=np.float32)

                        dur = len(samples) / float(sr)
                        # Allow the mandatory 'En definitiva,' closing up to 34.0s if sentences 1-2 used ~25s; hard-cap non-closing at 30.0s
                        hard_cap = (max_audio_sec + 4.0) if is_closing else max_audio_sec
                        if duration_sec <= 30 and planned_sec[0] + dur > hard_cap:
                            budget_reached = True
                            continue
                        elif duration_sec > 30 and planned_sec[0] + dur > max_audio_sec and not is_closing:
                            budget_reached = True
                            continue

                        planned_sec[0] += dur
                        if not stop_pipeline_event.is_set():
                            pcm_audio_queue.put((samples, sr, frase))
                    except Exception as ex:
                        logger.error(f"TTS synthesis worker error: {ex}")
                    finally:
                        text_queue.task_done()
            finally:
                pcm_audio_queue.put(None)

        player_thread = threading.Thread(target=audio_player_worker, daemon=True)
        synth_thread = threading.Thread(target=tts_synth_worker, daemon=True)
        player_thread.start()
        synth_thread.start()

        full_text = ""
        buffer_frase = ""
        first_chunk_sent = False
        in_think_block = False

        try:
            resp = requests.post(f"{self.server_url}/v1/chat/completions", json=payload, stream=True, timeout=35)
            self._active_http_resp = resp
            if resp.status_code == 200:
                for line in resp.iter_lines():
                    if stop_pipeline_event.is_set():
                        break
                    if not line:
                        continue
                    decoded = line.decode("utf-8", errors="replace")
                    if not decoded.startswith("data: "):
                        continue
                    data_str = decoded[6:]
                    if data_str == "[DONE]":
                        break
                    try:
                        delta = json.loads(data_str)["choices"][0].get("delta", {})
                        token = delta.get("content", "")
                        if not token:
                            continue
                        if "<think>" in token:
                            in_think_block = True
                            continue
                        if in_think_block:
                            if "</think>" in token:
                                in_think_block = False
                                token = token.split("</think>", 1)[-1]
                            else:
                                continue

                        buffer_frase += token
                        full_text += token
                        words_count = len(buffer_frase.strip().split())

                        if not first_chunk_sent:
                            if words_count >= 5:
                                if bool(re.search(r"[.?!;:]", token)):
                                    is_cut = True
                                else:
                                    safe_commas = []
                                    for m in re.finditer(r",", buffer_frase):
                                        c_idx = m.end()
                                        if c_idx == len(buffer_frase):
                                            if not re.search(r"\d+\s*,$", buffer_frase):
                                                safe_commas.append(m)
                                        else:
                                            after_s = buffer_frase[c_idx:].lstrip()
                                            if after_s and not after_s[0].isdigit():
                                                safe_commas.append(m)
                                    if safe_commas:
                                        fc = safe_commas[0]
                                        if len(buffer_frase[:fc.end()].split()) >= 5:
                                            chunk_0 = self._clean_text(buffer_frase[:fc.end()])
                                            remainder = buffer_frase[fc.end():].lstrip()
                                            if len(chunk_0) > 3:
                                                text_queue.put(chunk_0)
                                                first_chunk_sent = True
                                                buffer_frase = remainder
                                            continue
                                    is_cut = False
                            else:
                                is_cut = False
                        else:
                            is_cut = bool(re.search(r"[.?!\n]", token))

                        if is_cut:
                            clean_frase = self._clean_text(buffer_frase)
                            if len(clean_frase) > 3 and not stop_pipeline_event.is_set():
                                text_queue.put(clean_frase)
                                first_chunk_sent = True
                            buffer_frase = ""
                    except Exception:
                        pass

            clean_frase = self._clean_text(buffer_frase)
            if len(clean_frase) > 3 and not stop_pipeline_event.is_set():
                if not clean_frase.endswith((".", "!", "?")):
                    clean_frase += "."
                text_queue.put(clean_frase)

        except Exception as e:
            logger.error(f"Streaming error from llama-server: {e}")
        finally:
            self._active_http_resp = None
            self._active_stop_event = None
            text_queue.put(None)
            synth_thread.join()
            player_thread.join()
            if audio_mixer:
                audio_mixer._narrative_active = False
                audio_mixer.last_speech_end_time = time.time()
                audio_mixer.unduck_music()

        final_clean_text = " ".join(spoken_sentences).strip() or self._clean_text(full_text)
        if final_clean_text and not final_clean_text.endswith((".", "!", "?")):
            final_clean_text += "."
        return final_clean_text, total_spoken_sec[0], ttfa_ms[0]

    def _clean_text(self, text: str) -> str:
        t = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL)
        t = re.sub(r'[*#_~`«»"“”\']', "", t).strip()
        t = t.replace("[", "").replace("]", "")
        t = t.replace("(", ", ").replace(")", ", ")
        t = re.sub(r"\b(?:n|num|núm)\.\s*(\d+)\b", r"numeral \1", t, flags=re.IGNORECASE)
        t = re.sub(r"\b(en|de|a|por|con)\s*,\s*", r"\1 ", t, flags=re.IGNORECASE)
        t = re.sub(r"\s*,\s*,+", ", ", t)
        t = re.sub(r"\s*([,.:;?!])", r"\1", t)
        t = re.sub(r"\s+", " ", t).strip()
        return t

    def synthesize_with_watchdog(self, text: str, audio_mixer, kokoro_engine=None, max_sec: float = 30.0):
        """Sentence-by-sentence synthesis with strict <= max_sec watchdog."""
        sentences = [s.strip() for s in re.split(r"(?<=[.!?])\s+", text) if s.strip()]
        pcm_chunks = []
        total_sec = 0.0

        sample_rate = getattr(kokoro_engine, "sample_rate", config.SAMPLE_RATE_TTS)
        speed = config.SPEED_NORMAL

        for sentence in sentences:
            if not sentence:
                continue
            if kokoro_engine is not None:
                try:
                    tts_sentence = expand_biblical_citations_for_tts(sentence)
                    samples, sr = kokoro_engine.create(tts_sentence, voice=config.TTS_VOICE_NAME, speed=speed, lang=config.TTS_LANG)
                    dur = len(samples) / float(sr)
                except Exception:
                    dur = max(0.5, len(sentence.split()) / 2.2)
                    samples = np.zeros(int(dur * sample_rate), dtype=np.float32)
                    sr = sample_rate
            else:
                dur = max(0.5, len(sentence.split()) / 2.2)
                samples = np.zeros(int(dur * sample_rate), dtype=np.float32)
                sr = sample_rate

            if total_sec + dur > max_sec:
                break

            pcm_chunks.append((samples, sr, sentence))
            total_sec += dur

        return pcm_chunks, total_sec
