import os
import re
import time
import math
import queue
import hashlib
import threading
import numpy as np
import scipy.signal
import soundfile as sf
import logging

try:
    import pygame
    HAS_PYGAME = True
except ImportError:
    HAS_PYGAME = False

from . import config

logger = logging.getLogger("vox_verbi.audio_mixer")


def clean_prosodic_text(text: str) -> str:
    """Removes direction brackets ([VOZ SERENA], [VOZ ÍNTIMA], etc.) and markdown formatting."""
    direction_patterns = [
        r"\[VOZ\s+SERENA\]",
        r"\[VOZ\s+ÍNTIMA\]",
        r"\[VOZ\s+INTIMA\]",
        r"\[VOZ\s+PROFUNDA\]",
        r"\[ÉNFASIS\s+SUAVE\]",
        r"\[ENFASIS\s+SUAVE\]",
        r"\[TRANSICIÓN\s+SUAVE\]",
        r"\[TRANSICION\s+SUAVE\]",
    ]
    cleaned = text
    for p in direction_patterns:
        cleaned = re.sub(p, "", cleaned, flags=re.IGNORECASE)
    cleaned = cleaned.replace("**", "").replace("__", "")
    cleaned = re.sub(r"(?<!\w)_(?!\w)", "", cleaned)
    return cleaned


def int_to_spanish(n: int) -> str:
    """Converts an integer 0..9999 to natural spoken Spanish words (covers all CIC numerals up to 2865)."""
    units = ["cero", "uno", "dos", "tres", "cuatro", "cinco", "seis", "siete", "ocho", "nueve"]
    teens = ["diez", "once", "doce", "trece", "catorce", "quince", "dieciséis", "diecisiete", "dieciocho", "diecinueve"]
    tens = ["", "diez", "veinte", "treinta", "cuarenta", "cincuenta", "sesenta", "setenta", "ochenta", "noventa"]
    hundreds = ["", "ciento", "doscientos", "trescientos", "cuatrocientos", "quinientos", "seiscientos", "setecientos", "ochocientos", "novecientos"]

    if n < 0:
        return str(n)
    if n == 100:
        return "cien"
    if n < 10:
        return units[n]
    if n < 20:
        return teens[n - 10]
    if n < 30:
        if n == 20:
            return "veinte"
        if n == 21:
            return "veintiuno"
        if n == 22:
            return "veintidós"
        if n == 23:
            return "veintitrés"
        if n == 26:
            return "veintiséis"
        return f"veinti{units[n - 20]}"
    if n < 100:
        u = n % 10
        t = n // 10
        return f"{tens[t]} y {units[u]}" if u != 0 else tens[t]
    if n < 1000:
        h = n // 100
        rem = n % 100
        if rem == 0:
            return "cien" if h == 1 else hundreds[h]
        return f"{hundreds[h]} {int_to_spanish(rem)}"
    if n < 10000:
        th = n // 1000
        rem = n % 1000
        th_str = "mil" if th == 1 else f"{int_to_spanish(th)} mil"
        return th_str if rem == 0 else f"{th_str} {int_to_spanish(rem)}"
    return str(n)


def expand_biblical_citations_for_tts(text: str) -> str:
    """Expands biblical abbreviations, Roman numerals, and 1..9999 integers into clean spoken Spanish."""
    if not text:
        return ""
    t = text
    t = re.sub(r'[*#_~`«»"“”\']', "", t)
    t = t.replace("[", "").replace("]", "")

    # Salmos dual numbering
    t = re.sub(r"\b(Salmos?)\s+(\d+)\s*\(\s*(\d+)\s*\)\s*,\s*(\d+)\s*[-–]\s*(\d+)\b", r"\1 \2, versículos \4 al \5", t, flags=re.IGNORECASE)
    t = re.sub(r"\b(Salmos?)\s+(\d+)\s*\(\s*(\d+)\s*\)\s*,\s*(\d+)\b", r"\1 \2, versículo \4", t, flags=re.IGNORECASE)
    t = re.sub(r"\b(Salmos?)\s+(\d+)\s*\(\s*\d+\s*\)", r"\1 \2", t, flags=re.IGNORECASE)

    t = t.replace("(", ", ").replace(")", ", ")

    roman_entities = [
        (r"\bBenedicto\s+XVI\b", "Benedicto Dieciséis"),
        (r"\bLeón\s+XIV\b", "León Catorce"),
        (r"\bLeón\s+XIII\b", "León Trece"),
        (r"\bJuan\s+Pablo\s+II\b", "Juan Pablo Segundo"),
        (r"\bJuan\s+Pablo\s+I\b", "Juan Pablo Primero"),
        (r"\bPablo\s+VI\b", "Pablo Sexto"),
        (r"\bJuan\s+XXIII\b", "Juan Veintitrés"),
        (r"\bP[ií]o\s+XII\b", "Pío Doce"),
        (r"\bP[ií]o\s+XI\b", "Pío Once"),
        (r"\bP[ií]o\s+X\b", "Pío Décimo"),
        (r"\bVaticano\s+II\b", "Vaticano Segundo"),
        (r"\bVaticano\s+I\b", "Vaticano Primero"),
        (r"\bSiglo\s+XXI\b", "Siglo veintiuno"),
        (r"\bSiglo\s+XX\b", "Siglo veinte"),
        (r"\bSiglo\s+V\b", "Siglo quinto"),
        (r"\bSiglo\s+IV\b", "Siglo cuarto"),
        (r"\bSiglo\s+III\b", "Siglo tercero"),
        (r"\bSiglo\s+II\b", "Siglo segundo"),
        (r"\bSiglo\s+I\b", "Siglo primero"),
    ]
    for pattern, repl in roman_entities:
        t = re.sub(pattern, repl, t, flags=re.IGNORECASE)

    t = re.sub(r"\b(?:en el\s+)?(?:n|num|núm|número|numeral|ds)\.?\s*(\d+)\b", r"numeral \1", t, flags=re.IGNORECASE)
    t = re.sub(r"\bCIC\s*(\d+)\b", r"Catecismo numeral \1", t, flags=re.IGNORECASE)
    t = re.sub(r"\bCompendio\s*(\d+)\b", r"Compendio numeral \1", t, flags=re.IGNORECASE)

    # Expand Patristic / Conciliar Roman numerals after Libro / Capítulo / Carta / Sección or before verse commas
    roman_cardinals = [
        ("XXV", "veinticinco"), ("XXIV", "veinticuatro"), ("XXIII", "veintitrés"), ("XXII", "veintidós"),
        ("XXI", "veintiuno"), ("XX", "veinte"), ("XIX", "diecinueve"), ("XVIII", "dieciocho"),
        ("XVII", "diecisiete"), ("XVI", "dieciséis"), ("XV", "quince"), ("XIV", "catorce"),
        ("XIII", "trece"), ("XII", "doce"), ("XI", "once"), ("X", "diez"),
        ("IX", "nueve"), ("VIII", "ocho"), ("VII", "siete"), ("VI", "seis"),
        ("V", "cinco"), ("IV", "cuatro"), ("III", "tres"), ("II", "dos"), ("I", "uno"),
    ]
    for r_sym, r_word in roman_cardinals:
        t = re.sub(rf"\b(Libro|Capítulo|Carta|Parte|Tomo|Sección|Canon)\s+{r_sym}\b", rf"\1 {r_word}", t, flags=re.IGNORECASE)
        t = re.sub(rf",\s*{r_sym}\s*,", f", {r_word},", t)

    book_pattern = r"(Corintios|Cor|Juan|Jn|Pedro|Pe|Timoteo|Tim|Tesalonicenses|Tes|Reyes|Samuel|Macabeos|Crónicas)"
    t = re.sub(rf"\b1\s+{book_pattern}\b", r"Primera de \1", t, flags=re.IGNORECASE)
    t = re.sub(rf"\b2\s+{book_pattern}\b", r"Segunda de \1", t, flags=re.IGNORECASE)
    t = re.sub(rf"\b3\s+(Juan|Jn)\b", r"Tercera de \1", t, flags=re.IGNORECASE)

    t = re.sub(r"\bPrimera de Cor\b", "Primera de Corintios", t, flags=re.IGNORECASE)
    t = re.sub(r"\bSegunda de Cor\b", "Segunda de Corintios", t, flags=re.IGNORECASE)
    t = re.sub(r"\bPrimera de Jn\b", "Primera de Juan", t, flags=re.IGNORECASE)
    t = re.sub(r"\bSegunda de Jn\b", "Segunda de Juan", t, flags=re.IGNORECASE)
    t = re.sub(r"\bPrimera de Pe\b", "Primera de Pedro", t, flags=re.IGNORECASE)

    t = re.sub(r"(\b[A-Za-zÁÉÍÓÚáéíóúñ]+)\s+(\d+)\s*,\s*(\d+)\s*[-–]\s*(\d+)\b", r"\1 capítulo \2, versículos \3 al \4", t)
    t = re.sub(r"(\b[A-Za-zÁÉÍÓÚáéíóúñ]+)\s+(\d+)\s*,\s*(\d+)\b", r"\1 capítulo \2, versículo \3", t)
    t = re.sub(r"\b(\d+)\s*,\s*(\d+)\s*[-–]\s*(\d+)\b", r"capítulo \1, versículos \2 al \3", t)
    t = re.sub(r"\b(\d+)\s*[-–]\s*(\d+)\b", r"\1 al \2", t)

    def _num_to_es(m):
        try:
            return int_to_spanish(int(m.group(1)))
        except Exception:
            return m.group(1)

    t = re.sub(r"\b(\d{1,4})\b", _num_to_es, t)
    t = re.sub(r"\beros\b", "éros", t, flags=re.IGNORECASE)

    t = re.sub(r"\b(en|de|a|por|con)\s*,\s*", r"\1 ", t, flags=re.IGNORECASE)
    t = re.sub(r"\s*,\s*,+", ", ", t)
    t = re.sub(r"\s*([,.:;?!])", r"\1", t)
    t = re.sub(r"([,.:;?!])(?=[^\s\d])", r"\1 ", t)
    t = re.sub(r"\s+", " ", t).strip()
    return t


def _split_speech_block_into_sentences(text: str, max_chars: int = 230) -> list[str]:
    """
    Splits a speech block cleanly along paragraph and sentence boundaries so TTS synthesizes
    the first sentence in <250ms (eliminating multi-second upfront stalls on long sections)
    without altering a single word of the canonical text.
    """
    out = []
    paragraphs = [p.strip() for p in re.split(r"\n\s*\n+", text) if p.strip()]
    for p in paragraphs:
        p_single = re.sub(r"\s*\n\s*", " ", p).strip()
        sentences = [s.strip() for s in re.split(r"(?<=[.!?])\s+", p_single) if s.strip()]
        buf = ""
        for idx, s in enumerate(sentences):
            # Keep the very first sentence standalone for ultra-fast TTFA!
            if not out and not buf:
                out.append(s)
                continue
            if not buf:
                buf = s
            elif len(buf) + 1 + len(s) <= max_chars:
                buf = f"{buf} {s}"
            else:
                out.append(buf)
                buf = s
        if buf:
            out.append(buf)
    return out if out else ([text.strip()] if text.strip() else [])


def strip_editorial_headings_for_speech(text: str) -> str:
    """
    Removes printed document numbering/titles (e.g. '11. Vox Verbi en una definición',
    '1. El fundamento: ...', 'Presentación', 'I. EL TESORO DEL HOMBRE', 'CUENTO DE APERTURA: ...')
    so Vox Verbi never speaks 'once.' or raw chapter numbers aloud before the content!
    """
    if not text:
        return ""
    t = text
    # Strip top header of COMPLETA if present
    t = re.sub(
        r"^\s*VOX\s+VERBI\s*\n+Una propuesta de evangelización[^\n]*\n+",
        "",
        t,
        flags=re.IGNORECASE,
    )
    # Strip Guía Fundamental numbered or 'Presentación' section title lines (at start or after blank line)
    t = re.sub(
        r"(?:^|\n\n)\s*(?:Presentación|\d{1,2}\.\s+[^\n]+)\s*\n+",
        "\n\n",
        t,
        flags=re.IGNORECASE,
    )
    # Strip story editorial header lines ('CUENTO DE APERTURA: ...', 'El sueño de los tres árboles – ...', Roman numerals 'I. ...')
    t = re.sub(r"^\s*CUENTO\s+DE\s+APERTURA:[^\n]*\n+", "", t, flags=re.IGNORECASE)
    t = re.sub(r"^\s*El\s+sueño\s+de\s+los\s+tres\s+árboles\s*[–-][^\n]*\n+", "", t, flags=re.IGNORECASE)
    t = re.sub(r"(?m)^\s*[IVX]{1,4}\.\s+[A-ZÁÉÍÓÚÑ\s]{4,}\n+", "\n", t)
    return t.strip()


def parse_prosodic_segments(raw_text: str):
    """
    Parses a text containing prosodic pause markers and returns an ordered list of segments:
      {"type": "speech", "text": "..."}  or  {"type": "silence", "duration": float}
    Long speech blocks are cleanly split at sentence boundaries for low-latency streaming.
    """
    cleaned = strip_editorial_headings_for_speech(clean_prosodic_text(raw_text))

    marker_map = [
        (r"\[SILENCIO\s+DE\s+6\s+SEGUNDOS\]", config.SILENCIO_6_SEC),
        (r"\[PAUSA\s+DE\s+5\s+SEGUNDOS\]", config.PAUSE_5_SEC),
        (r"\[PAUSA\s+DE\s+4\s+SEGUNDOS\]", config.PAUSE_4_SEC),
        (r"\[PAUSA\s+DE\s+3\s+SEGUNDOS\]", config.PAUSE_3_SEC),
        (r"\[PAUSA\s+BREVE\]", config.PAUSE_BREVE),
        (r"---+", config.PAUSE_DASHES),
        (r"\.\.\.|…", config.PAUSE_ELLIPSIS),
    ]

    pattern_parts = [f"({p})" for p, _ in marker_map]
    combined_regex = re.compile("|".join(pattern_parts), flags=re.IGNORECASE)

    tokens = combined_regex.split(cleaned)
    segments = []

    for token in tokens:
        if not token:
            continue

        matched_marker = False
        for p, dur in marker_map:
            if re.fullmatch(p, token.strip(), flags=re.IGNORECASE):
                segments.append({"type": "silence", "duration": dur})
                matched_marker = True
                break

        if not matched_marker:
            text_str = token.strip()
            if text_str:
                for sub_s in _split_speech_block_into_sentences(text_str):
                    segments.append({"type": "speech", "text": sub_s})

    return segments


class AudioMixer:
    """Manages ALSA/pygame playback, audio ducking, interruptible speech, and low-latency narrative streaming."""

    def __init__(self, output_alsa=config.OUTPUT_ALSA):
        self.output_alsa = output_alsa
        self.current_track = None
        self.master_volume = float(getattr(config, "MASTER_OUTPUT_VOLUME", 0.20))
        self.music_volume = config.MUSIC_GAIN_NORMAL
        self.target_music_volume = config.MUSIC_GAIN_NORMAL
        self.is_playing_music = False
        self.is_paused = False
        self.ducking_active = False

        self.lock = threading.Lock()
        self._active_channel = None
        self._stop_speech_event = threading.Event()
        self._mem_tts_cache = {}
        self.is_speaking = False
        self._narrative_active = False
        self.last_speech_end_time = 0.0

        if HAS_PYGAME and output_alsa is not None:
            try:
                resolved_alsa = output_alsa
                if os.path.exists("/proc/asound/cards"):
                    with open("/proc/asound/cards", "r", encoding="utf-8", errors="ignore") as fc:
                        cards_txt = fc.read()
                    if "S330" in cards_txt:
                        resolved_alsa = "plughw:CARD=S330,DEV=0"
                    elif "Generic_1" in cards_txt and "S330" in str(output_alsa):
                        resolved_alsa = "plughw:CARD=Generic_1,DEV=0"
                self.output_alsa = resolved_alsa
                if resolved_alsa:
                    os.environ["SDL_AUDIODRIVER"] = "alsa"
                    os.environ["AUDIODEV"] = resolved_alsa
                pygame.mixer.init(frequency=config.SAMPLE_RATE_MUSIC, channels=2)
                logger.info(f"Initialized pygame.mixer on {resolved_alsa}")
            except Exception as e:
                logger.warning(f"Failed to init pygame.mixer with ALSA {output_alsa}: {e}. Retrying default.")
                try:
                    pygame.mixer.init()
                except Exception as ex:
                    logger.error(f"Cannot init pygame mixer: {ex}")

    def set_master_volume(self, vol: float):
        """Sets global output volume (0.03 .. 1.00) immediately on active speech and music."""
        with self.lock:
            self.master_volume = max(0.03, min(1.0, round(float(vol), 2)))
            cur_master = self.master_volume
            cur_music = self.music_volume
            try:
                if self._active_channel is not None and self._active_channel.get_busy():
                    self._active_channel.set_volume(cur_master)
                if HAS_PYGAME and self.output_alsa is not None and pygame.mixer.get_init() and self.is_playing_music:
                    pygame.mixer.music.set_volume(cur_music * cur_master)
            except Exception:
                pass
        logger.info(f"🔉 Master output volume set to {int(round(cur_master * 100))}% ({cur_master:.2f})")
        return cur_master

    def adjust_master_volume(self, delta: float):
        return self.set_master_volume(self.master_volume + delta)

    def is_busy(self) -> bool:
        return self.is_speaking or self._narrative_active

    def stop_speech(self):
        """Immediately stops any ongoing TTS narration or response."""
        self._stop_speech_event.set()
        try:
            if self._active_channel is not None:
                self._active_channel.stop()
        except Exception:
            pass

    def play_speech(self, samples: np.ndarray, sample_rate: int = None):
        """Plays speech PCM samples through pygame.mixer seamlessly with polyphase resampling."""
        sr = sample_rate or config.SAMPLE_RATE_TTS
        if self._stop_speech_event.is_set():
            return
        if not HAS_PYGAME or self.output_alsa is None or not pygame.mixer.get_init():
            return

        try:
            mixer_freq, _, _ = pygame.mixer.get_init()
            if sr != mixer_freq:
                gcd = math.gcd(mixer_freq, sr)
                resampled = scipy.signal.resample_poly(samples, mixer_freq // gcd, sr // gcd)
            else:
                resampled = samples.copy()

            peak = float(np.max(np.abs(resampled))) if len(resampled) else 0.0
            target_norm = getattr(config, "SPEECH_PEAK_NORM", 0.88)
            if peak > 0.01:
                target_gain = min(2.5, target_norm / peak)
                resampled = resampled * target_gain

            fade_len = int(mixer_freq * 0.005)
            if len(resampled) > 2 * fade_len:
                resampled[:fade_len] *= np.linspace(0.0, 1.0, fade_len, dtype=np.float32)
                resampled[-fade_len:] *= np.linspace(1.0, 0.0, fade_len, dtype=np.float32)

            clamped = np.clip(resampled, -1.0, 1.0)
            samples_i16 = (clamped * 32767).astype(np.int16)
            stereo_i16 = np.ascontiguousarray(np.column_stack((samples_i16, samples_i16)))

            sound = pygame.sndarray.make_sound(stereo_i16)
            self.is_speaking = True
            channel = sound.play()
            self._active_channel = channel
            if channel:
                channel.set_volume(self.master_volume)
                while channel.get_busy() and not self._stop_speech_event.is_set():
                    pygame.time.wait(10)
                if self._stop_speech_event.is_set():
                    channel.stop()
        except Exception as e:
            logger.error(f"Error playing speech through pygame: {e}")
        finally:
            self.is_speaking = False
            self.last_speech_end_time = time.time()

    def play_ready_chime(self):
        """Plays a gentle, contemplative 140ms two-note chime (A4 -> C#5) when 'Vox Verbi' arms active listening."""
        if not HAS_PYGAME or self.output_alsa is None or not pygame.mixer.get_init():
            return
        try:
            mixer_freq, _, _ = pygame.mixer.get_init()
            t1 = np.linspace(0.0, 0.065, int(mixer_freq * 0.065), endpoint=False, dtype=np.float32)
            t2 = np.linspace(0.0, 0.085, int(mixer_freq * 0.085), endpoint=False, dtype=np.float32)
            env1 = np.sin(np.pi * np.linspace(0.0, 1.0, len(t1), dtype=np.float32))
            env2 = np.sin(np.pi * np.linspace(0.0, 1.0, len(t2), dtype=np.float32))
            tone = np.concatenate([
                0.18 * np.sin(2.0 * np.pi * 440.0 * t1) * env1,
                0.20 * np.sin(2.0 * np.pi * 554.37 * t2) * env2,
            ])
            i16 = (tone * 32767).astype(np.int16)
            stereo = np.ascontiguousarray(np.column_stack((i16, i16)))
            ch = pygame.sndarray.make_sound(stereo).play()
            if ch:
                ch.set_volume(self.master_volume)
                while ch.get_busy():
                    pygame.time.wait(5)
            self.last_speech_end_time = time.time()
        except Exception:
            pass

    def _cache_filepath(self, voice_tag: str, speed: float, tts_text: str) -> str:
        cache_dir = getattr(config, "TTS_CACHE_DIR", "")
        if not cache_dir:
            return ""
        os.makedirs(cache_dir, exist_ok=True)
        digest = hashlib.sha1(f"{voice_tag}|{speed:.2f}|{tts_text}".encode("utf-8")).hexdigest()[:20]
        return os.path.join(cache_dir, f"{digest}.wav")

    def synthesize_cached_chunk(self, tts_text: str, tts_engine, speed: float, use_disk_cache: bool = False) -> tuple[np.ndarray, int]:
        voice_tag = f"{getattr(tts_engine, 'backend', 'tts')}_{getattr(tts_engine, 'voice_name', config.TTS_VOICE_NAME)}"
        mem_key = (voice_tag, round(speed, 2), tts_text)
        if mem_key in self._mem_tts_cache:
            return self._mem_tts_cache[mem_key]

        cpath = self._cache_filepath(voice_tag, speed, tts_text) if use_disk_cache else ""
        if cpath and os.path.exists(cpath):
            try:
                wav, sr = sf.read(cpath, dtype="float32")
                self._mem_tts_cache[mem_key] = (wav, sr)
                return wav, sr
            except Exception:
                pass

        wav, sr = tts_engine.create(tts_text, voice=config.TTS_VOICE_NAME, speed=speed, lang=config.TTS_LANG)
        if use_disk_cache and cpath:
            try:
                sf.write(cpath, wav, sr)
            except Exception:
                pass
            if len(self._mem_tts_cache) < 64:
                self._mem_tts_cache[mem_key] = (wav, sr)
        return wav, sr

    def prewarm_canonical_cache(self, canonical_data: dict, tts_engine):
        """
        Pre-synthesizes and caches on disk/RAM the opening speech chunk of every Guía Fundamental
        section so any fundamental question starts playing in < 50 ms!
        """
        if not canonical_data or tts_engine is None:
            return
        pres = canonical_data.get("presentacion", {})
        targets = {"COMPLETA": pres.get("completa", "")}
        targets.update(pres.get("secciones", {}))
        warmed = 0
        for sec_id, raw_txt in targets.items():
            if not raw_txt:
                continue
            segs = parse_prosodic_segments(raw_txt)
            first_speech = next((s["text"] for s in segs if s["type"] == "speech"), None)
            if first_speech:
                tts_t = expand_biblical_citations_for_tts(first_speech)
                self.synthesize_cached_chunk(tts_t, tts_engine, config.SPEED_NORMAL, use_disk_cache=True)
                warmed += 1
        logger.info(f"Pre-warmed opening TTS audio cache for {warmed} Guía Fundamental sections.")

    def stream_narrative(self, text: str, mode: str = "NORMAL", tts_engine=None, use_cache_first_chunk: bool = True) -> tuple[float, float]:
        """
        Producer-Consumer streaming narrative engine:
        Plays chunk 1 immediately (< 50ms if cached, ~180ms cold) while synthesizing subsequent
        sentences in a background thread, eliminating all upfront waiting on long readings!
        Returns (total_spoken_sec, ttfa_ms).
        """
        self._stop_speech_event.clear()
        segments = parse_prosodic_segments(text)
        if not segments:
            return 0.0, 0.0

        base_speed = config.SPEED_RELATOS if mode in ("RELATO", "AL_PARAISO") else config.SPEED_NORMAL
        base_gain = config.GAIN_RELATOS if mode in ("RELATO", "AL_PARAISO") else config.GAIN_NORMAL
        total_speech_segments = sum(1 for s in segments if s["type"] == "speech")

        pcm_q = queue.Queue(maxsize=4)
        t0 = time.time()
        ttfa_ms = [0.0]
        first_started = [False]
        total_sec = [0.0]

        self.duck_music()
        self._narrative_active = True

        def _producer():
            speech_idx = 0
            for seg in segments:
                if self._stop_speech_event.is_set():
                    break
                if seg["type"] == "silence":
                    pcm_q.put(("silence", seg["duration"], None, config.SAMPLE_RATE_TTS))
                elif seg["type"] == "speech":
                    speech_idx += 1
                    gain = base_gain
                    if mode == "AL_PARAISO":
                        fade_threshold = max(1, int(total_speech_segments * 0.75))
                        if speech_idx >= fade_threshold:
                            t_frac = (speech_idx - fade_threshold) / max(1, (total_speech_segments - fade_threshold))
                            gain = config.GAIN_AL_PARAISO_START - t_frac * (config.GAIN_AL_PARAISO_START - config.GAIN_AL_PARAISO_END)
                    if tts_engine is not None:
                        try:
                            tts_speech = expand_biblical_citations_for_tts(seg["text"])
                            cache_this = use_cache_first_chunk and (speech_idx == 1) and (mode == "NORMAL")
                            samples, sr = self.synthesize_cached_chunk(tts_speech, tts_engine, base_speed, use_disk_cache=cache_this)
                            samples = samples * gain
                        except Exception as e:
                            logger.error(f"Narrative TTS synthesis error: {e}")
                            sr = config.SAMPLE_RATE_TTS
                            samples = np.zeros(int(max(0.5, len(seg["text"].split()) / 2.2) * sr), dtype=np.float32)
                    else:
                        sr = config.SAMPLE_RATE_TTS
                        samples = np.zeros(int(max(0.5, len(seg["text"].split()) / 2.2) * sr), dtype=np.float32)
                    pcm_q.put(("speech", len(samples) / float(sr), samples, sr))
            pcm_q.put(None)

        prod_thread = threading.Thread(target=_producer, daemon=True)
        prod_thread.start()

        try:
            while True:
                item = pcm_q.get()
                if item is None:
                    break
                if self._stop_speech_event.is_set():
                    continue
                kind, dur, samples, sr = item
                if not first_started[0]:
                    first_started[0] = True
                    ttfa_ms[0] = (time.time() - t0) * 1000
                    logger.info(f"[NARRATIVE TTFA]: {ttfa_ms[0]:.1f} ms (mode={mode})")
                if kind == "silence":
                    if HAS_PYGAME and self.output_alsa is not None and pygame.mixer.get_init():
                        end_t = time.time() + dur
                        while time.time() < end_t and not self._stop_speech_event.is_set():
                            time.sleep(min(0.05, max(0.0, end_t - time.time())))
                    total_sec[0] += dur
                elif kind == "speech" and samples is not None:
                    self.play_speech(samples, sr)
                    total_sec[0] += dur
        finally:
            self._narrative_active = False
            self.last_speech_end_time = time.time()
            prod_thread.join(timeout=2.0)
            self.unduck_music()

        return total_sec[0], ttfa_ms[0]

    def play_music(self, track_file_or_id: str, fade_in_sec: float = 1.0, start_offset_sec: float = 0.0):
        with self.lock:
            filepath = track_file_or_id
            if not os.path.exists(filepath):
                cand = os.path.join(config.MUSIC_DIR, f"{track_file_or_id}.ogg")
                if os.path.exists(cand):
                    filepath = cand
                else:
                    logger.error(f"Music file not found: {track_file_or_id}")
                    return False

            self.current_track = filepath
            self.music_volume = config.MUSIC_GAIN_NORMAL
            self.target_music_volume = config.MUSIC_GAIN_NORMAL
            self.ducking_active = False

            if HAS_PYGAME and self.output_alsa is not None and pygame.mixer.get_init():
                try:
                    pygame.mixer.music.load(filepath)
                    pygame.mixer.music.set_volume(self.music_volume * self.master_volume)
                    pygame.mixer.music.play(loops=-1, start=start_offset_sec, fade_ms=int(fade_in_sec * 1000))
                    self.is_playing_music = True
                    self.is_paused = False
                    logger.info(f"Playing music: {filepath} at vol {self.music_volume} (master={self.master_volume:.2f})")
                    return True
                except Exception as e:
                    logger.error(f"Error loading music in pygame: {e}")
            else:
                self.is_playing_music = True
                self.is_paused = False
                return True
        return False

    def duck_music(self, target_vol: float = config.MUSIC_GAIN_DUCKED, duration_sec: float = 0.5):
        with self.lock:
            self.ducking_active = True
            self.target_music_volume = target_vol

        def _fade():
            steps = 10
            sleep_time = duration_sec / steps
            with self.lock:
                start_vol = self.music_volume
            for i in range(1, steps + 1):
                cur_vol = start_vol + (target_vol - start_vol) * (i / steps)
                with self.lock:
                    self.music_volume = cur_vol
                    if HAS_PYGAME and self.output_alsa is not None and pygame.mixer.get_init():
                        pygame.mixer.music.set_volume(cur_vol * self.master_volume)
                time.sleep(sleep_time)

        threading.Thread(target=_fade, daemon=True).start()

    def unduck_music(self, target_vol: float = config.MUSIC_GAIN_NORMAL, duration_sec: float = 0.5):
        with self.lock:
            self.ducking_active = False
            self.target_music_volume = target_vol

        def _fade():
            steps = 10
            sleep_time = duration_sec / steps
            with self.lock:
                start_vol = self.music_volume
            for i in range(1, steps + 1):
                cur_vol = start_vol + (target_vol - start_vol) * (i / steps)
                with self.lock:
                    self.music_volume = cur_vol
                    if HAS_PYGAME and self.output_alsa is not None and pygame.mixer.get_init():
                        pygame.mixer.music.set_volume(cur_vol * self.master_volume)
                time.sleep(sleep_time)

        threading.Thread(target=_fade, daemon=True).start()

    def set_ducking(self, active: bool, ducked_vol: float = 0.10):
        if active:
            self.duck_music(target_vol=ducked_vol, duration_sec=0.15)
        else:
            self.unduck_music(target_vol=config.MUSIC_GAIN_NORMAL, duration_sec=0.3)

    def pause_music(self):
        with self.lock:
            self.is_paused = True
            if HAS_PYGAME and self.output_alsa is not None and pygame.mixer.get_init():
                pygame.mixer.music.pause()
            logger.info("Music paused")

    def resume_music(self):
        with self.lock:
            self.is_paused = False
            if HAS_PYGAME and self.output_alsa is not None and pygame.mixer.get_init():
                pygame.mixer.music.unpause()
            logger.info("Music resumed")

    def stop_music(self, fade_out_sec: float = 4.0):
        with self.lock:
            self.is_playing_music = False
            self.is_paused = False
            if HAS_PYGAME and self.output_alsa is not None and pygame.mixer.get_init():
                pygame.mixer.music.fadeout(int(fade_out_sec * 1000))
            logger.info(f"Music stopped with {fade_out_sec}s fadeout")

    def synthesize_narrative_to_pcm(self, text: str, mode: str = "NORMAL", kokoro_engine=None):
        """Offline PCM chunk builder (kept for Test 2 suite validation)."""
        segments = parse_prosodic_segments(text)
        pcm_chunks = []
        sample_rate = getattr(kokoro_engine, "sample_rate", config.SAMPLE_RATE_TTS)

        if mode in ("RELATO", "AL_PARAISO"):
            base_speed = config.SPEED_RELATOS
            base_gain = config.GAIN_RELATOS
        else:
            base_speed = config.SPEED_NORMAL
            base_gain = config.GAIN_NORMAL

        total_speech_segments = sum(1 for s in segments if s["type"] == "speech")
        speech_idx = 0

        for seg in segments:
            if seg["type"] == "silence":
                dur = seg["duration"]
                num_samples = int(dur * config.SAMPLE_RATE_TTS)
                pcm_chunks.append({"type": "silence", "duration": dur, "pcm": np.zeros(num_samples, dtype=np.float32)})
            elif seg["type"] == "speech":
                speech_text = seg["text"]
                speech_idx += 1
                gain = base_gain
                if mode == "AL_PARAISO":
                    fade_threshold = max(1, int(total_speech_segments * 0.75))
                    if speech_idx >= fade_threshold:
                        t_frac = (speech_idx - fade_threshold) / max(1, (total_speech_segments - fade_threshold))
                        gain = config.GAIN_AL_PARAISO_START - t_frac * (config.GAIN_AL_PARAISO_START - config.GAIN_AL_PARAISO_END)
                if kokoro_engine is not None:
                    try:
                        tts_speech = expand_biblical_citations_for_tts(speech_text)
                        samples, sr = kokoro_engine.create(tts_speech, voice=config.TTS_VOICE_NAME, speed=base_speed, lang=config.TTS_LANG)
                        samples = samples * gain
                        pcm_chunks.append({"type": "speech", "text": speech_text, "duration": len(samples) / sr, "pcm": samples})
                    except Exception as e:
                        logger.error(f"TTS synthesis error: {e}")
                        dur = max(0.5, len(speech_text.split()) / 2.2)
                        pcm_chunks.append({"type": "speech", "text": speech_text, "duration": dur, "pcm": np.zeros(int(dur * sample_rate), dtype=np.float32)})
                else:
                    dur = max(0.5, len(speech_text.split()) / 2.2)
                    pcm_chunks.append({"type": "speech", "text": speech_text, "duration": dur, "pcm": np.zeros(int(dur * config.SAMPLE_RATE_TTS), dtype=np.float32)})

        return pcm_chunks
