import os
import sys
import io
import re
import wave
import time
import signal
import queue
import threading
import collections
import logging
import requests
import numpy as np
import scipy.signal
import scipy.ndimage
import pyaudio
import webrtcvad

from . import config
from . import hardware_hooks
from .state_machine import StateMachine
from .audio_mixer import AudioMixer, expand_biblical_citations_for_tts
from .command_router import CommandRouter, normalize_text, match_and_strip_wake_word
from .pre_filter import PreFilter
from .rag_engine import RAGEngine
from .llm_catequist import LLMCatequist
from .tts_engine import VoiceEngine

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger("vox_verbi.main")


def find_usb_microphone(pa: pyaudio.PyAudio):
    """Dynamically finds Anker PowerConf S330 / USB microphone, falling back to default input device."""
    device_count = pa.get_device_count()
    preferred_candidate = None
    usb_candidate = None

    for i in range(device_count):
        try:
            info = pa.get_device_info_by_index(i)
            if info.get("maxInputChannels", 0) > 0:
                name = info.get("name", "").lower()
                logger.info(f"Input device [{i}]: {info.get('name')}")
                if any(k in name for k in ["anker", "powerconf", "s330"]):
                    preferred_candidate = i
                elif usb_candidate is None and any(k in name for k in ["usb", "webcam", "camera"]):
                    usb_candidate = i
        except Exception:
            continue

    chosen = preferred_candidate if preferred_candidate is not None else usb_candidate
    if chosen is not None:
        logger.info(f"Selected USB Microphone index: {chosen}")
        return chosen

    try:
        default_idx = pa.get_default_input_device_info().get("index", 0)
        logger.info(f"Using default input device index: {default_idx}")
        return default_idx
    except Exception:
        return config.INPUT_DEVICE_INDEX


def match_wake_word(text: str) -> bool:
    matched, _ = match_and_strip_wake_word(text)
    if matched:
        return True
    return "vox verbi" in normalize_text(text)


def canonicalize_wake_word(text: str) -> str:
    matched, rest = match_and_strip_wake_word(text)
    if matched:
        return f"Vox Verbi, {rest}".strip()
    return text.strip()


def strip_wake_word(text: str) -> str:
    matched, rest = match_and_strip_wake_word(text)
    if matched:
        return rest
    return text.strip()


class BiasedWhisperRecognizer:
    """
    Low-latency offline STT wrapper powered by whisper.cpp Vulkan (ggml-vulkan on Radeon 680M)
    with Silero Neural VAD gating, adaptive dynamic-contrast gain leveling, dynamic audio_ctx
    (~218 ms avg transcription), and automatic fallback to faster-whisper / sherpa-onnx.
    """

    HALLUCINATION_PHRASES = {
        "subtitulos realizados por la comunidad de amara org",
        "subtitulos realizados por la comunidad de amara.org",
        "subtitulos por la comunidad de amara org",
        "subtitulos por la comunidad de amara.org",
        "gracias por ver el video",
        "suscribete al canal",
        "suscribete",
        "musica",
        "risas",
        "aplausos",
        "susurrando",
    }

    def __init__(self):
        self.whisper_server_url = None
        self._http = requests.Session()
        self.fw_model = None
        self.sherpa = None
        self._vad_ready = False
        self._load()

    def _warmup_vad(self):
        try:
            from faster_whisper.vad import VadOptions, get_speech_timestamps
            get_speech_timestamps(
                np.zeros(16000, dtype=np.float32),
                vad_options=VadOptions(threshold=0.48, min_speech_duration_ms=260),
                sampling_rate=16000,
            )
            self._vad_ready = True
        except Exception as e:
            logger.warning(f"Silero VAD warmup warning: {e}")

    def _load_fw_fallback(self):
        if self.fw_model is not None:
            return True
        try:
            from faster_whisper import WhisperModel
            model_size = getattr(config, "STT_FW_MODEL_SIZE", "small")
            self.fw_model = WhisperModel(
                model_size,
                device="cpu",
                compute_type="int8",
                cpu_threads=8,
                local_files_only=True,
            )
            list(self.fw_model.transcribe(np.zeros(16000, dtype=np.float32), language="es", beam_size=1))
            logger.info(f"faster-whisper '{model_size}' (int8 CPU) initialized.")
            return True
        except Exception as e:
            logger.warning(f"faster-whisper unavailable ({e}).")
            return False

    def _load(self):
        self._warmup_vad()
        engine = getattr(config, "STT_ENGINE", "whisper_cpp_vulkan")
        if engine == "whisper_cpp_vulkan":
            url = getattr(config, "WHISPER_SERVER_URL", "http://127.0.0.1:8090/inference")
            try:
                buf = io.BytesIO()
                with wave.open(buf, "wb") as wf:
                    wf.setnchannels(1)
                    wf.setsampwidth(2)
                    wf.setframerate(16000)
                    wf.writeframes(np.zeros(8000, dtype=np.int16).tobytes())
                resp = self._http.post(
                    url,
                    files={"file": ("warmup.wav", buf.getvalue(), "audio/wav")},
                    data={"response_format": "json", "temperature": "0.0", "language": "es", "audio_ctx": "192"},
                    timeout=5,
                )
                if resp.status_code == 200:
                    self.whisper_server_url = url
                    logger.info(f"whisper.cpp Vulkan server connected at {url} (Radeon 680M ggml-vulkan active).")
                    return
                logger.warning(f"whisper-server returned HTTP {resp.status_code}; falling back to faster-whisper.")
            except Exception as e:
                logger.warning(f"whisper-server unreachable at {url} ({e}); falling back to faster-whisper.")

        if self._load_fw_fallback():
            return

        try:
            import sherpa_onnx
            whisper_dir = config.STT_MODEL_DIR
            if os.path.exists(whisper_dir):
                enc = os.path.join(whisper_dir, "small-encoder.int8.onnx")
                dec = os.path.join(whisper_dir, "small-decoder.int8.onnx")
                tok = os.path.join(whisper_dir, "small-tokens.txt")
                if os.path.exists(tok):
                    self.sherpa = sherpa_onnx.OfflineRecognizer.from_whisper(
                        encoder=enc, decoder=dec, tokens=tok,
                        language="es", task="transcribe", num_threads=6,
                    )
                    logger.info("Sherpa-ONNX Whisper fallback loaded.")
        except Exception as e:
            logger.error(f"Could not initialize any STT engine: {e}")

    def _prepare_leveled_speech(self, samples: np.ndarray, peak: float, sample_rate: int) -> np.ndarray | None:
        # Step 1: Silero Neural VAD speech-presence gate (<3ms CPU ONNX)
        try:
            from faster_whisper.vad import VadOptions, get_speech_timestamps
            probe = np.clip(samples * min(3.2, 0.65 / peak), -1.0, 1.0).astype(np.float32)
            ts = get_speech_timestamps(
                probe,
                vad_options=VadOptions(threshold=0.48, min_speech_duration_ms=260),
                sampling_rate=sample_rate,
            )
            total_voiced_samples = sum((seg_t["end"] - seg_t["start"]) for seg_t in ts)
            if not ts or total_voiced_samples < int(0.35 * sample_rate):
                return None
            start_i = max(0, ts[0]["start"] - int(2.60 * sample_rate))
            end_i = min(len(samples), ts[-1]["end"] + int(0.35 * sample_rate))
            seg_samples = samples[start_i:end_i]
        except Exception:
            seg_samples = samples

        # Step 2: Preserve uniform linear gain on normal speech; use 650ms smooth envelope leveling
        # ONLY when speech has >2.6x dynamic contrast (soft 'Vox Verbi' + loud question)
        seg_peak = max(1e-4, float(np.max(np.abs(seg_samples))))
        win_env = int(0.65 * sample_rate)
        local_pk = scipy.ndimage.maximum_filter1d(np.abs(seg_samples), size=win_env, mode="nearest")
        voiced_pks = local_pk[local_pk >= 0.015]
        dyn_ratio = (
            float(np.percentile(voiced_pks, 95)) / max(0.015, float(np.percentile(voiced_pks, 5)))
            if len(voiced_pks) > int(0.5 * sample_rate)
            else 1.0
        )

        if dyn_ratio > 2.6:
            raw_gain = np.where(local_pk >= 0.014, np.minimum(16.0, 0.65 / np.maximum(local_pk, 0.014)), 1.0).astype(np.float32)
            smooth_len = int(0.25 * sample_rate)
            kernel = np.hanning(smooth_len).astype(np.float32)
            kernel /= np.sum(kernel)
            smooth_gain = np.convolve(raw_gain, kernel, mode="same")
            return np.clip(seg_samples * smooth_gain, -1.0, 1.0).astype(np.float32)
        return np.clip(seg_samples * min(4.5, 0.72 / seg_peak), -1.0, 1.0).astype(np.float32)

    def _transcribe_whisper_cpp_vulkan(self, s_leveled: np.ndarray, sample_rate: int) -> str | None:
        # Pad 0.25s tail silence so Whisper cleanly closes the final timestamp token
        padded = np.concatenate([s_leveled, np.zeros(int(0.25 * sample_rate), dtype=np.float32)])
        pcm_i16 = np.clip(padded * 32767.0, -32768, 32767).astype(np.int16)
        buf = io.BytesIO()
        with wave.open(buf, "wb") as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)
            wf.setframerate(sample_rate)
            wf.writeframes(pcm_i16.tobytes())

        dur_sec = len(padded) / float(sample_rate)
        dyn_ac = min(1500, max(256, ((int(dur_sec * 50) + 56 + 63) // 64) * 64))

        resp = self._http.post(
            self.whisper_server_url,
            files={"file": ("speech.wav", buf.getvalue(), "audio/wav")},
            data={
                "response_format": "verbose_json",
                "temperature": "0.0",
                "language": "es",
                "audio_ctx": str(dyn_ac),
            },
            timeout=8,
        )
        if resp.status_code != 200:
            raise RuntimeError(f"whisper-server HTTP {resp.status_code}")

        js = resp.json()
        segments = js.get("segments")
        if isinstance(segments, list) and segments:
            parts = []
            for seg in segments:
                nsp = float(seg.get("no_speech_prob", 0.0))
                alp = float(seg.get("avg_logprob", 0.0))
                if nsp > 0.55 or alp < -0.95:
                    continue
                seg_txt = seg.get("text", "").strip()
                if seg_txt:
                    parts.append(seg_txt)
            return " ".join(parts).strip()
        return js.get("text", "").strip()

    def transcribe(self, pcm_bytes: bytes, sample_rate: int = 16000) -> str:
        samples = np.frombuffer(pcm_bytes, dtype=np.int16).astype(np.float32) / 32768.0
        if len(samples) < int(sample_rate * 0.55):
            return ""

        samples = samples - float(np.mean(samples))
        peak = float(np.max(np.abs(samples)))
        if peak < 0.016:
            return ""

        s_leveled = self._prepare_leveled_speech(samples, peak, sample_rate)
        if s_leveled is None:
            return ""

        text = ""
        used_vulkan = False
        if self.whisper_server_url is not None:
            try:
                res = self._transcribe_whisper_cpp_vulkan(s_leveled, sample_rate)
                text = res or ""
                used_vulkan = True
            except Exception as e:
                logger.warning(f"whisper-server request failed ({e}); using CPU fallback.")
                self._load_fw_fallback()

        if not used_vulkan and self.fw_model is not None:
            segments, _ = self.fw_model.transcribe(
                s_leveled,
                language="es",
                beam_size=2,
                best_of=2,
                temperature=0.0,
                condition_on_previous_text=False,
                without_timestamps=False,
                vad_filter=True,
                vad_parameters={
                    "threshold": 0.42,
                    "min_speech_duration_ms": 200,
                    "min_silence_duration_ms": 500,
                    "speech_pad_ms": 650,
                },
                no_speech_threshold=0.50,
                log_prob_threshold=-0.78,
                compression_ratio_threshold=1.9,
                initial_prompt=config.STT_INITIAL_PROMPT,
                hotwords=config.STT_HOTWORDS,
            )
            parts = []
            for seg in segments:
                nsp = getattr(seg, "no_speech_prob", 0.0)
                alp = getattr(seg, "avg_logprob", 0.0)
                if nsp > 0.50 or alp < -0.82:
                    continue
                parts.append(seg.text)
            text = " ".join(parts).strip()
        elif not used_vulkan and self.sherpa is not None:
            s = self.sherpa.create_stream()
            s.accept_waveform(sample_rate, samples)
            self.sherpa.decode_stream(s)
            text = s.result.text.strip()

        if not text:
            return ""
        # Filter bracketed Whisper noise tags, single-letter noise artifacts & known silence hallucinations
        if re.fullmatch(r"[\[\(\*].*?[\]\)\*]\.?", text):
            return ""
        clean_check = normalize_text(text)
        if len(clean_check) <= 1 or clean_check in self.HALLUCINATION_PHRASES:
            return ""
        return text


class VoxVerbiApp:
    def __init__(self, init_audio_hw: bool = True):
        self.state = StateMachine()
        self.mixer = AudioMixer(output_alsa=config.OUTPUT_ALSA if init_audio_hw else None)
        self.router = CommandRouter()
        self.prefilter = PreFilter()
        self.rag = RAGEngine()
        self.llm = LLMCatequist()

        self.tts = VoiceEngine()
        self.kokoro = self.tts  # backward-compatible alias
        self.in_al_paraiso = False
        self._worker_thread = None

        try:
            self.mixer.prewarm_canonical_cache(self.router.canonical_data, self.tts)
        except Exception as e:
            logger.warning(f"Could not prewarm canonical audio cache: {e}")

        self.stt = BiasedWhisperRecognizer()

        try:
            self.llm.warmup()
        except Exception as e:
            logger.warning(f"Could not warm up LLM KV Cache: {e}")

    def is_protected_sacred_flow(self) -> bool:
        """True during 'Al Paraíso' preparation reading or active Eucharistic Adoration."""
        return self.in_al_paraiso or self.state.is_adoracion()

    def is_worker_active(self) -> bool:
        return self._worker_thread is not None and self._worker_thread.is_alive()

    def stop_everything(self) -> bool:
        """
        Immediately stops whatever Vox Verbi is currently doing (LLM stream, speech narration, or music),
        EXCEPT during 'Al Paraíso' or Eucharistic Adoration!
        """
        if self.is_protected_sacred_flow():
            logger.info("🕊️ [PROTECTED] 'Vox Verbi, detente' ignorado durante Al Paraíso / Adoración.")
            return False

        self.llm.abort_stream()
        self.mixer.stop_speech()
        if self.mixer.is_playing_music:
            self.mixer.stop_music(fade_out_sec=0.6)
        self.state.reset_duration()
        wt = self._worker_thread
        if wt is not None and wt.is_alive() and wt is not threading.current_thread():
            wt.join(timeout=0.5)
        logger.info("🛑 [STOP_ALL] Acción en curso detenida inmediatamente por 'Vox Verbi, detente'.")
        return True

    def dispatch_async(self, user_text: str, route_match=None):
        """Runs handle_query in a background worker thread so the main loop stays live for 'Vox Verbi, detente'."""
        def _run():
            try:
                self.handle_query(user_text, route_match=route_match)
            except Exception as e:
                logger.error(f"Error in async handle_query: {e}")

        t = threading.Thread(target=_run, daemon=True)
        self._worker_thread = t
        t.start()
        return t

    def handle_query(self, user_text: str, route_match=None):
        logger.info(f"[QUERY] Processing: '{user_text}'")

        norm = normalize_text(user_text)
        if self.state.is_adoracion():
            is_exit_adoracion = bool(
                "finalizamos la adoracion" in norm
                or re.search(
                    r"\b(finaliz\w*|termin\w*|conclu\w*|salir\w*|acaba\w*|deten\w*|para|parar|apaga\w*)\b.*\b(adoraci\w*|doraci\w*|oraci\w*|vibraci\w*|santisimo|musica)\b",
                    norm,
                )
                or re.search(r"\bfinal\s+a\s+la\s+musica\b", norm)
            )
            if is_exit_adoracion:
                logger.info("[ADORACION] Exit command received")
                self.state.exit_adoracion(mixer=self.mixer)
                self.speak("Hemos concluido el tiempo de adoración eucarística.")
            else:
                logger.info("[ADORACION] Query ignored (Rule 53 total silence)")
            return

        # Handle explicit percentage volume command ("Vox Verbi, volumen al 20 por ciento")
        m_vol_pct = re.search(r"\bvolumen\b.*?\b(\d{1,3})\s*(?:%|por\s+ciento)\b", user_text, flags=re.IGNORECASE)
        if m_vol_pct:
            pct = max(5, min(100, int(m_vol_pct.group(1))))
            self.mixer.set_master_volume(pct / 100.0)
            self.speak(f"Volumen ajustado al {pct} por ciento.")
            return

        match = route_match or self.router.route(user_text)
        if match:
            action = match.get("action")
            logger.info(f"[ROUTE] Action={action} Target={match.get('target') or match.get('track_id')}")

            if action == "VOLUME_DOWN":
                new_v = self.mixer.adjust_master_volume(-0.10)
                self.speak(f"Volumen al {int(round(new_v * 100))} por ciento.")
                return
            elif action == "VOLUME_UP":
                new_v = self.mixer.adjust_master_volume(+0.10)
                self.speak(f"Volumen al {int(round(new_v * 100))} por ciento.")
                return
            elif action in ("STOP_ALL", "STOP_SPEECH"):
                self.stop_everything()
                return
            elif action == "EXIT_ADORACION":
                self.state.exit_adoracion(mixer=self.mixer)
                return
            elif action == "ENTER_ADORACION_DIRECT":
                self.mixer.stop_speech()
                self.mixer.stop_music(fade_out_sec=1.0)
                self.state.enter_adoracion(mixer=self.mixer, track="MUS-005")
                return
            elif action == "ENTER_ADORACION_PARAISO":
                self.in_al_paraiso = True
                try:
                    self.mixer.stop_speech()
                    self.mixer.play_music("MUS-005", fade_in_sec=3.0)
                    time.sleep(3.0)
                    text = self.router.get_canonical_text(match)
                    self.speak_narrative(text, mode="AL_PARAISO")
                    self.state.enter_adoracion(mixer=self.mixer, track=None)
                finally:
                    self.in_al_paraiso = False
                return
            elif action == "PLAY_MUSIC":
                self.mixer.play_music(match.get("track_id", "MUS-003"))
                return
            elif action == "MUSIC_PAUSE":
                self.mixer.pause_music()
                return
            elif action == "MUSIC_RESUME":
                self.mixer.resume_music()
                return
            elif action == "MUSIC_STOP":
                self.mixer.stop_music()
                return
            elif action in ("READ_PRESENTACION", "READ_RELATO"):
                text = self.router.get_canonical_text(match)
                mode = "RELATO" if action == "READ_RELATO" else "NORMAL"
                self.speak_narrative(text, mode=mode)
                return

        duration_sec = self.state.inspect_and_set_duration(user_text)
        thematic_query = self.state.strip_duration_phrases(user_text)

        top_k = 3 if duration_sec <= 30 else (4 if duration_sec <= 60 else 5)
        rag_res = self.rag.search(thematic_query, top_k=top_k)

        allowed, literal_resp = self.prefilter.evaluate(thematic_query, rag_result=rag_res)
        if not allowed and literal_resp:
            self.speak(literal_resp)
            self.state.reset_duration()
            return

        logger.info(f"[SPEAK STREAMING]: duration_budget={duration_sec}s")
        self.mixer._stop_speech_event.clear()
        full_text, spoken_sec, ttfa_ms = self.llm.stream_and_speak(
            thematic_query, rag_res.get("results", []), self.mixer, self.tts, duration_sec=duration_sec
        )
        logger.info(f"[DONE] Spoken={spoken_sec:.2f}s | TTFA={ttfa_ms:.0f}ms | Text: {full_text}")
        self.state.reset_duration()

    def speak(self, text: str):
        logger.info(f"[SPEAK]: {text}")
        self.mixer._stop_speech_event.clear()
        self.mixer.duck_music()
        if self.tts:
            try:
                tts_text = expand_biblical_citations_for_tts(text)
                samples, sr = self.tts.create(tts_text, voice=config.TTS_VOICE_NAME, speed=config.SPEED_NORMAL, lang=config.TTS_LANG)
                self.mixer.play_speech(samples, sr)
            except Exception as e:
                logger.error(f"TTS speak error: {e}")
        self.mixer.unduck_music()

    def speak_narrative(self, text: str, mode: str = "NORMAL"):
        logger.info(f"[SPEAK NARRATIVE]: Mode={mode}, chars={len(text)}")
        self.mixer.stream_narrative(text, mode=mode, tts_engine=self.tts, use_cache_first_chunk=(mode == "NORMAL"))

    def speak_pedagogical(self, text: str, max_sec: float = 30.0):
        logger.info(f"[SPEAK PEDAGOGICAL]: max_sec={max_sec}, text={text}")
        self.mixer._stop_speech_event.clear()
        self.mixer.duck_music()
        chunks, total_sec = self.llm.synthesize_with_watchdog(text, self.mixer, kokoro_engine=self.tts, max_sec=max_sec)
        for samples, sr, _ in chunks:
            if len(samples) > 0:
                self.mixer.play_speech(samples, sr)
        self.mixer.unduck_music()


def main():
    app = VoxVerbiApp()
    # Restore SIGTERM handler after pygame/SDL2 init so systemd stops/restarts cleanly in <0.2s
    signal.signal(signal.SIGTERM, lambda signum, frame: (_ for _ in ()).throw(KeyboardInterrupt()))
    hardware_hooks.start_s330_volume_buttons_listener(app.mixer)
    logger.info(f"VOX VERBI v3.5 initialized and ready (master_volume={int(round(app.mixer.master_volume * 100))}%).")

    pa = pyaudio.PyAudio()
    mic_idx = find_usb_microphone(pa)

    vad = webrtcvad.Vad(config.VAD_AGGRESSIVENESS)
    rate = config.SAMPLE_RATE_VAD
    chunk_ms = 30
    chunk_size = int(rate * chunk_ms / 1000)
    hw_rate = rate
    hw_channels = 1
    hw_chunk_size = chunk_size

    try:
        stream = pa.open(
            format=pyaudio.paInt16, channels=1, rate=rate,
            input=True, input_device_index=mic_idx, frames_per_buffer=chunk_size,
        )
    except Exception:
        try:
            dev_info = pa.get_device_info_by_index(mic_idx)
            hw_rate = int(dev_info.get("defaultSampleRate", 48000))
            max_ch = int(dev_info.get("maxInputChannels", 2))
            hw_chunk_size = int(hw_rate * chunk_ms / 1000)
            try:
                hw_channels = min(max(max_ch, 1), 2)
                stream = pa.open(
                    format=pyaudio.paInt16, channels=hw_channels, rate=hw_rate,
                    input=True, input_device_index=mic_idx, frames_per_buffer=hw_chunk_size,
                )
            except Exception:
                hw_channels = 1
                stream = pa.open(
                    format=pyaudio.paInt16, channels=hw_channels, rate=hw_rate,
                    input=True, input_device_index=mic_idx, frames_per_buffer=hw_chunk_size,
                )
            logger.info(f"Opened microphone at native hardware format: {hw_rate} Hz, {hw_channels} ch (downsampling to {rate} Hz mono)")
        except Exception as e:
            logger.error(f"Error opening audio input: {e}")
            return

    # 85 Hz Butterworth high-pass filter removes table/fan/AC rumble (<85Hz) so room hum
    # never triggers false '[Música]' transcriptions while keeping 100% of human voice energy!
    hpf_sos = scipy.signal.butter(2, 85.0 / (0.5 * rate), btype="highpass", output="sos")
    hpf_zi = scipy.signal.sosfilt_zi(hpf_sos) * 0.0

    # Dedicated background capture thread so STT decoding never drops incoming microphone frames
    frame_queue = queue.Queue(maxsize=400)
    stop_capture = threading.Event()

    def _capture_loop():
        nonlocal hpf_zi
        while not stop_capture.is_set():
            try:
                raw_chunk = stream.read(hw_chunk_size, exception_on_overflow=False)
            except Exception:
                time.sleep(0.01)
                continue

            # During 'Al Paraíso' sacred preparation, ignore mic input completely so nothing interrupts it
            if app.in_al_paraiso:
                continue

            if hw_rate == rate and hw_channels == 1:
                arr16 = np.frombuffer(raw_chunk, dtype=np.int16).astype(np.float32)
            else:
                arr = np.frombuffer(raw_chunk, dtype=np.int16).astype(np.float32)
                if hw_channels > 1:
                    # Pick the stronger channel or clean average if balanced
                    ch_views = arr.reshape(-1, hw_channels)
                    e0 = float(np.mean(np.abs(ch_views[:, 0])))
                    e1 = float(np.mean(np.abs(ch_views[:, 1])))
                    if e1 < 0.2 * e0:
                        arr = ch_views[:, 0]
                    elif e0 < 0.2 * e1:
                        arr = ch_views[:, 1]
                    else:
                        arr = ch_views.mean(axis=1)
                if hw_rate == 48000 and rate == 16000:
                    arr16 = scipy.signal.resample_poly(arr, 1, 3).astype(np.float32)
                elif hw_rate != rate:
                    gcd = np.gcd(rate, hw_rate)
                    arr16 = scipy.signal.resample_poly(arr, rate // gcd, hw_rate // gcd).astype(np.float32)
                else:
                    arr16 = arr

            filtered, hpf_zi = scipy.signal.sosfilt(hpf_sos, arr16, zi=hpf_zi)
            pcm_i16 = np.clip(filtered, -32768, 32767).astype(np.int16)
            chunk_bytes = pcm_i16.tobytes()
            if len(chunk_bytes) != chunk_size * 2:
                continue
            try:
                frame_queue.put_nowait((chunk_bytes, float(np.sqrt(np.mean(filtered ** 2)))))
            except queue.Full:
                try:
                    frame_queue.get_nowait()
                    frame_queue.put_nowait((chunk_bytes, float(np.sqrt(np.mean(filtered ** 2)))))
                except Exception:
                    pass

    logger.info("Microphone stream opened. Listening for 'Vox Verbi'...")
    try:
        app.speak("Vox Verbi está listo.")
    except Exception as e:
        logger.warning(f"Startup announcement skipped: {e}")

    cap_thread = threading.Thread(target=_capture_loop, daemon=True)
    cap_thread.start()

    pre_roll_buffer = collections.deque(maxlen=config.VAD_PREROLL_LEN)
    ring_buffer = collections.deque(maxlen=config.VAD_RING_LEN)
    silence_buffer = collections.deque(maxlen=config.VAD_SILENCE_LEN)
    voiced_frames = []
    triggered = False
    started_during_speech = False
    was_raw_speaking = False
    was_speaking_now = False
    peak_rms = 0.0

    def _flush_after_response():
        while not frame_queue.empty():
            try:
                frame_queue.get_nowait()
            except Exception:
                break
        pre_roll_buffer.clear()
        ring_buffer.clear()
        silence_buffer.clear()

    def _is_genuine_question_or_request(rest_text: str) -> bool:
        """
        Ensures Vox Verbi ONLY answers real questions or explicit catechetical requests
        after 'Vox Verbi' at the beginning — rejecting repeated words ('no no no', 'por favor por favor'),
        isolated wake-word echoes, or random conversational noise!
        """
        norm_r = normalize_text(rest_text)
        if not norm_r or norm_r == "vox verbi":
            return False
        words = [w for w in norm_r.split() if len(w) > 1]
        unique_words = set(words) - {"vox", "verbi", "por", "favor", "no", "si", "eh", "ah"}
        if len(unique_words) < 2:
            return False
        # Require an interrogative, imperative request verb, or explicit religious/catechetical root
        request_cues = (
            r"\b(que|quien|quienes|como|cual|cuales|cuando|donde|por\s+que|para\s+que|"
            r"explic\w*|habl\w*|dime|cuent\w*|ensen\w*|dice|signific\w*|respond\w*|"
            r"profundiz\w*|desarroll\w*|describ\w*|resum\w*|defin\w*|present\w*|"
            r"quiero|puedes|podrias|leeme|lee|muestr\w*)\b"
        )
        if "?" in rest_text or "¿" in rest_text or re.search(request_cues, norm_r):
            return True
        if any(root in norm_r for root in app.prefilter.RELIGIOUS_ROOTS):
            return True
        return False

    def _check_barge_in_interrupt(text: str) -> str | None:
        """
        Detects explicit interrupt or volume commands ('Vox Verbi, detente', 'Vox Verbi, baja el volumen')
        with 'Vox Verbi' at the start while Vox Verbi is speaking!
        """
        if not text:
            return None
        has_wake, rest = match_and_strip_wake_word(text, strict_leading=True)
        if not has_wake or not rest:
            return None
        norm_rest = normalize_text(rest)
        if re.search(r"\b(baja\w*|menos|mas\s+bajo|bajito)\b", norm_rest) and "volumen" in norm_rest:
            return "VOLUME_DOWN"
        if re.search(r"\b(sube\w*|subir|mas\s+alto|mas\s+fuerte|mas)\b.*\bvolumen\b", norm_rest):
            return "VOLUME_UP"
        if re.search(r"^(detent\w*|deten\w*|para\b|parar\b|basta\b|calla\w*|cancela\w*|silencio\b|alto\b)", norm_rest):
            return "STOP_ALL"
        return None

    try:
        while True:
            chunk, rms = frame_queue.get()

            raw_speech = vad.is_speech(chunk, rate)
            raw_speaking = app.mixer.is_busy() or app.is_worker_active()
            if was_raw_speaking and not raw_speaking:
                # Clear any TTS echo frames from pre-roll the instant Vox Verbi finishes speaking
                pre_roll_buffer.clear()
                ring_buffer.clear()
                silence_buffer.clear()
            was_raw_speaking = raw_speaking

            speaking_now = raw_speaking or ((time.time() - app.mixer.last_speech_end_time) < 0.55)
            if was_speaking_now and not speaking_now:
                # Clear residual 100%-volume loudspeaker reverb tail after 0.55s ALSA drain
                pre_roll_buffer.clear()
                ring_buffer.clear()
                silence_buffer.clear()
            was_speaking_now = speaking_now
            music_active = app.mixer.is_playing_music and not app.mixer.is_paused

            if speaking_now:
                base_floor = getattr(config, "RMS_THRESHOLD_SPEAKING", 420.0)
            elif music_active:
                base_floor = config.RMS_THRESHOLD_WITH_MUSIC
            else:
                base_floor = config.RMS_THRESHOLD_IDLE

            if triggered:
                if rms > peak_rms:
                    peak_rms = rms
                active_floor = min(450.0, max(base_floor * 0.80, peak_rms * 0.14))
                is_speech_effective = raw_speech and (rms > active_floor)
            else:
                is_speech_effective = raw_speech and (rms > base_floor)

            if not triggered:
                pre_roll_buffer.append(chunk)
                ring_buffer.append((chunk, is_speech_effective))
                num_voiced = sum(1 for _, sp in ring_buffer if sp)

                if not speaking_now:
                    if music_active and num_voiced >= 3 and not app.mixer.ducking_active:
                        app.mixer.set_ducking(True, ducked_vol=0.08)
                    elif music_active and num_voiced == 0 and app.mixer.ducking_active:
                        app.mixer.set_ducking(False)

                if num_voiced >= config.VAD_ONSET_MIN_VOICED:
                    triggered = True
                    started_during_speech = speaking_now
                    peak_rms = rms
                    if not speaking_now:
                        logger.info(f"🗣️ Voz detectada (RMS={rms:.0f})")
                        if music_active:
                            app.mixer.set_ducking(True, ducked_vol=0.08)
                    voiced_frames = list(pre_roll_buffer)
                    ring_buffer.clear()
                    silence_buffer.clear()
            else:
                voiced_frames.append(chunk)
                silence_buffer.append((chunk, is_speech_effective))

                req_sil_len = 25 if started_during_speech else silence_buffer.maxlen
                recent_sil = list(silence_buffer)[-req_sil_len:]
                num_unvoiced = sum(1 for _, sp in recent_sil if not sp)
                silence_cut = (len(recent_sil) == req_sil_len and num_unvoiced >= int(0.82 * req_sil_len))
                timeout_cut = len(voiced_frames) >= (160 if started_during_speech else config.MAX_RECORD_CHUNKS)

                if silence_cut or timeout_cut:
                    triggered = False
                    was_barge_in = started_during_speech or app.mixer.is_busy() or app.is_worker_active()
                    started_during_speech = False
                    peak_rms = 0.0
                    audio_data = b"".join(voiced_frames)
                    tail_frames = voiced_frames[-20:] if len(voiced_frames) >= 20 else list(voiced_frames)
                    voiced_frames = []
                    ring_buffer.clear()
                    silence_buffer.clear()

                    t_stt0 = time.time()
                    result_text = app.stt.transcribe(audio_data, sample_rate=rate)
                    stt_ms = (time.time() - t_stt0) * 1000
                    if result_text and not was_barge_in:
                        logger.info(f"[STT {stt_ms:.0f}ms]: «{result_text}»")

                    # --- PATH A: Barge-in while Vox Verbi is actively speaking ---
                    if was_barge_in:
                        barge_action = _check_barge_in_interrupt(result_text)
                        if barge_action == "STOP_ALL":
                            logger.info(f"[BARGE-IN {stt_ms:.0f}ms]: «{result_text}» -> STOP_ALL")
                            app.stop_everything()
                            _flush_after_response()
                        elif barge_action == "VOLUME_DOWN":
                            logger.info(f"[BARGE-IN {stt_ms:.0f}ms]: «{result_text}» -> VOLUME_DOWN")
                            app.mixer.adjust_master_volume(-0.10)
                        elif barge_action == "VOLUME_UP":
                            logger.info(f"[BARGE-IN {stt_ms:.0f}ms]: «{result_text}» -> VOLUME_UP")
                            app.mixer.adjust_master_volume(+0.10)
                        continue

                    # --- PATH B: Strict Activation Gate (MUST begin with 'Vox Verbi' + real question/command!) ---
                    handled = False
                    if result_text:
                        has_wake, rest = match_and_strip_wake_word(result_text, strict_leading=True)
                        if has_wake and rest:
                            direct_route = app.router.route(rest) or app.router.route(f"vox verbi {rest}")
                            single_cmds = (
                                "defínete", "definete", "preséntate", "presentate", "silencio",
                                "detente", "detén", "deten", "para", "parar", "basta", "cancela", "cancelar",
                            )
                            clean_single = re.sub(r"[^\w]", "", rest).lower()

                            if direct_route and direct_route.get("action") in ("STOP_ALL", "STOP_SPEECH"):
                                logger.info(f"[WAKE+STOP]: «Vox Verbi, {rest}»")
                                app.stop_everything()
                                _flush_after_response()
                                handled = True
                            elif direct_route:
                                canonical_query = f"Vox Verbi, {rest}".strip()
                                logger.info(f"[WAKE+ROUTE]: «{canonical_query}»")
                                app.dispatch_async(canonical_query, route_match=direct_route)
                                _flush_after_response()
                                handled = True
                            elif clean_single in single_cmds or _is_genuine_question_or_request(rest):
                                canonical_query = f"Vox Verbi, {rest}".strip()
                                logger.info(f"[WAKE+QUERY]: «{canonical_query}»")
                                app.dispatch_async(canonical_query)
                                _flush_after_response()
                                handled = True
                            else:
                                logger.info(f"[IGNORED] 'Vox Verbi' sin pregunta/comando real: «{rest}»")
                        elif has_wake and not rest:
                            logger.info("[IGNORED] 'Vox Verbi' pronunciado sin pregunta posterior.")

                    if not handled:
                        for tf in tail_frames[-15:]:
                            pre_roll_buffer.append(tf)
                        if music_active:
                            app.mixer.set_ducking(False)

    except KeyboardInterrupt:
        logger.info("Terminating Vox Verbi main loop.")
    finally:
        stop_capture.set()
        cap_thread.join(timeout=1.0)
        stream.stop_stream()
        stream.close()
        pa.terminate()


if __name__ == "__main__":
    main()
