import os

# Force offline mode for HuggingFace / Transformers at runtime
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"

# --- BASE DIRECTORIES & PATHS ---
BASE_DIR = "/home/wildu/llama.cpp"
VOICE_MODELS_DIR = os.path.join(BASE_DIR, "voice_models")

# Primary contemplative voice engine: Supertonic ONNX ("M5" warm reflective baritone, 44.1 kHz)
TTS_ENGINE = "supertonic"
TTS_VOICE_NAME = "M5"
TTS_LANG = "es"
SUPERTONIC_MODEL_DIR = os.path.join(VOICE_MODELS_DIR, "tts/supertonic")

# Fallback Kokoro ONNX paths (kept for backward compatibility)
TTS_MODEL_PATH = os.path.join(VOICE_MODELS_DIR, "tts/kokoro-v1.0.onnx")
TTS_VOICES_PATH = os.path.join(VOICE_MODELS_DIR, "tts/voices-v1.0.bin")

SPEECH_PEAK_NORM = 0.88
MASTER_OUTPUT_VOLUME = 1.00       # 100% master output volume (adjustable via physical S330 buttons or voice)

# Primary STT engine: whisper.cpp compiled with Vulkan (ggml-vulkan) on AMD Radeon 680M
# Uses Silero Neural VAD gate (<3ms) + whisper-server (ggml-small-q8_0.bin) with dynamic audio_ctx (~218ms avg),
# with automatic fallback to faster-whisper (int8 CPU) if whisper-server is unreachable.
# CRITICAL: Keep BOTH STT_INITIAL_PROMPT = None and STT_HOTWORDS = None so background speech never hallucinates 'Vox Verbi'.
STT_ENGINE = "whisper_cpp_vulkan"
WHISPER_SERVER_URL = "http://127.0.0.1:8090/inference"
WHISPER_CPP_MODEL_PATH = "/home/wildu/voice_models/stt/whisper_cpp/ggml-small-q8_0.bin"
STT_FW_MODEL_SIZE = "small"
STT_MODEL_DIR = os.path.join(VOICE_MODELS_DIR, "stt/sherpa-onnx-whisper-small")
STT_INITIAL_PROMPT = None
STT_HOTWORDS = None

# --- VAD & WAKE-WORD SENSITIVITY SETTINGS ---
VAD_AGGRESSIVENESS = 2            # Balanced sensitivity (preserves soft /v/ and /b/ onsets in pre-roll)
RMS_THRESHOLD_IDLE = 360.0        # Ignores distant room murmurs (<360) while 1.65s pre-roll captures soft 'Vox Verbi' onsets
RMS_THRESHOLD_WITH_MUSIC = 400.0  # Threshold over ducked background music
RMS_THRESHOLD_SPEAKING = 780.0    # Barge-in threshold calibrated for 100% speakerphone volume (for 'Vox Verbi, detente')
VAD_ONSET_MIN_VOICED = 6          # 6 of 12 chunks (180ms) of clear speech to trigger
VAD_RING_LEN = 12                 # 360ms onset window
VAD_PREROLL_LEN = 55              # 1.65s pre-roll audio preserved before VAD trigger
VAD_SILENCE_LEN = 50              # 1.50s post-speech pause tolerance (allows natural pause after 'Vox Verbi,' in 1 recording)
MAX_RECORD_CHUNKS = 400           # 12.0s max utterance length

DATA_DIR = os.path.join(BASE_DIR, "vox_verbi/data")
CANONICAL_JSON_PATH = os.path.join(DATA_DIR, "canonical_readings.json")
MUSIC_CATALOG_PATH = os.path.join(DATA_DIR, "music_catalog.json")
MUSIC_DIR = os.path.join(BASE_DIR, "vox_verbi/media/musica")
TTS_CACHE_DIR = os.path.join(DATA_DIR, "tts_cache")

FAISS_INDEX_PATH = os.path.join(DATA_DIR, "base_vox_verbi.index")
META_PKL_PATH = os.path.join(DATA_DIR, "textos_teologicos_meta.pkl")
BM25_PKL_PATH = os.path.join(DATA_DIR, "bm25_lexical.pkl")
EMBEDDER_MODEL_NAME = "intfloat/multilingual-e5-small"

CORPUS_WHITELIST_DIR = os.path.join(DATA_DIR, "corpus_whitelist")
BIBLIA_PATH = os.path.join(BASE_DIR, "biblia.txt")
BIBLIA_PROJECT_DIR = os.path.join(BASE_DIR, "La-Biblia-de-Jerusalen-Project")
CATECISMO_PATH = os.path.join(BASE_DIR, "catecismo.txt")

# --- LLM INFERENCE SETTINGS (Vulkan Radeon 680M) ---
LLAMA_SERVER_URL = "http://127.0.0.1:8080"
LLAMA_MODEL_NAME = "gemma-4-E4B-it-Q4_K_M.gguf"
LLAMA_MODEL_PATH = os.path.join(BASE_DIR, "models", LLAMA_MODEL_NAME)

# --- AUDIO & HARDWARE SETTINGS ---
INPUT_DEVICE_INDEX = 4
OUTPUT_ALSA = "plughw:CARD=S330,DEV=0"
SAMPLE_RATE_TTS = 44100
SAMPLE_RATE_MUSIC = 44100
SAMPLE_RATE_VAD = 16000

# --- AUDIO DUCKING PARAMETERS ---
MUSIC_GAIN_NORMAL = 0.45
MUSIC_GAIN_DUCKED = 0.18
LONG_PAUSE_SEC = 2.5

# --- PROSODIC INTERVALS (SECONDS) ---
PAUSE_BREVE = 1.2
PAUSE_3_SEC = 3.0
PAUSE_4_SEC = 4.0
PAUSE_5_SEC = 5.0
SILENCIO_6_SEC = 6.0
PAUSE_DASHES = 2.0
PAUSE_ELLIPSIS = 0.7

# --- CONTEMPLATIVE SPEED & GAIN MODULATION ---
SPEED_NORMAL = 1.00        # Natural serene M5 baritone cadence (ensures 3 full sentences fit <= 30s)
GAIN_NORMAL = 1.0

SPEED_RELATOS = 0.87       # Intimate narrative cadence for stories & Al Paraíso
GAIN_RELATOS = 0.84

GAIN_AL_PARAISO_START = 0.84
GAIN_AL_PARAISO_END = 0.58

# --- WORD & TOKEN BUDGETS BY DURATION ---
DURATION_BUDGETS = {
    30: {"max_words": 50, "max_tokens": 180, "target_spoken_sec": (18.0, 26.0), "max_sec": 30.0},
    60: {"max_words": 130, "max_tokens": 280, "target_spoken_sec": (48.0, 58.0), "max_sec": 62.0},
    120: {"max_words": 255, "max_tokens": 520, "target_spoken_sec": (95.0, 116.0), "max_sec": 124.0},
    180: {"max_words": 385, "max_tokens": 740, "target_spoken_sec": (145.0, 172.0), "max_sec": 184.0},
    240: {"max_words": 510, "max_tokens": 960, "target_spoken_sec": (195.0, 230.0), "max_sec": 244.0},
}
