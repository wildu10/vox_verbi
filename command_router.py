import json
import os
import re
import unicodedata
from difflib import SequenceMatcher

try:
    from . import config
except ImportError:
    import config


def convert_digits_to_words(text: str) -> str:
    """Converts digits to Spanish word equivalents before matching commands or durations."""
    if not text:
        return ""
    t = text
    t = re.sub(r"\b30\s*(segundos|seg|s)\b", "treinta segundos", t, flags=re.IGNORECASE)
    t = re.sub(r"\b60\s*(segundos|seg|s)\b", "sesenta segundos", t, flags=re.IGNORECASE)
    t = re.sub(r"\b120\s*(segundos|seg|s)\b", "ciento veinte segundos", t, flags=re.IGNORECASE)
    t = re.sub(r"\b1\s*(minuto|min)\b", "un minuto", t, flags=re.IGNORECASE)
    t = re.sub(r"\b2\s*(minutos|min)\b", "dos minutos", t, flags=re.IGNORECASE)
    t = re.sub(r"\b3\s*(minutos|min)\b", "tres minutos", t, flags=re.IGNORECASE)
    t = re.sub(r"\b4\s*(minutos|min)\b", "cuatro minutos", t, flags=re.IGNORECASE)
    t = re.sub(r"\b3\s*cruces\b", "tres cruces", t, flags=re.IGNORECASE)

    num_map = [
        (r"\b0?1\b", "uno"),
        (r"\b0?2\b", "dos"),
        (r"\b0?3\b", "tres"),
        (r"\b0?4\b", "cuatro"),
        (r"\b0?5\b", "cinco"),
        (r"\b0?6\b", "seis"),
        (r"\b0?7\b", "siete"),
        (r"\b0?8\b", "ocho"),
        (r"\b0?9\b", "nueve"),
        (r"\b10\b", "diez"),
        (r"\b11\b", "once"),
    ]
    for pattern, word in num_map:
        t = re.sub(pattern, word, t)
    return t


# Comprehensive phonetic targets covering Spanish, Latin, and Whisper cross-lingual confusions
WAKE_TARGETS_3 = [
    "vos ver mi", "bos ver mi", "voz ver bi", "bos berg b", "bos ber bi",
    "vos ver bi", "box ver mi", "bos ver b", "vos ver b", "vox ver bi",
    "bos ber b", "voz ver mi", "pos ver mi", "boso better be", "vos better be",
    "boss better be", "mos ber bi", "was ber bi",
    "box verb y", "bos verb y", "vos verb y", "vox verb y", "vox ver vi", "box ver vi",
]

WAKE_TARGETS_2 = [
    "vox verbi", "vos verbi", "box verbi", "bos verbi", "voz verbi",
    "boss berby", "boss verbi", "bos verb", "vox verb", "box verb", "box berbi",
    "bos berbi", "pos verbi", "bos berg", "vos verb", "bos barbi",
    "box barbi", "ghost burby", "ghost verbi", "ghost berbi", "gost verbi",
    "gost berbi", "vox berbi", "vox derbi", "vos pervi", "bos pervi",
    "vox vervi", "box vervi", "bos vervi", "vos vervi", "box bervi", "vox bervi",
    "los berbi", "los verbis", "dos verbi", "dos berbi",
    "mos berbi", "mos verbi", "was berbi", "was verbi", "mas verbi",
    "pues verbi", "por verbi", "boso verbi",
]

WAKE_TARGETS_1 = [
    "busverby", "busverbi", "bosberbi", "bosbergvi", "bosberg", "voxverbi",
    "vosverbi", "boxverbi", "boxverby", "bosverby", "bosbiervi", "bosbierby", "busberby",
    "voxvervi", "boxvervi", "bosvervi", "vosvervi", "boxbervi", "voxbervi",
    "busverb", "bosverb", "ghostburby", "ghostberbi", "ghostverbi",
    "gostburby", "gostberbi", "gosberbi", "gosverbi", "oxverbi", "oxberbi",
    "osperbi", "osberbi", "obsberby", "obsverbi", "bosberti", "vosberti",
    "bosberghi", "vosberghi", "bosbergy", "vosbergy", "mosberbi", "mosberby",
    "wasberby", "wasberbi", "wosberbi", "wosberby",
]

# Tokens that belong to the QUESTION itself and must NEVER be swallowed as part of the wake-word
NON_WAKE_FOLLOW_TOKENS = {
    "que", "qué", "quien", "quién", "como", "cómo", "cual", "cuál", "cuales", "cuáles",
    "donde", "dónde", "cuando", "cuándo", "por", "para", "explicame", "explícame",
    "muestrame", "muéstrame", "hablame", "háblame", "cuentame", "cuéntame", "quiero",
    "pon", "ponga", "activa", "activar", "silencio", "continua", "continúa", "termina",
    "detente", "deten", "detén", "detener", "para", "parar", "basta", "cancela", "cancelar",
    "al", "el", "la", "las", "en", "un", "una", "definete", "defínete", "definity",
    "apartado", "iniciamos", "finalizamos", "dime", "enseñame", "enséñame", "leeme",
    "léeme", "responde", "profundiza", "es", "era", "son", "fue",
}

# Allowed conversational fillers BEFORE a wake-word (never interrogatives!)
ALLOWED_PRE_WAKE_FILLERS = {
    "oye", "hola", "eh", "em", "ah", "bueno", "entonces", "escucha",
    "disculpa", "perdona", "por", "favor", "mira", "a", "ver", "hermano", "bien",
}


def _strip_accents_basic(s: str) -> str:
    n = unicodedata.normalize("NFD", s.lower())
    return "".join(c for c in n if unicodedata.category(c) != "Mn")


def _repair_fused_post_wake(rest: str) -> str:
    """
    Repairs common Whisper syllable fusions right after 'Vox Verbi':
      - 'Bikin era Jesús' -> 'quién era Jesús'
      - 'cos la Biblia'   -> 'qué es la Biblia'
      - 'Definity' / 'define te' / 'definito' -> 'defínete'
      - 'que es la fea'   -> 'qué es la fe'
    """
    if not rest:
        return ""
    r = rest.strip()
    r = re.sub(r"^(?:bikin|vikin|pikin)\b", "quién", r, flags=re.IGNORECASE)
    r = re.sub(r"^(?:cos|kes|ques|case)\b", "qué es", r, flags=re.IGNORECASE)
    r = re.sub(r"^definitivamente([.!?]*)$", r"defínete\1", r, flags=re.IGNORECASE)
    r = re.sub(
        r"\b(?:definity|definite|definirte|definitiva|definito|de\s+finete|define\s+te|defini\s+te)\b",
        "defínete",
        r,
        count=1,
        flags=re.IGNORECASE,
    )
    r = re.sub(r"\b(?:que|qué)\s+es\s+la\s+fea\b", "qué es la fe", r, flags=re.IGNORECASE)
    return r.strip()


def _match_wake_at_start(words: list[str]) -> tuple[bool, str]:
    if not words:
        return False, ""

    # 0. Check fused syllable contractions on first 1-2 words:
    #    e.g. "Boxverbipara." -> (True, "para")
    #         "Mosbér Bikin era Jesús" -> words[0]="Mosbér", words[1]="Bikin"
    #         "los verbicos la Biblia" -> words[0]="los", words[1]="verbicos"
    w0_clean = _strip_accents_basic(re.sub(r"[^\w]", "", words[0]))
    m_fused1 = re.fullmatch(
        r"(?:vox|vos|bos|box|voz|mos|was)[bv]er[bv][iy](para|parar|detente|deten|basta|silencio|definete)",
        w0_clean,
    )
    if m_fused1:
        rest = (m_fused1.group(1) + " " + " ".join(words[1:])).strip()
        return True, _repair_fused_post_wake(rest)

    if len(words) >= 2:
        w1_clean = _strip_accents_basic(re.sub(r"[^\w]", "", words[1]))
        if re.fullmatch(r"(?:mos|bos|vos|box|vox|was|wos|pos|los|dos)ber", w0_clean) and re.match(r"^[bv]i(kin|quien|que)", w1_clean):
            tail_q = "quién" if "ki" in w1_clean or "qui" in w1_clean else "qué"
            rest = f"{tail_q} " + " ".join(words[2:])
            return True, _repair_fused_post_wake(rest)
        if w0_clean in ("vox", "vos", "bos", "box", "voz", "los", "mos", "was", "dos") and re.match(r"^[bv]er[bv]i(cos|kes|ques|para|detente)$", w1_clean):
            suffix = re.sub(r"^[bv]er[bv]i", "", w1_clean)
            lead = "qué es" if suffix in ("cos", "kes", "ques") else suffix
            rest = f"{lead} " + " ".join(words[2:])
            return True, _repair_fused_post_wake(rest)

    best_sim = 0.0
    best_n = 0

    for n in (3, 2, 1):
        if len(words) >= n:
            if n >= 2:
                last_tok = _strip_accents_basic(re.sub(r"[^\w]", "", words[n - 1]))
                if last_tok in NON_WAKE_FOLLOW_TOKENS:
                    continue
            prefix = " ".join(words[:n])
            prefix_clean = _strip_accents_basic(re.sub(r"[^\w\s]", "", prefix)).strip()
            if len(prefix_clean) < 5:
                continue

            targets = WAKE_TARGETS_3 if n == 3 else (WAKE_TARGETS_2 if n == 2 else WAKE_TARGETS_1)
            thresh = 0.74 if n == 3 else 0.72
            for target in targets:
                sim = SequenceMatcher(None, prefix_clean, target).ratio()
                if sim >= thresh and sim > best_sim:
                    best_sim = sim
                    best_n = n

    if best_n > 0:
        rest = " ".join(words[best_n:]).lstrip(" ,.:;!?¿¡-")
        rest = re.sub(r"^(?:mi|b|bi|vi|y|i)\b[\s,.:;!?¿¡-]*", "", rest, flags=re.IGNORECASE).lstrip(" ,.:;!?¿¡-")
        return True, _repair_fused_post_wake(rest)

    # Broad phonetic syllable matcher for 1 or 2 leading tokens
    for n in (1, 2):
        if len(words) >= n:
            if n == 2:
                last_tok = _strip_accents_basic(re.sub(r"[^\w]", "", words[1]))
                if last_tok in NON_WAKE_FOLLOW_TOKENS:
                    continue
            prefix_clean = _strip_accents_basic(re.sub(r"[^\w\s]", "", " ".join(words[:n]))).strip()
            norm = prefix_clean.replace(" ", "")
            starts = (
                "bo", "vo", "po", "bu", "vu", "mo", "ma", "wa", "wo", "lo", "do",
                "box", "vox", "bos", "bus", "vos", "voz", "mos", "was", "wos", "gh", "go", "ox",
            )
            bodies = (
                "erb", "erbi", "erby", "verbi", "berbi", "verby", "berby", "vervi", "bervi",
                "biervi", "bierby", "burb", "derb", "berg", "perv", "perb",
            )
            endings = ("bi", "by", "vi", "vy", "gy", "gi", "ti", "erbi", "erby", "erb", "erv", "erg", "is")
            if 6 <= len(norm) <= 13 and any(norm.startswith(s) for s in starts) and any(b in norm for b in bodies) and any(norm.endswith(e) for e in endings):
                rest = " ".join(words[n:]).lstrip(" ,.:;!?¿¡-")
                rest = re.sub(r"^(?:mi|b|bi|vi|y|i)\b[\s,.:;!?¿¡-]*", "", rest, flags=re.IGNORECASE).lstrip(" ,.:;!?¿¡-")
                return True, _repair_fused_post_wake(rest)

    return False, ""


WAKE_WORD_REGEX = re.compile(
    r"\b(?:"
    r"(?:vox|box|voz|vos|boss|bos|pos|mos|was|wos|los|dos|ghost|gost)\s*"
    r"(?:verbi|berbi|derbi|verby|berby|bergy|burby|barbi|barbie|ver\s+vi|ber\s+vi|verbis|ver\s+mi|better\s+be)"
    r"|busverby|busverbi|bosbergvi|bosberbi|bosbergy|bosberg|busberby|mosberby|mosberbi|wasberby|wasberbi|ghostburby|ghostberbi|ghostverbi|gostburby"
    r")\b",
    flags=re.IGNORECASE,
)


def match_and_strip_wake_word(text: str, strict_leading: bool = False) -> tuple[bool, str]:
    """
    Wake-word detector and extractor:
      - When strict_leading=True (live microphone mode): ONLY accepts 'Vox Verbi' at the
        beginning of the utterance (index 0, or preceded by at most 1 greeting word 'oye'/'hola'),
        and strips any repeated consecutive wake-words so 'Vox Verbi, Vox Verbi' -> (True, '').
      - When strict_leading=False (router text normalization): also normalizes self-referential
        questions like '¿Qué es Vox Verbi?'.
    """
    if not text:
        return False, ""
    cleaned = re.sub(r'^[¿¡"«\s]+', "", text.strip())
    cleaned_no_quotes = re.sub(r'["«»“”]', "", cleaned).strip()
    words = cleaned_no_quotes.split()
    if not words:
        return False, text

    # 1. Check leading wake-word at index 0
    matched, rest = _match_wake_at_start(words)
    if matched:
        # Strip any accidental repeated wake-words ("Vox Verbi, Vox Verbi" -> "")
        # UNLESS preceded by an interrogative ("¿qué es Vox Verbi?")!
        while rest:
            r_words = re.sub(r'^[¿¡"«\s,.:;-]+', "", rest).split()
            if not r_words:
                rest = ""
                break
            m2, rest2 = _match_wake_at_start(r_words)
            if m2:
                rest = rest2
            else:
                break
        rest = WAKE_WORD_REGEX.sub("Vox Verbi", rest).strip(" ,.:;-")
        return True, rest

    # 2. Check after 1 brief vocative ("oye", "hola") ONLY
    allowed_vocatives = {"oye", "hola"} if strict_leading else ALLOWED_PRE_WAKE_FILLERS
    max_offset = min(len(words) - 1, 1 if strict_leading else 3)
    for offset in range(1, max_offset + 1):
        prev_tok = _strip_accents_basic(re.sub(r"[^\w]", "", words[offset - 1]))
        if prev_tok not in allowed_vocatives:
            break
        matched, rest = _match_wake_at_start(words[offset:])
        if matched:
            while rest:
                r_words = re.sub(r'^[¿¡"«\s,.:;-]+', "", rest).split()
                if not r_words:
                    rest = ""
                    break
                m2, rest2 = _match_wake_at_start(r_words)
                if m2:
                    rest = rest2
                else:
                    break
            rest = WAKE_WORD_REGEX.sub("Vox Verbi", rest).strip(" ,.:;-")
            return True, rest

    if strict_leading:
        return False, text

    # 3. Non-strict mode (for internal router normalization / test cases with mid-sentence wake-word)
    m = WAKE_WORD_REGEX.search(cleaned_no_quotes)
    if m:
        before = cleaned_no_quotes[:m.start()].strip(" ,.:;!?¿¡-")
        after = cleaned_no_quotes[m.end():].strip(" ,.:;!?¿¡-")
        after = re.sub(r"^(?:mi|b|bi|vi)\b[\s,.:;!?¿¡-]*", "", after, flags=re.IGNORECASE).strip()

        if before:
            raw_before = cleaned_no_quotes[:m.start()].rstrip()
            if after:
                before_words = before.split()
                after_words = after.split()
                if len(before_words) <= 2 and len(after_words) >= 2 and _strip_accents_basic(before_words[0]) not in NON_WAKE_FOLLOW_TOKENS:
                    return True, _repair_fused_post_wake(WAKE_WORD_REGEX.sub("Vox Verbi", after))
            elif raw_before.endswith((",", ".", ";", ":")):
                return True, _repair_fused_post_wake(before)
            full_q = WAKE_WORD_REGEX.sub("Vox Verbi", cleaned_no_quotes).strip()
            return True, _repair_fused_post_wake(full_q)

        return True, _repair_fused_post_wake(after)

    return False, text


def canonicalize_wake_word(text: str) -> str:
    if not text:
        return ""
    matched, rest = match_and_strip_wake_word(text)
    if matched:
        return f"vox verbi {rest}".strip()
    t = WAKE_WORD_REGEX.sub("vox verbi", text)
    return re.sub(r'["«»“”]', " ", t)


def normalize_text(text: str) -> str:
    """Removes accents, punctuation, converts digits to words, canonicalizes wake word, and lowercases."""
    if not text:
        return ""
    t = canonicalize_wake_word(text)
    t = t.lower().strip()
    t = convert_digits_to_words(t)
    t = unicodedata.normalize("NFD", t)
    t = "".join(c for c in t if unicodedata.category(c) != "Mn")
    t = re.sub(r"[^a-z0-9\s]", " ", t)
    t = re.sub(r"\s+", " ", t).strip()
    return t


class CommandRouter:
    """
    Deterministic O(1) + Intent-Aware router for:
      1. Adoration / Entronización
      2. Music Catalog & Controls
      3. Guía Fundamental de Vox Verbi (12 sections + natural preacher questions in < 1 ms)
      4. Canonical Narratives / Relatos
    """

    def __init__(self, canonical_json_path=config.CANONICAL_JSON_PATH, music_catalog_path=config.MUSIC_CATALOG_PATH):
        self.canonical_json_path = canonical_json_path
        self.music_catalog_path = music_catalog_path
        self.canonical_data = {}
        self.music_catalog = {}
        self.routes = []
        self.exact_map = {}
        self._load_data()

    def _load_data(self):
        if os.path.exists(self.canonical_json_path):
            with open(self.canonical_json_path, "r", encoding="utf-8") as f:
                self.canonical_data = json.load(f)

        if os.path.exists(self.music_catalog_path):
            with open(self.music_catalog_path, "r", encoding="utf-8") as f:
                self.music_catalog = json.load(f)

        self._build_routes()

    def _build_routes(self):
        self.routes = []

        # 1. Adoration commands
        for cmd in [
            "vox verbi finalizamos la adoracion", "vox verbi finalizamos adoracion",
            "vox verbi finaliza la adoracion", "vox verbi finaliza adoracion",
            "vox verbi finalizamos la doracion", "vox verbi finaliza la doracion",
            "vox verbi terminamos la adoracion", "vox verbi termina la adoracion",
        ]:
            self.routes.append({"command": cmd, "action": "EXIT_ADORACION", "priority": 100})

        for cmd in [
            "vox verbi iniciamos la entronizacion del santisimo",
            "vox verbi iniciamos la entronizacion",
            "vox verbi entronizacion del santisimo",
        ]:
            self.routes.append({"command": cmd, "action": "ENTER_ADORACION_DIRECT", "priority": 95})

        for cmd in [
            "vox verbi al paraiso", "vox verbi el paraiso",
            "al paraiso", "el paraiso", "vox verbi al paraíso", "al paraíso",
        ]:
            self.routes.append({"command": cmd, "action": "ENTER_ADORACION_PARAISO", "priority": 100})

        # 2. Music control commands
        for cmd in ["vox verbi silencio", "vox verbi pausa la musica", "vox verbi pausar la musica"]:
            self.routes.append({"command": cmd, "action": "MUSIC_PAUSE", "priority": 85})
        for cmd in ["vox verbi continua la musica", "vox verbi continuar la musica", "vox verbi reanuda la musica"]:
            self.routes.append({"command": cmd, "action": "MUSIC_RESUME", "priority": 85})
        for cmd in [
            "vox verbi termina la musica", "vox verbi terminar la musica", "vox verbi terminaron la musica",
            "vox verbi finaliza la musica", "vox verbi finalizar la musica", "vox verbi final a la musica",
            "vox verbi para la musica", "vox verbi parar la musica", "vox verbi deten la musica",
            "vox verbi detener la musica", "vox verbi apaga la musica", "vox verbi quita la musica",
        ]:
            self.routes.append({"command": cmd, "action": "MUSIC_STOP", "priority": 85})

        for trk in self.music_catalog.get("tracks", []):
            t_id = trk.get("id")
            for cmd in trk.get("commands", []):
                self.routes.append({"command": cmd, "action": "PLAY_MUSIC", "track_id": t_id, "priority": 80})

        gen_cmds = list(self.music_catalog.get("general_commands", [])) + [
            "vox verbi pon musica suave para orar", "vox verbi pon musica para orar",
            "vox verbi musica suave para orar", "vox verbi musica para orar",
            "vox verbi pon musica suave", "vox verbi pon musica de fondo",
            "vox verbi pon musica", "vox verbi con musica",
            "vox verbi poner musica", "vox verbi reproduce musica",
        ]
        for cmd in gen_cmds:
            self.routes.append({"command": cmd, "action": "PLAY_MUSIC", "track_id": "MUS-003", "priority": 75})

        # 2B. Volume & Speech-Stop controls
        for cmd in [
            "vox verbi baja el volumen", "vox verbi bajar el volumen", "vox verbi menos volumen",
            "vox verbi habla mas bajo", "vox verbi mas bajito",
        ]:
            self.routes.append({"command": cmd, "action": "VOLUME_DOWN", "priority": 88})
        for cmd in [
            "vox verbi sube el volumen", "vox verbi subir el volumen", "vox verbi mas volumen",
            "vox verbi habla mas fuerte", "vox verbi mas alto",
        ]:
            self.routes.append({"command": cmd, "action": "VOLUME_UP", "priority": 88})
        for cmd in [
            "vox verbi detente", "vox verbi deten", "vox verbi detener", "vox verbi detenga",
            "vox verbi detengase", "vox verbi basta", "vox verbi para", "vox verbi parar",
            "vox verbi para ya", "vox verbi para de hablar", "vox verbi deja de hablar",
            "vox verbi callate", "vox verbi cancela", "vox verbi cancelar", "vox verbi alto",
        ]:
            self.routes.append({"command": cmd, "action": "STOP_ALL", "priority": 92})

        # 3. Guía Fundamental de Vox Verbi (12 canonical sections + natural questions)
        pres_mappings = [
            ("vox verbi muestrame para que fuiste creado", "COMPLETA"),
            ("vox verbi lee tu presentacion completa", "COMPLETA"),
            ("vox verbi presentacion completa", "COMPLETA"),
            ("vox verbi muestrame para que fuiste hecho", "SEC_PRESENTACION"),
            ("vox verbi para que fuiste hecho", "SEC_PRESENTACION"),
            ("vox verbi cual es tu funcionamiento", "SEC_PRESENTACION"),
            ("vox verbi cual es su funcionamiento", "SEC_PRESENTACION"),
            ("vox verbi como funcionas", "SEC_PRESENTACION"),
            ("vox verbi como funciona vox verbi", "SEC_PRESENTACION"),
            ("vox verbi de que conviccion nace vox verbi", "SEC_PRESENTACION"),
            ("vox verbi presentate", "SEC_PRESENTACION"),
            ("vox verbi apartado cero", "SEC_PRESENTACION"),
            ("vox verbi explicame tu fundamento", "SEC_01"),
            ("vox verbi cual es tu fundamento", "SEC_01"),
            ("vox verbi cual es su fundamento", "SEC_01"),
            ("vox verbi cual es el fundamento de vox verbi", "SEC_01"),
            ("vox verbi en que te fundamentas", "SEC_01"),
            ("vox verbi en que se fundamenta vox verbi", "SEC_01"),
            ("vox verbi apartado uno", "SEC_01"),
            ("vox verbi como entiendes la evangelizacion", "SEC_02"),
            ("vox verbi que entiendes por evangelizacion", "SEC_02"),
            ("vox verbi como entiende vox verbi la evangelizacion", "SEC_02"),
            ("vox verbi apartado dos", "SEC_02"),
            ("vox verbi como conduces al encuentro con cristo", "SEC_03"),
            ("vox verbi como conduce vox verbi al encuentro con cristo", "SEC_03"),
            ("vox verbi llevame al encuentro con cristo", "SEC_03"),
            ("vox verbi como llevas al encuentro con cristo", "SEC_03"),
            ("vox verbi apartado tres", "SEC_03"),
            ("vox verbi que ofreces a quien no cree", "SEC_04"),
            ("vox verbi que le ofreces a quien no cree", "SEC_04"),
            ("vox verbi que ofreces al no creyente", "SEC_04"),
            ("vox verbi apartado cuatro", "SEC_04"),
            ("vox verbi que lugar tiene la belleza en tu mision", "SEC_05"),
            ("vox verbi que lugar ocupa la belleza en tu mision", "SEC_05"),
            ("vox verbi que papel tiene la belleza en vox verbi", "SEC_05"),
            ("vox verbi apartado cinco", "SEC_05"),
            ("vox verbi cuales son tus criterios eclesiales", "SEC_06"),
            ("vox verbi cuales son sus criterios eclesiales", "SEC_06"),
            ("vox verbi cuales son los criterios eclesiales de vox verbi", "SEC_06"),
            ("vox verbi apartado seis", "SEC_06"),
            ("vox verbi cual es tu responsabilidad eclesial", "SEC_07"),
            ("vox verbi cual es su responsabilidad eclesial", "SEC_07"),
            ("vox verbi apartado siete", "SEC_07"),
            ("vox verbi cual es tu finalidad pastoral", "SEC_08"),
            ("vox verbi cual es su finalidad pastoral", "SEC_08"),
            ("vox verbi cual es tu objetivo pastoral", "SEC_08"),
            ("vox verbi apartado ocho", "SEC_08"),
            ("vox verbi que propones a la iglesia de hoy", "SEC_09"),
            ("vox verbi que le propones a la iglesia de hoy", "SEC_09"),
            ("vox verbi apartado nueve", "SEC_09"),
            ("vox verbi cual es tu vision", "SEC_10"),
            ("vox verbi cual es su vision", "SEC_10"),
            ("vox verbi cual es la vision de vox verbi", "SEC_10"),
            ("vox verbi apartado diez", "SEC_10"),
            ("vox verbi definete", "SEC_11"),
            ("vox verbi define", "SEC_11"),
            ("vox verbi definicion", "SEC_11"),
            ("vox verbi definity", "SEC_11"),
            ("vox verbi que es vox verbi", "SEC_11"),
            ("vox verbi quien es vox verbi", "SEC_11"),
            ("vox verbi que eres", "SEC_11"),
            ("vox verbi quien eres", "SEC_11"),
            ("vox verbi quien eres tu", "SEC_11"),
            ("vox verbi explicame que es vox verbi", "SEC_11"),
            ("vox verbi dime que es vox verbi", "SEC_11"),
            ("vox verbi hablame de vox verbi", "SEC_11"),
            ("vox verbi para que sirve vox verbi", "SEC_11"),
            ("vox verbi cual es tu mision", "SEC_11"),
            ("vox verbi apartado once", "SEC_11"),
        ]
        for cmd, sec_id in pres_mappings:
            self.routes.append({
                "command": cmd,
                "action": "READ_PRESENTACION",
                "target": sec_id,
                "priority": 72,
            })

        # 4. Stories
        stories_mappings = [
            ("vox verbi las tres cruces", "VV-HIST-001"),
            ("vox verbi cuentame las tres cruces", "VV-HIST-001"),
            ("vox verbi hablame de las tres cruces", "VV-HIST-001"),
            ("vox verbi el sueno de los tres arboles", "VV-HIST-001"),
            ("vox verbi la cruz que nadie queria", "VOX_VERBI_RELATO_02_CRUZ_QUE_NADIE_QUERIA"),
            ("la cruz que nadie queria", "VOX_VERBI_RELATO_02_CRUZ_QUE_NADIE_QUERIA"),
            ("activar la cruz que nadie queria", "VOX_VERBI_RELATO_02_CRUZ_QUE_NADIE_QUERIA"),
            ("leer la cruz que nadie queria", "VOX_VERBI_RELATO_02_CRUZ_QUE_NADIE_QUERIA"),
            ("vox verbi el jardin descuidado", "VV-HIST-003"),
            ("vox verbi cuentame el jardin descuidado", "VV-HIST-003"),
        ]
        for cmd, story_id in stories_mappings:
            self.routes.append({
                "command": cmd,
                "action": "READ_RELATO",
                "target": story_id,
                "priority": 70,
            })

        self.exact_map = {}
        for r in self.routes:
            r["norm_cmd"] = normalize_text(r["command"])
            if r["norm_cmd"] not in self.exact_map or r["priority"] > self.exact_map[r["norm_cmd"]]["priority"]:
                self.exact_map[r["norm_cmd"]] = r
        self.routes.sort(key=lambda x: x.get("priority", 50), reverse=True)

    def _match_guia_intent(self, norm_query: str):
        body = re.sub(r"^vox\s+verbi\s+", "", norm_query).strip()
        if not body:
            return None

        self_cues = (
            "vox verbi" in body
            or bool(re.search(
                r"\b(tu|tus|su|sus|ti|eres|fuiste|naciste|entiendes?|conduces?|llevame|ofreces?|propones?|fundamentas?|definete|definity|presentate|funcionas?|funcionamiento|apartado)\b",
                body,
            ))
        )
        if not self_cues:
            return None

        intent_rules = [
            ("COMPLETA", [r"para\s+que\s+fuiste\s+creado", r"presentacion\s+completa", r"documento\s+completo"]),
            ("SEC_PRESENTACION", [r"para\s+que\s+fuiste\s+hecho", r"conviccion\s+fundamental", r"\bpresentate\b", r"\bfuncionamiento\b", r"\bcomo\s+funcionas?\b"]),
            ("SEC_01", [r"\bfundamento\b", r"\bfundamentas?\b", r"no\s+inventar\s+sino\s+transmitir"]),
            ("SEC_02", [r"(?:como\s+entiendes?|que\s+entiendes?|entiende\s+vox\s+verbi).*evangelizacion", r"rebaja.*contenido\s+de\s+la\s+fe"]),
            ("SEC_03", [r"encuentro\s+con\s+cristo", r"conocimiento\s+de\s+la\s+fe\s+al\s+encuentro"]),
            ("SEC_04", [r"(?:no\s+cree|no\s+creyente|alejados?\s+de\s+la\s+fe|distante\s+de\s+la\s+fe)"]),
            ("SEC_05", [r"\bbelleza\b.*\b(mision|misterio|vox\s+verbi|lugar|papel)\b", r"\b(mision|vox\s+verbi)\b.*\bbelleza\b"]),
            ("SEC_06", [r"criterios?\s+(?:eclesiales?|ecl\s+sociales?|de\s+la\s+iglesia)", r"sometida\s+a\s+criterios"]),
            ("SEC_07", [r"responsabilidad\s+eclesial"]),
            ("SEC_08", [r"(?:finalidad|horizonte|objetivo)\s+pastoral"]),
            ("SEC_09", [r"iglesia\s+de\s+hoy", r"propones?\s+a\s+la\s+iglesia"]),
            ("SEC_10", [r"\b(?:tu|su)\s+vision\b", r"vision\s+de\s+vox\s+verbi", r"infraestructura\s+pastoral"]),
            ("SEC_11", [
                r"^(?:que|quien)\s+es\s+vox\s+verbi$",
                r"\b(?:que|quien)\s+es\s+vox\s+verbi\b",
                r"^(?:quien|que)\s+eres(?:\s+tu)?$",
                r"\ben\s+una\s+definicion\b",
                r"\bdefinete\b",
                r"\bdefinity\b",
                r"\bcomo\s+se\s+define\s+vox\s+verbi\b",
                r"\bpara\s+que\s+sirve\s+vox\s+verbi\b",
            ]),
        ]
        for target_sec, patterns in intent_rules:
            for pat in patterns:
                if re.search(pat, body):
                    return {"command": norm_query, "action": "READ_PRESENTACION", "target": target_sec, "priority": 72}
        return None

    def route(self, user_query: str, threshold: float = 0.88):
        norm_query = normalize_text(user_query)
        if not norm_query or norm_query == "vox verbi":
            return None

        if norm_query in self.exact_map:
            return self.exact_map[norm_query]
        prefixed = norm_query if norm_query.startswith("vox verbi ") else f"vox verbi {norm_query}"
        if prefixed in self.exact_map:
            return self.exact_map[prefixed]

        guia_hit = self._match_guia_intent(prefixed)
        if guia_hit:
            return guia_hit

        best_match = None
        best_ratio = 0.0

        for r in self.routes:
            target_cmd = r["norm_cmd"]
            q_cmp = prefixed if target_cmd.startswith("vox verbi ") else norm_query
            if len(target_cmd) > 10 and (target_cmd in q_cmp or q_cmp in target_cmd):
                ratio = SequenceMatcher(None, q_cmp, target_cmd).ratio()
                if ratio >= threshold and ratio > best_ratio:
                    best_ratio = ratio
                    best_match = r
                elif ratio < threshold:
                    words_cmd = set(target_cmd.split())
                    words_query = set(q_cmp.split())
                    substantive_cmd_words = words_cmd - {"vox", "verbi", "vos", "bos"}
                    if len(substantive_cmd_words) >= 2 and substantive_cmd_words.issubset(words_query):
                        return r
            else:
                ratio = SequenceMatcher(None, q_cmp, target_cmd).ratio()
                if ratio >= threshold and ratio > best_ratio:
                    best_ratio = ratio
                    best_match = r

        return best_match

    def get_canonical_text(self, route_match):
        if not route_match or not self.canonical_data:
            return ""
        action = route_match.get("action")
        target = route_match.get("target")
        if action == "READ_PRESENTACION":
            if target == "COMPLETA":
                return self.canonical_data.get("presentacion", {}).get("completa", "")
            return self.canonical_data.get("presentacion", {}).get("secciones", {}).get(target, "")
        if action == "READ_RELATO":
            return self.canonical_data.get("relatos", {}).get(target, {}).get("texto", "")
        if action == "ENTER_ADORACION_PARAISO":
            return self.canonical_data.get("adoracion", {}).get("AL_PARAISO", "")
        return ""
