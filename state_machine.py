import re
import logging
from . import config
from . import hardware_hooks
from .command_router import normalize_text

logger = logging.getLogger("vox_verbi.state_machine")


class StateMachine:
    """Manages system modes (CATEQUESIS vs ADORACION) and dynamic oral duration budgets."""

    MODE_CATEQUESIS = "CATEQUESIS"
    MODE_ADORACION = "ADORACION"

    def __init__(self):
        self.mode = self.MODE_CATEQUESIS
        self.duration_limit_sec = 30
        self.is_custom_duration = False

    def is_adoracion(self) -> bool:
        return self.mode == self.MODE_ADORACION

    def enter_adoracion(self, mixer=None, track="MUS-005"):
        """Transitions into ADORACION mode, stops any ongoing speech, and triggers hardware hooks."""
        logger.info("[STATE] Entering ADORACION mode")
        if mixer and hasattr(mixer, "stop_speech"):
            mixer.stop_speech()
        self.mode = self.MODE_ADORACION
        hardware_hooks.on_enter_adoracion()
        if mixer and track:
            mixer.play_music(track, fade_in_sec=3.0)

    def exit_adoracion(self, mixer=None):
        """Exits ADORACION mode, triggers 4s fadeout and restores CATEQUESIS mode."""
        logger.info("[STATE] Exiting ADORACION mode")
        if mixer:
            mixer.stop_music(fade_out_sec=4.0)
        hardware_hooks.on_exit_adoracion()
        self.mode = self.MODE_CATEQUESIS
        self.reset_duration()

    def inspect_and_set_duration(self, user_query: str) -> int:
        """
        Inspects preacher query for natural oral duration requests:
        - 'en 30 segundos' / 'treinta segundos' / 'medio minuto' / 'brevemente' -> 30s
        - 'durante un minuto' / 'en un minuto' / 'por un minuto' / '60 segundos' -> 60s
        - 'durante dos minutos' / 'profundiza durante dos minutos' / '120 segundos' -> 120s
        - 'durante tres minutos' / 'por tres minutos' / '180 segundos' -> 180s
        - 'durante cuatro minutos' / 'por cuatro minutos' / '240 segundos' -> 240s
        Defaults to 30s and resets automatically after the response.
        """
        q_raw = (user_query or "").lower()
        query_norm = normalize_text(user_query)

        if re.search(r"\b(cuatro\s+minutos|240\s*segundos|doscientos\s+cuarenta\s+segundos)\b", query_norm) or re.search(r"\b4\s*(minutos|min)\b", q_raw):
            self.duration_limit_sec = 240
            self.is_custom_duration = True
        elif re.search(r"\b(tres\s+minutos|180\s*segundos|ciento\s+ochenta\s+segundos)\b", query_norm) or re.search(r"\b3\s*(minutos|min)\b", q_raw):
            self.duration_limit_sec = 180
            self.is_custom_duration = True
        elif re.search(r"\b(dos\s+minutos|120\s*segundos|ciento\s+veinte\s+segundos)\b", query_norm) or re.search(r"\b2\s*(minutos|min)\b", q_raw):
            self.duration_limit_sec = 120
            self.is_custom_duration = True
        elif re.search(r"\b(un\s+minuto|uno\s+minuto|60\s*segundos|sesenta\s+segundos)\b", query_norm) or re.search(r"\b1\s*(minuto|min)\b", q_raw):
            self.duration_limit_sec = 60
            self.is_custom_duration = True
        elif re.search(r"\b(treinta\s+segundos|30\s*segundos|medio\s+minuto|en\s+breve|brevemente)\b", query_norm) or re.search(r"\b30\s*(segundos|seg|s)\b", q_raw):
            self.duration_limit_sec = 30
            self.is_custom_duration = True
        else:
            self.duration_limit_sec = 30
            self.is_custom_duration = False

        logger.info(f"[STATE] Duration limit set to {self.duration_limit_sec}s (custom: {self.is_custom_duration})")
        return self.duration_limit_sec

    @staticmethod
    def strip_duration_phrases(user_query: str) -> str:
        """Removes temporal instruction clauses from the query so RAG focuses purely on the theological topic."""
        if not user_query:
            return ""
        t = user_query
        patterns = [
            r"\b(?:responde|hablame|háblame|explicame|explícame|profundiza|desarrolla|dime|comentame|coméntame)\s+(?:durante|por|en)\s+(?:un|uno|dos|tres|cuatro|1|2|3|4|30|60|120|180|240|treinta|sesenta|medio)\s+(?:minutos?|min|segundos?|seg)\s*(?:sobre|acerca\s+de|en\s+torno\s+a|de|del|la|el)?\b",
            r"\b(?:durante|por|en)\s+(?:un|uno|dos|tres|cuatro|1|2|3|4|30|60|120|180|240|treinta|sesenta|medio)\s+(?:minutos?|min|segundos?|seg)\b",
            r"\b(?:en\s+medio\s+minuto|brevemente)\b",
        ]
        for pat in patterns:
            t = re.sub(pat, " ", t, flags=re.IGNORECASE)
        t = re.sub(r"\s+,", ",", t)
        t = re.sub(r"\s+", " ", t).strip(" ,.-")
        return t or user_query

    def reset_duration(self):
        """Unconditionally resets duration limit to default 30 seconds after intervention."""
        self.duration_limit_sec = 30
        self.is_custom_duration = False
        logger.debug("[STATE] Duration limit reset to 30s")

    def get_duration_budget(self):
        return config.DURATION_BUDGETS.get(self.duration_limit_sec, config.DURATION_BUDGETS[30])
