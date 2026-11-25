import re
import logging
from typing import Optional, Tuple, Dict, Any

try:
    from .command_router import normalize_text
except ImportError:
    from command_router import normalize_text

logger = logging.getLogger("vox_verbi.pre_filter")

# --- LITERAL MANDATORY RESPONSES (INMUTABLES SEGÚN ESPECIFICACIÓN) ---
RESP_AUTORIA_SAGRADA = "No hablaré en nombre de esa persona; puedo explicarte lo que enseñan las fuentes autorizadas sobre este tema."
RESP_CONFESION = "Vox Verbi no administra el sacramento de la Reconciliación. Para la confesión sacramental debes acudir a un sacerdote."
RESP_ADIVINACION = "No puedo atribuir a Dios una respuesta personal que no esté contenida en la Revelación o en las fuentes autorizadas. Sí puedo mostrarte qué principios ofrece la Escritura y la enseñanza de la Iglesia sobre este tema."
RESP_MEDICO = "Vox Verbi no realiza diagnósticos médicos ni psicológicos. Por favor, acude a un profesional de la salud cualificado."
RESP_IMPERTINENTE = "Esta pregunta no corresponde al tema catequético de Vox Verbi. Permíteme continuar con el contenido de la fe que estamos profundizando."
RESP_SIN_FUENTES = "No encuentro en las fuentes autorizadas de Vox Verbi información suficiente para responder con seguridad a esta pregunta. Puedo ayudarte a formular la pregunta desde la Sagrada Escritura, el Magisterio o la Tradición."


class PreFilter:
    """Implements mandatory 5-step gatekeeper filter before LLM invocation."""

    IMPERSONATION_PATTERNS = [
        r"(habla|responde|dime|actua|hablame)\s+(como\s+si\s+fueras|como)\s+(jesus|jesucristo|dios|el\s+espiritu\s+santo|la\s+virgen|maria|san\s+|santa\s+|el\s+papa|francisco|un\s+santo|un\s+angel)",
        r"que\s+me\s+dirias\s+tu\s*(jesus|maria|dios|senor)?",
        r"\bse\s+(jesus|dios|maria|el\s+papa)\b",
        r"ponte\s+en\s+el\s+lugar\s+de\s+(jesus|dios|maria)",
    ]

    # Personal sacramental confession / absolution requests (Regla 10.F)
    # Carefully distinguishes personal sacramental solicitation from doctrinal questions ("¿qué es la reconciliación?")
    CONFESSION_SOLICITATION_PATTERNS = [
        r"\b(confesarme|confiesame|confieso\s+mis|quiero\s+confesar|me\s+quiero\s+confesar|vengo\s+a\s+confesar)\b",
        r"\b(absuelveme|dame\s+la\s+absolucion|darme\s+la\s+absolucion)\b",
        r"\bperdon(a|ame)\s+(mis\s+)?pecados\b",
        r"\b(mis\s+)?pecados\s+contigo\b",
        r"\bescucha\s+mi\s+confesion\b",
    ]

    DIVINATION_PATTERNS = [
        r"\b(sueno|suenos|sonar|sone)\b.*(dios|jesus|senor|virgen|revel|quiere\s+decir|significa|mensaje|interpret)",
        r"(dios|jesus|senor|virgen|revel|quiere\s+decir|significa|mensaje|interpret).*\b(sueno|suenos|sonar|sone)\b",
        r"\binterpret(ar?|ame|a|acion)?\s+(este|mi|el|un)?\s*sueno\b",
        r"\bque\s+significa\s+(mi|el|este|un)\s+sueno\b",
        r"\bque\s+me\s+(quiere\s+decir|dice|dijo)\s+dios\b",
        r"\b(que\s+)?(quiere|espera|tiene|busca|depara)\s+(dios|jesus|el\s+senor)\s+para\s+(mi\s+|el\s+)?futuro\b",
        r"\b(dios|jesus|el\s+senor)\b.*\bmi\s+futuro\b",
        r"\bmi\s+futuro\b.*\b(dios|jesus|el\s+senor)\b",
        r"\b(que\s+)?(decision|camino)\s+personal\s+quiere\s+dios\b",
        r"\bque\s+(me\s+depara|espera)\s+(el\s+futuro|mi\s+futuro)\b",
        r"\bque\s+va\s+a\s+pasar\s+(conmigo|manana|en\s+mi\s+futuro|en\s+mi\s+vida)\b",
        r"\bcon\s+quien\s+(me\s+debo\s+casar|debo\s+casarme|me\s+voy\s+a\s+casar)\b",
        r"\b(debo|tengo\s+que)\s+(vender|comprar|mudarme|estudiar|renunciar|dejar|casarme)\s+.*(dios|jesus)\b",
        r"\bdios\s+quiere\s+que\s+(yo\s+)?(venda|compre|me\s+mude|renuncie|viaje|deje)\b",
        r"\b(adivina|adivinacion|profetiz(a|ame|ar)|horoscopo|astrologia|tarot|quiromancia|videncia|vidente|bola\s+de\s+cristal)\b",
        r"\b(loteria|lotto|chance|baloto|numero\s+ganador|ganar\s+la\s+loteria)\b",
    ]

    MEDICAL_PATTERNS = [
        r"(diagnostico\s+medico|tengo\s+sintomas\s+de|recetame|que\s+enfermedad\s+tengo|tratamiento\s+para\s+la\s+depresion)"
    ]

    SECULAR_PATTERNS = [
        r"\b(futbol|futbolista|partido\s+de|marcador|campeonato|champions|mundial\s+de|gol|goles|messi|ronaldo|baloncesto|beisbol|tenis|deporte|deportes)\b",
        r"\b(receta|recetas|cocina|cocinar|ingredientes|pizza|hamburguesa|postre|restaurante)\b",
        r"\b(clima|temperatura|va\s+a\s+llover|pronostico\s+del\s+tiempo|meteorolog)\b",
        r"\b(elecciones|candidato|candidatos|partido\s+politico|votar\s+por|diputado|senador|alcalde)\b",
        r"\b(dolar|dolares|euro|bolsa\s+de\s+valores|cripto|criptomoneda|bitcoin|cotizacion|precio\s+del)\b",
        r"\b(chiste|chistes|broma|bromas)\b",
        r"\b(pelicula|peliculas|cine|netflix|hollywood)\b",
        r"\b(videojuego|videojuegos|playstation|xbox|nintendo)\b",
        r"\b(programacion|javascript|python|css|html|software|hardware|computadora|computador|ordenador|laptop|celular|telefono|smartphone|router|wifi)\b",
        r"\b(carro|carros|coche|automovil|camion|helicóptero|helicoptero|avion|motocicleta|moto|bicicleta|motor|neumatico)\b",
    ]

    RELIGIOUS_ROOTS = (
        "ensena", "sacrosan", "concili", "litur", "lumen", "gentium", "verbum",
        "verbo", "berbun", "gaudium", "caritas", "mysterium", "dives", "spe salvi",
        "fides", "evangelii", "ecclesia", "consustancial", "nicea", "calcedonia",
        "efeso", "trento", "filioque", "paraclito", "epiclesis", "ruah", "didache",
        "catecismo", "compendio", "biblia", "escritura", "evangelio", "padre", "hijo",
        "espiritu", "iglesia", "dios", "jesus", "cristo", "mesias", "senor", "fe",
        "esperanza", "caridad", "amor", "amar", "misericord", "perdon", "reconcilia",
        "penitencia", "confesion", "gracia", "justifica", "pecado", "conversion",
        "discipul", "bautis", "confirmacion", "eucarist", "transustancia", "transubstancia",
        "presencia real", "comunion", "misa", "matrimonio", "sacerdocio", "orden sagrado",
        "uncion", "sacramento", "santo", "santa", "virgen", "maria", "theotokos",
        "papa", "obispo", "oracion", "padrenuestro", "salvacion", "redencion",
        "revelacion", "encarnacion", "resurreccion", "resucit", "pascua", "pentecost",
        "ascension", "cruz", "mandamiento", "bienaventuranza", "parabola", "reino",
        "virtud", "conciencia", "libertad", "dignidad", "bien comun", "moral",
        "apostol", "profeta", "salmo", "proverbio", "genesis", "exodo", "isaias",
        "jeremias", "mateo", "marcos", "lucas", "juan", "hechos", "romanos",
        "corintios", "galatas", "efesios", "filipenses", "colosenses", "hebreos",
        "santiago", "apocalipsis", "pablo", "saulo", "pedro", "moises", "abraham",
        "david", "magdalena", "patristic", "tradicion", "magisterio", "enciclica",
        "ireneo", "ignacio", "agustin", "atanasio", "basilio", "gregorio", "crisostomo",
        "jeronimo", "ambrosio", "clemente", "policarpo", "justino", "tertuliano",
        "origenes", "cipriano", "cirilo", "leon magno", "damasceno", "doctrina", "teolog",
    )

    def evaluate(self, user_query: str, rag_result: Optional[Dict[str, Any]] = None) -> Tuple[bool, Optional[str]]:
        q = normalize_text(user_query)

        # PASO 1: Filtro de Autoría Sagrada (Regla 49)
        for pattern in self.IMPERSONATION_PATTERNS:
            if re.search(pattern, q):
                logger.info(f"[PREFILTER] Triggered Sacred Authorship Filter on: {user_query}")
                return False, RESP_AUTORIA_SAGRADA

        # PASO 2: Filtro de Solicitud Personal de Confesión / Absolución (Regla 10.F)
        is_doctrinal_inquiry = bool(re.search(
            r"\b(que\s+es|que\s+significa|que\s+ensena|explicame|hablame\s+de|cual\s+es|como\s+instituyo|por\s+que\s+existe)\b",
            q
        ))
        for pattern in self.CONFESSION_SOLICITATION_PATTERNS:
            if re.search(pattern, q):
                logger.info(f"[PREFILTER] Triggered Confession Filter on: {user_query}")
                return False, RESP_CONFESION
        if not is_doctrinal_inquiry and re.search(r"\b(confesar|absolucion)\b", q):
            logger.info(f"[PREFILTER] Triggered Confession Filter on: {user_query}")
            return False, RESP_CONFESION

        # PASO 3: Filtro de Pertinencia Secular Inmediata (Regla 48)
        for pattern in self.SECULAR_PATTERNS:
            if re.search(pattern, q):
                logger.info(f"[PREFILTER] Triggered Immediate Secular Pertinence Filter on: {user_query}")
                return False, RESP_IMPERTINENTE

        # PASO 4: Adivinación, Sueños y Decisiones Personales Atribuidas a Dios (Regla 52)
        group_a = (
            "sueno", "suenos", "sone", "sonar", "decision", "decisiones", "decidir",
            "mi futuro", "trabajo", "empleo", "laboral", "pareja", "novio", "novia",
            "casarme", "negocio", "negocios", "empresa", "loteria", "lotto", "suerte",
        )
        group_b = (
            "quiere dios", "voluntad de dios para mi", "me dice dios", "me quiere decir dios",
            "debo tomar", "que hago", "debo hacer", "me pide dios", "espera dios de mi",
            "quiere jesus para mi", "quiere el senor para mi",
        )
        if any(w in q for w in group_a) and any(w in q for w in group_b):
            logger.info(f"[PREFILTER] Triggered Rule 52 on: {user_query}")
            return False, RESP_ADIVINACION

        for pattern in self.DIVINATION_PATTERNS:
            if re.search(pattern, q):
                logger.info(f"[PREFILTER] Triggered Divination/Prophecy Filter on: {user_query}")
                return False, RESP_ADIVINACION

        for pattern in self.MEDICAL_PATTERNS:
            if re.search(pattern, q):
                logger.info(f"[PREFILTER] Triggered Medical Filter on: {user_query}")
                return False, RESP_MEDICO

        # PASO 5: Evaluación de Respaldo Documental RAG (Reglas 5, 6 y 48)
        if rag_result is not None:
            confidence = rag_result.get("confidence", "NIVEL D")
            min_dist = rag_result.get("min_distance", 1.0)
            results = rag_result.get("results", [])

            has_religious_words = any(r in q for r in self.RELIGIOUS_ROOTS)

            if not has_religious_words and (confidence == "NIVEL D" or not results):
                logger.info(f"[PREFILTER] Non-religious topic without religious keywords on: {user_query}")
                return False, RESP_IMPERTINENTE

            if confidence == "NIVEL D" and (not results or min_dist > 0.95):
                logger.info(f"[PREFILTER] Triggered Anti-hallucination Filter (NIVEL D) on: {user_query}")
                return False, RESP_SIN_FUENTES

        return True, None
