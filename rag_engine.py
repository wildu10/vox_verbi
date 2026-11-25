import os
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"

import re
import math
import pickle
import logging
import unicodedata
from collections import OrderedDict
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

from . import config

logger = logging.getLogger("vox_verbi.rag_engine")

STOPWORDS_ES = {
    "el", "la", "los", "las", "un", "una", "unos", "unas", "de", "del", "al", "en",
    "por", "para", "con", "sin", "sobre", "acerca", "que", "quien", "quienes", "cual",
    "cuales", "como", "cuando", "donde", "es", "son", "fue", "ser", "esta", "este",
    "estos", "estas", "ese", "esa", "su", "sus", "tu", "tus", "mi", "mis", "se", "le",
    "lo", "me", "te", "nos", "ya", "si", "no", "mas", "pero", "o", "y", "e", "u", "a",
    "vox", "verbi", "hablame", "explicame", "dime", "responde", "profundiza", "desarrolla",
    "ensena", "ensenanza", "dice", "significa", "iglesia", "segun", "durante", "minuto",
    "minutos", "segundos", "treinta", "sesenta",
}


def tokenize_es(text: str) -> list[str]:
    t = unicodedata.normalize("NFD", (text or "").lower())
    t = "".join(c for c in t if unicodedata.category(c) != "Mn")
    return [w for w in re.findall(r"[a-z0-9]{2,}", t) if w not in STOPWORDS_ES]


# Document cue -> canonical short citation substring for precise Magisterial/Patristic targeting
DOC_TARGET_MAP = [
    (r"\b(dei\s*verb\w*|de\s*verb\w*|dei\s*berb\w*|de\s*berb\w*|de\s*iberb\w*|deiberb\w*|iberb\w*|berbun\w*|david\s*boom)\b", "dei verbum"),
    (r"\b(lumen\s*[gjy]en[tc]i\w*)\b", "lumen gentium"),
    (r"\b(sacrosan\w*|sacro\s*san\w*|concilium)\b", "sacrosanctum concilium"),
    (r"\b(gaudi\w*\s*et\s*spes|gaudium)\b", "gaudium et spes"),
    (r"\b(deus\s*caritas\w*|de\s*uscaritas\w*|e?us[ck]aritas\w*)\b", "deus caritas est"),
    (r"\b(e?spe\s*salvi)\b", "spe salvi"),
    (r"\b(lumen\s*fidei)\b", "lumen fidei"),
    (r"\b(dives\s*in\s*misericordia)\b", "dives in misericordia"),
    (r"\b(e[cc]*lesia\s*de\s*euc[h]?aristia)\b", "ecclesia de eucharistia"),
    (r"\b(fides\s*et\s*ratio|fides\s*et\s*racio)\b", "fides et ratio"),
    (r"\b(evangelii?\s*gaudium)\b", "evangelii gaudium"),
    (r"\b(m[iy]steri\w*\s*fidei)\b", "mysterium fidei"),
    (r"\b(didach\w*|didaqu\w*)\b", "didach"),
    (r"\b(ireneo)\b", "ireneo"),
    (r"\b(ignacio\s+de\s+antioqu\w*)\b", "ignacio de antioqu"),
    (r"\b(atanasio)\b", "atanasio"),
    (r"\b(basilio)\b", "basilio"),
    (r"\b(gregorio\s+nacianceno)\b", "gregorio nacianceno"),
    (r"\b(gregorio\s+de\s+nisa)\b", "gregorio de nisa"),
    (r"\b(crisostomo|juan\s+crisostomo)\b", "cris"),
    (r"\b(ambrosio)\b", "ambrosio"),
    (r"\b(jeronimo)\b", "jer"),
    (r"\b(agustin)\b", "agust"),
    (r"\b(clemente\s+romano|1\s+clemente)\b", "clemente"),
    (r"\b(policarpo)\b", "policarpo"),
    (r"\b(justino)\b", "justino"),
    (r"\b(tertuliano)\b", "tertuliano"),
    (r"\b(origenes)\b", "or"),
    (r"\b(cipriano)\b", "cipriano"),
    (r"\b(cirilo)\b", "cirilo"),
    (r"\b(leon\s+magno)\b", "le"),
]

# Theological concept expansions to enrich both E5 dense and BM25 lexical recall
CONCEPT_EXPANSIONS = [
    (r"\bmisericord\w*\b", "misericordia compasión perdón Dios Padre misericordioso Lucas 15 hijo pródigo Lucas 6, 36 Mateo 5, 7 Dives in misericordia obras de misericordia Catecismo 1846"),
    (r"\bperd[oó]n\w*\b|\breconcilia\w*\b|\bpenitencia\b", "perdón de los pecados reconciliación penitencia setenta veces siete Mateo 18, 21-22 Juan 20, 22-23 Lucas 15 Catecismo 1422 1441 Compendio 296"),
    (r"\bamor\b|\bcaridad\b|\bamar\b", "amor cristiano Dios es amor caridad ágape eros 1 Juan 4, 7-16 1 Corintios 13, 4-8 Deus caritas est Catecismo 1822"),
    (r"\besperanza\b", "virtud teologal esperanza vida eterna promesas de Cristo Romanos 5, 5 Romanos 8, 24 Hebreos 10, 23 Spe salvi Catecismo 1817"),
    (r"\b\bfe\b", "virtud teologal de la fe creer adhesión a Dios Hebreos 11, 1 Romanos 10, 17 Lumen fidei Dei Verbum Catecismo 150 1814"),
    (r"\bsan\s+pablo\b|\bpablo\b|\bsaulo\b", "apóstol San Pablo Saulo conversión camino de Damasco Hechos 9, 1-15 apóstol de los gentiles gracia justificación Romanos 3 Gálatas 2, 20 1 Corintios"),
    (r"\bgracia\b|\bjustifica\w*\b", "gracia santificante justificación gratuita don sobrenatural participación naturaleza divina Romanos 3, 24 Efesios 2, 8 Catecismo 1996 Compendio 422"),
    (r"\beucarist\w*\b|\btransu[bs]tancia\w*\b|\bpresencia\s+real\b|\bmisa\b", "Eucaristía presencia real Cuerpo y Sangre de Cristo transustanciación Última Cena Juan 6, 51-56 1 Corintios 11, 23-26 Lucas 22, 19 Catecismo 1374 Mysterium fidei Ecclesia de Eucharistia"),
    (r"\bbautis\w*\b", "sacramento del Bautismo renacer del agua y del Espíritu iniciación cristiana Mateo 28, 19 Juan 3, 5 Romanos 6, 3-4 Catecismo 1213 Compendio 252"),
    (r"\besp[ií]ritu\s+santo\b|\bpar[aá]clito\b|\bpentecost\w*\b", "Espíritu Santo Paráclito Consolador Espíritu de la verdad Pentecostés dones frutos Juan 14, 26 Hechos 2, 1-4 Romanos 8 Lumen gentium 4 Catecismo 685"),
    (r"\btrinidad\b|\bconsustancial\b|\bnicea\b", "Santísima Trinidad un solo Dios en tres Personas Padre Hijo y Espíritu Santo consustancial Concilio de Nicea Mateo 28, 19 2 Corintios 13, 13 Juan 10, 30 Catecismo 253"),
    (r"\bbiblia\b|\bsagrada\s+escritura\b|\bescrituras?\b", "Sagrada Escritura Palabra de Dios inspiración divina canon bíblico Antiguo y Nuevo Testamento 2 Timoteo 3, 16 Hebreos 4, 12 Dei Verbum 9 11 Catecismo 101 105"),
    (r"\bquien\s+(?:es|era)\s+jesus\b|\bjesucristo\b|\bmesias\b|\bhijo\s+de\s+dios\b", "Jesucristo Hijo de Dios vivo Verbo encarnado verdadero Dios y verdadero hombre Mesías Salvador Mateo 16, 16 Juan 1, 14 Filipenses 2, 6-11 Catecismo 422 456 Compendio 87"),
    (r"\bsacramentos?\b", "siete sacramentos de la Nueva Ley instituidos por Cristo Bautismo Confirmación Eucaristía Penitencia Reconciliación Unción de los enfermos Orden sacerdotal Matrimonio Catecismo 1113 1210 Compendio 224 250"),
    (r"\bpadres\s+de\s+la\s+iglesia\b|\bpatr[ií]stic\w*\b", "Tradición Patrística Padres de la Iglesia San Agustín San Ireneo San Ignacio de Antioquía San Atanasio San Juan Crisóstomo San Jerónimo Didaché"),
]

# Canonical locus-classicus citation boosts so foundational Gospel/Catechism texts lead over peripheral verses
LOCUS_ANCHORS = [
    (r"\bmisericord\w*\b", ("lucas 15", "lucas 6", "mateo 5", "dives in misericordia", "1846", "1847")),
    (r"\bperdon\w*\b|\breconcilia\w*\b|\bpenitencia\b", ("juan 20", "mateo 18", "lucas 15", "1422", "1441", "1446", "296", "300")),
    (r"\beucarist\w*\b|\bpresencia\s+real\b|\btransu[bs]tancia\w*\b", ("juan 6", "lucas 22", "1 corintios 11", "1374", "1376", "mysterium fidei", "ecclesia de eucharistia")),
    (r"\bsacramentos\b", ("1210", "1113", "224", "250")),
    (r"\bamor\b|\bcaridad\b", ("1 juan 4", "1 corintios 13", "juan 13", "deus caritas est", "1822")),
    (r"\besperanza\b", ("romanos 5", "romanos 8", "hebreos 10", "spe salvi", "1817", "387")),
    (r"\b\bfe\b", ("hebreos 11", "romanos 10", "lumen fidei", "dei verbum", "150", "1814")),
]


class RAGEngine:
    """
    Hybrid Multilingual-E5 Dense + Inverted-Index BM25 Lexical RAG Engine across the 4 Canonical Columns:
      1. Sagrada Escritura (Nivel 1)
      2. Magisterio Conciliar y Pontificio (Nivel 2)
      3. Catecismo de la Iglesia Católica (Nivel 3)
      4. Tradición Patrística (Nivel 4)
    """

    def __init__(
        self,
        index_path=config.FAISS_INDEX_PATH,
        meta_path=config.META_PKL_PATH,
        bm25_path=getattr(config, "BM25_PKL_PATH", ""),
    ):
        self.index_path = index_path
        self.meta_path = meta_path
        self.bm25_path = bm25_path
        self.index = None
        self.metadata = []
        self.bm25 = None
        self.embedder = None
        self.is_e5 = False
        self._cache = OrderedDict()
        self._max_cache = 128
        self._load()

    def _load(self):
        if os.path.exists(self.index_path) and os.path.exists(self.meta_path):
            logger.info("Loading FAISS index and theological metadata...")
            self.index = faiss.read_index(self.index_path)
            with open(self.meta_path, "rb") as f:
                self.metadata = pickle.load(f)
            logger.info(f"Loaded {len(self.metadata)} records in RAG Engine (index ntotal={self.index.ntotal})")
        else:
            logger.warning("FAISS index or metadata not found. RAG will return empty.")

        if self.bm25_path and os.path.exists(self.bm25_path):
            try:
                with open(self.bm25_path, "rb") as f:
                    self.bm25 = pickle.load(f)
                logger.info(f"Loaded BM25 lexical index ({len(self.bm25.get('postings', {}))} terms).")
            except Exception as e:
                logger.warning(f"Could not load BM25 index: {e}")

        try:
            self._get_embedder()
        except Exception as e:
            logger.warning(f"Could not preload embedder: {e}")

    def _get_embedder(self):
        if self.embedder is None:
            preferred = getattr(config, "EMBEDDER_MODEL_NAME", "intfloat/multilingual-e5-small")
            for mname in (preferred, "all-MiniLM-L6-v2"):
                try:
                    self.embedder = SentenceTransformer(mname, local_files_only=True)
                    self.is_e5 = "e5" in mname.lower()
                    logger.info(f"Loaded SentenceTransformer '{mname}' (offline, e5={self.is_e5}).")
                    break
                except Exception:
                    try:
                        self.embedder = SentenceTransformer(mname)
                        self.is_e5 = "e5" in mname.lower()
                        logger.info(f"Loaded SentenceTransformer '{mname}' (e5={self.is_e5}).")
                        break
                    except Exception:
                        continue
        return self.embedder

    def _bm25_scores(self, query_text: str) -> dict[int, float]:
        """Computes fast BM25 scores over the inverted index in < 3 ms."""
        if not self.bm25 or self.bm25.get("N", 0) != len(self.metadata):
            return {}
        toks = tokenize_es(query_text)
        if not toks:
            return {}
        postings = self.bm25["postings"]
        idf_map = self.bm25["idf"]
        doc_lens = self.bm25["doc_lens"]
        avgdl = self.bm25["avgdl"]
        k1, b = 1.5, 0.75

        acc = {}
        for tok in set(toks):
            if tok not in postings:
                continue
            w_idf = idf_map.get(tok, 0.0)
            if w_idf <= 0.05:
                continue
            d_arr, tf_arr = postings[tok]
            # Vectorized BM25 term contribution
            tfs = tf_arr.astype(np.float32)
            dl = doc_lens[d_arr]
            contrib = w_idf * (tfs * (k1 + 1.0)) / (tfs + k1 * (1.0 - b + b * (dl / avgdl)))
            # Keep top 400 per term for speed
            if len(contrib) > 400:
                top_i = np.argpartition(contrib, -400)[-400:]
                for did, sc in zip(d_arr[top_i].tolist(), contrib[top_i].tolist()):
                    acc[did] = acc.get(did, 0.0) + sc
            else:
                for did, sc in zip(d_arr.tolist(), contrib.tolist()):
                    acc[did] = acc.get(did, 0.0) + sc
        return acc

    @staticmethod
    def _clean_thematic_query(query: str) -> str:
        q = re.sub(r'["«»\'“”¿¡?!]', " ", query or "")
        q = re.sub(
            r"^(vox\s+verbi|box\s+berbi|vos\s+verbi|bosbergvi|bosberbi|voz\s+verbi|busverby|bosberg|vos\s+ver\s+mi|boss\s+berby)\s*,?\s*",
            "", q, flags=re.IGNORECASE,
        ).strip()
        q = re.sub(
            r"\b(?:responde|hablame|háblame|explicame|explícame|profundiza|desarrolla|dime)\s+(?:durante|por|en)\s+\w+\s+(?:minutos?|min|segundos?|seg)\s*(?:sobre|acerca\s+de|de|del)?\b",
            " ", q, flags=re.IGNORECASE,
        )
        q = re.sub(
            r"\b(?:durante|por|en)\s+(?:un|uno|dos|tres|cuatro|1|2|3|4|30|60|120|180|240|treinta|sesenta)\s+(?:minutos?|min|segundos?|seg)\b",
            " ", q, flags=re.IGNORECASE,
        )
        q = re.sub(
            r"^(hablame\s+de|háblame\s+de|hablame\s+del|háblame\s+del|hablame\s+sobre|háblame\s+sobre|explicame\s+sobre|explícame\s+sobre|explicame\s+que\s+es|explícame\s+qué\s+es|explicame|explícame|que\s+dice\s+la\s+iglesia\s+sobre|que\s+enseña\s+la\s+iglesia\s+sobre|qué\s+enseña\s+la\s+iglesia\s+sobre|cual\s+es\s+la\s+ensenanza\s+sobre|cual\s+es\s+la\s+doctrina\s+sobre)\s+",
            "", q, flags=re.IGNORECASE,
        ).strip()
        return re.sub(r"\s+", " ", q).strip(" ,.-")

    def search(self, query: str, top_k: int = 3, max_words_per_chunk: int = 95):
        if not self.index or not self.metadata:
            return {"confidence": "NIVEL D", "results": [], "min_distance": 99.0}

        cache_key = (query.strip().lower(), top_k, max_words_per_chunk)
        if cache_key in self._cache:
            self._cache.move_to_end(cache_key)
            return self._cache[cache_key]

        clean_q = self._clean_thematic_query(query)
        q_norm = unicodedata.normalize("NFD", clean_q.lower())
        q_norm = "".join(c for c in q_norm if unicodedata.category(c) != "Mn")

        # Detect explicit Magisterial / Patristic document or author target
        requested_doc_sub = None
        for pat, doc_sub in DOC_TARGET_MAP:
            if re.search(pat, q_norm):
                requested_doc_sub = doc_sub
                break

        asks_patristic = bool(
            re.search(r"\b(padres?\s+de\s+la\s+iglesia|patristic\w*|tradicion\s+patristica)\b", q_norm)
            or (requested_doc_sub in (
                "didach", "ireneo", "ignacio de antioqu", "atanasio", "basilio",
                "gregorio nacianceno", "gregorio de nisa", "cris", "ambrosio",
                "jer", "agust", "clemente", "policarpo", "justino", "tertuliano",
                "or", "cipriano", "cirilo", "le",
            ))
        )

        # Build expanded semantic + lexical query
        expanded_q = clean_q if len(clean_q) > 2 else query
        for pat, exp in CONCEPT_EXPANSIONS:
            if re.search(pat, clean_q.lower()) or re.search(pat, q_norm):
                expanded_q = f"{expanded_q} — {exp}"

        embedder = self._get_embedder()
        dense_input = f"query: {expanded_q}" if self.is_e5 else expanded_q
        q_vec = embedder.encode([dense_input], convert_to_numpy=True, normalize_embeddings=True).astype(np.float32)

        # Also encode unexpanded raw query so we have true semantic relevance for off-topic detection!
        raw_input = f"query: {clean_q or query}" if self.is_e5 else (clean_q or query)
        q_vec_raw = embedder.encode([raw_input], convert_to_numpy=True, normalize_embeddings=True).astype(np.float32)
        raw_scores, _ = self.index.search(q_vec_raw, 5)
        best_raw_cos = float(raw_scores[0][0]) if len(raw_scores[0]) else 0.0
        # Calibrate E5 similarity (where unrelated Spanish sentences sit around ~0.74-0.78, and relevant hits are >= 0.81)
        if self.is_e5:
            # Map E5 cosine [0.76 .. 0.92] smoothly onto standard L2 distance scale [1.15 .. 0.25]
            eff_cos = max(0.0, min(1.0, (best_raw_cos - 0.75) / 0.17))
            raw_min_dist = math.sqrt(max(0.0, 2.0 * (1.0 - eff_cos)))
        else:
            raw_min_dist = math.sqrt(max(0.0, 2.0 * (1.0 - best_raw_cos)))

        search_depth = min(800, len(self.metadata))
        d_scores, d_indices = self.index.search(q_vec, search_depth)

        bm25_map = self._bm25_scores(expanded_q)
        bm25_ranked = sorted(bm25_map.items(), key=lambda x: x[1], reverse=True)[:400]
        bm25_rank_pos = {did: r for r, (did, _) in enumerate(bm25_ranked, 1)}
        max_bm25 = bm25_ranked[0][1] if bm25_ranked else 1.0

        candidate_ids = {}
        for rank, (cos_s, idx) in enumerate(zip(d_scores[0], d_indices[0]), 1):
            if 0 <= idx < len(self.metadata):
                candidate_ids[int(idx)] = {"dense_rank": rank, "cos": float(cos_s)}
        for rank, (did, _) in enumerate(bm25_ranked, 1):
            if did not in candidate_ids and 0 <= did < len(self.metadata):
                candidate_ids[did] = {"dense_rank": search_depth + 100, "cos": 0.78 if self.is_e5 else 0.35}

        # If a specific document/author was named, ensure ALL its matching chunks enter the candidate pool
        if requested_doc_sub:
            for idx, rec in enumerate(self.metadata):
                hay = f"{rec.get('cita_corta', '')} {rec.get('titulo', '')} {rec.get('autor', '')}".lower()
                hay_n = "".join(c for c in unicodedata.normalize("NFD", hay) if unicodedata.category(c) != "Mn")
                if requested_doc_sub in hay_n and idx not in candidate_ids:
                    candidate_ids[idx] = {"dense_rank": search_depth + 50, "cos": 0.83 if self.is_e5 else 0.55}

        q_tokens = set(tokenize_es(clean_q))
        scored_candidates = []
        min_dist = raw_min_dist

        for idx, info in candidate_ids.items():
            rec = dict(self.metadata[idx])
            d_rank = info["dense_rank"]
            b_rank = bm25_rank_pos.get(idx, 600)
            b_val = bm25_map.get(idx, 0.0) / max(1.0, max_bm25)

            # Hybrid RRF + normalized cosine + lexical boost
            rrf = (1.0 / (45.0 + d_rank)) + (0.95 / (45.0 + b_rank)) + 0.012 * b_val

            meta_str = f"{rec.get('cita_corta', '')} {rec.get('titulo', '')} {rec.get('autor', '')}".lower()
            meta_norm = "".join(c for c in unicodedata.normalize("NFD", meta_str) if unicodedata.category(c) != "Mn")

            # Exact document/author match boost (ranks the specific section within that document highest!)
            if requested_doc_sub and requested_doc_sub in meta_norm:
                rrf += 0.08
                min_dist = min(min_dist, 0.25)

            # Locus-classicus boost for foundational Gospel/Catechism passages
            for l_pat, anchor_subs in LOCUS_ANCHORS:
                if re.search(l_pat, q_norm):
                    if any(a_sub in meta_norm for a_sub in anchor_subs):
                        rrf += 0.018
                        min_dist = min(min_dist, 0.45)

            # Title / citation lexical overlap boost
            if q_tokens:
                overlap = sum(1 for qt in q_tokens if qt in meta_norm)
                rrf += 0.006 * overlap

            # Canonical New Testament preference over obscure OT historical wars when asking general Christian concepts
            if int(rec.get("nivel_autoridad", 1)) == 1 and not any(
                ot in q_norm for ot in ("macabeos", "reyes", "cronicas", "josue", "jueces", "levitico", "numeros")
            ):
                if any(nt in meta_norm for nt in (
                    "mateo", "marcos", "lucas", "juan", "hechos", "romanos", "corintios",
                    "galatas", "efesios", "filipenses", "colosenses", "hebreos", "santiago", "pedro",
                )):
                    rrf += 0.005

            if self.is_e5:
                eff_c = max(0.0, min(1.0, (info["cos"] - 0.75) / 0.17))
                dist = math.sqrt(max(0.0, 2.0 * (1.0 - eff_c)))
            else:
                dist = math.sqrt(max(0.0, 2.0 * (1.0 - info["cos"])))

            rec["score"] = float(rrf)
            rec["cos_sim"] = info["cos"]
            rec["distance"] = round(dist, 4)
            rec["nivel_autoridad"] = int(rec.get("nivel_autoridad", 1))
            scored_candidates.append(rec)

        if not scored_candidates or (min_dist > 1.08 and not requested_doc_sub and max_bm25 < 4.5):
            out = {
                "confidence": "NIVEL D",
                "results": [],
                "min_distance": round(min_dist, 4),
                "summary": "Sin fundamento suficiente en el corpus autorizado",
            }
            self._cache[cache_key] = out
            return out

        scored_candidates.sort(key=lambda x: x["score"], reverse=True)

        esc = [c for c in scored_candidates if c["nivel_autoridad"] == 1]
        mag = [c for c in scored_candidates if c["nivel_autoridad"] == 2]
        cat = [c for c in scored_candidates if c["nivel_autoridad"] == 3]
        pat = [c for c in scored_candidates if c["nivel_autoridad"] >= 4]

        # Stratified selection across the 4 Canonical Columns avoiding duplicate citations
        selected = []
        used_citas = set()

        def add_unique(cand_list):
            for c in cand_list:
                ck = c.get("cita_corta", "").lower()
                if ck and ck not in used_citas:
                    used_citas.add(ck)
                    selected.append(c)
                    return True
            return False

        if asks_patristic and pat:
            add_unique(pat)
            add_unique(esc)
            add_unique(mag if (mag and (not cat or mag[0]["score"] >= cat[0]["score"])) else cat)
        elif requested_doc_sub:
            # Place requested Magisterial/Catechism source first, followed by Scripture & complementary column
            doc_hits = [
                c for c in scored_candidates
                if requested_doc_sub in "".join(
                    ch for ch in unicodedata.normalize("NFD", f"{c.get('cita_corta','')} {c.get('titulo','')} {c.get('autor','')}".lower())
                    if unicodedata.category(ch) != "Mn"
                )
            ]
            add_unique(doc_hits)
            add_unique(esc)
            add_unique(cat if (cat and (not mag or cat[0]["score"] >= mag[0]["score"])) else mag)
        else:
            # Standard doctrinal query: 1. Sagrada Escritura -> 2. Magisterio/Catecismo -> 3. Catecismo/Magisterio o Patrística
            add_unique(esc)
            if mag and cat:
                if mag[0]["score"] >= cat[0]["score"] * 0.96:
                    add_unique(mag)
                    if len(selected) < top_k:
                        add_unique(cat)
                else:
                    add_unique(cat)
                    if len(selected) < top_k:
                        add_unique(mag)
            elif mag:
                add_unique(mag)
            elif cat:
                add_unique(cat)

        # Fill remaining slots up to top_k (for 1-min / 2-min deep dives) across Patristics and top remaining witnesses
        if len(selected) < top_k and pat:
            add_unique(pat)
        for c in scored_candidates:
            if len(selected) >= top_k:
                break
            ck = c.get("cita_corta", "").lower()
            if ck not in used_citas:
                used_citas.add(ck)
                selected.append(c)

        cleaned_selected = []
        for rec in selected[:top_k]:
            item = dict(rec)
            raw_t = item.get("texto", "")
            clean_t = re.sub(r"^((?:[123]\s+)?[A-Za-zÁÉÍÓÚáéíóúñ]+\s+\d+)\s+\1\s*", r"\1 ", raw_t).strip()
            words = clean_t.split()
            if len(words) > max_words_per_chunk:
                item["texto"] = " ".join(words[:max_words_per_chunk]) + "..."
            else:
                item["texto"] = clean_t
            cleaned_selected.append(item)

        # Determine confidence level per Regla 6
        if min_dist < 0.88 and any(s["nivel_autoridad"] in (1, 2, 3) for s in cleaned_selected):
            confidence = "NIVEL A"
        elif len(cleaned_selected) >= 2 and min_dist <= 1.04 and any(s["nivel_autoridad"] in (1, 2, 3) for s in cleaned_selected):
            confidence = "NIVEL B"
        elif any(s["nivel_autoridad"] >= 4 for s in cleaned_selected) and min_dist <= 1.06:
            confidence = "NIVEL C"
        else:
            confidence = "NIVEL D"

        result = {
            "confidence": confidence,
            "results": cleaned_selected,
            "min_distance": round(min_dist, 4),
        }
        if len(self._cache) >= self._max_cache:
            self._cache.popitem(last=False)
        self._cache[cache_key] = result
        return result
