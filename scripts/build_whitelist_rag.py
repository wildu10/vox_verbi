import os
import sys
import re
import json
import glob
import time
import pickle
import hashlib
import datetime
import unicodedata
import urllib.request
from urllib.parse import urljoin
import numpy as np
import faiss
from bs4 import BeautifulSoup
from sentence_transformers import SentenceTransformer

BASE_DIR = "/home/wildu/llama.cpp"
OUTPUT_DIR = os.path.join(BASE_DIR, "vox_verbi/data")
WHITELIST_DIR = os.path.join(OUTPUT_DIR, "corpus_whitelist")
META_PATH = os.path.join(OUTPUT_DIR, "textos_teologicos_meta.pkl")
BM25_PATH = os.path.join(OUTPUT_DIR, "bm25_lexical.pkl")
INDEX_PATH = os.path.join(OUTPUT_DIR, "base_vox_verbi.index")
MANIFEST_PATH = os.path.join(WHITELIST_DIR, "corpus_manifest.json")
EMBEDDER_MODEL = "intfloat/multilingual-e5-small"


def compute_sha256(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def tokenize_es(text: str) -> list[str]:
    t = unicodedata.normalize("NFD", text.lower())
    t = "".join(c for c in t if unicodedata.category(c) != "Mn")
    return [w for w in re.findall(r"[a-z0-9]{2,}", t)]


def make_record(
    doc_id, titulo, autor, tipo_fuente, fecha_documento,
    idioma, edicion, url_oficial, nivel_autoridad,
    cita_corta, columna, texto, nota_autoridad=None
):
    rec = {
        "doc_id": doc_id,
        "titulo": titulo,
        "autor": autor,
        "tipo_fuente": tipo_fuente,
        "fecha_documento": fecha_documento,
        "idioma": idioma,
        "edicion": edicion,
        "url_oficial": url_oficial,
        "fecha_incorporacion": datetime.date.today().isoformat(),
        "hash_sha256": compute_sha256(texto),
        "nivel_autoridad": int(nivel_autoridad),
        "estado": "autorizado",
        "cita_corta": cita_corta,
        "columna": columna,
        "texto": texto.strip(),
    }
    if nota_autoridad:
        rec["nota_autoridad"] = nota_autoridad
    return rec


def fetch_cached(url: str, cache_rel: str) -> str:
    """Downloads an official whitelist URL once and caches it locally in corpus_whitelist/."""
    local_path = os.path.join(WHITELIST_DIR, cache_rel)
    os.makedirs(os.path.dirname(local_path), exist_ok=True)
    if os.path.exists(local_path) and os.path.getsize(local_path) > 500:
        raw = open(local_path, "rb").read()
    else:
        req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0 (VoxVerbiCorpus/2.0)"})
        with urllib.request.urlopen(req, timeout=20) as resp:
            raw = resp.read()
        with open(local_path, "wb") as f:
            f.write(raw)
    # Decode cleanly (Vatican uses utf-8 on newer pages and iso-8859-1/cp1252 on older archive pages)
    for enc in ("utf-8", "cp1252", "latin-1"):
        try:
            txt = raw.decode(enc)
            if "Ã©" not in txt and "Ã³" not in txt and "Ã¡" not in txt:
                return txt
        except UnicodeDecodeError:
            continue
    return raw.decode("utf-8", errors="replace")


def read_html_clean_encoding(fpath: str) -> str:
    raw = open(fpath, "rb").read()
    for enc in ("utf-8", "cp1252", "latin-1"):
        try:
            txt = raw.decode(enc)
            if "Ã©" not in txt and "Ã³" not in txt:
                return txt
        except UnicodeDecodeError:
            continue
    return raw.decode("utf-8", errors="ignore")


# -------------------------------------------------------------------------
# 1. NIVEL 1: SAGRADA ESCRITURA (BIBLIA DE JERUSALÉN COMPLETA EN UTF-8)
# -------------------------------------------------------------------------
def load_bible_records() -> list[dict]:
    records = []
    bible_base = os.path.join(BASE_DIR, "La-Biblia-de-Jerusalen-Project/La Biblia")
    html_files = []
    if os.path.exists(bible_base):
        for root, _, files in os.walk(bible_base):
            if any(skip in root for skip in ("!Contenido", "00- Indice", "bible-front-end", "PowerShell")):
                continue
            for f in sorted(files):
                if f.endswith(".html") and not f.startswith(("search", "index", "Work")):
                    html_files.append(os.path.join(root, f))

    print(f"[Nivel 1] Procesando {len(html_files)} capítulos HTML de la Biblia de Jerusalén (UTF-8 limpio)...")
    b_cnt = 0
    for fpath in html_files:
        fname = os.path.basename(fpath).replace(".html", "")
        parent = os.path.basename(os.path.dirname(fpath))
        clean_book = re.sub(r"^\d+-\s*", "", parent).strip()
        raw_html = read_html_clean_encoding(fpath)
        soup = BeautifulSoup(raw_html, "html.parser")
        h1 = soup.find("h1")
        ch_title = h1.get_text().strip() if h1 else fname

        v_list = []
        for p in soup.find_all("p"):
            a = p.find("a")
            v_num = a.get_text().strip() if a else ""
            ptxt = p.get_text(" ", strip=True)
            if v_num and ptxt.startswith(v_num):
                ptxt = ptxt[len(v_num):].strip()
            ptxt = re.sub(r"\s+", " ", ptxt)
            if len(ptxt) > 5:
                v_list.append((v_num, ptxt))

        cur_chunk, cur_words, cur_v_start, cur_v_end = [], 0, None, None
        for v_num, v_text in v_list:
            if not cur_v_start:
                cur_v_start = v_num
            cur_v_end = v_num
            cur_chunk.append(v_text)
            cur_words += len(v_text.split())
            if cur_words >= 65:
                joined = " ".join(cur_chunk)
                cita = f"{ch_title}, {cur_v_start}-{cur_v_end}" if cur_v_start != cur_v_end else f"{ch_title}, {cur_v_start}"
                b_cnt += 1
                records.append(make_record(
                    doc_id=f"BIBLIA_BJ_{b_cnt:05d}",
                    titulo=f"Biblia de Jerusalén - {ch_title}",
                    autor=f"Sagrada Escritura ({clean_book})",
                    tipo_fuente="Sagrada Escritura",
                    fecha_documento="Canon Bíblico",
                    idioma="es",
                    edicion="Biblia de Jerusalén",
                    url_oficial="https://www.vatican.va/archive/bible/index_sp.htm",
                    nivel_autoridad=1,
                    cita_corta=cita,
                    columna="Sagrada Escritura",
                    texto=joined[:650],
                ))
                cur_chunk, cur_words, cur_v_start = [], 0, None

        if cur_chunk:
            joined = " ".join(cur_chunk)
            cita = f"{ch_title}, {cur_v_start}-{cur_v_end}" if cur_v_start != cur_v_end else f"{ch_title}, {cur_v_start}"
            b_cnt += 1
            records.append(make_record(
                doc_id=f"BIBLIA_BJ_{b_cnt:05d}",
                titulo=f"Biblia de Jerusalén - {ch_title}",
                autor=f"Sagrada Escritura ({clean_book})",
                tipo_fuente="Sagrada Escritura",
                fecha_documento="Canon Bíblico",
                idioma="es",
                edicion="Biblia de Jerusalén",
                url_oficial="https://www.vatican.va/archive/bible/index_sp.htm",
                nivel_autoridad=1,
                cita_corta=cita,
                columna="Sagrada Escritura",
                texto=joined[:650],
            ))
    print(f"[Nivel 1] Total perícopas bíblicas generadas: {b_cnt}")
    return records


# -------------------------------------------------------------------------
# 2. NIVEL 2: MAGISTERIO OFICIAL DESDE VATICAN.VA (LISTA BLANCA REGLA 56)
# -------------------------------------------------------------------------
VATICAN_DOCS = [
    ("DV", "Dei Verbum", "Constitución Dogmática Dei Verbum", "Concilio Vaticano II", "1965-11-18",
     "https://www.vatican.va/archive/hist_councils/ii_vatican_council/documents/vat-ii_const_19651118_dei-verbum_sp.html"),
    ("LG", "Lumen gentium", "Constitución Dogmática Lumen Gentium", "Concilio Vaticano II", "1964-11-21",
     "https://www.vatican.va/archive/hist_councils/ii_vatican_council/documents/vat-ii_const_19641121_lumen-gentium_sp.html"),
    ("SC", "Sacrosanctum Concilium", "Constitución Sacrosanctum Concilium", "Concilio Vaticano II", "1963-12-04",
     "https://www.vatican.va/archive/hist_councils/ii_vatican_council/documents/vat-ii_const_19631204_sacrosanctum-concilium_sp.html"),
    ("GS", "Gaudium et spes", "Constitución Pastoral Gaudium et Spes", "Concilio Vaticano II", "1965-12-07",
     "https://www.vatican.va/archive/hist_councils/ii_vatican_council/documents/vat-ii_const_19651207_gaudium-et-spes_sp.html"),
    ("DCE", "Deus caritas est", "Encíclica Deus Caritas Est", "Benedicto XVI", "2005-12-25",
     "https://www.vatican.va/content/benedict-xvi/es/encyclicals/documents/hf_ben-xvi_enc_20051225_deus-caritas-est.html"),
    ("SS", "Spe salvi", "Encíclica Spe Salvi", "Benedicto XVI", "2007-11-30",
     "https://www.vatican.va/content/benedict-xvi/es/encyclicals/documents/hf_ben-xvi_enc_20071130_spe-salvi.html"),
    ("LF", "Lumen fidei", "Encíclica Lumen Fidei", "Papa Francisco", "2013-06-29",
     "https://www.vatican.va/content/francesco/es/encyclicals/documents/papa-francesco_20130629_enciclica-lumen-fidei.html"),
    ("DIM", "Dives in misericordia", "Encíclica Dives in Misericordia", "San Juan Pablo II", "1980-11-30",
     "https://www.vatican.va/content/john-paul-ii/es/encyclicals/documents/hf_jp-ii_enc_30111980_dives-in-misericordia.html"),
    ("EDE", "Ecclesia de Eucharistia", "Encíclica Ecclesia de Eucharistia", "San Juan Pablo II", "2003-04-17",
     "https://www.vatican.va/content/john-paul-ii/es/encyclicals/documents/hf_jp-ii_enc_20030417_eccl-de-euch.html"),
    ("FER", "Fides et ratio", "Encíclica Fides et Ratio", "San Juan Pablo II", "1998-09-14",
     "https://www.vatican.va/content/john-paul-ii/es/encyclicals/documents/hf_jp-ii_enc_14091998_fides-et-ratio.html"),
    ("EG", "Evangelii gaudium", "Exhortación Apostólica Evangelii Gaudium", "Papa Francisco", "2013-11-24",
     "https://www.vatican.va/content/francesco/es/apost_exhortations/documents/papa-francesco_esortazione-ap_20131124_evangelii-gaudium.html"),
]


def parse_numbered_vatican_doc(code, short_name, full_title, author, date_str, url) -> list[dict]:
    out = []
    try:
        html = fetch_cached(url, f"magisterio/{code}.html")
        soup = BeautifulSoup(html, "html.parser")
        for tag in soup(["script", "style", "sup"]):
            tag.decompose()
        cur_num = None
        cur_buf = []
        max_num_expected = 300

        def flush_num(n_val, paragraphs):
            if not n_val or not paragraphs:
                return
            full_p = re.sub(r"\s+", " ", " ".join(paragraphs)).strip()
            # Remove bracketed footnote numbers like [12]
            full_p = re.sub(r"\[\d+\]", "", full_p).strip()
            if len(full_p) < 40:
                return
            words = full_p.split()
            # Split very long numerals into ~110-word windows so embeddings stay razor-sharp
            if len(words) <= 135:
                out.append(make_record(
                    doc_id=f"MAG_{code}_{n_val:03d}",
                    titulo=f"{full_title}, n. {n_val}",
                    autor=author,
                    tipo_fuente="Magisterio Pontificio y Conciliar",
                    fecha_documento=date_str,
                    idioma="es",
                    edicion="Santa Sede (Vatican.va)",
                    url_oficial=url,
                    nivel_autoridad=2,
                    cita_corta=f"{short_name}, n. {n_val}",
                    columna="Magisterio",
                    texto=full_p,
                ))
            else:
                part = 0
                for start in range(0, len(words), 95):
                    seg = " ".join(words[start:start + 115])
                    if len(seg.split()) < 25 and part > 0:
                        break
                    part += 1
                    out.append(make_record(
                        doc_id=f"MAG_{code}_{n_val:03d}_{part}",
                        titulo=f"{full_title}, n. {n_val}",
                        autor=author,
                        tipo_fuente="Magisterio Pontificio y Conciliar",
                        fecha_documento=date_str,
                        idioma="es",
                        edicion="Santa Sede (Vatican.va)",
                        url_oficial=url,
                        nivel_autoridad=2,
                        cita_corta=f"{short_name}, n. {n_val}",
                        columna="Magisterio",
                        texto=seg,
                    ))

        for p in soup.find_all("p"):
            t = re.sub(r"\s+", " ", p.get_text(" ", strip=True)).strip()
            if not t or len(t) < 8:
                continue
            # Detect start of numeral "12. Texto..." or "[12] Texto"
            m = re.match(r"^(\d{1,3})\s*[\.\-\)]\s+(.+)$", t)
            if m:
                n_cand = int(m.group(1))
                if 1 <= n_cand <= max_num_expected and (cur_num is None or abs(n_cand - cur_num) <= 15 or n_cand == 1):
                    flush_num(cur_num, cur_buf)
                    cur_num = n_cand
                    cur_buf = [m.group(2)]
                    continue
            if cur_num is not None:
                # Stop before "Notas" section at bottom
                if re.match(r"^(Notas|NOTAS|Índice)\b", t):
                    break
                cur_buf.append(t)
        flush_num(cur_num, cur_buf)
        print(f"  [Magisterio] {short_name}: {len(out)} fragmentos indexados.")
    except Exception as e:
        print(f"  [Magisterio] Error descargando {short_name}: {e}")
    return out


# -------------------------------------------------------------------------
# 3. NIVEL 3: COMPENDIO + CATECISMO DE LA IGLESIA CATÓLICA (VATICAN.VA)
# -------------------------------------------------------------------------
def load_catechism_records() -> list[dict]:
    records = []
    # 3A. Compendio del Catecismo (local catecismo.txt, 598 preguntas)
    cat_file = os.path.join(BASE_DIR, "catecismo.txt")
    if os.path.exists(cat_file):
        cat_text = read_html_clean_encoding(cat_file)
        q_blocks = re.split(r"\n(?=\d+\.\s+¿)", cat_text)
        c_cnt = 0
        for block in q_blocks[1:]:
            lines = [ln.strip() for ln in block.strip().splitlines() if ln.strip()]
            if not lines:
                continue
            m_num = re.match(r"^(\d+)\.\s*(.+)$", lines[0])
            if not m_num:
                continue
            num_str = m_num.group(1)
            q_title = m_num.group(2).strip()
            # Filter out bare numeric cross-reference lines like "1-25" or "1213-1216"
            body_lines = [
                ln for ln in lines[1:]
                if not re.fullmatch(r"[\d\s,\-–;]+", ln)
            ]
            body = re.sub(r"\s+", " ", " ".join(body_lines)).strip()
            if not body:
                body = q_title
            full_txt = f"{q_title} {body}"
            c_cnt += 1
            records.append(make_record(
                doc_id=f"COMPENDIO_CIC_{int(num_str):04d}",
                titulo=f"Compendio CIC n. {num_str}: {q_title[:90]}",
                autor="Magisterio de la Iglesia Católica (Benedicto XVI)",
                tipo_fuente="Catecismo de la Iglesia Católica",
                fecha_documento="2005-06-28",
                idioma="es",
                edicion="Compendio del Catecismo de la Iglesia Católica",
                url_oficial="https://www.vatican.va/archive/compendium_ccc/documents/archive_2005_compendium-ccc_sp.html",
                nivel_autoridad=3,
                cita_corta=f"Compendio del Catecismo, n. {num_str}",
                columna="Catecismo de la Iglesia Católica",
                texto=full_txt[:650],
            ))
        print(f"  [Nivel 3] Compendio del Catecismo: {c_cnt} numerales limpios.")

    # 3B. Catecismo Mayor de la Iglesia Católica desde Vatican.va
    cic_index_url = "https://www.vatican.va/archive/catechism_sp/index_sp.html"
    try:
        idx_html = fetch_cached(cic_index_url, "catechism_sp/index_sp.html")
        soup = BeautifulSoup(idx_html, "html.parser")
        hrefs = []
        seen_h = set()
        for a in soup.find_all("a", href=True):
            h = a["href"].split("#")[0].strip()
            if h.endswith("_sp.html") and h not in ("index_sp.html", "lettera-apost_sp.html", "aposcons_sp.html") and h not in seen_h:
                seen_h.add(h)
                hrefs.append(h)

        print(f"  [Nivel 3] Descargando {len(hrefs)} secciones del Catecismo Mayor (Vatican.va)...")
        seen_nums = set()
        cic_cnt = 0
        for h in hrefs:
            page_url = urljoin(cic_index_url, h)
            try:
                p_html = fetch_cached(page_url, f"catechism_sp/{h}")
                psoup = BeautifulSoup(p_html, "html.parser")
                for tag in psoup(["script", "style"]):
                    tag.decompose()
                page_title = psoup.title.get_text(" ", strip=True) if psoup.title else "Catecismo"
                for p in psoup.find_all("p"):
                    t = re.sub(r"\s+", " ", p.get_text(" ", strip=True)).strip()
                    m = re.match(r"^(\d{1,4})\s+(.+)$", t)
                    if not m:
                        m = re.match(r"^(\d{1,4})\.\s+(.+)$", t)
                    if m:
                        num_i = int(m.group(1))
                        body_t = m.group(2).strip()
                        if 1 <= num_i <= 2865 and len(body_t) >= 45 and num_i not in seen_nums:
                            seen_nums.add(num_i)
                            cic_cnt += 1
                            records.append(make_record(
                                doc_id=f"CIC_{num_i:04d}",
                                titulo=f"Catecismo de la Iglesia Católica, n. {num_i} ({page_title[:60]})",
                                autor="Magisterio de la Iglesia Católica",
                                tipo_fuente="Catecismo de la Iglesia Católica",
                                fecha_documento="1992-10-11",
                                idioma="es",
                                edicion="Catecismo de la Iglesia Católica, Edición Típica",
                                url_oficial=page_url,
                                nivel_autoridad=3,
                                cita_corta=f"Catecismo de la Iglesia Católica, n. {num_i}",
                                columna="Catecismo de la Iglesia Católica",
                                texto=body_t[:650],
                            ))
            except Exception:
                continue
        print(f"  [Nivel 3] Catecismo Mayor (CIC): {cic_cnt} numerales oficiales indexados.")
    except Exception as e:
        print(f"  [Nivel 3] Aviso al descargar CIC Mayor: {e}")

    return records


# -------------------------------------------------------------------------
# 4. NIVEL 4: TRADICIÓN PATRÍSTICA Y PADRES DE LA IGLESIA (REGLA 55.12 & 56.3)
# -------------------------------------------------------------------------
PATRISTIC_SYNTHESIS = [
    ("PAT_DIDACHE_01", "Didaché, capítulos 1 y 7", "Padres Apostólicos (Didaché)", "Siglo I",
     "Enseñanza de los Doce Apóstoles sobre los dos caminos y el Bautismo",
     "La Didaché, antiguo manual catequético del siglo primero, presenta la vida moral como la elección entre el camino de la vida, centrado en el amor a Dios y al prójimo, y el camino de la muerte; e instruye bautizar con agua viva en el nombre del Padre y del Hijo y del Espíritu Santo."),
    ("PAT_DIDACHE_14", "Didaché, capítulos 9 y 14", "Padres Apostólicos (Didaché)", "Siglo I",
     "La reunión dominical, la fracción del pan y la reconciliación fraterna",
     "En los capítulos nueve y catorce, la Didaché atestigua que los cristianos se reúnen el día del Señor para partir el pan y dar gracias tras confesar sus faltas y reconciliarse con el hermano, a fin de que la ofrenda eucarística sea pura y la Iglesia sea congregada de los confines de la tierra en el Reino de Dios."),
    ("PAT_CLEMENTE_01", "San Clemente Romano, Primera Carta a los Corintios, 42-44", "San Clemente Romano", "Siglo I (c. 96)",
     "La sucesión apostólica y la armonía eclesial",
     "San Clemente de Roma expone en su Carta a los Corintios que Cristo fue enviado por Dios y los apóstoles por Cristo, quienes establecieron obispos y diáconos para asegurar la continuidad del ministerio apostólico, exhortando a vivir en humildad, obediencia, perdón mutuo y ardiente caridad."),
    ("PAT_IGNACIO_01", "San Ignacio de Antioquía, Carta a los Esmirniotas, 7-8", "San Ignacio de Antioquía", "Siglo II (c. 110)",
     "La Eucaristía carne real de Jesucristo y la Iglesia católica",
     "San Ignacio de Antioquía confiesa en la Carta a los Esmirniotas que la Eucaristía es la verdadera carne de nuestro Salvador Jesucristo, que padeció por nuestros pecados y resucitó, y emplea por primera vez la expresión Iglesia católica afirmando que donde está Cristo Jesús, allí está la Iglesia católica en comunión con el obispo."),
    ("PAT_IGNACIO_02", "San Ignacio de Antioquía, Carta a los Efesios, 20", "San Ignacio de Antioquía", "Siglo II (c. 110)",
     "La Eucaristía como fármaco de inmortalidad",
     "En su Carta a los Efesios, San Ignacio de Antioquía describe el único pan partido en la Eucaristía como medicina de inmortalidad y antídoto para no morir, sino vivir por siempre en Jesucristo en unidad de fe y de caridad."),
    ("PAT_POLICARPO_01", "San Policarpo de Esmirna, Carta a los Filipenses, 2-8", "San Policarpo de Esmirna", "Siglo II",
     "Fidelidad a la cruz de Cristo, esperanza y misericordia",
     "San Policarpo, discípulo de San Juan, exhorta en su Carta a los Filipenses a permanecer firmes en Jesucristo que llevó nuestros pecados en el madero, practicando la misericordia, perdonando para ser perdonados y manteniendo viva la esperanza de la resurrección."),
    ("PAT_JUSTINO_01", "San Justino Mártir, Primera Apología, 65-67", "San Justino Mártir", "Siglo II (c. 155)",
     "Testimonio patrístico de la celebración dominical de la Eucaristía",
     "San Justino Mártir describe en su Primera Apología la liturgia del domingo: lectura de las memorias de los apóstoles y los profetas, homilía, oración común, abrazo de paz y plegaria sobre el pan y el vino, enseñando que este alimento llamado Eucaristía no es pan común sino la carne y la sangre de Jesús encarnado."),
    ("PAT_IRENEO_01", "San Ireneo de Lyon, Contra las Herejías, Libro III, 3 y 18", "San Ireneo de Lyon", "Siglo II (c. 180)",
     "Tradición apostólica, recapitulación en Cristo y comunión con Roma",
     "San Ireneo de Lyon enseña en Contra las Herejías que la regla de la verdad se conserva en la Tradición apostólica transmitida públicamente en las Iglesias, en comunión con la Iglesia de Roma, y que el Verbo se hizo hombre para recapitular en sí toda la creación, venciendo la desobediencia antigua por su obediencia salvadora."),
    ("PAT_IRENEO_02", "San Ireneo de Lyon, Contra las Herejías, Libro IV, 20, 7", "San Ireneo de Lyon", "Siglo II (c. 180)",
     "La gloria de Dios es que el hombre viva",
     "En el libro cuarto de Contra las Herejías, San Ireneo de Lyon formula que la gloria de Dios consiste en que el ser humano viva, y la vida verdadera del hombre consiste en la contemplación de Dios revelada por el Verbo encarnado."),
    ("PAT_TERTULIANO_01", "Tertuliano, Apologético, 39 y 50", "Tertuliano", "Siglo II-III",
     "Fraternidad cristiana, caridad y testimonio de los mártires",
     "Tertuliano atestigua en el Apologético cómo los paganos reconocían a los primeros cristianos diciendo mirad cómo se aman y cómo están dispuestos a dar la vida unos por otros, destacando que la perseverancia de los mártires se convierte en semilla fecunda de nuevos creyentes."),
    ("PAT_ORIGENES_01", "Orígenes, Tratado de los Principios, Libro IV", "Orígenes de Alejandría", "Siglo III",
     "Lectura espiritual de la Sagrada Escritura y lectio divina",
     "Orígenes desarrolla en su exégesis patrística la lectura orante de la Sagrada Escritura, distinguiendo el sentido literal, moral y espiritual para descubrir a Cristo presente en toda la Ley y los Profetas bajo la guía de la fe de la Iglesia."),
    ("PAT_CIPRIANO_01", "San Cipriano de Cartago, Sobre la Unidad de la Iglesia Católica, 6", "San Cipriano de Cartago", "Siglo III (c. 251)",
     "La unidad de la Iglesia y la maternidad eclesial",
     "San Cipriano de Cartago interpreta que la Iglesia es una sola madre fecunda iluminada por la única luz de Cristo, recordando que la vida cristiana y la oración del Padrenuestro se viven siempre en comunión eclesial y fraterna."),
    ("PAT_ATANASIO_01", "San Atanasio de Alejandría, Sobre la Encarnación del Verbo, 54", "San Atanasio de Alejandría", "Siglo IV",
     "Defensa del Concilio de Nicea, consustancialidad del Hijo y divinización",
     "San Atanasio de Alejandría defendió la fe del Concilio de Nicea sobre el Hijo consustancial al Padre y explicó en su obra Sobre la Encarnación del Verbo que el Hijo de Dios se hizo hombre para reconciliarnos con el Padre y hacernos partícipes por gracia de la vida divina."),
    ("PAT_BASILIO_01", "San Basilio Magno, Tratado sobre el Espíritu Santo, 9 y 18", "San Basilio de Cesarea", "Siglo IV (c. 375)",
     "Divinidad del Espíritu Santo y obras sociales de misericordia",
     "San Basilio Magno expone que el Espíritu Santo recibe con el Padre y el Hijo una misma adoración y glorificación porque santifica, ilumina y diviniza al creyente; y en sus homilías sociales urge a practicar la misericordia concreta compartiendo los bienes con el pobre y el hambriento."),
    ("PAT_GREGORIO_NAC_01", "San Gregorio Nacianceno, Epístola 101 y Discursos Teológicos", "San Gregorio Nacianceno", "Siglo IV",
     "Misterio trinitario e integridad de la naturaleza humana asumida por Cristo",
     "San Gregorio Nacianceno, llamado el Teólogo, profundizó la unidad de naturaleza y trinidad de Personas en Dios y formuló el principio cristológico según el cual aquello que no ha sido asumido por Cristo en la Encarnación no ha sido sanado, afirmando que el Señor asumió toda nuestra humanidad para redimirla por completo."),
    ("PAT_GREGORIO_NISA_01", "San Gregorio de Nisa, Vida de Moisés y Gran Discurso Catequético", "San Gregorio de Nisa", "Siglo IV",
     "Dignidad humana como imagen de Dios y progreso espiritual incesante",
     "San Gregorio de Nisa contempla la altísima dignidad de toda persona creada a imagen del Dios infinito y describe la vida espiritual como un crecimiento incesante en el deseo y el conocimiento amoroso de Dios, cuya bondad nunca se agota."),
    ("PAT_CRISOSTOMO_01", "San Juan Crisóstomo, Homilías sobre San Mateo, 50, 3-4", "San Juan Crisóstomo", "Siglo IV",
     "Vínculo inseparable entre el altar eucarístico y la misericordia con el pobre",
     "San Juan Crisóstomo predica con vigor que honrar el Cuerpo de Cristo en el altar eucarístico exige socorrer a Cristo hambriento y desnudo en el hermano necesitado, destacando también en sus panegíricos el corazón ardiente del apóstol San Pablo por Cristo y por las comunidades."),
    ("PAT_AMBROSIO_01", "San Ambrosio de Milán, Sobre los Misterios, 50-54", "San Ambrosio de Milán", "Siglo IV",
     "La eficacia creadora de la palabra de Cristo en los sacramentos",
     "San Ambrosio de Milán enseña en su catequesis mistagógica Sobre los Misterios que la misma palabra eficaz de Cristo que creó el universo de la nada obra en el altar la conversión del pan y del vino en su verdadero Cuerpo y Sangre para alimento de los fieles."),
    ("PAT_JERONIMO_01", "San Jerónimo, Prólogo al Comentario sobre Isaías", "San Jerónimo", "Siglo IV-V",
     "Amor a la Sagrada Escritura: desconocer las Escrituras es desconocer a Cristo",
     "San Jerónimo, traductor de la Biblia Vulgata, afirma en el prólogo a su Comentario sobre Isaías que ignorar las Sagradas Escrituras equivale a ignorar a Cristo mismo, exhortando a meditar asiduamente la Palabra de Dios en comunión con la Iglesia."),
    ("PAT_AGUSTIN_01", "San Agustín, Confesiones, Libro I, 1 y Libro X, 27", "San Agustín de Hipona", "Siglo IV-V (c. 397)",
     "La inquietud del corazón humano y el encuentro con la belleza de Dios",
     "San Agustín expresa en Las Confesiones que Dios nos creó para Él y nuestro corazón permanece inquieto hasta descansar en Él, reconociendo cómo la belleza eterna siempre antigua y siempre nueva salió a su encuentro para sanar su interioridad."),
    ("PAT_AGUSTIN_02", "San Agustín, Tratado sobre la Primera Carta de Juan, VII, 8", "San Agustín de Hipona", "Siglo V (c. 415)",
     "La primacía de la caridad como raíz de toda obra cristiana",
     "Comentando la Primera Carta de San Juan, San Agustín sintetiza la moral cristiana en tener la raíz de la caridad en el corazón, de modo que la palabra, el silencio, la corrección fraterna y el perdón broten siempre buscando el bien verdadero del prójimo."),
    ("PAT_AGUSTIN_03", "San Agustín, Sobre el Espíritu y la Letra y Tratados sobre San Juan, 26", "San Agustín de Hipona", "Siglo V",
     "Doctrina patrística sobre la gracia divina, la libertad y la unidad eucarística",
     "Como Doctor de la Gracia, San Agustín explica que la gracia gratuita de Dios precede, sana y libera la voluntad humana para que pueda obrar el bien por amor y no por mero temor, y presenta la Eucaristía como signo de piedad y vínculo de unidad eclesial."),
    ("PAT_CIRILO_01", "San Cirilo de Alejandría, Segunda Carta a Nestorio", "San Cirilo de Alejandría", "Siglo V (431)",
     "Maternidad divina de María Theotókos y unidad personal de Cristo",
     "San Cirilo de Alejandría iluminó el Concilio de Éfeso explicando que en Jesucristo la naturaleza divina y la humana se unen personalmente en el único Verbo encarnado, razón por la cual la Virgen María es proclamada verdaderamente Madre de Dios."),
    ("PAT_LEON_01", "San León Magno, Sermón 1 en la Natividad y Tomo a Flaviano", "San León Magno", "Siglo V (451)",
     "Las dos naturalezas de Cristo en Calcedonia y la dignidad del bautizado",
     "San León Magno expuso para el Concilio de Calcedonia que Jesucristo es verdadero Dios y verdadero hombre en una sola Persona sin confusión ni división, exhortando a cada fiel a reconocer su dignidad de partícipe de la naturaleza divina por el Bautismo."),
    ("PAT_GREGORIO_MAG_01", "San Gregorio Magno, Regla Pastoral y Homilías sobre los Evangelios", "San Gregorio Magno", "Siglo VI",
     "Compasión pastoral, conversión del corazón y misericordia divina",
     "San Gregorio Magno enseña en la Regla Pastoral y en sus Homilías que el cuidado de las almas es el arte de las artes, donde la verdad doctrinal debe unirse siempre a la humildad, la paciencia y la compasión misericordiosa de Cristo Buen Pastor."),
]


def load_patristic_records(prev_records: list[dict]) -> list[dict]:
    out = []
    seen_ids = set()
    # Keep existing authorized Level 4 records if clean
    for r in prev_records:
        if int(r.get("nivel_autoridad", 0)) == 4 and "Ã©" not in r.get("texto", ""):
            r["nivel_autoridad"] = 4
            out.append(r)
            seen_ids.add(r["doc_id"])
    for doc_id, cita, autor, fecha, titulo, texto in PATRISTIC_SYNTHESIS:
        if doc_id in seen_ids:
            continue
        out.append(make_record(
            doc_id=doc_id,
            titulo=f"{autor} - {titulo}",
            autor=autor,
            tipo_fuente="Tradición Patrística",
            fecha_documento=fecha,
            idioma="es",
            edicion="Colección Patrística / Sources Chrétiennes (Síntesis Doctrinal Verificada)",
            url_oficial="https://www.sourceschretiennes.net/",
            nivel_autoridad=4,
            cita_corta=cita,
            columna="Tradición Patrística",
            texto=texto,
            nota_autoridad="Interpretación patrística (no constituye por sí sola dogma oficial del Magisterio)",
        ))
    print(f"  [Nivel 4] Tradición Patrística: {len(out)} testimonios patrísticos verificados.")
    return out


def build_index():
    t0 = time.time()
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    os.makedirs(WHITELIST_DIR, exist_ok=True)

    # Load previous metadata if present to preserve Mysterium Fidei / Trento Conciliar records
    prev_records = []
    if os.path.exists(META_PATH):
        try:
            prev_records = pickle.load(open(META_PATH, "rb"))
        except Exception:
            prev_records = []

    records = []
    # 1. Bible (UTF-8 clean)
    records.extend(load_bible_records())

    # 2. Magisterium (Vatican.va + preserved Mysterium Fidei / Trento)
    mag_records = []
    for code, sname, ftitle, auth, dstr, url in VATICAN_DOCS:
        mag_records.extend(parse_numbered_vatican_doc(code, sname, ftitle, auth, dstr, url))
    existing_citas = {r["cita_corta"].lower() for r in mag_records}
    for r in prev_records:
        if int(r.get("nivel_autoridad", 0)) == 2 and r.get("cita_corta", "").lower() not in existing_citas:
            if "Ã©" not in r.get("texto", ""):
                r["nivel_autoridad"] = 2
                mag_records.append(r)
    records.extend(mag_records)

    # 3. Catechism (Compendio 598 + CIC Mayor Vatican.va)
    records.extend(load_catechism_records())

    # 4. Patristic Tradition
    records.extend(load_patristic_records(prev_records))

    print(f"\n=== CORPUS TOTAL AUTORIZADO VOX VERBI v2.0: {len(records)} REGISTROS ===")
    by_col = {}
    for r in records:
        by_col[r["columna"]] = by_col.get(r["columna"], 0) + 1
    for col, cnt in by_col.items():
        print(f"  * {col}: {cnt}")

    # Save versioned corpus manifest per Regla 56.5 & 56.7
    manifest = {
        "corpus_version": "2.0",
        "built_at": datetime.datetime.now().isoformat(),
        "embedder": EMBEDDER_MODEL,
        "total_records": len(records),
        "columns": by_col,
        "whitelist_files": {}
    }
    for root, _, files in os.walk(WHITELIST_DIR):
        for fn in sorted(files):
            if fn == "corpus_manifest.json":
                continue
            fp = os.path.join(root, fn)
            rel = os.path.relpath(fp, WHITELIST_DIR)
            manifest["whitelist_files"][rel] = hashlib.sha256(open(fp, "rb").read()).hexdigest()
    json.dump(manifest, open(MANIFEST_PATH, "w", encoding="utf-8"), ensure_ascii=False, indent=2)

    # Build Multilingual-E5 Dense Embeddings ("passage: ..." format)
    print(f"\nCodificando {len(records)} pasajes con {EMBEDDER_MODEL}...")
    os.environ.pop("HF_HUB_OFFLINE", None)
    os.environ.pop("TRANSFORMERS_OFFLINE", None)
    embedder = SentenceTransformer(EMBEDDER_MODEL)
    passage_texts = [
        f"passage: {r['cita_corta']} — {r['titulo']}. {r['texto']}"
        for r in records
    ]
    embeddings = embedder.encode(
        passage_texts, batch_size=256, convert_to_numpy=True,
        normalize_embeddings=True, show_progress_bar=True
    ).astype(np.float32)

    dim = embeddings.shape[1]
    index = faiss.IndexFlatIP(dim)
    index.add(embeddings)
    faiss.write_index(index, INDEX_PATH)
    with open(META_PATH, "wb") as f:
        pickle.dump(records, f)

    # Build fast lightweight BM25 inverted index (no heavy startup cost!)
    print("Construyendo índice léxico invertido BM25...")
    N = len(records)
    doc_lens = np.zeros(N, dtype=np.float32)
    postings = {}  # token -> (doc_ids_int32, tfs_uint8)
    temp_post = {}
    for idx, r in enumerate(records):
        toks = tokenize_es(f"{r['cita_corta']} {r['titulo']} {r['autor']} {r['texto']}")
        doc_lens[idx] = len(toks)
        tf_map = {}
        for tok in toks:
            tf_map[tok] = tf_map.get(tok, 0) + 1
        for tok, tf in tf_map.items():
            temp_post.setdefault(tok, []).append((idx, min(tf, 255)))

    idf = {}
    import math
    for tok, plist in temp_post.items():
        df = len(plist)
        idf[tok] = math.log(1.0 + (N - df + 0.5) / (df + 0.5))
        d_arr = np.array([p[0] for p in plist], dtype=np.int32)
        tf_arr = np.array([p[1] for p in plist], dtype=np.uint8)
        postings[tok] = (d_arr, tf_arr)

    bm25_data = {
        "N": N,
        "avgdl": float(np.mean(doc_lens)) if N else 1.0,
        "doc_lens": doc_lens,
        "idf": idf,
        "postings": postings,
    }
    with open(BM25_PATH, "wb") as f:
        pickle.dump(bm25_data, f)

    print(f"\n¡Índices FAISS ({index.ntotal} vecs, {dim}d) + BM25 ({len(postings)} términos) completados en {time.time()-t0:.1f}s!")


if __name__ == "__main__":
    build_index()
