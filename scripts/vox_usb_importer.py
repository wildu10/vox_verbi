#!/usr/bin/env python3
"""
Importador Automático y Manual de Contenidos por Memoria USB para Vox Verbi.
- Detecta memorias USB con la carpeta VOX_VERBI/fuentes_nuevas (o relatos_nuevos / musica_nueva).
- Procesa documentos .txt, .md y .html, los fragmenta por numerales o bloques teológicos (~105 palabras),
  calcula su huella SHA-256 para evitar duplicados, codifica únicamente los nuevos vectores con
  intfloat/multilingual-e5-small offline y actualiza base_vox_verbi.index + textos_teologicos_meta.pkl + bm25_lexical.pkl.
- Notifica a asistente-vivo.service (SIGHUP) para recargar el índice en memoria sin reiniciar.
"""
import os
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"

import re
import sys
import glob
import json
import math
import time
import shutil
import pickle
import signal
import hashlib
import datetime
import subprocess
import unicodedata
from html.parser import HTMLParser
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer

BASE_DIR = "/home/wildu/llama.cpp"
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

OUTPUT_DIR = os.path.join(BASE_DIR, "vox_verbi/data")
META_PATH = os.path.join(OUTPUT_DIR, "textos_teologicos_meta.pkl")
BM25_PATH = os.path.join(OUTPUT_DIR, "bm25_lexical.pkl")
INDEX_PATH = os.path.join(OUTPUT_DIR, "base_vox_verbi.index")
CANONICAL_JSON_PATH = os.path.join(OUTPUT_DIR, "canonical_readings.json")
MUSIC_DIR = os.path.join(BASE_DIR, "vox_verbi/media/musica")
EMBEDDER_MODEL = "intfloat/multilingual-e5-small"


def compute_sha256(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def tokenize_es(text: str) -> list[str]:
    t = unicodedata.normalize("NFD", (text or "").lower())
    t = "".join(c for c in t if unicodedata.category(c) != "Mn")
    return [w for w in re.findall(r"[a-z0-9]{2,}", t)]


class SimpleHTMLTextExtractor(HTMLParser):
    def __init__(self):
        super().__init__()
        self.skip = 0
        self.blocks = []
        self.cur = []

    def handle_starttag(self, tag, attrs):
        tl = tag.lower()
        if tl in ("script", "style", "sup"):
            self.skip += 1
        elif tl in ("p", "h1", "h2", "h3", "h4", "li", "div", "br"):
            if self.cur:
                s = re.sub(r"\s+", " ", "".join(self.cur)).strip()
                if s:
                    self.blocks.append(s)
                self.cur = []

    def handle_data(self, data):
        if self.skip == 0:
            self.cur.append(data)

    def handle_endtag(self, tag):
        tl = tag.lower()
        if tl in ("script", "style", "sup") and self.skip > 0:
            self.skip -= 1
        elif tl in ("p", "h1", "h2", "h3", "h4", "li", "div"):
            if self.cur:
                s = re.sub(r"\s+", " ", "".join(self.cur)).strip()
                if s:
                    self.blocks.append(s)
                self.cur = []


def read_text_auto_encoding(fpath: str) -> str:
    raw = open(fpath, "rb").read()
    for enc in ("utf-8-sig", "utf-8", "cp1252", "latin-1"):
        try:
            txt = raw.decode(enc)
            if "Ã©" not in txt and "Ã³" not in txt:
                return txt
        except UnicodeDecodeError:
            continue
    return raw.decode("utf-8", errors="replace")


def parse_document_file(fpath: str) -> list[dict]:
    """
    Parses a .txt, .md, or .html file into clean Vox Verbi theological records.
    Supports optional metadata header before '---'.
    """
    ext = os.path.splitext(fpath)[1].lower()
    fname = os.path.splitext(os.path.basename(fpath))[0]
    raw_txt = read_text_auto_encoding(fpath)

    meta = {
        "titulo": fname.replace("_", " ").strip(),
        "autor": "Comisión Teológica Internacional (Santa Sede)",
        "columna": "Magisterio",
        "nivel_autoridad": 2,
        "cita_corta": fname.replace("_", " ").strip()[:70],
        "fecha": datetime.date.today().isoformat(),
        "url": "local://usb_import",
    }

    body = raw_txt
    if "\n---\n" in raw_txt[:1500] or "\r\n---\r\n" in raw_txt[:1500]:
        parts = re.split(r"\r?\n---\r?\n", raw_txt, maxsplit=1)
        hdr, body = parts[0], parts[1]
        for line in hdr.splitlines():
            if ":" in line:
                k, v = line.split(":", 1)
                kl = k.strip().upper()
                vl = v.strip()
                if not vl:
                    continue
                if kl in ("TITULO", "TÍTULO"):
                    meta["titulo"] = vl
                elif kl == "AUTOR":
                    meta["autor"] = vl
                elif kl == "COLUMNA":
                    meta["columna"] = vl
                elif kl == "NIVEL_AUTORIDAD" and vl.isdigit():
                    meta["nivel_autoridad"] = int(vl)
                elif kl == "CITA_CORTA":
                    meta["cita_corta"] = vl
                elif kl == "FECHA":
                    meta["fecha"] = vl
                elif kl == "URL":
                    meta["url"] = vl

    if ext in (".html", ".htm"):
        ex = SimpleHTMLTextExtractor()
        ex.feed(body)
        paragraphs = [p for p in ex.blocks if len(p) >= 30]
    else:
        paragraphs = [
            re.sub(r"\s+", " ", p).strip()
            for p in re.split(r"\r?\n\s*\r?\n", body)
            if len(p.strip()) >= 30
        ]

    # Group or split paragraphs into ~95-125 word theological records, preserving numeral numbers if present
    records = []
    slug_code = re.sub(r"[^A-Za-z0-9]+", "_", fname).strip("_").upper()[:28]
    cur_num = None
    cur_words = []
    rec_seq = 0

    def emit_chunk(num_tag, words_list):
        nonlocal rec_seq
        if not words_list:
            return
        txt_chunk = " ".join(words_list).strip()
        txt_chunk = re.sub(r"\[\d+\]", "", txt_chunk).strip()
        if len(txt_chunk) < 45:
            return
        rec_seq += 1
        if num_tag is not None:
            cita = f"{meta['cita_corta']}, n. {num_tag}"
            tit = f"{meta['titulo']}, n. {num_tag}"
            did = f"USB_{slug_code}_N{num_tag}_{rec_seq:03d}"
        else:
            cita = f"{meta['cita_corta']}, sección {rec_seq}"
            tit = f"{meta['titulo']} (sección {rec_seq})"
            did = f"USB_{slug_code}_S{rec_seq:03d}"

        records.append({
            "doc_id": did,
            "titulo": tit,
            "autor": meta["autor"],
            "tipo_fuente": meta["columna"],
            "fecha_documento": meta["fecha"],
            "idioma": "es",
            "edicion": "Importación Autorizada Vox Verbi",
            "url_oficial": meta["url"],
            "fecha_incorporacion": datetime.date.today().isoformat(),
            "hash_sha256": compute_sha256(txt_chunk),
            "nivel_autoridad": int(meta["nivel_autoridad"]),
            "estado": "autorizado",
            "cita_corta": cita,
            "columna": meta["columna"],
            "texto": txt_chunk[:680],
        })

    for p in paragraphs:
        m_num = re.match(r"^(\d{1,3})\s*[\.\-\)]\s+(.+)$", p)
        if m_num:
            if cur_words:
                emit_chunk(cur_num, cur_words)
                cur_words = []
            cur_num = int(m_num.group(1))
            p_body = m_num.group(2).strip()
        else:
            p_body = p

        p_words = p_body.split()
        for w in p_words:
            cur_words.append(w)
            if len(cur_words) >= 110:
                emit_chunk(cur_num, cur_words)
                cur_words = []

    if cur_words:
        emit_chunk(cur_num, cur_words)

    return records


def rebuild_bm25(records: list[dict]):
    N = len(records)
    doc_lens = np.zeros(N, dtype=np.float32)
    postings = {}
    temp_post = {}
    for idx, r in enumerate(records):
        toks = tokenize_es(f"{r.get('cita_corta','')} {r.get('titulo','')} {r.get('autor','')} {r.get('texto','')}")
        doc_lens[idx] = len(toks)
        tf_map = {}
        for tok in toks:
            tf_map[tok] = tf_map.get(tok, 0) + 1
        for tok, tf in tf_map.items():
            temp_post.setdefault(tok, []).append((idx, min(tf, 255)))

    idf = {}
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


def notify_running_service():
    """Sends SIGHUP to running vox_verbi.main so it hot-reloads FAISS + BM25 without restarting."""
    try:
        out = subprocess.check_output(["pgrep", "-f", "vox_verbi.main"], text=True).strip()
        for pid_str in out.splitlines():
            if pid_str.strip().isdigit():
                pid = int(pid_str.strip())
                if pid != os.getpid():
                    os.kill(pid, signal.SIGHUP)
    except Exception:
        pass


def import_from_directory(source_dir: str, notify_service: bool = True) -> dict:
    """
    Imports new theological documents from source_dir (either VOX_VERBI root or fuentes_nuevas folder).
    Returns summary dict: {docs_found, docs_added, chunks_added, total_records}.
    """
    t0 = time.time()
    fuentes_dir = os.path.join(source_dir, "fuentes_nuevas") if os.path.isdir(os.path.join(source_dir, "fuentes_nuevas")) else source_dir
    if not os.path.isdir(fuentes_dir):
        return {"docs_found": 0, "docs_added": 0, "chunks_added": 0, "total_records": 0, "error": f"No existe {fuentes_dir}"}

    candidate_files = sorted(
        glob.glob(os.path.join(fuentes_dir, "*.txt"))
        + glob.glob(os.path.join(fuentes_dir, "*.md"))
        + glob.glob(os.path.join(fuentes_dir, "*.html"))
    )
    candidate_files = [
        f for f in candidate_files
        if not os.path.basename(f).upper().startswith(("LEEME", "README", "REPORTE"))
    ]

    if not os.path.exists(META_PATH) or not os.path.exists(INDEX_PATH):
        return {"docs_found": len(candidate_files), "docs_added": 0, "chunks_added": 0, "total_records": 0, "error": "Índice base no encontrado"}

    with open(META_PATH, "rb") as f:
        existing_records = pickle.load(f)
    existing_hashes = {r.get("hash_sha256") for r in existing_records if r.get("hash_sha256")}

    new_records = []
    docs_added = 0
    doc_summaries = []

    for fpath in candidate_files:
        recs = parse_document_file(fpath)
        unique_recs = [r for r in recs if r["hash_sha256"] not in existing_hashes]
        if unique_recs:
            docs_added += 1
            for r in unique_recs:
                existing_hashes.add(r["hash_sha256"])
            new_records.extend(unique_recs)
            doc_summaries.append(f"  + {os.path.basename(fpath)}: {len(unique_recs)} pasajes nuevos ({unique_recs[0]['titulo'][:60]})")

    if not new_records:
        return {
            "docs_found": len(candidate_files),
            "docs_added": 0,
            "chunks_added": 0,
            "total_records": len(existing_records),
            "elapsed_sec": round(time.time() - t0, 2),
        }

    # Backup before modifying
    shutil.copy2(META_PATH, META_PATH + ".bak")
    shutil.copy2(INDEX_PATH, INDEX_PATH + ".bak")

    print(f"Codificando {len(new_records)} nuevos pasajes teológicos con {EMBEDDER_MODEL} (offline)...")
    try:
        embedder = SentenceTransformer(EMBEDDER_MODEL, local_files_only=True)
    except Exception:
        embedder = SentenceTransformer(EMBEDDER_MODEL)

    passage_texts = [
        f"passage: {r['cita_corta']} — {r['titulo']}. {r['texto']}"
        for r in new_records
    ]
    new_embs = embedder.encode(
        passage_texts,
        batch_size=128,
        convert_to_numpy=True,
        normalize_embeddings=True,
        show_progress_bar=False,
    ).astype(np.float32)

    index = faiss.read_index(INDEX_PATH)
    index.add(new_embs)
    faiss.write_index(index, INDEX_PATH)

    all_records = existing_records + new_records
    with open(META_PATH, "wb") as f:
        pickle.dump(all_records, f)

    rebuild_bm25(all_records)

    elapsed = round(time.time() - t0, 2)
    report_path = os.path.join(source_dir, "REPORTE_IMPORTACION.txt")
    try:
        with open(report_path, "w", encoding="utf-8") as fr:
            fr.write(
                f"REPORTE DE IMPORTACIÓN VOX VERBI ({datetime.datetime.now().isoformat(timespec='seconds')})\n"
                f"Documentos analizados : {len(candidate_files)}\n"
                f"Documentos nuevos     : {docs_added}\n"
                f"Pasajes incorporados  : {len(new_records)}\n"
                f"Total corpus activo   : {len(all_records)} registros\n"
                f"Tiempo de indexación  : {elapsed} s\n\n"
                + "\n".join(doc_summaries) + "\n"
            )
    except Exception:
        pass

    if notify_service:
        notify_running_service()

    return {
        "docs_found": len(candidate_files),
        "docs_added": docs_added,
        "chunks_added": len(new_records),
        "total_records": len(all_records),
        "elapsed_sec": elapsed,
        "details": doc_summaries,
    }


def find_or_mount_vox_usb() -> tuple[str | None, str | None]:
    """
    Finds any mounted or unmounted USB partition containing VOX_VERBI/fuentes_nuevas.
    Returns (vox_verbi_dir_path, temp_mount_point_if_mounted_by_us).
    """
    # 1. Check already mounted media/mnt paths
    for pattern in ("/media/*/*/VOX_VERBI", "/media/*/VOX_VERBI", "/mnt/*/VOX_VERBI", "/run/media/*/*/VOX_VERBI"):
        for cand in glob.glob(pattern):
            if os.path.isdir(cand):
                return cand, None

    # 2. Check removable / USB block partitions (/dev/sd[a-z][1-9])
    try:
        out = subprocess.check_output(["lsblk", "-rn", "-o", "NAME,FSTYPE,MOUNTPOINT"], text=True)
        for line in out.splitlines():
            parts = line.split()
            if len(parts) < 2:
                continue
            dev_name, fstype = parts[0], parts[1]
            mnt = parts[2] if len(parts) >= 3 else ""
            if not dev_name.startswith("sd") or not any(ch.isdigit() for ch in dev_name):
                continue
            if mnt:
                cand = os.path.join(mnt, "VOX_VERBI")
                if os.path.isdir(cand):
                    return cand, None
            elif fstype in ("vfat", "ntfs", "exfat", "ext4"):
                tmp_mnt = f"/mnt/vox_usb_{dev_name}"
                os.makedirs(tmp_mnt, exist_ok=True)
                rc = subprocess.call(["sudo", "mount", f"/dev/{dev_name}", tmp_mnt])
                if rc == 0:
                    cand = os.path.join(tmp_mnt, "VOX_VERBI")
                    if os.path.isdir(cand):
                        return cand, tmp_mnt
                    subprocess.call(["sudo", "umount", tmp_mnt])
    except Exception:
        pass
    return None, None


def import_from_usb(notify_service: bool = True) -> dict:
    vox_dir, temp_mnt = find_or_mount_vox_usb()
    if not vox_dir:
        # Also check staging folder /home/wildu/llama.cpp/vox_verbi/data/usb_staging/VOX_VERBI if present
        staging = os.path.join(OUTPUT_DIR, "usb_staging/VOX_VERBI")
        if os.path.isdir(staging):
            return import_from_directory(staging, notify_service=notify_service)
        return {"docs_found": 0, "docs_added": 0, "chunks_added": 0, "total_records": 0, "error": "No se detectó memoria USB con carpeta VOX_VERBI"}

    try:
        res = import_from_directory(vox_dir, notify_service=notify_service)
        subprocess.call(["sync"])
        return res
    finally:
        if temp_mnt:
            subprocess.call(["sudo", "umount", temp_mnt])


if __name__ == "__main__":
    target_dir = None
    if "--dir" in sys.argv:
        idx = sys.argv.index("--dir")
        if idx + 1 < len(sys.argv):
            target_dir = sys.argv[idx + 1]

    print("=" * 68)
    print("  VOX VERBI - IMPORTADOR DE FUENTES Y CONTENIDOS (USB / LOCAL)")
    print("=" * 68)
    if target_dir:
        res = import_from_directory(target_dir)
    else:
        res = import_from_usb()

    if res.get("error"):
        print(f"  [!] {res['error']}")
    else:
        print(f"  Documentos encontrados : {res['docs_found']}")
        print(f"  Documentos nuevos      : {res['docs_added']}")
        print(f"  Pasajes indexados      : {res['chunks_added']}")
        print(f"  Total corpus activo    : {res['total_records']} registros")
        print(f"  Tiempo de proceso      : {res.get('elapsed_sec', 0)} s")
        for d in res.get("details", [])[:40]:
            print(d)
    print("=" * 68)
