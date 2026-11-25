import os
import json
import re

SPEC_FILE = "/home/wildu/llama.cpp/SPEC_AGENT_ANTIGRAVITY.md"
CANONICAL_OUTPUT = "/home/wildu/llama.cpp/vox_verbi/data/canonical_readings.json"
MUSIC_OUTPUT = "/home/wildu/llama.cpp/vox_verbi/data/music_catalog.json"

def extract_between(text, start_tag, end_tag):
    pattern = rf"<<<{start_tag}>>>(.*?)<<<{end_tag}>>>"
    match = re.search(pattern, text, re.DOTALL)
    if not match:
        raise ValueError(f"Tag {start_tag} not found in spec")
    return match.group(1).strip("\r\n")

def build_canonical_json(spec_path=SPEC_FILE, out_path=CANONICAL_OUTPUT, music_path=MUSIC_OUTPUT):
    with open(spec_path, "r", encoding="utf-8") as f:
        spec_content = f.read()

    # Extract all sections
    header = extract_between(spec_content, "START_DOC_PRESENTACION_HEADER", "END_DOC_PRESENTACION_HEADER")
    sec_presentacion = extract_between(spec_content, "START_SEC_PRESENTACION", "END_SEC_PRESENTACION")
    
    sections = {}
    sections["SEC_PRESENTACION"] = sec_presentacion
    
    for i in range(1, 12):
        tag = f"SEC_{i:02d}"
        sections[tag] = extract_between(spec_content, f"START_{tag}", f"END_{tag}")
        
    footer = extract_between(spec_content, "START_DOC_PRESENTACION_FOOTER", "END_DOC_PRESENTACION_FOOTER")
    
    # Full presentation
    # Reads from title VOX VERBI and Presentacion until end of section 11 and closure (...fe, conversión, comunión y misión.)
    full_pres_parts = [header, sec_presentacion]
    for i in range(1, 12):
        full_pres_parts.append(sections[f"SEC_{i:02d}"])
    full_pres_parts.append(footer)
    doc_presentacion_completa = "\n\n".join(full_pres_parts)
    
    al_paraiso = extract_between(spec_content, "START_AL_PARAISO", "END_AL_PARAISO")
    
    vv_hist_001 = extract_between(spec_content, "START_VV_HIST_001", "END_VV_HIST_001")
    # Per technical note 397:
    # "En el primer párrafo de VV-HIST-001 el documento original dice 'anlo profundo';
    # conservar exactamente 'anhelo profundo' en el JSON canónico y limpiar '**' únicamente en la capa de síntesis de voz."
    vv_hist_001 = vv_hist_001.replace("anlo profundo", "anhelo profundo")
    
    relato_02 = extract_between(spec_content, "START_RELATO_02", "END_RELATO_02")
    vv_hist_003 = extract_between(spec_content, "START_VV_HIST_003", "END_VV_HIST_003")
    
    canonical_data = {
        "metadata": {
            "version": "2.5",
            "description": "Textos literales inmutables del corpus canonico de Vox Verbi (Apendice A)"
        },
        "presentacion": {
            "header": header,
            "footer": footer,
            "completa": doc_presentacion_completa,
            "secciones": sections
        },
        "adoracion": {
            "AL_PARAISO": al_paraiso
        },
        "relatos": {
            "VV-HIST-001": {
                "id": "VV-HIST-001",
                "titulo": "El sueño de los tres árboles – Un cuento con alma",
                "texto": vv_hist_001,
                "comandos": [
                    "vox verbi las tres cruces",
                    "vox verbi cuentame las tres cruces",
                    "vox verbi hablame de las tres cruces"
                ]
            },
            "VOX_VERBI_RELATO_02_CRUZ_QUE_NADIE_QUERIA": {
                "id": "VOX_VERBI_RELATO_02_CRUZ_QUE_NADIE_QUERIA",
                "titulo": "CUENTO DE APERTURA: “LA CRUZ QUE NADIE QUERÍA”",
                "texto": relato_02,
                "comandos": [
                    "vox verbi la cruz que nadie queria",
                    "la cruz que nadie queria",
                    "activar la cruz que nadie queria",
                    "leer la cruz que nadie queria"
                ]
            },
            "VV-HIST-003": {
                "id": "VV-HIST-003",
                "titulo": "EL JARDÍN DESCUIDADO",
                "texto": vv_hist_003,
                "comandos": [
                    "vox verbi el jardin descuidado"
                ]
            }
        },
        "command_mappings": {
            "vox verbi muestrame para que fuiste creado": {"type": "presentacion", "target": "completa"},
            "vox verbi muestrame para que fuiste hecho": {"type": "presentacion", "target": "SEC_PRESENTACION"},
            "vox verbi explicame tu fundamento": {"type": "presentacion", "target": "SEC_01"},
            "vox verbi como entiendes la evangelizacion": {"type": "presentacion", "target": "SEC_02"},
            "vox verbi como conduces al encuentro con cristo": {"type": "presentacion", "target": "SEC_03"},
            "vox verbi que ofreces a quien no cree": {"type": "presentacion", "target": "SEC_04"},
            "vox verbi que lugar tiene la belleza en tu mision": {"type": "presentacion", "target": "SEC_05"},
            "vox verbi cuales son tus criterios eclesiales": {"type": "presentacion", "target": "SEC_06"},
            "vox verbi cual es tu responsabilidad eclesial": {"type": "presentacion", "target": "SEC_07"},
            "vox verbi cual es tu finalidad pastoral": {"type": "presentacion", "target": "SEC_08"},
            "vox verbi que propones a la iglesia de hoy": {"type": "presentacion", "target": "SEC_09"},
            "vox verbi cual es tu vision": {"type": "presentacion", "target": "SEC_10"},
            "vox verbi definete": {"type": "presentacion", "target": "SEC_11"},
            "vox verbi las tres cruces": {"type": "relato", "target": "VV-HIST-001"},
            "vox verbi cuentame las tres cruces": {"type": "relato", "target": "VV-HIST-001"},
            "vox verbi hablame de las tres cruces": {"type": "relato", "target": "VV-HIST-001"},
            "vox verbi la cruz que nadie queria": {"type": "relato", "target": "VOX_VERBI_RELATO_02_CRUZ_QUE_NADIE_QUERIA"},
            "la cruz que nadie queria": {"type": "relato", "target": "VOX_VERBI_RELATO_02_CRUZ_QUE_NADIE_QUERIA"},
            "activar la cruz que nadie queria": {"type": "relato", "target": "VOX_VERBI_RELATO_02_CRUZ_QUE_NADIE_QUERIA"},
            "leer la cruz que nadie queria": {"type": "relato", "target": "VOX_VERBI_RELATO_02_CRUZ_QUE_NADIE_QUERIA"},
            "vox verbi el jardin descuidado": {"type": "relato", "target": "VV-HIST-003"},
            "vox verbi al paraiso": {"type": "adoracion", "target": "AL_PARAISO"}
        }
    }
    
    if os.path.dirname(out_path):
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(canonical_data, f, ensure_ascii=False, indent=2)
    print(f"Generated canonical readings at {out_path} ({os.path.getsize(out_path)} bytes)")
    
    # Extract Appendix B JSON using code_master anchor
    cm_pos = spec_content.find('"code_master"')
    if cm_pos != -1:
        start_brace = spec_content.rfind("{", 0, cm_pos)
        end_brace = spec_content.find("```", cm_pos)
        end_brace = spec_content.rfind("}", cm_pos, end_brace if end_brace != -1 else len(spec_content)) + 1
        music_json_str = spec_content[start_brace:end_brace]
        music_catalog = json.loads(music_json_str)
        if os.path.dirname(music_path):
            os.makedirs(os.path.dirname(music_path), exist_ok=True)
        with open(music_path, "w", encoding="utf-8") as f:
            json.dump(music_catalog, f, ensure_ascii=False, indent=2)
        print(f"Generated music catalog at {music_path} ({os.path.getsize(music_path)} bytes)")
    else:
        print("Warning: Could not find 'code_master' in spec!")

if __name__ == "__main__":
    import sys
    spec_p = sys.argv[1] if len(sys.argv) > 1 else SPEC_FILE
    out_p = sys.argv[2] if len(sys.argv) > 2 else CANONICAL_OUTPUT
    music_p = sys.argv[3] if len(sys.argv) > 3 else MUSIC_OUTPUT
    build_canonical_json(spec_p, out_p, music_p)
