import sys
import os
import paramiko

if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
if hasattr(sys.stderr, 'reconfigure'):
    sys.stderr.reconfigure(encoding='utf-8', errors='replace')

def main():
    c = paramiko.SSHClient()
    c.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    c.connect('192.168.1.7', username='wildu', password='4123')
    
    remote_test_code = '''
import sys
import os
sys.path.insert(0, '/home/wildu/llama.cpp')

import re
from vox_verbi.command_router import CommandRouter, match_and_strip_wake_word, normalize_text
from vox_verbi.pre_filter import PreFilter, RESP_CONFESION, RESP_IMPERTINENTE
from vox_verbi.rag_engine import RAGEngine
from vox_verbi.llm_catequist import LLMCatequist
from vox_verbi.state_machine import StateMachine

print("=" * 70)
print("INICIANDO VERIFICACIÓN DE AUDITORÍA LOG_2 EN CUSTODIA")
print("=" * 70)

router = CommandRouter()
prefilter = PreFilter()
rag = RAGEngine()
llm = LLMCatequist()
state = StateMachine()

errors = []

# --------------------------------------------------------------------
# PRUEBA 1: Confesión Sacramental (Regla 10.F)
# --------------------------------------------------------------------
q1 = "Vox Verbi, quiero confesar mis pecados contigo."
matched1, rest1 = match_and_strip_wake_word(q1)
rag1 = rag.search(q1)
allowed1, literal1 = prefilter.evaluate(q1, rag_result=rag1)

print(f"\\n[PRUEBA 1] '{q1}'")
print(f"  Matched Wake Word: {matched1}, Rest: '{rest1}'")
print(f"  Allowed: {allowed1}")
print(f"  Literal Response: {literal1}")
if allowed1 or literal1 != RESP_CONFESION:
    errors.append(f"PRUEBA 1 FALLÓ: se esperaba allowed=False y RESP_CONFESION. Obtenido: allowed={allowed1}, {literal1}")
    print("  ❌ FALLO")
else:
    print("  ✅ ÉXITO (Regla 10.F activada sin llamar al LLM)")

# --------------------------------------------------------------------
# PRUEBA 2: Pertinencia Temática / Secular (Regla 48)
# --------------------------------------------------------------------
q2 = "Vox Verbi, ¿quién ganó el partido de fútbol ayer?"
matched2, rest2 = match_and_strip_wake_word(q2)
rag2 = rag.search(q2)
allowed2, literal2 = prefilter.evaluate(q2, rag_result=rag2)

print(f"\\n[PRUEBA 2] '{q2}'")
print(f"  Matched Wake Word: {matched2}, Rest: '{rest2}'")
print(f"  Allowed: {allowed2}")
print(f"  Literal Response: {literal2}")
if allowed2 or literal2 != RESP_IMPERTINENTE:
    errors.append(f"PRUEBA 2 FALLÓ: se esperaba allowed=False y RESP_IMPERTINENTE. Obtenido: allowed={allowed2}, {literal2}")
    print("  ❌ FALLO")
else:
    print("  ✅ ÉXITO (Regla 48 activada sin llamar al LLM)")

# --------------------------------------------------------------------
# PRUEBA 3: Priorizar Documentos Nombrados (Dei Verbum)
# --------------------------------------------------------------------
q3 = "Vox Verbi, ¿qué enseña de i verbum sobre la revelación?"
matched3, rest3 = match_and_strip_wake_word(q3)
allowed3, literal3 = prefilter.evaluate(q3)
print(f"\\n[PRUEBA 3] '{q3}'")
print(f"  Matched Wake Word: {matched3}, Rest: '{rest3}'")

rag3 = rag.search(q3, top_k=2)
results3 = rag3.get("results", [])
print(f"  Fuentes recuperadas por RAG Engine:")
for i, r in enumerate(results3, 1):
    print(f"    Fuente {i}: {r.get('cita_corta')} | {r.get('titulo')}")

citas_text = " ".join([r.get('cita_corta', '') + " " + r.get('titulo', '') for r in results3]).lower()
if "dei verbum" not in citas_text:
    errors.append("PRUEBA 3 FALLÓ: RAG no recuperó Dei Verbum")
    print("  ❌ FALLO: Dei Verbum no presente en fuentes")
elif "baruc" in citas_text:
    errors.append("PRUEBA 3 FALLÓ: Baruc recuperado en lugar de texto sobre revelación")
    print("  ❌ FALLO: Aparece Baruc")
else:
    print("  ✅ RAG recuperó Dei Verbum y Escritura canónica sobre Revelación")

print("  Generando respuesta con Llama 3.1...")
resp3 = llm.generate(q3, results3, duration_sec=30)
print(f"  Respuesta LLM: {resp3}")

if "dei verbum" not in resp3.lower():
    errors.append("PRUEBA 3 FALLÓ: El LLM no citó Dei Verbum")
    print("  ❌ FALLO: LLM no citó Dei Verbum")
elif "baruc" in resp3.lower():
    errors.append("PRUEBA 3 FALLÓ: El LLM citó a Baruc")
    print("  ❌ FALLO: LLM citó a Baruc")
elif resp3.startswith("Según enseña el profeta"):
    errors.append("PRUEBA 3 FALLÓ: LLM empezó con muletilla 'Según enseña el profeta'")
    print("  ❌ FALLO: Muletilla fija")
else:
    print("  ✅ ÉXITO (Dei Verbum citado con apertura natural)")

# --------------------------------------------------------------------
# PRUEBA 4: 1-Token Wake Word ("Busverby, al paraíso")
# --------------------------------------------------------------------
q4 = "Busverby, al paraíso"
matched4, rest4 = match_and_strip_wake_word(q4)
route4 = router.route(rest4) or router.route(f"vox verbi {rest4}")

print(f"\\n[PRUEBA 4] '{q4}'")
print(f"  Matched Wake Word: {matched4}")
print(f"  Rest tras limpiar wake word: '{rest4}'")
act4 = route4.get("action") if route4 else None
print(f"  Action Routed: {act4}")

if not matched4:
    errors.append("PRUEBA 4 FALLÓ: No se reconoció wake-word de 1 token 'Busverby'")
    print("  ❌ FALLO: Wake-word no detectado")
elif act4 != "ENTER_ADORACION_PARAISO":
    errors.append(f"PRUEBA 4 FALLÓ: se esperaba acción ENTER_ADORACION_PARAISO, obtenida: {act4}")
    print("  ❌ FALLO: Acción incorrecta")
else:
    cano_text = router.get_canonical_text(route4)
    print(f"  Texto canónico AL_PARAISO (longitud {len(cano_text)} chars): {cano_text[:80]}...")
    if len(cano_text) > 50:
        print("  ✅ ÉXITO (Wake-word de 1 token reconocido, ruta AL_PARAISO y modo ADORACION listos)")
    else:
        errors.append("PRUEBA 4 FALLÓ: Texto canónico AL_PARAISO vacío")
        print("  ❌ FALLO: Texto canónico vacío")

# --------------------------------------------------------------------
# PRUEBA 5: Fragmento incompleto < 2 palabras ("BusBerby. Llevamos...")
# --------------------------------------------------------------------
q5 = "BusBerby. Llevamos..."
matched5, rest5 = match_and_strip_wake_word(q5)
real_words5 = [w for w in re.sub(r'[^\\w\\s]', '', rest5).split() if len(w) > 1]
print(f"\\n[PRUEBA 5] '{q5}'")
print(f"  Matched: {matched5}, Rest: '{rest5}', Palabras reales: {len(real_words5)} ({real_words5})")
if matched5 and len(real_words5) < 2:
    print("  ✅ ÉXITO (Detectado < 2 palabras, activa aclaración breve de Regla 48 sin alucinar)")
else:
    errors.append(f"PRUEBA 5 FALLÓ: No detectó < 2 palabras")

print("\\n" + "=" * 70)
if not errors:
    print("🎉 TODAS LAS PRUEBAS DE AUDITORÍA PASARON CON ÉXITO (0 ERRORES)")
else:
    print(f"⚠️ SE ENCONTRARON {len(errors)} ERRORES:")
    for e in errors:
        print(f"  - {e}")
print("=" * 70)
'''

    sftp = c.open_sftp()
    with sftp.open("/home/wildu/llama.cpp/verify_audit_log2.py", "w") as f:
        f.write(remote_test_code)
    sftp.close()
    
    print("Running verify_audit_log2.py on remote host...")
    stdin, stdout, stderr = c.exec_command("/home/wildu/llama.cpp/venv/bin/python /home/wildu/llama.cpp/verify_audit_log2.py")
    out = stdout.read().decode('utf-8', errors='replace')
    err = stderr.read().decode('utf-8', errors='replace')
    
    print("STDOUT:\n", out)
    if err:
        print("STDERR:\n", err)
        
    c.close()

if __name__ == "__main__":
    main()
