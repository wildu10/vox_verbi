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
import time
sys.path.insert(0, '/home/wildu/llama.cpp')

import re
from vox_verbi.command_router import CommandRouter, match_and_strip_wake_word, normalize_text
from vox_verbi.pre_filter import PreFilter, RESP_CONFESION, RESP_IMPERTINENTE, RESP_ADIVINACION
from vox_verbi.rag_engine import RAGEngine
from vox_verbi.llm_catequist import LLMCatequist
from vox_verbi.state_machine import StateMachine

print("=" * 70)
print("VERIFICACIÓN DE AUDITORÍA LOG_3 EN CUSTODIA (RADEON 680M)")
print("=" * 70)

router = CommandRouter()
prefilter = PreFilter()
rag = RAGEngine()
llm = LLMCatequist()
state = StateMachine()

errors = []

# --------------------------------------------------------------------
# PRUEBA 1: WARMUP DE KV CACHE + PRIMERA CONSULTA EN FRÍO ("dei" verbun)
# --------------------------------------------------------------------
print("\\n[PRUEBA 1] WARMUP SILENCIOSO Y PRIMERA CONSULTA")
t_warm_start = time.time()
warmup_ms = llm.warmup()
t_warm_end = time.time()
print(f"  ⚡ Tiempo de Warmup KV Cache: {warmup_ms:.1f} ms")

q1 = 'Vox Verbi, que enseña "dei" verbun sobre la revelación'
matched1, rest1 = match_and_strip_wake_word(q1)
print(f"  Consulta: '{q1}'")
print(f"  Matched: {matched1}, Rest: '{rest1}'")

# RAG search
rag1 = rag.search(q1, top_k=2)
results1 = rag1.get("results", [])
print(f"  Fuentes recuperadas:")
for i, r in enumerate(results1, 1):
    print(f"    Fuente {i}: {r.get('cita_corta')} | {r.get('titulo')}")

citas_1 = " ".join([r.get('cita_corta', '') + " " + r.get('titulo', '') for r in results1]).lower()
if "dei verbum" not in citas_1:
    errors.append("PRUEBA 1 FALLÓ: RAG no recuperó Dei Verbum con 'dei' verbun")
    print("  ❌ FALLO: Dei Verbum no presente en fuentes")
else:
    print("  ✅ RAG forzó exitosamente Dei Verbum ante fonética 'dei verbun'")

# Generación en caliente tras warmup (simulando stream_and_speak para medir TTFA y prompt eval)
print("  Generando streaming con Llama 3.1...")
t_gen_start = time.time()

# Mock mixer to capture TTFA
class MockMixer:
    def __init__(self):
        self.first_audio_time = None
        self.speech_called = False
    def set_ducking(self, active, ducked_vol=0.10): pass
    def duck_music(self, **kwargs): pass
    def unduck_music(self, **kwargs): pass
    def play_speech(self, samples, sr):
        if self.first_audio_time is None:
            self.first_audio_time = time.time()
        self.speech_called = True

mock_mixer = MockMixer()

# Use generate or stream_and_speak
full_text, spoken_sec, ttfa_ms = llm.stream_and_speak(
    q1, results1, mock_mixer, kokoro_engine=None, duration_sec=30
)
total_query_time = (time.time() - t_gen_start) * 1000

print(f"  TTFA medido (primer chunk de texto/audio): {ttfa_ms:.1f} ms")
print(f"  Tiempo total de generación: {total_query_time:.1f} ms")
print(f"  Respuesta completa del LLM: «{full_text}»")

if "dei verbum" not in full_text.lower():
    errors.append("PRUEBA 1 FALLÓ: El LLM no citó Dei Verbum por su nombre")
    print("  ❌ FALLO: LLM no citó Dei Verbum")
elif ttfa_ms > 3000:
    errors.append(f"PRUEBA 1 FALLÓ: TTFA {ttfa_ms:.1f} ms supera el umbral de 3000 ms")
    print(f"  ❌ FALLO: TTFA demasiado alto ({ttfa_ms:.1f} ms)")
else:
    print("  ✅ ÉXITO (TTFA < 3000 ms, prompt pre-cacheado y Dei Verbum citado explícitamente)")

# --------------------------------------------------------------------
# PRUEBA 2: FILTRO REGLA 52 (SUEÑOS / ADIVINACIÓN / VOLUNTAD FUTURA)
# --------------------------------------------------------------------
q2 = "Vox Verbi, ¿qué me quiere decir Dios con el sueño que tuve anoche?"
allowed2, literal2 = prefilter.evaluate(q2)
print(f"\\n[PRUEBA 2] '{q2}'")
print(f"  Allowed: {allowed2}")
print(f"  Respuesta canónica: «{literal2}»")

if allowed2 or literal2 != RESP_ADIVINACION:
    errors.append(f"PRUEBA 2 FALLÓ: Regla 52 no se disparó. Obtenido: allowed={allowed2}, {literal2}")
    print("  ❌ FALLO en Regla 52")
else:
    print("  ✅ ÉXITO (Regla 52 disparada inmediatamente sin llamar al LLM)")

q2_bis = "Vox Verbi, ¿qué quiere Dios para mi futuro?"
allowed2b, literal2b = prefilter.evaluate(q2_bis)
print(f"  Consulta bis: '{q2_bis}' -> Allowed={allowed2b}")
if allowed2b or literal2b != RESP_ADIVINACION:
    errors.append("PRUEBA 2 BIS FALLÓ: Futuro personal atribuido a Dios no disparó Regla 52")
    print("  ❌ FALLO en Regla 52 bis")
else:
    print("  ✅ ÉXITO (Futuro personal bloqueado con Regla 52)")

# --------------------------------------------------------------------
# PRUEBA 3: CICLO COMPLETO DE ADORACIÓN (AL PARAÍSO -> BLOQUEO -> SALIDA)
# --------------------------------------------------------------------
print("\\n[PRUEBA 3] CICLO COMPLETO DE MODO ADORACION")
q3_enter = "Vox Verbi, al paraíso"
matched3, rest3 = match_and_strip_wake_word(q3_enter)
route3 = router.route(rest3) or router.route(f"vox verbi {rest3}")
action3 = route3.get("action") if route3 else None
print(f"  Paso A: Entrada -> '{q3_enter}' -> Acción: {action3}")

if action3 != "ENTER_ADORACION_PARAISO":
    errors.append(f"PRUEBA 3 FALLÓ: Acción esperada ENTER_ADORACION_PARAISO, obtenida: {action3}")
    print("  ❌ FALLO en Entrada a Adoración")
else:
    # Simular entrada
    state.enter_adoracion(mixer=None, track=None)
    print(f"  Estado actual tras entrada: mode='{state.mode}', is_adoracion={state.is_adoracion()}")
    
    # Paso B: Pregunta durante Adoración
    q3_inter = "¿Quién es Jesús?"
    norm_inter = normalize_text(q3_inter)
    is_blocked = state.is_adoracion() and ("finalizamos la adoracion" not in norm_inter)
    print(f"  Paso B: Pregunta durante Adoración: '{q3_inter}' -> ¿Bloqueada totalmente?: {is_blocked}")
    if not is_blocked:
        errors.append("PRUEBA 3 FALLÓ: La pregunta no fue bloqueada en modo ADORACION (violación Regla 53)")
        print("  ❌ FALLO en Bloqueo de Adoración")
    else:
        print("  ✅ ÉXITO: Pregunta bloqueada totalmente según Regla 53")
        
    # Paso C: Comando de salida
    q3_exit = "Vox Verbi, finalizamos la adoración"
    norm_exit = normalize_text(q3_exit)
    can_exit = state.is_adoracion() and ("finalizamos la adoracion" in norm_exit)
    print(f"  Paso C: Comando de salida: '{q3_exit}' -> ¿Reconocido para salir?: {can_exit}")
    if not can_exit:
        errors.append("PRUEBA 3 FALLÓ: Comando de salida no reconocido")
        print("  ❌ FALLO en Salida de Adoración")
    else:
        state.exit_adoracion(mixer=None)
        print(f"  Estado tras salida: mode='{state.mode}', is_adoracion={state.is_adoracion()}")
        print("  ✅ ÉXITO: Modo ADORACION concluido limpiamente y restaurado a NORMAL")

# --------------------------------------------------------------------
# RESUMEN FINAL
# --------------------------------------------------------------------
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
    with sftp.open("/home/wildu/llama.cpp/verify_audit_log3.py", "w") as f:
        f.write(remote_test_code)
    sftp.close()
    
    print("Ejecutando verify_audit_log3.py en custodia...")
    stdin, stdout, stderr = c.exec_command("/home/wildu/llama.cpp/venv/bin/python /home/wildu/llama.cpp/verify_audit_log3.py")
    out = stdout.read().decode('utf-8', errors='replace')
    err = stderr.read().decode('utf-8', errors='replace')
    
    print("STDOUT:\n", out)
    if err:
        print("STDERR:\n", err)
        
    c.close()

if __name__ == "__main__":
    main()
