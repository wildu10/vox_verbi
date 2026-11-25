# ESPECIFICACIÓN MAESTRA AUTOCONTENIDA PARA AGENTE IA (ANTIGRAVITY IDE)
## Proyecto: Refactorización Integral a "VOX VERBI v2.5" (Flujo de Audio, RAG, LLM y Multimedia)

### 1. CONTEXTO DEL ENTORNO Y REGLAS DE EJECUCIÓN DEL AGENTE
- **Hardware:** Mini PC Bosgame P3 (AMD Ryzen 9, GPU Radeon 680M, 16GB RAM).
- **Directorio de trabajo:** `/home/wildu/llama.cpp` con entorno virtual Python 3.12 en `/home/wildu/llama.cpp/venv`.
- **Componentes locales ya instalados:**
  - Servidor LLM: `/home/wildu/llama.cpp/build/bin/llama-server` con modelo `models/Meta-Llama-3.1-8B-Instruct-Q4_K_M.gguf` en puerto `8080`.
  - STT: `sherpa_onnx` Whisper Small en `./voice_models/stt/sherpa-onnx-whisper-small/`.
  - TTS: `kokoro_onnx` en `./voice_models/tts/kokoro-v1.0.onnx` y `./voice_models/tts/voices-v1.0.bin` (voz `em_alex`).
  - Corpus base ya existentes en raíz: `biblia.txt` (y carpeta `La-Biblia-de-Jerusalen-Project`) y `catecismo.txt`.
  - Salida de audio ALSA: `plughw:1,0`.
- **Alcance actual:** Implementar el 100% del flujo de captura de micrófono, VAD, STT, enrutador de estados, módulos de lectura canónica inmutable, catálogo musical con *audio ducking*, RAG jerárquico de 4 columnas, filtro previo de 5 pasos y generación pedagógica con control estricto de tiempo. El control físico de luces/movimiento/incienso se dejará como funciones stub listas para futura conexión serial/GPIO en `hardware_hooks.py`.

---

### 2. ARQUITECTURA DE DIRECTORIOS A CONSTRUIR
El agente debe crear el paquete modular `/home/wildu/llama.cpp/vox_verbi/` sin borrar `asistente_vivo.py` (conservar como respaldo):

vox_verbi/
├── __init__.py
├── config.py                   # Constantes, umbrales, rutas y presupuestos de palabras/tiempo
├── hardware_hooks.py           # Stubs para futuro control de luces, custodia e incienso
├── state_machine.py            # Máquina de estados (CATEQUESIS vs ADORACION) y temporizador (30s / 1-4m)
├── audio_mixer.py              # Mezclador ALSA (Voz + Música con Audio Ducking) e intérprete prosódico
├── command_router.py           # Enrutador determinista de comandos (Lecturas, Cuentos, Música, Adoración)
├── pre_filter.py               # Filtro obligatorio de 5 pasos (Pertinencia, Autoría Sagrada, Seguridad)
├── rag_engine.py               # Motor FAISS con metadatos de las 4 columnas y trazabilidad exacta
├── llm_catequist.py            # Cliente SSE Llama 3.1 con Prompt Maestro v2.5 y watchdog de tiempo
├── main.py                     # Bucle principal (Micrófono USB dinámico + WebRTCVAD + Pipeline completo)
├── scripts/
│   ├── build_canonical_json.py # Genera canonical_readings.json y music_catalog.json desde este archivo
│   ├── download_music.py       # Descarga MUS-001 a MUS-006 con yt-dlp y convierte a .ogg/.wav
│   ├── build_whitelist_rag.py  # Descarga documentos del Vaticano e indexa las 4 columnas en FAISS
│   └── test_vox_verbi_suite.py # Batería de 7 pruebas automatizadas de validación
├── data/
│   ├── canonical_readings.json # Textos literales inmutables extraídos del Apéndice A de este documento
│   ├── music_catalog.json      # Catálogo MUS-001 a MUS-006 extraído del Apéndice B
│   └── corpus_whitelist/       # Textos y metadatos verificados de Escritura, Magisterio y Patrística
└── media/
    └── musica/                 # Archivos locales MUS-001.ogg a MUS-006.ogg

---

### 3. PLAN DE IMPLEMENTACIÓN PASO A PASO

#### FASE 1: GENERACIÓN DE DATOS CANÓNICOS INMUTABLES (`canonical_readings.json`)
1. Crear y ejecutar `vox_verbi/scripts/build_canonical_json.py` para volcar exactamente los textos del **APÉNDICE A** de este documento en `vox_verbi/data/canonical_readings.json`.
2. **REGLA ABSOLUTA DE INTEGRIDAD TEXTUAL:** Está prohibido resumir, corregir, alterar palabras, cambiar signos de puntuación o pasar estos textos por el LLM. Cuando se active un comando de lectura o cuento, el texto de `canonical_readings.json` va directamente al motor narrativo de `audio_mixer.py`.
3. Mapeo exacto de comandos de activación (normalizados sin tildes ni signos para comparación *fuzzy* >= 88%):
   - **Presentación Completa:**
     - Comandos: `"vox verbi muestrame para que fuiste creado"` -> Lee desde el título `VOX VERBI` y `Presentación` hasta el final del apartado 11 y el cierre (*"...fe, conversión, comunión y misión."*).
   - **Activación Individual por Sección (Protocolo de Lectura Selectiva):**
     - `SEC_PRESENTACION`: `"vox verbi muestrame para que fuiste hecho"` -> Lee únicamente la sección `Presentación` (detenerse antes de `1. El fundamento: no inventar, sino transmitir`). *(Nota: si el usuario pide específicamente "la presentación completa", leer todo el documento).*
     - `SEC_01`: `"vox verbi explicame tu fundamento"` -> Lee apartado 1 completo.
     - `SEC_02`: `"vox verbi como entiendes la evangelizacion"` -> Lee apartado 2 completo.
     - `SEC_03`: `"vox verbi como conduces al encuentro con cristo"` -> Lee apartado 3 completo.
     - `SEC_04`: `"vox verbi que ofreces a quien no cree"` -> Lee apartado 4 completo.
     - `SEC_05`: `"vox verbi que lugar tiene la belleza en tu mision"` -> Lee apartado 5 completo.
     - `SEC_06`: `"vox verbi cuales son tus criterios eclesiales"` -> Lee apartado 6 completo.
     - `SEC_07`: `"vox verbi cual es tu responsabilidad eclesial"` -> Lee apartado 7 completo.
     - `SEC_08`: `"vox verbi cual es tu finalidad pastoral"` -> Lee apartado 8 completo.
     - `SEC_09`: `"vox verbi que propones a la iglesia de hoy"` -> Lee apartado 9 completo.
     - `SEC_10`: `"vox verbi cual es tu vision"` -> Lee apartado 10 completo.
     - `SEC_11`: `"vox verbi definete"` -> Lee apartado 11 completo y se detiene.
   - **Repertorio Narrativo (3 Historias):**
     - `VV-HIST-001` (*El sueño de los tres árboles – Un cuento con alma*): Comandos `"vox verbi las tres cruces"`, `"vox verbi cuentame las tres cruces"`, `"vox verbi hablame de las tres cruces"`.
     - `VOX_VERBI_RELATO_02_CRUZ_QUE_NADIE_QUERIA` (*CUENTO DE APERTURA: “LA CRUZ QUE NADIE QUERÍA”*): Comandos `"vox verbi la cruz que nadie queria"`, `"la cruz que nadie queria"`, `"activar la cruz que nadie queria"`, `"leer la cruz que nadie queria"`.
     - `VV-HIST-003` (*EL JARDÍN DESCUIDADO*): Comando `"vox verbi el jardin descuidado"`.
   - **Transición Eucarística:**
     - `AL_PARAISO`: Comando `"vox verbi al paraiso"`.

#### FASE 2: DESCARGA MUSICAL, MEZCLADOR MULTICANAL Y MOTOR PROSÓDICO
1. Instalar dependencias en `./venv`: `pip install yt-dlp pygame sounddevice soundfile beautifulsoup4 lxml`. Asegurar que `ffmpeg` esté instalado en el sistema (`sudo apt-get install -y ffmpeg libsdl2-mixer-2.0-0`).
2. Ejecutar `vox_verbi/scripts/download_music.py` para descargar a `vox_verbi/media/musica/` los 6 recursos oficiales definidos en el **APÉNDICE B** (`MUS-001.ogg` a `MUS-006.ogg`), normalizados a 24000 Hz o 44100 Hz.
3. Implementar `vox_verbi/audio_mixer.py`:
   - Solucionar el bloqueo exclusivo de ALSA abriendo un único mezclador de audio hacia `plughw:1,0` (usando `pygame.mixer` o un stream continuo en `sounddevice` que sume matemáticamente los buffers PCM de música y voz).
   - **Audio Ducking:** Mientras Kokoro TTS está reproduciendo voz, el canal de música reduce su ganancia suavemente al `18%`. En las pausas largas (> 2.5s) y al terminar de hablar, la música vuelve suavemente al `45%`.
   - **Comandos de control musical:**
     - `"vox verbi silencio"` (`MUS-CTRL-04`) -> Pausa/detiene inmediatamente la música.
     - `"vox verbi continua la musica"` (`MUS-CTRL-05`) -> Reanuda la reproducción.
     - `"vox verbi termina la musica"` (`MUS-CTRL-06`) -> Finaliza la reproducción con *fade-out*.
   - **Preprocesador Prosódico para Kokoro TTS (`em_alex`):**
     - **Limpieza obligatoria:** Jamás enviar al sintetizador etiquetas de dirección entre corchetes (`[VOZ SERENA]`, `[VOZ ÍNTIMA]`, `[VOZ PROFUNDA]`, `[ÉNFASIS SUAVE]`, `[TRANSICIÓN SUAVE]`) ni asteriscos `**` o `_`.
     - **Traducción de etiquetas a silencios PCM exactos:**
       - `[PAUSA BREVE]` -> Insertar `1.2` segundos de silencio.
       - `[PAUSA DE 3 SEGUNDOS]` -> Insertar `3.0` segundos de silencio.
       - `[PAUSA DE 4 SEGUNDOS]` -> Insertar `4.0` segundos de silencio.
       - `[PAUSA DE 5 SEGUNDOS]` -> Insertar `5.0` segundos de silencio.
       - `[SILENCIO DE 6 SEGUNDOS]` -> Insertar `6.0` segundos de silencio.
       - Separador `---` en *La cruz que nadie quería* -> Insertar `2.0` segundos de silencio.
       - Puntos suspensivos `…` o `...` -> Asegurar pausa expresiva de `0.7` segundos.
     - **Modulación de velocidad y volumen:**
       - En modo catequesis normal: `speed = 1.0`, ganancia `1.0`.
       - En relatos (`VV-HIST-001`, `RELATO_02`, `VV-HIST-003`) y `AL_PARAISO`: `speed = 0.88`, ganancia base `0.82` (voz íntima/serena).
       - En `AL_PARAISO`: aplicar atenuación progresiva en las últimas frases (*"El Señor nos espera..."* hasta *"Permanezcamos en su presencia."*) reduciendo la ganancia de `0.82` a `0.58`.

#### FASE 3: MÁQUINA DE ESTADOS, ADORACIÓN EUCARÍSTICA Y CONTROL DE DURACIÓN
1. Implementar `vox_verbi/state_machine.py`:
   - Estado inicial: `mode = "CATEQUESIS"`, `duration_limit_sec = 30`.
2. **Transición a Modo 2 (Custodia / Adoración Eucarística):**
   - **Vía 1:** Comando `"vox verbi iniciamos la entronizacion del santisimo"`. Acción: detiene cualquier voz, ejecuta `hardware_hooks.on_enter_adoracion()` y cambia inmediatamente a `mode = "ADORACION"`.
   - **Vía 2:** Comando `"vox verbi al paraiso"`. Acción: inicia suavemente la pista `MUS-005` (`wZ7KaCZ6i44`) durante 3 segundos, proclama una sola vez el texto canónico `AL_PARAISO` respetando sus pausas, pronuncia el cierre *"Cristo está aquí. Permanezcamos en su presencia."*, ejecuta `hardware_hooks.on_enter_adoracion()` y pasa a `mode = "ADORACION"` manteniendo `MUS-005` de fondo.
3. **Bloqueo Total en `ADORACION` (Regla 53):**
   - Mientras `mode == "ADORACION"`, queda prohibido responder preguntas (pertinentes o impertinentes), iniciar relatos o invocar al LLM.
   - El bucle de escucha ignora absolutamente todo el audio capturado, excepto el comando exacto de salida: `"vox verbi finalizamos la adoracion"`.
   - Al detectar `"vox verbi finalizamos la adoracion"`, desvanece la música (`fadeout` de 4 segundos), ejecuta `hardware_hooks.on_exit_adoracion()` y restablece `mode = "CATEQUESIS"`.
4. **Control Dinámico de Duración (Reglas 16, 23-28):**
   - Por defecto, toda respuesta catequética tiene `duration_limit_sec = 30` (buscando durar entre 10 y 20 segundos, con tope absoluto de 30 segundos).
   - Si la petición del predicador incluye explícitamente:
     - `"durante un minuto"` / `"por un minuto"` -> `duration_limit_sec = 60`
     - `"durante dos minutos"` / `"por dos minutos"` -> `duration_limit_sec = 120`
     - `"durante tres minutos"` / `"por tres minutos"` -> `duration_limit_sec = 180`
     - `"durante cuatro minutos"` / `"por cuatro minutos"` -> `duration_limit_sec = 240`
   - **Restablecimiento automático incondicional:** En cuanto termina esa única intervención ampliada, `state_machine.reset_duration()` devuelve `duration_limit_sec = 30`.

#### FASE 4: CORPUS CERRADO (LISTA BLANCA) Y MOTOR RAG DE 4 COLUMNAS
1. Ejecutar `vox_verbi/scripts/build_whitelist_rag.py` para construir el corpus cerrado versionado (`Vox Verbi Corpus v1.0`) siguiendo las Secciones 2, 55 y 56:
   - **Nivel 1 — Sagrada Escritura:** Procesar `La-Biblia-de-Jerusalen-Project` extrayendo Libro, Capítulo y Versículo exacto para cada fragmento (ej. `"cita": "1 Corintios 13, 4-7"`, `"columna": "Sagrada Escritura"`, `"nivel_autoridad": 1`).
   - **Nivel 2 — Magisterio Conciliar y Pontificio:** Descargar desde las URLs oficiales de `vatican.va` listadas en el **APÉNDICE C** los textos en español de:
     - *Dei Verbum*, *Lumen gentium*, *Sacrosanctum Concilium*, *Gaudium et spes* (Concilio Vaticano II).
     - *Deus caritas est* (Benedicto XVI) y *Mysterium fidei* (Pablo VI).
     - Etiquetar cada fragmento con documento y numeral (`"columna": "Magisterio"`, `"nivel_autoridad": 2`).
   - **Nivel 3 — Catecismo de la Iglesia Católica:** Procesar `catecismo.txt` y el archivo del Catecismo en `vatican.va` conservando el numeral exacto (`"columna": "Catecismo de la Iglesia Católica"`, `"nivel_autoridad": 3`).
   - **Nivel 4 — Tradición y Padres de la Iglesia:** Incorporar los fragmentos patrísticos base definidos en el script (*Didaché*, *Cartas de San Ignacio de Antioquía*, *San Agustín*, *San Ireneo*, *San Gregorio de Nisa*) con metadato estricto: `"columna": "Tradición Patrística"`, `"nivel_autoridad": 4`, `"nota_autoridad": "Interpretación patrística (no constituye por sí sola dogma oficial del Magisterio)"`.
2. Cada registro en `vox_verbi/data/textos_teologicos_meta.pkl` debe contener los campos obligatorios de la Regla 56.5:
   `doc_id`, `titulo`, `autor`, `tipo_fuente`, `fecha_documento`, `idioma`, `edicion`, `url_oficial`, `fecha_incorporacion`, `hash_sha256`, `nivel_autoridad`, `estado` (`"autorizado"`), `cita_corta` y `texto`.
3. Compilar el índice vectorial en `vox_verbi/data/base_vox_verbi.index` usando `SentenceTransformer('all-MiniLM-L6-v2')`.
4. Configurar `vox_verbi/rag_engine.py`:
   - Recuperar candidatos con FAISS y ordenarlos respetando la prioridad (Escritura -> Magisterio/Catecismo -> Tradición -> Teología).
   - Asignar Nivel de Confianza (Regla 6):
     - `NIVEL A` (Distancia < 0.85 en fuente directa).
     - `NIVEL B` (Múltiples fuentes con distancia < 1.02 que permiten síntesis).
     - `NIVEL C` (Fuentes de nivel 4/5 patrísticas o teológicas).
     - `NIVEL D` (Distancia > 1.05 en todas las fuentes -> Sin fundamento suficiente).

#### FASE 5: FILTRO PREVIO DE 5 PASOS Y GENERACIÓN PEDAGÓGICA (LLAMA 3.1)
1. Implementar `vox_verbi/pre_filter.py` (Reglas 10, 48, 49, 51, 52):
   - **Paso 1 y 3 (Filtro de Autoría Sagrada - Regla 49):** Si el usuario pide que Vox Verbi hable en primera persona como Dios, Jesucristo, el Espíritu Santo, la Virgen María, un santo, el Papa o cualquier figura histórica (ej. *"Háblame como si fueras Jesús"*, *"¿Qué me dirías tú, María?"*), abortar llamada al LLM y responder literalmente:
     `"No hablaré en nombre de esa persona; puedo explicarte lo que enseñan las fuentes autorizadas sobre este tema."`
   - **Paso 5 (Filtro de Confesión, Profecía, Adivinación y Revelación Privada - Reglas 10 y 52):**
     - Si solicita confesión o absolución de pecados -> Responder literalmente:
       `"Vox Verbi no administra el sacramento de la Reconciliación. Para la confesión sacramental debes acudir a un sacerdote."`
     - Si solicita adivinación, interpretación de sueños como revelación, profecías futuras o saber qué decisión personal quiere Dios para su vida -> Responder literalmente:
       `"No puedo atribuir a Dios una respuesta personal que no esté contenida en la Revelación o en las fuentes autorizadas. Sí puedo mostrarte qué principios ofrece la Escritura y la enseñanza de la Iglesia sobre este tema."`
     - Si solicita diagnóstico médico o psicológico -> Declinar e indicar acudir a un profesional de la salud.
   - **Paso 2 (Filtro de Pertinencia Temática - Regla 48):** Si la consulta trata sobre temas seculares ajenos a la fe/catequesis/moral (cocina, deportes, programación, política partidista, etc.) y el RAG devuelve `NIVEL D`, abortar llamada al LLM y responder literalmente:
     `"Esta pregunta no corresponde al tema catequético de Vox Verbi. Permíteme continuar con el contenido de la fe que estamos profundizando."`
   - **Paso 4 (Filtro de Fuentes / Regla Anti-Alucinación - Regla 5):** Si la pregunta es de índole religiosa pero el RAG devuelve `NIVEL D` (no existe sustento en el corpus autorizado), responder literalmente:
     `"No encuentro en las fuentes autorizadas de Vox Verbi información suficiente para responder con seguridad a esta pregunta. Puedo ayudarte a formular la pregunta desde la Sagrada Escritura, el Magisterio o la Tradición."`
2. Implementar `vox_verbi/llm_catequist.py`:
   - Mapeo de presupuesto de palabras según `duration_limit_sec`:
     - `30s`: `max_words = 55`, `max_tokens = 105` (diseñado para durar 15-25s, máximo absoluto 30s).
     - `60s`: `max_words = 130`, `max_tokens = 220`.
     - `120s`: `max_words = 260`, `max_tokens = 440`.
     - `180s`: `max_words = 390`, `max_tokens = 650`.
     - `240s`: `max_words = 520`, `max_tokens = 860`.
   - **Prompt Maestro del Sistema:**
     ```text
     IDENTIDAD: Eres Vox Verbi («Voz de la Palabra»), una herramienta de apoyo catequético católico subordinada al predicador. No eres Jesús, no eres el Espíritu Santo, no eres la Virgen María, no eres el Papa, no eres un sacerdote y no posees autoridad doctrinal propia. Vox Verbi no toma la palabra; presta la palabra.
     REGLA DE ORO (NO INVENTAR): Utiliza ÚNICAMENTE las fuentes del corpus autorizado inyectadas en esta consulta. No inventes citas, numerales ni enseñanzas. Si el predicador atribuye por error una frase a una fuente que no la sostiene, corrígelo con respeto.
     PROHIBICIÓN DE SUPLANTACIÓN: Nunca hables en primera persona como Dios, Jesús, María, los santos ni ningún autor. Usa siempre atribución externa verificable («En la Primera Carta de Juan, capítulo 4, leemos...», «San Pablo, en 1 Corintios 13, enseña...», «El Catecismo de la Iglesia Católica, en el numeral...», «Benedicto XVI, en Deus caritas est, explica...», «San Agustín interpreta...»).
     DISTINCIÓN DE AUTORIDAD: Distingue siempre entre texto bíblico (Escritura), enseñanza oficial (Magisterio/Catecismo), interpretación patrística (nunca la presentes como dogma automático) y opinión teológica.
     SECUENCIA PEDAGÓGICA OBLIGATORIA (RESPONDER PARA ENSEÑAR):
     1. Responde directamente a la pregunta sin saludos ni preámbulos.
     2. Presenta la fuente o fuentes relevantes citando su referencia y explica claramente qué significan.
     3. Cuando uses dos o más fuentes, relaciona qué aporta cada una.
     4. SÍNTESIS PEDAGÓGICA FINAL OBLIGATORIA: Termina SIEMPRE con una conclusión integradora breve y memorable introducida por «En definitiva,» o «En conjunto,» que responda al núcleo de la pregunta.
     5. Silencio final: No hagas preguntas de vuelta ni intentes dirigir la catequesis.
     LÍMITE ESTRICTO DE EXTENSIÓN: Tu respuesta completa NO puede superar las {max_words} palabras (equivalente a un máximo de {duration_limit_sec} segundos hablados).
     ```
   - **Watchdog de Tiempo en el Stream TTS:** Mientras Kokoro sintetiza las oraciones del LLM, acumular `duracion_acumulada += len(samples) / sample_rate`. En modo de 30s, si la siguiente oración haría que `duracion_acumulada > 30.0`, truncar limpiamente en la síntesis final para garantizar que jamás se violen los 30 segundos.

#### FASE 6: INTEGRACIÓN EN `main.py`, DETECCIÓN USB DINÁMICA Y SERVICIOS SYSTEMD
1. En `vox_verbi/main.py`:
   - Implementar detección automática del micrófono USB buscando en `pyaudio` dispositivos con `maxInputChannels > 0` cuyo nombre contenga `"USB"`, `"Webcam"` o `"Camera"`, con fallback al dispositivo de entrada por defecto.
   - Mantener WebRTCVAD nivel 3 con *debounce* de 15 chunks (450 ms) para evitar disparos por ruido.
   - Requerir prefijo de activación `"vox verbi"` (con tolerancia fonética a transcripciones de Whisper como `"box berbi"`, `"voz verbi"`, `"vos verbi"`, `"vox derbi"`) para procesar cualquier comando o pregunta.
2. Configurar y activar `/etc/systemd/system/llama-server.service` y `/etc/systemd/system/asistente-vivo.service` (apuntando a `/home/wildu/llama.cpp/venv/bin/python -m vox_verbi.main`).

#### FASE 7: EJECUCIÓN DE `test_vox_verbi_suite.py`
El agente debe crear y correr `vox_verbi/scripts/test_vox_verbi_suite.py` validando automáticamente:
1. Integridad exacta (hash/string match) de los 11 apartados de la Presentación y los 3 cuentos.
2. Limpieza total de etiquetas `[...]` y conversión a silencios PCM en *El jardín descuidado* y *La cruz que nadie quería*.
3. Activación, pausa y *audio ducking* de `MUS-001` a `MUS-006`.
4. Rechazo exacto ante preguntas fuera de tema y ante solicitudes de hablar en primera persona como Jesús o María.
5. Respuesta a *"Vox Verbi, háblame del amor"* cumpliendo: cita de fuentes, explicación, cierre con *"En definitiva,"* y duración de audio `<= 30.0` segundos.
6. Ampliación temporal a 1 minuto (*"Vox Verbi, háblame durante un minuto del amor"*) y verificación de que la siguiente pregunta sin duración regresa automáticamente a `<= 30.0` segundos.
7. Transición a `ADORACION` con *"Vox Verbi, al paraíso"*, verificación de bloqueo total ante preguntas durante la adoración, y reactivación únicamente con *"Vox Verbi, finalizamos la adoración"*.

---

## APÉNDICE A: TEXTOS CANÓNICOS LITERALES E INMUTABLES
*(Instrucción para el Agente: Extraer estos bloques literalmente hacia `vox_verbi/data/canonical_readings.json` sin alterar ni una sola letra, tilde o signo de puntuación).*

### A.1. DOCUMENTO: PRESENTACIÓN TEOLÓGICO-PASTORAL DE VOX VERBI (SECCIONES 0 A 11)

<<<START_DOC_PRESENTACION_HEADER>>>
VOX VERBI
Una propuesta de evangelización, formación y maduración de la fe al servicio de la Iglesia
<<<END_DOC_PRESENTACION_HEADER>>>

<<<START_SEC_PRESENTACION>>>
Presentación
Vox Verbi nace de una convicción fundamental: la nueva evangelización exige nuevos modos de comunicación, pero no un nuevo Evangelio.
La Iglesia está llamada en cada época a anunciar a Jesucristo de manera que el hombre concreto pueda escuchar, comprender y acoger la verdad del Evangelio dentro de las circunstancias históricas, culturales e intelectuales que le son propias. Esta tarea no consiste en modificar el contenido de la fe para hacerlo atractivo, sino en hacer accesible, inteligible y existencialmente significativo el depósito de la fe que la Iglesia ha recibido y custodia.
Vox Verbi quiere situarse precisamente en ese ámbito.
No pretende sustituir la acción pastoral de la Iglesia, ni reemplazar la relación personal con el sacerdote, la vida sacramental, la comunidad cristiana o los procesos ordinarios de catequesis y formación.
Pretende ser un instrumento al servicio de esa misión, utilizando las posibilidades contemporáneas de la tecnología y de la inteligencia artificial bajo criterios de rigor teológico, fidelidad doctrinal, discernimiento eclesial y responsabilidad pastoral.
<<<END_SEC_PRESENTACION>>>

<<<START_SEC_01>>>
1. El fundamento: no inventar, sino transmitir
El punto de partida de Vox Verbi es profundamente eclesial.
La Iglesia no recibe su misión de anunciar un mensaje elaborado por ella misma. Recibe y custodia el Evangelio de Jesucristo.
Por ello, Vox Verbi quiere establecer una relación rigurosa con las fuentes que constituyen el patrimonio de la fe: Sagrada Escritura, Tradición viva de la Iglesia, enseñanza del Magisterio, doctrina, liturgia, pensamiento de los Padres de la Iglesia y reflexión teológica en continuidad con la fe apostólica.
En este sentido, Dei Verbum resulta fundamental: la revelación cristiana no puede reducirse a una información religiosa, porque Dios mismo se comunica y se manifiesta en la historia de la salvación, alcanzando su plenitud en Jesucristo.
Por eso, el objetivo de Vox Verbi no es simplemente proporcionar información religiosa.
Es favorecer un acercamiento a la fe que conduzca desde la comprensión hacia la contemplación, desde la contemplación hacia la adhesión y desde la adhesión hacia una existencia cristiana concreta.
<<<END_SEC_01>>>

<<<START_SEC_02>>>
2. Una evangelización que no rebaja el contenido de la fe
Existe hoy un desafío pastoral evidente: muchas personas se encuentran alejadas de la Iglesia no necesariamente porque hayan rechazado conscientemente a Jesucristo, sino porque nunca han tenido acceso a una presentación suficientemente profunda, inteligible y existencialmente significativa de la fe cristiana.
Al mismo tiempo, numerosos creyentes viven una fe recibida sacramentalmente pero insuficientemente profundizada.
Entre ambos escenarios aparece una necesidad pastoral: la iniciación y la maduración de la fe.
Vox Verbi quiere contribuir precisamente a ese proceso.
No pretende responder a la secularización mediante una simplificación excesiva del cristianismo. Tampoco mediante una estrategia de entretenimiento religioso. Y mucho menos mediante una adaptación del Evangelio a las tendencias culturales del momento.
La propuesta es distinta: presentar la riqueza de la fe cristiana con profundidad, claridad, belleza y rigor.
Porque la inteligencia humana también puede abrirse al misterio. Porque la razón no es enemiga de la fe. Porque la tradición intelectual cristiana posee una riqueza capaz de entrar en diálogo con las preguntas fundamentales del hombre contemporáneo. Y porque la evangelización auténtica no teme las preguntas difíciles.
<<<END_SEC_02>>>

<<<START_SEC_03>>>
3. Del conocimiento de la fe al encuentro con Cristo
Vox Verbi parte de una distinción pastoral esencial: conocer contenidos sobre Cristo no equivale todavía a encontrarse con Cristo.
La evangelización no puede quedarse en la transmisión de conceptos. Pero tampoco puede prescindir de ellos.
La fe cristiana posee una dimensión doctrinal, pero también una dimensión existencial, litúrgica, espiritual, comunitaria y moral.
Por ello, Vox Verbi busca integrar estas dimensiones.
La persona puede comenzar con una pregunta intelectual: ¿Qué cree realmente la Iglesia?
Puede continuar con una pregunta antropológica: ¿Qué tiene que ver esta fe con mi existencia?
Puede llegar entonces a una pregunta cristológica: ¿Quién es Jesucristo para mí?
Y finalmente encontrarse ante la cuestión propiamente discipular: ¿Qué significa seguirlo?
De esta manera, la propuesta no pretende producir simplemente usuarios mejor informados, sino favorecer procesos de búsqueda, comprensión, discernimiento y maduración de la fe.
<<<END_SEC_03>>>

<<<START_SEC_04>>>
4. La persona que no cree también merece una propuesta seria
Uno de los destinatarios fundamentales de Vox Verbi es la persona que se encuentra distante de la fe.
Pero aquí resulta indispensable una precisión pastoral.
No queremos aproximarnos al no creyente desde una actitud de conquista, presión o proselitismo.
Queremos ofrecerle un espacio intelectualmente honesto y humanamente respetuoso en el que pueda formular sus preguntas, expresar sus objeciones y encontrarse con la propuesta cristiana en su integridad.
La evangelización comienza por reconocer la dignidad y libertad de la persona.
Por ello, Vox Verbi no pretende eliminar las preguntas del interlocutor. Pretende hacer posible que las preguntas puedan ser escuchadas y respondidas desde la tradición viva de la Iglesia.
La finalidad no es producir una adhesión artificial. Es disponer un camino en el que la persona pueda acercarse libremente a la verdad del Evangelio.
<<<END_SEC_04>>>

<<<START_SEC_05>>>
5. La belleza como vía de acceso al misterio
La evangelización tampoco puede reducirse a una exposición académica.
La tradición cristiana siempre ha comprendido que la verdad posee una dimensión de belleza.
La liturgia, la arquitectura, la música sacra, la iconografía, la espiritualidad, la vida de los santos y la tradición patrística manifiestan que el cristianismo no es solamente un sistema conceptual.
Es una forma de existencia configurada por el encuentro con Dios.
Por ello, Vox Verbi quiere integrar verdad, belleza y experiencia.
No para sustituir la verdad por la emoción, sino para permitir que la verdad pueda ser percibida también en su capacidad de transformar la existencia.
En términos pastorales: no se trata de hacer entretenida la fe; se trata de hacer visible su riqueza.
<<<END_SEC_05>>>

<<<START_SEC_06>>>
6. Una inteligencia artificial sometida a criterios eclesiales
Aquí se encuentra uno de los elementos más delicados y, al mismo tiempo, más importantes de Vox Verbi.
La inteligencia artificial no puede constituirse en autoridad doctrinal.
No puede convertirse en fuente autónoma de revelación.
No puede sustituir al Magisterio.
No puede reemplazar el discernimiento pastoral de la Iglesia.
Por ello, Vox Verbi debe concebirse como una herramienta subordinada a fuentes, criterios y controles previamente establecidos.
La respuesta que proporcione el sistema deberá distinguir entre: Sagrada Escritura; enseñanza del Magisterio; Tradición; doctrina católica; interpretación teológica; testimonios patrísticos; opinión de autores; cuestiones abiertas al debate teológico; y cuestiones sobre las cuales exista legítima diversidad de posiciones.
Esta distinción es fundamental.
Una inteligencia artificial aplicada a la pastoral no debe crear la apariencia de certeza donde la Iglesia no la posee. Debe aprender a distinguir entre lo definido, lo enseñado, lo teológicamente propuesto y lo simplemente opinable.
Ese principio constituye uno de los pilares de Vox Verbi.
<<<END_SEC_06>>>

<<<START_SEC_07>>>
7. La responsabilidad eclesial
Vox Verbi quiere nacer desde una actitud de servicio y comunión eclesial.
Por eso aspira a establecer mecanismos que permitan identificar claramente las fuentes utilizadas, favorecer la trazabilidad de la información y establecer criterios de revisión teológica y doctrinal.
El proyecto no pretende hablar “en nombre de la Iglesia” por cuenta propia.
Pretende ofrecer una herramienta que pueda ser discernida, acompañada y eventualmente integrada por responsables pastorales, sacerdotes, teólogos y autoridades eclesiales.
Esta diferencia es esencial.
Vox Verbi no busca constituirse en instancia de autoridad. Busca ponerse al servicio de quienes poseen legítimamente esa autoridad.
<<<END_SEC_07>>>

<<<START_SEC_08>>>
8. La finalidad pastoral
El horizonte último de Vox Verbi no es tecnológico. Es pastoral.
Y su finalidad tampoco puede medirse únicamente por número de usuarios, consultas realizadas o contenidos consumidos.
La pregunta fundamental debe ser otra:
¿Esta herramienta ayuda realmente a una persona a acercarse a Cristo, comprender la fe, discernir su vida y participar más plenamente en la comunidad de la Iglesia?
Si la respuesta es afirmativa, entonces la tecnología habrá cumplido su función.
Si no lo hace, la tecnología habrá fracasado, aunque tenga éxito desde el punto de vista técnico.
Por eso Vox Verbi quiere colocar la tecnología detrás de la misión, y no la misión detrás de la tecnología.
<<<END_SEC_08>>>

<<<START_SEC_09>>>
9. Una propuesta para la Iglesia de hoy
Vox Verbi quiere situarse en la dinámica de una Iglesia que anuncia, acompaña, discierne y sale al encuentro.
Una Iglesia que no renuncia a la profundidad de su tradición para dialogar con el mundo contemporáneo.
Una Iglesia que no teme la razón.
Una Iglesia que no oculta su doctrina.
Una Iglesia que sabe escuchar las preguntas del hombre contemporáneo.
Una Iglesia que anuncia a Jesucristo no como una idea entre otras, sino como el centro de la existencia cristiana.
En este sentido, Vox Verbi quiere contribuir a una pastoral que permita realizar un movimiento doble:
Hacia dentro: ayudar al creyente a profundizar, madurar y dar razón de su fe.
Hacia fuera: ofrecer al hombre y a la mujer alejados de la Iglesia un primer espacio serio, respetuoso y fundamentado de encuentro con la propuesta cristiana.
Ambos movimientos pertenecen a una misma misión evangelizadora.
<<<END_SEC_09>>>

<<<START_SEC_10>>>
10. La visión
Vox Verbi aspira a convertirse en una infraestructura pastoral de formación y evangelización capaz de poner al alcance de las personas, mediante herramientas contemporáneas, la riqueza de la tradición cristiana.
Una herramienta que no diga simplemente: “Esto es lo que debes pensar.”
Sino que pueda acompañar: “Esto es lo que la Iglesia enseña; estas son sus fuentes; esta es su tradición; estas son las preguntas que surgen; estos son los argumentos; y este es el horizonte cristiano desde el cual puedes discernir.”
De este modo, la tecnología deja de ser simplemente un mecanismo de respuesta y puede convertirse en un instrumento de acompañamiento intelectual y pastoral.
<<<END_SEC_10>>>

<<<START_SEC_11>>>
11. Vox Verbi en una definición
Vox Verbi es una propuesta de evangelización y maduración de la fe que integra inteligencia artificial, Sagrada Escritura, Tradición, Magisterio, patrimonio patrístico y reflexión teológica, con criterios de rigor doctrinal, trazabilidad de fuentes y discernimiento eclesial, para favorecer el encuentro con Jesucristo, la profundización de la fe de los creyentes y un acercamiento respetuoso y fundamentado de quienes se encuentran alejados de la Iglesia.
No pretende producir una nueva forma de cristianismo. Pretende servir a la transmisión de la fe recibida.
No pretende reemplazar la pastoral. Pretende fortalecerla.
No pretende sustituir al sacerdote. Pretende poner herramientas contemporáneas al servicio de su misión.
No pretende hablar en lugar de la Iglesia. Pretende escuchar a la Iglesia y aprender de su patrimonio para servir a su misión evangelizadora.
<<<END_SEC_11>>>

<<<START_DOC_PRESENTACION_FOOTER>>>
VOX VERBI La Palabra que la Iglesia ha recibido, custodiado y anunciado; comunicada hoy con profundidad, rigor y responsabilidad.
Porque el desafío no consiste simplemente en conseguir que el hombre contemporáneo vuelva a escuchar hablar de Dios.
El desafío pastoral es que, al escuchar nuevamente el Evangelio, pueda descubrir que detrás de la palabra anunciada existe una Persona: Jesucristo.
Y que ese encuentro pueda abrir un camino de fe, conversión, comunión y misión.
<<<END_DOC_PRESENTACION_FOOTER>>>

---

### A.2. DOCUMENTO: AL PARAÍSO (TRANSICIÓN A ADORACIÓN EUCARÍSTICA)

<<<START_AL_PARAISO>>>
Hermanos, hemos llegado hasta aquí después de un camino.
Hemos escuchado la Palabra, hemos permitido que nos interrogue, hemos descubierto algo más del amor de Dios y hemos dejado que su mensaje toque, poco a poco, las puertas de nuestro corazón.
Pero ahora ha llegado el momento de dar un paso diferente.
Hasta ahora hemos escuchado para comprender. Ahora queremos permanecer para contemplar.
Hasta ahora hemos buscado palabras que nos hablen de Dios. Ahora queremos abrir el corazón para descubrir cómo Dios nos habla en el silencio.
Por eso, no tengas prisa. No necesitas demostrar nada. No necesitas encontrar las palabras perfectas ni sentir algo extraordinario.
Simplemente, permanece.
Mira el lugar en el que te encuentras. Contempla la luz, los detalles, la belleza de este momento. Permite que tus ojos descubran aquello que tantas veces pasa inadvertido cuando vivimos apresurados.
Y después, ve un poco más adentro.
Abre los ojos del corazón.
Tal vez llegas con una preocupación, con una herida que todavía no ha sanado, con una pregunta que no has podido responder o con una gratitud que no has sabido expresar.
No necesitas esconder nada. Ven tal como eres.
Porque ante Dios no tienes que representar ningún papel. Puedes presentarte con tu historia, con tus luchas, con tus esperanzas y también con esos silencios que nadie más conoce.
Ahora, la Palabra que hemos escuchado nos conduce hacia el encuentro.
La enseñanza ha preparado el camino. La reflexión nos ha ayudado a comprender. Pero ahora queremos abrir un espacio para que el amor de Dios no sea solamente algo que conocemos, sino Alguien a quien acogemos.
En unos instantes, la custodia se entronizará solemnemente, el incienso elevará su perfume como signo de nuestra oración y nuestra mirada se dirigirá hacia Cristo, presente en la Eucaristía.
No te quedes solamente mirando con los ojos del cuerpo. Mira también con los ojos del corazón.
Contempla. Adora. Permanece.
Y si no sabes qué decir, no digas nada.
Si necesitas llorar, deja que tus lágrimas sean oración.
Si llevas mucho tiempo buscando respuestas, permite que este momento sea, sencillamente, un tiempo para estar con Él.
No tienes que llenar el silencio. No tienes que apresurar el encuentro.
Deja que Dios encuentre espacio en ti.
Ahora, hermanos, comienza nuestro camino hacia el silencio.
Dejemos que se aquiete el ruido exterior. Dejemos descansar nuestras palabras. Permitamos que nuestra mirada, nuestros pensamientos y nuestro corazón se vuelvan hacia Cristo.
Que cada uno pueda encontrarse con Él desde su propia historia, sin compararse con nadie, sin temor y con plena libertad interior.
El Señor nos espera.
Y nosotros queremos estar aquí, ante Él, con todo lo que somos.
Cristo está aquí.
Permanezcamos en su presencia.
<<<END_AL_PARAISO>>>

---

### A.3. RELATO 01 (VV-HIST-001): LAS TRES CRUCES

<<<START_VV_HIST_001>>>
El sueño de los tres árboles – Un cuento con alma
En lo alto de una colina, donde el cielo besa la tierra, crecían tres árboles. Eran jóvenes, fuertes, y como toda creación, soñaban con grandeza. Cada uno albergaba en su corazón un anlo profundo, aún sin saber que el Creador ya los había visto.
**El primer árbol**, mirando las estrellas, deseaba ser transformado en un cofre precioso. Quería contener lo más valioso del mundo: oro, incienso, piedras finas. Su sueño era brillar con la riqueza de los reyes.
**El segundo árbol**, sintiendo el viento entre sus ramas, deseaba ser tallado en una gran embarcación. Soñaba con navegar los océanos llevando a personas poderosas, quizás a un rey. Quería conocer tierras lejanas y vivir aventuras.
**El tercer árbol**, más silencioso, deseaba no ser cortado nunca. Quería permanecer en la colina, elevando sus ramas al cielo. Su deseo era sencillo y puro: señalar a Dios con su sola presencia, y que al verlo, los hombres pensaran en lo eterno.
Pasaron los años. Los árboles crecieron. Y un día, llegaron los leñadores. Los sueños comenzaron a romperse.
Al primer árbol lo cortaron y lo llevaron a un taller. No se convirtió en un cofre de tesoros, sino en un **pesebre humilde**, rústico, tosco. El árbol lloró en silencio. ¿Cómo podía contener riqueza si estaba destinado a los animales?
Al segundo árbol lo convirtieron en una **pequeña barca de pesca**, para cruzar lagos tranquilos. Ningún rey, ningún palacio, sólo pescadores con manos ásperas. Su madera crujía de tristeza.
Al tercero lo partieron en vigas y lo arrojaron a un depósito. Nadie miraba al cielo por su causa. Estaba olvidado entre otras maderas.
Pero los sueños de Dios son más altos que los nuestros…
Una noche serena, **una madre colocó en el humilde pesebre a un niño envuelto en pañales**. Ese niño era el **Hijo de Dios**. El primer árbol comprendió entonces que contenía al **Tesoro más precioso del universo**.
Años más tarde, hombres subieron a la pequeña barca y entre ellos, uno dormía con paz en medio de una tormenta. Cuando se levantó, **ordenó al viento y al mar que se calmaran**. Era el **Rey de reyes**, y la barca lo llevaba.
Y llegó un día oscuro, cuando un hombre fue clavado en una cruz hecha de las vigas del tercer árbol. Era una escena de dolor, pero en ese sacrificio, **el mundo encontró esperanza y redención**. El tercer árbol entendió que su deseo de señalar a Dios se cumplía: su madera abrazaba al Salvador, elevándolo ante la humanidad entera.
<<<END_VV_HIST_001>>>
*(Nota técnica para el parser: En el primer párrafo de `VV-HIST-001` el documento original dice "anhelo profundo"; conservar exactamente `anhelo profundo` en el JSON canónico y limpiar `**` únicamente en la capa de síntesis de voz).*

---

### A.4. RELATO 02 (VOX_VERBI_RELATO_02_CRUZ_QUE_NADIE_QUERIA): LA CRUZ QUE NADIE QUERÍA

<<<START_RELATO_02>>>
CUENTO DE APERTURA: “LA CRUZ QUE NADIE QUERÍA”
Había un hombre…
que estaba cansado de su vida.
No era una mala persona…
pero estaba agotado.
Cansado de sus problemas…
de sus errores…
de sus luchas internas…
de cargar cosas que nadie veía.
Una noche… en medio de su desesperación…
hizo una oración:
“Señor… ya no quiero esta cruz…
es demasiado pesada para mí.”
Y esa noche soñó…
---
En su sueño…
llegaba a un lugar enorme…
como un salón infinito…
Y allí… había miles de cruces.
Grandes… pequeñas…
de madera fina… otras ásperas…
unas brillaban… otras parecían destruidas.
Y una voz le dijo:
“Puedes escoger la cruz que quieras.”
---
El hombre… sorprendido…
dijo:
“Por fin… voy a elegir una más fácil.”
Empezó a caminar…
Tomó una cruz grande…
y al levantarla… era imposible de cargar.
Tomó otra más pequeña…
pero tenía astillas que le herían las manos.
Vio una muy elegante…
pero pesaba más de lo que parecía.
Probó muchas…
Y ninguna le servía.
---
Después de mucho intentar…
encontró una cruz sencilla…
no muy grande…
no perfecta…
pero… era soportable.
La cargó…
y por primera vez… no sintió rechazo.
Entonces dijo:
“Esta… esta sí puedo llevarla.”
---
Y en ese momento…
la voz volvió a hablar:
“Hijo… esa es la misma cruz con la que llegaste.”
---
---
El hombre despertó…
y entendió algo que nunca había entendido:
No era que su cruz era demasiado pesada…
era que nunca había aprendido a abrazarla.
<<<END_RELATO_02>>>

---

### A.5. RELATO 03 (VV-HIST-003): EL JARDÍN DESCUIDADO

<<<START_VV_HIST_003>>>
I. EL TESORO DEL HOMBRE
Había una vez un hombre que tenía un pequeño jardín en el patio de su casa.
No era un jardín extraordinario. No aparecía en las revistas, ni recibía visitantes de lugares lejanos. Era, sencillamente, un pequeño rincón de tierra que el hombre había aprendido a amar.
Allí había sembrado flores de muchos colores. Unos rosales adornaban el camino, un árbol frutal extendía sus ramas hacia el cielo y, en un rincón, una pequeña fuente dejaba caer el agua con un sonido suave y constante.
Cada mañana, el hombre abría la puerta de su casa y salía a contemplarlo.
Respiraba profundamente.
Observaba las flores.
Escuchaba el agua.
Y, al mirar los frutos que comenzaban a crecer en el árbol, sentía una gratitud difícil de explicar.
A veces sonreía sin decir nada. Otras veces daba gracias a Dios por aquel pequeño lugar que le recordaba que la vida podía ser hermosa.
Aquel jardín era más que un pedazo de tierra.
Era el resultado de muchas horas de dedicación, de semillas sembradas con esperanza, de manos que habían aprendido a podar las ramas y de paciencia para esperar que cada flor llegara a su tiempo.
El hombre sabía que la belleza del jardín no había aparecido por casualidad.
Alguien había tenido que sembrar.
Alguien había tenido que regar.
Alguien había tenido que cuidar.
Y ese alguien había sido él.
[PAUSA DE 3 SEGUNDOS]
II. CUANDO OTRAS COSAS COMENZARON A OCUPAR SU LUGAR
Pero el tiempo fue pasando.
Y, poco a poco, la vida comenzó a llenarse de otras ocupaciones.
El trabajo exigía más horas. Llegaron nuevos compromisos, viajes, responsabilidades y distracciones. Siempre había algo urgente que resolver, una llamada que atender o una tarea que no podía esperar.
Al principio, el hombre se decía:
«Mañana cuidaré el jardín».
Y aquel mañana parecía una promesa inofensiva.
Un día no regó las flores.
Al siguiente, no retiró las hojas secas.
Después dejó para otra ocasión la poda de los rosales.
Y cuando veía crecer alguna hierba entre las plantas, pensaba que todavía no era necesario preocuparse.
No había dejado de amar su jardín.
Eso era lo que él se repetía.
Simplemente, tenía demasiadas cosas que hacer.
Y así, sin darse cuenta, comenzó a confundir el amor que sentía por su jardín con el cuidado que había dejado de darle.
Las primeras hierbas crecieron junto el camino.
Eran pequeñas, casi insignificantes. El hombre las miraba de vez en cuando y pensaba que podría arrancarlas después.
Pero aquellas hierbas no esperaban.
Crecían mientras él trabajaba.
Se extendían mientras él viajaba.
Echaban raíces mientras él aplazaba, una y otra vez, el momento de ocuparse de ellas.
Y lo que al principio parecía pequeño comenzó a ocupar cada vez más espacio.
Los rosales quedaron cubiertos por las malas hierbas. Las flores dejaron de recibir suficiente luz. Las hojas secas se acumularon en la fuente, hasta que el agua perdió su movimiento y comenzó a estancarse.
El árbol frutal, que antes ofrecía frutos dulces, empezó a perder sus hojas.
Primero unas pocas.
Después muchas.
Hasta que sus ramas quedaron cada vez más desnudas.
Y, sin embargo, el hombre seguía diciendo:
«Cuando tenga tiempo, volveré a cuidar de todo».
[PAUSA DE 4 SEGUNDOS]
III. EL DÍA DEL REGRESO
Pasaron los meses.
Hasta que, una mañana, el hombre sintió el deseo de regresar a aquel lugar que tantas alegrías le había dado.
Había tenido días difíciles. Estaba cansado de correr de un lado para otro y necesitaba recuperar la tranquilidad que tantas veces había encontrado en su jardín.
Pensó en los rosales.
Recordó el sonido de la fuente.
Y casi pudo saborear aquellos frutos dulces que recogía directamente del árbol.
Entonces sonrió y se dirigió al patio.
Pero, cuando abrió la puerta, se quedó inmóvil.
El jardín seguía allí.
Y, sin embargo, parecía otro lugar.
El camino estaba cubierto de hierbas y espinas. Las ramas secas se entrelazaban unas con otras. Los rosales, antes llenos de color, apenas podían reconocerse bajo aquella maraña de plantas invasoras.
La fuente estaba sucia, cubierta de hojas y sin aquel sonido de agua que solía acompañar sus mañanas.
El árbol frutal permanecía de pie, pero sus ramas estaban débiles y casi desnudas.
No había desaparecido todo.
Pero la belleza que él recordaba parecía haberse escondido bajo años de abandono.
El hombre dio unos pasos.
Miró alrededor.
Y sintió un dolor profundo.
No era solamente tristeza por las flores perdidas o por el árbol que había dejado de dar frutos.
Era algo más íntimo.
Aquel jardín le estaba mostrando, sin pronunciar una sola palabra, las consecuencias de sus propias decisiones.
Se acercó a uno de los rosales y apartó cuidadosamente algunas ramas secas.
Encontró una espina.
Se pinchó un dedo.
Y, mientras observaba la pequeña gota de sangre, sus ojos se llenaron de lágrimas.
Entonces comprendió algo que hasta aquel momento no había querido reconocer.
El jardín no se había deteriorado en un solo día.
No había amanecido destruido de repente.
Cada hierba había crecido poco a poco.
Cada hoja seca se había acumulado con el paso del tiempo.
Cada planta había sufrido la ausencia de un cuidado que parecía poder esperar.
Y cada vez que él había dicho «mañana», había permitido que el abandono avanzara un poco más.
[PAUSA DE 4 SEGUNDOS]
IV. LO QUE EL JARDÍN LE ENSEÑÓ
El hombre se sentó junto a la fuente.
Por primera vez en mucho tiempo, no tenía prisa.
No buscó una excusa.
No culpó al clima, ni al paso de los años, ni a las dificultades de la vida.
Simplemente, permaneció allí.
Mirando lo que había sucedido.
Y, en el silencio, comenzó a hacerse preguntas.
¿Cuántas veces había dejado para después aquello que realmente amaba?
¿Cuántas cosas importantes había descuidado mientras se ocupaba de otras que parecían más urgentes?
¿Cuántas veces había pensado que todavía tenía tiempo?
Quizás aquel jardín se parecía más a su vida de lo que había imaginado.
Porque también en el corazón humano hay flores que necesitan cuidado.
Hay amistades que necesitan una llamada.
Hay familias que necesitan tiempo.
Hay heridas que necesitan ser escuchadas.
Hay sueños que necesitan volver a recibir atención.
Y hay una vida espiritual que no puede alimentarse únicamente de los recuerdos de lo que un día sentimos.
La fe, como una semilla, necesita ser cultivada.
El amor necesita gestos concretos.
La reconciliación necesita humildad.
La esperanza necesita espacio para crecer.
Y el corazón necesita momentos de silencio para volver a descubrir aquello que verdaderamente importa.
El hombre comprendió que no había perdido su jardín porque hubiera dejado de amarlo en un instante.
Lo había perdido poco a poco, porque dejó de expresarle aquel amor mediante el cuidado cotidiano.
Y aquella verdad le dolió.
Pero también comenzó a liberarlo.
Porque, si el abandono había ocurrido día tras día, quizá la recuperación también podía comenzar de la misma manera.
[PAUSA DE 3 SEGUNDOS]
V. VOLVER A CUIDAR
A la mañana siguiente, el hombre volvió al jardín.
Esta vez no llegó para recordar cómo había sido antes.
Llegó para comenzar a cuidarlo tal como estaba.
Tomó unas herramientas.
Se puso unos guantes.
Y empezó por el camino.
Arrancó las hierbas que podía retirar sin dañar las plantas. Apartó las ramas secas. Recogió las hojas que cubrían la fuente y limpió cuidadosamente su interior.
El trabajo era lento.
Mucho más lento de lo que había imaginado.
A veces encontraba raíces profundas que no podía arrancar a la primera. Otras veces debía detenerse para proteger una planta que todavía conservaba vida.
Y, por momentos, sentía frustración.
El jardín no recuperaba su belleza de inmediato.
Pero el hombre ya no buscaba resultados instantáneos.
Había aprendido que no podía exigirle a la tierra que reparara en un día aquello que había sufrido durante tantos meses.
Así que regresó al día siguiente.
Y al otro.
Regó las flores que todavía podían recuperarse.
Podó con cuidado los rosales.
Preparó la tierra alrededor del árbol.
Y siguió retirando las hierbas que amenazaban con ocupar nuevamente el jardín.
Poco a poco, comenzó a distinguirse el camino.
La fuente volvió a dejar escuchar el sonido del agua.
En una de las ramas, apareció un pequeño brote verde.
Y, cerca de la tierra húmeda, una flor que el hombre creía perdida comenzó a abrir sus pétalos.
No todo volvió a ser como antes.
Algunas plantas no lograron recuperarse. Hubo ramas que tuvieron que ser retiradas y flores que no volvieron a crecer.
El hombre comprendió que cuidar también significaba aceptar las pérdidas, aprender de los errores y tener paciencia con aquello que todavía podía renacer.
Ya no cuidaba su jardín por costumbre.
Lo cuidaba con una conciencia nueva.
Ahora sabía que cada pequeño gesto tenía un valor.
Una gota de agua.
Una raíz protegida.
Una rama podada a tiempo.
Un momento dedicado a permanecer allí, sin permitir que las prisas le robaran nuevamente lo esencial.
[PAUSA DE 3 SEGUNDOS]
VI. EL JARDÍN DEL CORAZÓN
Con el paso de los días, el jardín comenzó a transformarse.
Las flores recuperaron lentamente sus colores.
La fuente volvió a reflejar la luz de la mañana.
El árbol comenzó a mostrar nuevos brotes.
Y el hombre volvió a sentarse junto a sus plantas, pero esta vez comprendía algo que antes había pasado por alto.
La belleza no consistía solamente en tener un jardín hermoso.
Consistía también en estar dispuesto a cuidarlo.
[SILENCIO DE 6 SEGUNDOS]
<<<END_VV_HIST_003>>>

---

## APÉNDICE B: TABLA MAESTRA DEL CATÁLOGO MUSICAL (`music_catalog.json`)

```json
{
  "code_master": "VOX_VERBI_MUSICA_INSTRUMENTAL",
  "general_commands": [
    "vox verbi inspiranos",
    "vox verbi pon musica instrumental",
    "vox verbi ambienta la palabra",
    "vox verbi acompaname con musica de fondo",
    "vox verbi musica para predicar"
  ],
  "tracks": [
    {
      "id": "MUS-001",
      "name": "Huellas de San Francisco",
      "category": "Franciscana / sencillez y contemplación",
      "commands": ["vox verbi llevanos a la sencillez"],
      "url": "[https://www.youtube.com/watch?v=LKTFPq8yTxg](https://www.youtube.com/watch?v=LKTFPq8yTxg)",
      "local_file": "media/musica/MUS-001.ogg"
    },
    {
      "id": "MUS-002",
      "name": "Serenidad interior",
      "category": "Música tranquila / relajación",
      "commands": ["vox verbi serenanos el alma"],
      "url": "[https://www.youtube.com/watch?v=fODMS7WQWho](https://www.youtube.com/watch?v=fODMS7WQWho)",
      "start_offset_sec": 2054,
      "local_file": "media/musica/MUS-002.ogg"
    },
    {
      "id": "MUS-003",
      "name": "La Palabra que ilumina",
      "category": "Fondo instrumental para predicación",
      "commands": [
        "vox verbi inspiranos",
        "vox verbi pon musica instrumental",
        "vox verbi ambienta la palabra",
        "vox verbi acompaname con musica de fondo",
        "vox verbi musica para predicar"
      ],
      "url": "[https://www.youtube.com/watch?v=otBWC_4owDs](https://www.youtube.com/watch?v=otBWC_4owDs)",
      "local_file": "media/musica/MUS-003.ogg"
    },
    {
      "id": "MUS-004",
      "name": "Silencio del corazón",
      "category": "Oración y recogimiento",
      "commands": ["vox verbi abre un espacio de silencio"],
      "url": "[https://www.youtube.com/watch?v=9qGr40bhyMk](https://www.youtube.com/watch?v=9qGr40bhyMk)",
      "local_file": "media/musica/MUS-004.ogg"
    },
    {
      "id": "MUS-005",
      "name": "En tus manos, Señor",
      "category": "Oración emotiva / entrega y transición Al Paraíso",
      "commands": ["vox verbi abre el corazon"],
      "url": "[https://www.youtube.com/watch?v=wZ7KaCZ6i44](https://www.youtube.com/watch?v=wZ7KaCZ6i44)",
      "local_file": "media/musica/MUS-005.ogg"
    },
    {
      "id": "MUS-006",
      "name": "Al encuentro de Jesús",
      "category": "Contemplación y encuentro personal",
      "commands": ["vox verbi sumergenos en tu presencia"],
      "url": "[https://www.youtube.com/watch?v=2gaeVMz_1x0](https://www.youtube.com/watch?v=2gaeVMz_1x0)",
      "local_file": "media/musica/MUS-006.ogg"
    }
  ],
  "control_commands": {
    "MUS-CTRL-04": {
      "commands": ["vox verbi silencio"],
      "action": "PAUSE_OR_STOP_IMMEDIATELY"
    },
    "MUS-CTRL-05": {
      "commands": ["vox verbi continua la musica"],
      "action": "RESUME"
    },
    "MUS-CTRL-06": {
      "commands": ["vox verbi termina la musica"],
      "action": "STOP_FADEOUT"
    }
  }
}