# Vox Verbi v3.0 — Asistente Catequético Católico Vivo

Vox Verbi («Voz de la Palabra») es un asistente de voz catequético católico de alta fidelidad doctrinal, trazabilidad de fuentes y baja latencia, ejecutado 100% en local sobre hardware embebido (Mini PC AMD Ryzen 9 6900HX con GPU integrada Radeon 680M vía Vulkan y altavoz/micrófono USB Anker PowerConf S330).

---

## 📁 Estructura del Proyecto

```text
vox_verbi/
├── __init__.py               # Paquete Python vox_verbi
├── config.py                 # Rutas, modelo Gemma 4 E4B, voz Supertonic M5 y presupuestos temporales
├── hardware_hooks.py         # Stubs de iluminación, custodia e incienso para modo Adoración
├── tts_engine.py             # Motor unificado de voz contemplativa (Supertonic M5 44.1 kHz + fallback Kokoro)
├── audio_mixer.py            # Mezclador ALSA/Pygame, Ducking (45% -> 18%), streaming narrativo y caché TTS
├── state_machine.py          # Estados (CATEQUESIS vs ADORACION) y control dinámico de duración (30s / 1-4m)
├── command_router.py         # Enrutador O(1) e intencional (Guía Fundamental, Música, Relatos, Adoración)
├── pre_filter.py             # Filtro previo de 5 pasos (Autoría Sagrada, Pertinencia, Confesión vs Doctrina)
├── rag_engine.py             # Motor RAG Híbrido (Multilingual-E5 + BM25 Léxico) sobre 4 Columnas Canónicas
├── llm_catequist.py          # Cliente SSE Gemma 4 E4B con streaming en 3 hilos y watchdog de duración oral
├── main.py                   # Bucle continuo VAD + Whisper STT + Pre-Ducking + manejo sin falsos positivos
├── muestras_voz/             # Muestras MP3 de la nueva voz contemplativa barítono (Supertonic M5)
├── scripts/
│   ├── build_whitelist_rag.py  # Constructor del Corpus Cerrado v2.0 (Biblia UTF-8, Vatican.va, CIC, Patrística)
│   ├── build_canonical_json.py # Extractor inmutable de textos canónicos y catálogo musical
│   ├── test_vox_verbi_suite.py # Batería automatizada de 9 pruebas integrales de validación
│   ├── run_live_benchmark.py   # Benchmark de latencia TTFA, RAG híbrido y voz en vivo
│   ├── upload_modules.py       # Despliegue SFTP hacia la Mini PC (192.168.1.9)
│   ├── deploy_services.py      # Instalador de unidades systemd endurecidas en arranque en frío
│   └── remote_shell.py         # Cliente SSH administrativo (192.168.1.9)
├── systemd/
│   ├── asistente-vivo.service  # Servicio principal con detección USB S330 y verificación /health
│   └── llama-server.service    # Servidor Vulkan Gemma 4 E4B (-ngl 99 -fa on -np 1 -ctk q8_0 -rea off)
└── docs/
    └── SPEC_AGENT_ANTIGRAVITY.md
```

---

## ⚙️ Arquitectura v3.0

1. **Inferencia LLM Acelerada por GPU Vulkan (`gemma-4-E4B-it-Q4_K_M.gguf`):**
   - ~20 tokens/s de generación (+45% frente a Llama 3.1 8B) y ~165 tokens/s de evaluación de prompt (+43%), con modo pensamiento desactivado (`-rea off --reasoning-budget 0`) y caché KV cuantizada (`-ctk q8_0 -ctv q8_0`) en slot único (`-np 1`).
2. **Voz Contemplativa Barítono (`Supertonic M5` ONNX a 44.1 kHz):**
   - Tono barítono sereno, cálido y reflexivo (~92 Hz F0, IQR 82–98 Hz, ~2.2 palabras/s a `speed=0.92`), sin acento extranjero ni artefactos robóticos, y 41% más rápida en CPU (`RTF ~0.168`) que Kokoro.
3. **Guía Fundamental Instantánea (`TTFA < 50 ms`):**
   - Acceso directo O(1) e intencional a las 12 secciones del documento fundacional (tanto comandos exactos como preguntas naturales del predicador: *«¿Qué es Vox Verbi?»*, *«¿Cuál es su fundamento?»*, etc.) con streaming oración por oración y pre-caché del primer segmento en disco/RAM.
4. **Corpus Cerrado v2.0 (16,951 registros UTF-8) y RAG Híbrido (Multilingual-E5 + BM25):**
   - **Sagrada Escritura:** 10,473 perícopas de la Biblia de Jerusalén en UTF-8 limpio.
   - **Magisterio:** 3,005 numerales oficiales de `vatican.va` (*Dei Verbum*, *Lumen Gentium*, *Sacrosanctum Concilium*, *Gaudium et Spes*, *Deus Caritas Est*, *Spe Salvi*, *Lumen Fidei*, *Dives in Misericordia*, *Ecclesia de Eucharistia*, *Fides et Ratio*, *Evangelii Gaudium*, *Mysterium Fidei*, *Trento*).
   - **Catecismo:** 3,448 registros (2,863 numerales del *Catecismo Mayor* + 585 del *Compendio*).
   - **Tradición Patrística:** 25 testimonios documentados (*Didaché*, *San Ignacio*, *San Ireneo*, *San Atanasio*, *San Basilio*, *San Gregorio Nacianceno*, *San Gregorio de Nisa*, *San Juan Crisóstomo*, *San Ambrosio*, *San Jerónimo*, *San Agustín*, *San León Magno*, etc.).
5. **Control Estricto de Duración Oral (30 s, 1 min, 2 min, 3–4 min):**
   - Reconocimiento flexible de instrucciones temporales, limpieza temática previa al RAG, escalado de fuentes y *watchdog* en vivo en el hilo sintetizador con restablecimiento automático a 30 s.
