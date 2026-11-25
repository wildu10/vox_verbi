# Vox Verbi v3.6 — Asistente Catequético y Contemplativo Católico

**Vox Verbi («Voz de la Palabra»)** es una herramienta de apoyo catequético y pastoral 100% offline para predicadores, fundamentada en las **4 Columnas Autorizadas de la Tradición Católica** más los documentos de la **Comisión Teológica Internacional (Santa Sede)**:

1. **Nivel 1 — Sagrada Escritura:** Biblia de Jerusalén completa en UTF-8.
2. **Nivel 2 — Magisterio Conciliar, Pontificio y Comisión Teológica Internacional:** Concilio Vaticano II (*Dei Verbum*, *Lumen Gentium*, *Sacrosanctum Concilium*, *Gaudium et Spes*), Encíclicas y Exhortaciones (*Deus Caritas Est*, *Spe Salvi*, *Lumen Fidei*, *Dives in Misericordia*, *Ecclesia de Eucharistia*, *Fides et Ratio*, *Evangelii Gaudium*, *Mysterium Fidei*), Concilio de Trento y los **33 documentos en español de la Comisión Teológica Internacional (1969–2026)**.
3. **Nivel 3 — Catecismo de la Iglesia Católica:** Catecismo Mayor (CIC) + Compendio del Catecismo (598 numerales).
4. **Nivel 4 — Tradición Patrística:** Padres Apostólicos y Padres de la Iglesia (*Didaché*, San Clemente, San Ignacio de Antioquía, San Ireneo, San Justino, San Atanasio, San Basilio, San Gregorio Nacianceno, San Gregorio de Nisa, San Juan Crisóstomo, San Ambrosio, San Jerónimo, San Agustín, San Cirilo, San León Magno, San Gregorio Magno).

---

## Arquitectura de Hardware y Software (Mini PC Ryzen 9 6900HX + Radeon 680M)

- **LLM en GPU Vulkan:** `gemma-4-E4B-it-Q4_K_M.gguf` sobre `llama-server` (`ggml-vulkan`, ~185 t/s prefill, ~19 t/s generación, KV Cache persistente en VRAM).
- **STT en GPU Vulkan:** `whisper.cpp` (`whisper-server` con `ggml-vulkan` en Radeon 680M, `ggml-small-q8_0.bin`, latencia **~210 ms**) + compuerta neuronal Silero VAD (`< 3 ms`).
- **TTS Contemplativo:** `Supertonic 3` voz barítona **`M5`** a 44.1 kHz con caché instantánea (`~1.7 ms` TTFA) para los 12 apartados de la *Guía Fundamental*.
- **RAG Híbrido:** Índice vectorial denso `FAISS` (`intfloat/multilingual-e5-small`) + índice léxico invertido `BM25` + anclas canónicas (*locus classicus*).
- **Control por Voz y Botones Físicos:**
  - Activación estricta con `«Vox Verbi, ...»` al inicio de la frase.
  - Interrupción inmediata con `«Vox Verbi, para»` o `«Vox Verbi, detente»` (excepto durante *Al Paraíso*).
  - Botones físicos `+` / `-` / `Mute` del altavoz USB Anker PowerConf S330 (`/dev/input/event3`).

---

## Utilidades de Terminal y Mantenimiento

- **`agregar-red` (o `wifi`):** Menú interactivo en español para escanear redes Wi-Fi, conectarse y fijar automáticamente la IP estática (`192.168.1.11` / `.11` + `custodia.local`).
- **`agregar-contenido` (o `importar-usb`):** Importador automático y manual de nuevos documentos teológicos desde una memoria USB (`VOX_VERBI/fuentes_nuevas/`), también activable por voz diciendo *«Vox Verbi, actualiza las fuentes desde la USB»* o automáticamente al insertar la memoria USB (`udev`).
- **`scripts/vox_clone_and_restore.sh`:** Clonador e instalador de imagen maestra en memoria USB de 64 GB (`--backup`, `--install-on-ubuntu`, `--full-disk`).
