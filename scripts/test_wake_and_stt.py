import os
import sys
import numpy as np
import scipy.signal

sys.path.insert(0, "/home/wildu/llama.cpp")

from vox_verbi import config
from vox_verbi.audio_mixer import parse_prosodic_segments, expand_biblical_citations_for_tts
from vox_verbi.command_router import CommandRouter, match_and_strip_wake_word
from vox_verbi.main import BiasedWhisperRecognizer, VoxVerbiApp
from vox_verbi.tts_engine import VoiceEngine


def test_historical_log_failures():
    print("=== 1. Testing All Historical Live Log Wake-Word & Routing Cases ===")
    router = CommandRouter()

    positive_cases = [
        # (raw_stt, expected_rest_substring, expected_action, expected_target)
        ("Mosbér Bikin era Jesús.", "quién era Jesús", None, None),
        ('¿Qué es "Boss Verbi"?', "Qué es Vox Verbi", "READ_PRESENTACION", "SEC_11"),
        ('¿Qué es "bos verbi"?', "Qué es Vox Verbi", "READ_PRESENTACION", "SEC_11"),
        ("Vox Verbi, ¿qué es Vox Verbi?", "qué es Vox Verbi", "READ_PRESENTACION", "SEC_11"),
        ("Wasberby Definity", "defínete", "READ_PRESENTACION", "SEC_11"),
        ("los verbicos la Biblia.", "qué es la Biblia", None, None),
        ("Boso better be.", "", None, None),
        ("Vox Verbi.", "", None, None),
        ("Vos ver mi, explícame tu fundamento.", "explícame tu fundamento", "READ_PRESENTACION", "SEC_01"),
        ("Vox Verbi, ¿cuál es su funcionamiento?", "cuál es su funcionamiento", "READ_PRESENTACION", "SEC_PRESENTACION"),
        ("Vox Verbi, ¿cómo conduces al encuentro con Cristo?", "cómo conduces al encuentro con Cristo", "READ_PRESENTACION", "SEC_03"),
        ("Defínete, Vox Verbi.", "Defínete", "READ_PRESENTACION", "SEC_11"),
        ("¿Quién era Jesús, Vox Verbi?", "Quién era Jesús", None, None),
    ]

    for raw, exp_sub, exp_action, exp_target in positive_cases:
        has_wake, rest = match_and_strip_wake_word(raw)
        assert has_wake, f"FAIL: Expected wake=True for {raw!r}, got False"
        if exp_sub == "":
            assert rest == "", f"FAIL: Expected empty rest for {raw!r}, got {rest!r}"
        else:
            assert exp_sub.lower() in rest.lower(), f"FAIL: Expected {exp_sub!r} in rest={rest!r} for {raw!r}"

        if exp_action:
            route = router.route(rest) or router.route(f"vox verbi {rest}")
            assert route is not None, f"FAIL: Expected route for {raw!r} (rest={rest!r}), got None"
            assert route.get("action") == exp_action, f"FAIL: Action mismatch for {raw!r}: {route}"
            assert route.get("target") == exp_target, f"FAIL: Target mismatch for {raw!r}: {route}"
        print(f"  [PASS] {raw!r} -> wake=True | rest={rest!r}" + (f" | {exp_action}:{exp_target}" if exp_action else ""))

    print("\n=== 2. Testing Strict Non-Activation Without 'Vox Verbi' ===")
    negative_cases = [
        "Bueno, amiguitos.",
        "un amigo de color hermoso",
        "¿Pero qué es eso?",
        "Estamos estudiando los verbos en español.",
        "La soberbia y los proverbios antiguos.",
        "¿Quién era Jesús?",
        "¿Qué es la Biblia?",
        "Dúces alevame al encuentro con Cristo.",
    ]
    for neg in negative_cases:
        has_wake, _ = match_and_strip_wake_word(neg, strict_leading=True)
        assert not has_wake, f"FAIL: False wake trigger on non-wake phrase {neg!r}!"
        print(f"  [PASS] Ignored non-wake phrase: {neg!r}")

    # Verify strict_leading=True rejects non-leading wake-words and collapses repeated 'Vox Verbi Vox Verbi' to ''
    hw_mid, _ = match_and_strip_wake_word("hablando en la sala sobre Vox Verbi hoy", strict_leading=True)
    assert not hw_mid, "FAIL: strict_leading=True must reject mid-sentence 'Vox Verbi'!"
    hw_rep, rest_rep = match_and_strip_wake_word("Vox Verbi  Vox Verbi  Vox Verbi", strict_leading=True)
    assert hw_rep and rest_rep == "", f"FAIL: Repeated 'Vox Verbi' must collapse to empty rest, got {rest_rep!r}"
    print("  [PASS] strict_leading=True rejects mid-sentence mentions and collapses 'Vox Verbi Vox Verbi' -> ''")


def test_distant_noisy_acoustic_stt():
    print("\n=== 3. Testing Distant/Noisy Acoustic STT + 85Hz HPF + Hallucination Suppression ===")
    tts = VoiceEngine()
    stt = BiasedWhisperRecognizer()
    router = CommandRouter()

    rate = 16000
    hpf_sos = scipy.signal.butter(2, 85.0 / (0.5 * rate), btype="highpass", output="sos")

    # 3A. Pure low-frequency room/AC rumble (<85 Hz) AND transient click/thump noise must transcribe to ""
    t = np.linspace(0.0, 2.5, int(rate * 2.5), endpoint=False, dtype=np.float32)
    rumble = 0.08 * np.sin(2 * np.pi * 52.0 * t) + 0.05 * np.sin(2 * np.pi * 68.0 * t) + 0.004 * np.random.randn(len(t)).astype(np.float32)
    # Add synthetic desk click/chair creak burst at t=1.0s (RMS ~230) to verify zero prompt-echo ('defínete')
    rumble[16000:17600] += 0.04 * np.sin(2 * np.pi * 220.0 * np.arange(1600) / rate) * np.hanning(1600)
    rumble_filt = scipy.signal.sosfilt(hpf_sos, rumble)
    rumble_i16 = np.clip(rumble_filt * 32767.0, -32768, 32767).astype(np.int16).tobytes()
    rumble_txt = stt.transcribe(rumble_i16, sample_rate=rate)
    assert "defínete" not in rumble_txt.lower() and "vox verbi" not in rumble_txt.lower(), (
        f"FAIL: Noise/click produced prompt-leakage hallucination: {rumble_txt!r}"
    )
    print(f"  [PASS] Room rumble + transient desk click -> {rumble_txt!r} (0 wake/prompt hallucinations)")

    # 3A-ter. Muffled distant room conversation (-30 dBFS + lowpass muffle + noise) MUST NEVER hallucinate 'Vox Verbi'!
    muf_wav, sr = tts.create("Estos dientes quedan más afuera y si lo miramos de lado se ven distintos.", voice=config.TTS_VOICE_NAME, speed=1.05, lang="es")
    gcd = np.gcd(rate, sr)
    muf16 = scipy.signal.resample_poly(muf_wav, rate // gcd, sr // gcd).astype(np.float32)
    lpf = scipy.signal.butter(2, 900.0 / (0.5 * rate), btype="lowpass", output="sos")
    muf16 = scipy.signal.sosfilt(lpf, muf16)
    muf16 = muf16 * (0.03 / max(1e-4, float(np.max(np.abs(muf16))))) + 0.003 * np.random.randn(len(muf16)).astype(np.float32)
    muf_pcm = np.clip(scipy.signal.sosfilt(hpf_sos, muf16) * 32767.0, -32768, 32767).astype(np.int16).tobytes()
    muf_txt = stt.transcribe(muf_pcm, sample_rate=rate)
    muf_wake, _ = match_and_strip_wake_word(muf_txt, strict_leading=True)
    assert not muf_wake, f"FAIL: Muffled room conversation hallucinated wake-word: {muf_txt!r}"
    print(f"  [PASS] Muffled background conversation -> STT={muf_txt!r} | wake=False (0 phantom triggers!)")
    sec11_raw = router.get_canonical_text({"action": "READ_PRESENTACION", "target": "SEC_11"})
    sec11_segs = parse_prosodic_segments(sec11_raw)
    sec11_first_spoken = expand_biblical_citations_for_tts(sec11_segs[0]["text"])
    assert not sec11_first_spoken.lower().startswith("once"), f"FAIL: SEC_11 still starts with 'once': {sec11_first_spoken!r}"
    assert sec11_first_spoken.startswith("Vox Verbi es una propuesta"), f"Unexpected SEC_11 opening: {sec11_first_spoken!r}"
    print(f"  [PASS] SEC_11 clean opening (no 'once'): {sec11_first_spoken[:70]!r}...")

    # 3B. Distant speech (-24 dBFS) mixed with 55Hz hum
    phrases = [
        ("Vox Verbi, defínete.", True, "SEC_11"),
        ("Vox Verbi, ¿qué es Vox Verbi?", True, "SEC_11"),
        ("Vox Verbi, ¿quién era Jesús?", True, None),
        ("Vox Verbi.", True, "ARMED_ONLY"),
    ]
    for phrase, exp_wake, exp_target in phrases:
        wav, sr = tts.create(phrase, voice=config.TTS_VOICE_NAME, speed=1.0, lang="es")
        gcd = np.gcd(rate, sr)
        wav16 = scipy.signal.resample_poly(wav, rate // gcd, sr // gcd).astype(np.float32)
        pad = np.zeros(int(rate * 0.25), dtype=np.float32)
        wav16 = np.concatenate([pad, wav16, pad])
        # Attenuate voice to -24 dBFS (~0.06 peak) + add 55 Hz hum
        peak = max(1e-4, float(np.max(np.abs(wav16))))
        distant = wav16 * (0.06 / peak)
        tt = np.arange(len(distant), dtype=np.float32) / rate
        noisy = distant + 0.04 * np.sin(2 * np.pi * 55.0 * tt) + 0.001 * np.random.randn(len(distant)).astype(np.float32)
        filt = scipy.signal.sosfilt(hpf_sos, noisy)
        pcm_bytes = np.clip(filt * 32767.0, -32768, 32767).astype(np.int16).tobytes()

        transcribed = stt.transcribe(pcm_bytes, sample_rate=rate)
        has_wake, rest = match_and_strip_wake_word(transcribed)
        assert has_wake == exp_wake, f"FAIL: Acoustic STT on {phrase!r} -> {transcribed!r} (has_wake={has_wake})"
        if exp_target == "ARMED_ONLY":
            assert rest == "", f"FAIL: Expected standalone wake on {phrase!r}, got rest={rest!r}"
        elif exp_target:
            route = router.route(rest) or router.route(f"vox verbi {rest}")
            assert route and route.get("target") == exp_target, f"FAIL: Route mismatch for {transcribed!r}: {route}"
        print(f"  [PASS] Distant -24dBFS {phrase!r} -> STT={transcribed!r} | wake={has_wake} | rest={rest!r}")

    # 3C. Soft leading 'Vox Verbi' (-30 dBFS) + 0.55s pause + Loud question (-2 dBFS, 12x amplitude contrast!)
    w_wake, sr = tts.create("Vox Verbi.", voice=config.TTS_VOICE_NAME, speed=1.0, lang="es")
    w_q, _ = tts.create("¿Qué enseña Lumen Gentium sobre el pueblo de Dios?", voice=config.TTS_VOICE_NAME, speed=1.0, lang="es")
    gcd = np.gcd(rate, sr)
    w_wake16 = scipy.signal.resample_poly(w_wake, rate // gcd, sr // gcd).astype(np.float32)
    w_q16 = scipy.signal.resample_poly(w_q, rate // gcd, sr // gcd).astype(np.float32)
    w_wake16 = w_wake16 * (0.032 / max(1e-4, float(np.max(np.abs(w_wake16)))))  # soft -30 dBFS wake!
    w_q16 = w_q16 * (0.75 / max(1e-4, float(np.max(np.abs(w_q16)))))            # booming -2.5 dBFS question!
    combo = np.concatenate([
        np.zeros(int(rate * 0.35), dtype=np.float32),
        w_wake16,
        np.zeros(int(rate * 0.55), dtype=np.float32),
        w_q16,
        np.zeros(int(rate * 0.35), dtype=np.float32),
    ])
    combo_pcm = np.clip(scipy.signal.sosfilt(hpf_sos, combo) * 32767.0, -32768, 32767).astype(np.int16).tobytes()
    combo_txt = stt.transcribe(combo_pcm, sample_rate=rate)
    cw, crest = match_and_strip_wake_word(combo_txt)
    assert cw and "lumen" in crest.lower(), f"FAIL: Soft 'Vox Verbi' + loud question clipped wake-word: {combo_txt!r}"
    print(f"  [PASS] Soft -30dBFS 'Vox Verbi' + 0.55s pause + Loud -2.5dBFS question -> STT={combo_txt!r}")


def test_stop_everything_and_paraiso_protection():
    print("\n=== 4. Testing 'Vox Verbi, detente' Barge-In Stop & 'Al Paraíso' Protection ===")
    app = VoxVerbiApp(init_audio_hw=False)

    # 4A. Route check for 'Vox Verbi, detente'
    r_stop = app.router.route("Vox Verbi, detente")
    assert r_stop and r_stop.get("action") == "STOP_ALL", f"Expected STOP_ALL, got {r_stop}"

    # 4B. Start music + async long question, then interrupt with stop_everything()
    app.mixer.play_music("MUS-003")
    assert app.mixer.is_playing_music is True
    t = app.dispatch_async("Vox Verbi, cuéntame las tres cruces", route_match=app.router.route("Vox Verbi, cuéntame las tres cruces"))
    import time
    time.sleep(0.25)
    stopped = app.stop_everything()
    assert stopped is True, "Expected stop_everything() == True during story/music"
    assert app.mixer.is_playing_music is False, "Expected music to stop on 'Vox Verbi, detente'"
    assert app.mixer._stop_speech_event.is_set(), "Expected speech stop event set"
    t.join(timeout=2.0)

    # 4B-bis. CRITICAL: Verify that AFTER 'Vox Verbi, detente', the NEXT streaming query clears _stop_speech_event and plays!
    app.handle_query("Vox Verbi, responde en 30 segundos qué es la fe")
    assert not app.mixer._stop_speech_event.is_set(), "FAIL: _stop_speech_event remained set on subsequent streaming query!"
    print("  [PASS] 'Vox Verbi, detente' stops active action AND subsequent streaming queries play cleanly!")

    # 4C. Verify 'Al Paraíso' and 'Adoración' CANNOT be stopped by 'Vox Verbi, detente'!
    app.mixer.play_music("MUS-005")
    app.in_al_paraiso = True
    try:
        stopped_paraiso = app.stop_everything()
        assert stopped_paraiso is False, "FAIL: 'Vox Verbi, detente' must NOT stop Al Paraíso!"
        assert app.mixer.is_playing_music is True, "FAIL: MUS-005 must remain playing during Al Paraíso!"
    finally:
        app.in_al_paraiso = False

    app.state.enter_adoracion(mixer=app.mixer, track="MUS-005")
    stopped_ador = app.stop_everything()
    assert stopped_ador is False, "FAIL: 'Vox Verbi, detente' must NOT stop Adoración!"
    app.state.exit_adoracion(mixer=app.mixer)
    print("  [PASS] 'Vox Verbi, detente' strictly ignored during 'Al Paraíso' and 'Adoración'")

    # 4D. Verify secular off-topic filter blocks 'computador' and 'carro' immediately
    for off_q in ("Vox Verbi, ¿qué es un computador?", "Vox Verbi, ¿qué es un carro?"):
        allowed, _ = app.prefilter.evaluate(off_q, rag_result=app.rag.search(off_q))
        assert allowed is False, f"FAIL: Secular off-topic query not blocked: {off_q!r}"
    print("  [PASS] Secular off-topic queries ('computador', 'carro') blocked cleanly by PreFilter")


if __name__ == "__main__":
    test_historical_log_failures()
    test_distant_noisy_acoustic_stt()
    test_stop_everything_and_paraiso_protection()
    print("\nALL WAKE-WORD, ACOUSTIC STT & BARGE-IN STOP REGRESSION TESTS PASSED!")
