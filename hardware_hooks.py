import glob
import os
import struct
import threading
import logging

logger = logging.getLogger("vox_verbi.hardware_hooks")

# Linux input_event constants (64-bit struct: tv_sec, tv_usec, type, code, value)
EVENT_STRUCT_FMT = "llHHi"
EVENT_SIZE = struct.calcsize(EVENT_STRUCT_FMT)
EV_KEY = 0x01
KEY_MUTE = 113
KEY_VOLUMEDOWN = 114
KEY_VOLUMEUP = 115


def start_s330_volume_buttons_listener(audio_mixer):
    """
    Listens to the physical +/- volume buttons on the Anker PowerConf S330 USB HID interface
    (/dev/input/event* matching 'Anker' & 'Consumer Control') so hardware buttons adjust
    Vox Verbi's master output volume in real time even without a desktop environment!
    """
    target_dev = None
    for name_path in sorted(glob.glob("/sys/class/input/event*/device/name")):
        try:
            with open(name_path, "r", encoding="utf-8", errors="ignore") as f:
                dev_name = f.read().strip()
            if "anker" in dev_name.lower() and "consumer control" in dev_name.lower():
                ev_dir = os.path.basename(os.path.dirname(os.path.dirname(name_path)))
                cand = f"/dev/input/{ev_dir}"
                if os.path.exists(cand):
                    target_dev = cand
                    break
        except Exception:
            continue

    if not target_dev:
        logger.info("[HARDWARE] Anker S330 Consumer Control input device not found; skipping button listener.")
        return None

    def _listener():
        try:
            with open(target_dev, "rb", buffering=0) as fd:
                logger.info(f"[HARDWARE] Listening to physical S330 volume buttons on {target_dev}")
                while True:
                    data = fd.read(EVENT_SIZE)
                    if not data or len(data) < EVENT_SIZE:
                        break
                    _, _, ev_type, code, value = struct.unpack(EVENT_STRUCT_FMT, data)
                    if ev_type == EV_KEY and value in (1, 2):  # press or hold-repeat
                        if code == KEY_VOLUMEDOWN:
                            audio_mixer.adjust_master_volume(-0.05)
                        elif code == KEY_VOLUMEUP:
                            audio_mixer.adjust_master_volume(+0.05)
                        elif code == KEY_MUTE and value == 1:
                            if audio_mixer.is_busy():
                                audio_mixer.stop_speech()
        except PermissionError:
            logger.warning(f"[HARDWARE] No read permission on {target_dev} (ensure user is in 'input' group).")
        except Exception as e:
            logger.warning(f"[HARDWARE] S330 button listener stopped: {e}")

    t = threading.Thread(target=_listener, daemon=True)
    t.start()
    return t


def on_enter_adoracion():
    """Triggered upon entering ADORACION mode.
    Prepares monstrance, adjusts lighting to intimate/sacred, activates incense."""
    logger.info("[HARDWARE] on_enter_adoracion: Dimming lights, illuminating Custodia, activating incense.")
    set_lights("SACRED_ADORATION")
    open_custodia()
    activate_incense(duration_sec=30)


def on_exit_adoracion():
    """Triggered upon exiting ADORACION mode.
    Restores normal lighting, closes custodia."""
    logger.info("[HARDWARE] on_exit_adoracion: Restoring normal lights, closing Custodia.")
    set_lights("NORMAL_CATECHESIS")
    close_custodia()


def set_lights(preset):
    """Stub to control DMX / Serial / GPIO lights."""
    logger.info(f"[HARDWARE STUB] set_lights: {preset}")


def open_custodia():
    """Stub for servo / actuator that reveals the Blessed Sacrament."""
    logger.info("[HARDWARE STUB] open_custodia: Mechanical door opened")


def close_custodia():
    """Stub for servo / actuator that conceals or protects the monstrance."""
    logger.info("[HARDWARE STUB] close_custodia: Mechanical door closed")


def activate_incense(duration_sec=30):
    """Stub for electronic thurible / ultrasonic incense diffuser."""
    logger.info(f"[HARDWARE STUB] activate_incense: Active for {duration_sec}s")
