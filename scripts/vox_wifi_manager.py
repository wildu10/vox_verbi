#!/usr/bin/env python3
"""
Gestor de Red Wi-Fi e IP Fija Multi-Red para Vox Verbi Mini PC (custodia).
Comando en terminal:  agregar-red   (o: wifi)
Modo guardia al arranque:  vox-wifi-manager --auto-guard
"""
import codecs
import getpass
import ipaddress
import json
import os
import re
import shutil
import subprocess
import sys
import time
import yaml

NETPLAN_PATH = "/etc/netplan/50-cloud-init.yaml"
STATE_PATH = "/etc/vox_verbi_network.json"
DEFAULT_IFACE = "wlp3s0"
DEFAULT_PRIMARY_IP = "192.168.1.11/24"
DEFAULT_HOST_OCTET = 11
DEFAULT_GATEWAY = "192.168.1.1"
DEFAULT_DNS = ["1.1.1.1", "8.8.8.8"]


def run_cmd(cmd: list[str], timeout: int = 15) -> tuple[int, str, str]:
    try:
        p = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
        return p.returncode, p.stdout.strip(), p.stderr.strip()
    except Exception as e:
        return 1, "", str(e)


def ensure_root():
    if os.geteuid() != 0:
        os.execvp("sudo", ["sudo", sys.executable, *sys.argv])


def detect_wifi_iface() -> str:
    rc, out, _ = run_cmd(["iw", "dev"])
    if rc == 0:
        m = re.search(r"Interface\s+(\S+)", out)
        if m:
            return m.group(1)
    return DEFAULT_IFACE


def load_state() -> dict:
    default_st = {
        "interface": detect_wifi_iface(),
        "primary_static_ip": DEFAULT_PRIMARY_IP,
        "fixed_host_octet": DEFAULT_HOST_OCTET,
        "active_subnet_ip": DEFAULT_PRIMARY_IP,
        "gateway": DEFAULT_GATEWAY,
    }
    if os.path.exists(STATE_PATH):
        try:
            with open(STATE_PATH, "r", encoding="utf-8") as f:
                data = json.load(f)
            default_st.update(data)
        except Exception:
            pass
    return default_st


def save_state(st: dict):
    with open(STATE_PATH, "w", encoding="utf-8") as f:
        json.dump(st, f, indent=2, ensure_ascii=False)
    os.chmod(STATE_PATH, 0o644)


def load_saved_access_points(iface: str) -> dict:
    if not os.path.exists(NETPLAN_PATH):
        return {}
    try:
        with open(NETPLAN_PATH, "r", encoding="utf-8") as f:
            doc = yaml.safe_load(f) or {}
        wifis = doc.get("network", {}).get("wifis", {})
        wcfg = wifis.get(iface) or next(iter(wifis.values()), {})
        aps = wcfg.get("access-points", {}) or {}
        return dict(aps)
    except Exception:
        return {}


def build_static_addresses(primary_ip: str, active_ip: str) -> list[str]:
    addrs = []
    if active_ip:
        addrs.append(active_ip)
    if primary_ip and primary_ip not in addrs:
        # Keep 192.168.1.11/24 always bound even if router uses another subnet (e.g. 192.168.0.x)
        addrs.append(primary_ip)
    return addrs


def write_netplan(iface: str, access_points: dict, static_mode: bool, addresses: list[str] = None, gateway: str = None):
    if os.path.exists(NETPLAN_PATH):
        shutil.copy2(NETPLAN_PATH, NETPLAN_PATH + ".bak")

    wcfg: dict = {
        "optional": True,
        "access-points": access_points,
    }
    if static_mode and addresses:
        wcfg["dhcp4"] = False
        wcfg["addresses"] = addresses
        if gateway:
            wcfg["routes"] = [{"to": "default", "via": gateway}]
        dns_list = list(DEFAULT_DNS)
        if gateway and gateway not in dns_list:
            dns_list.append(gateway)
        wcfg["nameservers"] = {"addresses": dns_list}
    else:
        wcfg["dhcp4"] = True

    doc = {
        "network": {
            "version": 2,
            "renderer": "networkd",
            "wifis": {iface: wcfg},
        }
    }
    with open(NETPLAN_PATH, "w", encoding="utf-8") as f:
        yaml.safe_dump(doc, f, default_flow_style=False, allow_unicode=True, sort_keys=False)
    os.chmod(NETPLAN_PATH, 0o600)


def apply_netplan() -> bool:
    rc, _, err = run_cmd(["netplan", "apply"], timeout=25)
    if rc != 0:
        print(f"  [!] Aviso de netplan: {err}")
        return False
    return True


def update_etc_issue(ips_str: str):
    issue_text = (
        "\\S \\n \\l\n"
        "============================================================\n"
        f"  VOX VERBI Mini PC  |  IP Fija: {ips_str}  (custodia.local)\n"
        "  Para conectar o agregar Wi-Fi inicia sesion y escribe: agregar-red\n"
        "============================================================\n\n"
    )
    try:
        with open("/etc/issue", "w", encoding="utf-8") as f:
            f.write(issue_text)
    except Exception:
        pass


def get_current_status(iface: str) -> dict:
    _, wpa_out, _ = run_cmd(["wpa_cli", "-i", iface, "status"])
    ssid = ""
    wpa_state = ""
    for line in wpa_out.splitlines():
        if line.startswith("ssid="):
            ssid = decode_wpa_ssid(line.split("=", 1)[1])
        elif line.startswith("wpa_state="):
            wpa_state = line.split("=", 1)[1]

    _, ip_out, _ = run_cmd(["ip", "-4", "-o", "addr", "show", "dev", iface])
    ips = re.findall(r"inet\s+(\d+\.\d+\.\d+\.\d+/\d+)", ip_out)

    _, rt_out, _ = run_cmd(["ip", "-4", "route", "show", "default", "dev", iface])
    m_gw = re.search(r"default\s+via\s+(\d+\.\d+\.\d+\.\d+)", rt_out)
    gw = m_gw.group(1) if m_gw else ""
    return {"ssid": ssid, "wpa_state": wpa_state, "ips": ips, "gateway": gw}


def ping_ok(target_ip: str) -> bool:
    if not target_ip:
        return False
    rc, _, _ = run_cmd(["ping", "-c", "1", "-W", "2", target_ip], timeout=4)
    return rc == 0


def decode_wpa_ssid(raw: str) -> str:
    if "\\x" not in raw:
        return raw
    try:
        b = codecs.escape_decode(raw.encode("latin-1"))[0]
        return b.decode("utf-8", errors="replace")
    except Exception:
        return raw


def scan_wifi_networks(iface: str) -> list[dict]:
    run_cmd(["ip", "link", "set", iface, "up"])
    run_cmd(["wpa_cli", "-i", iface, "scan"])
    time.sleep(2.5)
    rc, out, _ = run_cmd(["wpa_cli", "-i", iface, "scan_results"])
    seen = {}
    if rc == 0 and out:
        for line in out.splitlines()[1:]:
            parts = line.split("\t")
            if len(parts) >= 5:
                try:
                    sig = int(parts[2])
                except ValueError:
                    sig = -90
                flags = parts[3]
                ssid = decode_wpa_ssid(parts[4]).strip()
                if not ssid or ssid.startswith("\\x00"):
                    continue
                open_net = ("WPA" not in flags and "WEP" not in flags and "SAE" not in flags)
                if ssid not in seen or sig > seen[ssid]["signal"]:
                    seen[ssid] = {"ssid": ssid, "signal": sig, "open": open_net}

    if not seen:
        # Fallback to iw scan
        rc2, iw_out, _ = run_cmd(["iw", "dev", iface, "scan"], timeout=15)
        if rc2 == 0:
            cur_sig = -85
            for ln in iw_out.splitlines():
                ls = ln.strip()
                if ls.startswith("signal:"):
                    m = re.search(r"(-?\d+)", ls)
                    if m:
                        cur_sig = int(m.group(1))
                elif ls.startswith("SSID:"):
                    s = ls.split("SSID:", 1)[1].strip()
                    if s and (s not in seen or cur_sig > seen[s]["signal"]):
                        seen[s] = {"ssid": s, "signal": cur_sig, "open": False}

    return sorted(seen.values(), key=lambda x: x["signal"], reverse=True)


def signal_label(dbm: int) -> str:
    if dbm >= -55:
        return "████ Excelente"
    if dbm >= -67:
        return "███░ Buena    "
    if dbm >= -77:
        return "██░░ Regular  "
    return "█░░░ Débil    "


def probe_router_and_lock_static(iface: str, aps: dict, st: dict) -> tuple[bool, str, str]:
    """
    Probes the active Wi-Fi router via brief DHCP to discover its subnet & gateway,
    then locks fixed host octet (.11 by default) on that subnet PLUS keeps primary_static_ip (192.168.1.11/24).
    """
    print("  -> Conectando al router Wi-Fi y detectando puerta de enlace...")
    write_netplan(iface, aps, static_mode=False)
    apply_netplan()
    run_cmd(["wpa_cli", "-i", iface, "reconfigure"])

    discovered_gw = ""
    discovered_cidr = ""
    for _ in range(18):
        time.sleep(1.0)
        info = get_current_status(iface)
        if info["wpa_state"] == "COMPLETED" and info["ips"] and info["gateway"]:
            discovered_gw = info["gateway"]
            discovered_cidr = info["ips"][0]
            break

    primary_ip = st.get("primary_static_ip", DEFAULT_PRIMARY_IP)
    host_octet = int(st.get("fixed_host_octet", DEFAULT_HOST_OCTET))

    if discovered_gw and discovered_cidr:
        iface_obj = ipaddress.IPv4Interface(discovered_cidr)
        net = iface_obj.network
        gw_last = int(discovered_gw.split(".")[-1])
        target_octet = host_octet if host_octet != gw_last else (host_octet + 1)
        base_int = int(net.network_address) + target_octet
        active_ip = f"{ipaddress.IPv4Address(base_int)}/{net.prefixlen}"
        st["active_subnet_ip"] = active_ip
        st["gateway"] = discovered_gw
    else:
        # Fallback to stored static configuration so static IP is never lost even if DHCP is disabled on router
        active_ip = st.get("active_subnet_ip", primary_ip)
        discovered_gw = st.get("gateway", DEFAULT_GATEWAY)

    addrs = build_static_addresses(primary_ip, active_ip)
    print(f"  -> Fijando IP estática permanente: {', '.join(addrs)} (puerta de enlace: {discovered_gw})...")
    write_netplan(iface, aps, static_mode=True, addresses=addrs, gateway=discovered_gw)
    ok = apply_netplan()
    save_state(st)
    update_etc_issue(", ".join(a.split("/")[0] for a in addrs))
    time.sleep(2.0)
    return ok and ping_ok(discovered_gw), ", ".join(addrs), discovered_gw


def run_auto_guard():
    """
    Boot / background check:
    Ensures static IP is active, and if the Mini PC moved to a saved Wi-Fi on a DIFFERENT router subnet
    where the old gateway doesn't respond, adapts the subnet IP (.11) while keeping 192.168.1.11/24 bound!
    """
    st = load_state()
    iface = st["interface"]
    aps = load_saved_access_points(iface)
    if not aps:
        return

    for _ in range(15):
        info = get_current_status(iface)
        if info["wpa_state"] == "COMPLETED":
            break
        time.sleep(1.0)

    info = get_current_status(iface)
    primary_ip = st.get("primary_static_ip", DEFAULT_PRIMARY_IP)
    active_ip = st.get("active_subnet_ip", primary_ip)
    gw = st.get("gateway", DEFAULT_GATEWAY)

    if info["wpa_state"] == "COMPLETED":
        if ping_ok(gw):
            # Ensure static addresses are applied (not dynamic DHCP)
            addrs = build_static_addresses(primary_ip, active_ip)
            _, ip_detail, _ = run_cmd(["ip", "-4", "addr", "show", "dev", iface])
            if "dynamic" in ip_detail or primary_ip.split("/")[0] not in ip_detail:
                write_netplan(iface, aps, static_mode=True, addresses=addrs, gateway=gw)
                apply_netplan()
            update_etc_issue(", ".join(a.split("/")[0] for a in addrs))
            return
        # Connected to Wi-Fi, yet previous gateway doesn't answer -> router changed subnet!
        probe_router_and_lock_static(iface, aps, st)


def show_status_banner(st: dict):
    iface = st["interface"]
    info = get_current_status(iface)
    aps = load_saved_access_points(iface)
    print("\n" + "=" * 64)
    print("   VOX VERBI - CONFIGURACIÓN FÁCIL DE RED WI-FI E IP FIJA")
    print("=" * 64)
    print(f"  Interfaz Wi-Fi   : {iface}")
    print(f"  Estado conexión  : {'CONECTADO a «' + info['ssid'] + '»' if info['ssid'] else info['wpa_state'] or 'Sin conexión'}")
    print(f"  IP(s) activas    : {', '.join(info['ips']) if info['ips'] else 'Ninguna'}")
    print(f"  Nombre en red    : custodia.local  (ssh wildu@custodia.local)")
    print(f"  Puerta de enlace : {info['gateway'] or st.get('gateway', '-')}")
    print(f"  Redes guardadas  : {', '.join(aps.keys()) if aps else 'Ninguna'}")
    print("=" * 64)


def add_or_connect_ssid(st: dict, ssid: str, password: str, open_net: bool = False):
    iface = st["interface"]
    aps = load_saved_access_points(iface)
    if open_net or not password:
        aps[ssid] = {}
    else:
        aps[ssid] = {
            "auth": {
                "key-management": "psk",
                "password": password,
            }
        }
    ok, addrs_str, gw = probe_router_and_lock_static(iface, aps, st)
    info = get_current_status(iface)
    print("\n" + "-" * 64)
    if info["ssid"] == ssid or ok:
        print(f"  [OK] Red «{ssid}» guardada y conectada correctamente.")
        print(f"  [OK] IP Fija asignada : {addrs_str}")
        print(f"  [OK] Acceso SSH       : ssh wildu@{addrs_str.split('/')[0]}  (o custodia.local)")
    else:
        print(f"  [!] Red «{ssid}» guardada en el sistema (se conectará en cuanto esté al alcance).")
        print(f"      Verifica la contraseña si la red está encendida cerca.")
    print("-" * 64)


def interactive_menu():
    ensure_root()
    st = load_state()
    while True:
        show_status_banner(st)
        print("  Elige una opción (escribe el número y presiona Enter):")
        print("    1) Buscar redes Wi-Fi cercanas y conectarse")
        print("    2) Agregar una red Wi-Fi manualmente (escribir nombre y clave)")
        print("    3) Ver estado detallado y probar conexión")
        print("    4) Cambiar la IP fija manualmente")
        print("    5) Eliminar una red Wi-Fi guardada")
        print("    0) Salir")
        choice = input("\n  Opción > ").strip()

        if choice == "0" or choice.lower() in ("q", "salir", "exit"):
            print("\n  Saliendo del gestor Wi-Fi de Vox Verbi.\n")
            break

        elif choice == "1":
            print("\n  Escaneando redes Wi-Fi cercanas, espera 3 segundos...")
            nets = scan_wifi_networks(st["interface"])
            aps = load_saved_access_points(st["interface"])
            info = get_current_status(st["interface"])
            if not nets:
                print("  [!] No se detectaron redes en el escaneo. Usa la opción 2 para escribir el nombre.")
                continue
            print("\n  Redes Wi-Fi disponibles:")
            for idx, n in enumerate(nets, 1):
                mark = " [CONECTADA]" if n["ssid"] == info["ssid"] else (" [GUARDADA]" if n["ssid"] in aps else "")
                sec = "Abierta" if n["open"] else "Con clave"
                print(f"    {idx:2d}) {signal_label(n['signal'])} ({n['signal']} dBm) | {n['ssid']} ({sec}){mark}")
            sel = input("\n  Número de la red a conectar (o Enter para cancelar): ").strip()
            if not sel.isdigit() or not (1 <= int(sel) <= len(nets)):
                continue
            chosen = nets[int(sel) - 1]
            ssid = chosen["ssid"]
            if chosen["open"]:
                add_or_connect_ssid(st, ssid, "", open_net=True)
            else:
                if ssid in aps:
                    reuse = input(f"  «{ssid}» ya tiene clave guardada. ¿Usar la clave guardada? [S/n]: ").strip().lower()
                    if reuse in ("", "s", "si", "sí", "y"):
                        old_pw = aps[ssid].get("auth", {}).get("password", aps[ssid].get("password", ""))
                        add_or_connect_ssid(st, ssid, old_pw, open_net=False)
                        input("\n  Presiona Enter para continuar...")
                        continue
                ver = input("  ¿Mostrar la contraseña mientras escribes para evitar errores? [S/n]: ").strip().lower()
                if ver in ("", "s", "si", "sí", "y"):
                    pw = input(f"  Contraseña Wi-Fi para «{ssid}»: ").strip()
                else:
                    pw = getpass.getpass(f"  Contraseña Wi-Fi para «{ssid}»: ").strip()
                if len(pw) < 8:
                    print("  [!] La contraseña WPA debe tener al menos 8 caracteres.")
                    input("\n  Presiona Enter para continuar...")
                    continue
                add_or_connect_ssid(st, ssid, pw, open_net=False)
            input("\n  Presiona Enter para continuar...")

        elif choice == "2":
            ssid = input("\n  Nombre exacto de la red Wi-Fi (SSID): ").strip()
            if not ssid:
                continue
            pw = input(f"  Contraseña Wi-Fi para «{ssid}» (deja vacío si es abierta): ").strip()
            if pw and len(pw) < 8:
                print("  [!] La contraseña WPA debe tener al menos 8 caracteres.")
                input("\n  Presiona Enter para continuar...")
                continue
            add_or_connect_ssid(st, ssid, pw, open_net=(pw == ""))
            input("\n  Presiona Enter para continuar...")

        elif choice == "3":
            info = get_current_status(st["interface"])
            gw = info["gateway"] or st.get("gateway", "")
            print(f"\n  Probando comunicación con el router ({gw})...")
            print(f"    Router ({gw})   : {'RESPONDE OK' if ping_ok(gw) else 'SIN RESPUESTA'}")
            print(f"    Internet (1.1.1.1): {'CONECTADO OK' if ping_ok('1.1.1.1') else 'SIN INTERNET (modo local offline activo)'}")
            input("\n  Presiona Enter para continuar...")

        elif choice == "4":
            cur_ip = st.get("primary_static_ip", DEFAULT_PRIMARY_IP)
            cur_gw = st.get("gateway", DEFAULT_GATEWAY)
            print(f"\n  IP fija actual: {cur_ip}  |  Puerta de enlace: {cur_gw}")
            new_ip = input("  Nueva IP fija (ejemplo 192.168.1.11): ").strip()
            if not new_ip:
                continue
            if "/" not in new_ip:
                new_ip += "/24"
            try:
                ip_iface = ipaddress.IPv4Interface(new_ip)
            except ValueError:
                print("  [!] Formato de IP inválido.")
                input("\n  Presiona Enter para continuar...")
                continue
            def_gw = str(ip_iface.network.network_address + 1)
            new_gw = input(f"  Puerta de enlace del router [{def_gw}]: ").strip() or def_gw
            st["primary_static_ip"] = new_ip
            st["active_subnet_ip"] = new_ip
            st["fixed_host_octet"] = int(str(ip_iface.ip).split(".")[-1])
            st["gateway"] = new_gw
            aps = load_saved_access_points(st["interface"])
            addrs = build_static_addresses(new_ip, new_ip)
            write_netplan(st["interface"], aps, static_mode=True, addresses=addrs, gateway=new_gw)
            apply_netplan()
            save_state(st)
            update_etc_issue(new_ip.split("/")[0])
            print(f"  [OK] Nueva IP fija aplicada: {new_ip} (gateway {new_gw})")
            input("\n  Presiona Enter para continuar...")

        elif choice == "5":
            aps = load_saved_access_points(st["interface"])
            names = list(aps.keys())
            if not names:
                print("\n  No hay redes guardadas.")
                continue
            print("\n  Redes guardadas:")
            for idx, nm in enumerate(names, 1):
                print(f"    {idx}) {nm}")
            sel = input("  Número de la red a eliminar (o Enter para cancelar): ").strip()
            if sel.isdigit() and 1 <= int(sel) <= len(names):
                rem = names[int(sel) - 1]
                if len(aps) == 1:
                    conf = input("  [!] Es la única red guardada. ¿Seguro que deseas borrarla? [s/N]: ").strip().lower()
                    if conf not in ("s", "si", "sí"):
                        continue
                del aps[rem]
                addrs = build_static_addresses(st["primary_static_ip"], st["active_subnet_ip"])
                write_netplan(st["interface"], aps, static_mode=True, addresses=addrs, gateway=st.get("gateway"))
                apply_netplan()
                print(f"  [OK] Red «{rem}» eliminada.")
            input("\n  Presiona Enter para continuar...")


if __name__ == "__main__":
    if "--auto-guard" in sys.argv:
        run_auto_guard()
    else:
        interactive_menu()
