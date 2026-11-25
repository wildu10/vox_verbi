#!/usr/bin/env bash
# ==============================================================================
# VOX VERBI v3.6 - CLONADOR E INSTALADOR DE IMAGEN MAESTRA (64 GB USB)
# ==============================================================================
# Modos de uso:
#   1. Crear imagen maestra del Mini PC actual en la USB de 64 GB:
#        sudo bash vox_clone_and_restore.sh --backup /dev/sda1
#   2. Instalar todo Vox Verbi (código, modelos, drivers, servicios) en un Ubuntu 24.04:
#        sudo bash RESTAURAR_EN_NUEVO_MINIPC.sh --install-on-ubuntu
#   3. Clonar disco completo desde un Live USB en un Mini PC nuevo vacío:
#        sudo bash RESTAURAR_EN_NUEVO_MINIPC.sh --full-disk /dev/nvme0n1
# ==============================================================================
set -e

MODE="${1:---help}"
TARGET_DEV="${2:-/dev/sda1}"

if [ "$MODE" = "--backup" ]; then
    echo "=================================================================="
    echo "  VOX VERBI - CREANDO IMAGEN MAESTRA EN USB ($TARGET_DEV)"
    echo "=================================================================="
    MNT_DIR="/mnt/vox_clone_usb"
    mkdir -p "$MNT_DIR"
    if ! mountpoint -q "$MNT_DIR"; then
        mount "$TARGET_DEV" "$MNT_DIR"
    fi

    DEST="$MNT_DIR/VOX_VERBI_IMAGEN_MAESTRA"
    mkdir -p "$DEST/tabla_particiones" "$DEST/imagen_sistema" "$DEST/VOX_VERBI/fuentes_nuevas"

    echo "[1/5] Guardando esquema de particiones GPT, LVM y configuración de arranque..."
    sfdisk -d /dev/nvme0n1 > "$DEST/tabla_particiones/nvme0n1.sfdisk" 2>/dev/null || true
    lsblk -o NAME,SIZE,FSTYPE,UUID,MOUNTPOINT > "$DEST/tabla_particiones/lsblk_original.txt"
    cp /etc/fstab "$DEST/tabla_particiones/fstab.original"

    echo "[2/5] Clonando particiones de arranque (/boot/efi y /boot)..."
    tar --zstd -cf "$DEST/imagen_sistema/efi_partition.tar.zst" -C /boot/efi .
    tar --zstd -cf "$DEST/imagen_sistema/boot_partition.tar.zst" -C /boot .

    echo "[3/5] Empaquetando Stack Completo de Vox Verbi (Gemma 4 E4B, Whisper Vulkan, Supertonic M5, Corpus Teológico + CTI, Python venv y Servicios)..."
    tar -I "zstd -T0 -3" -cf "$DEST/imagen_sistema/vox_verbi_stack_completo.tar.zst" \
        --exclude="/home/wildu/llama.cpp/models/Llama-3.2-3B-Instruct-Q4_K_M.gguf" \
        --exclude="/home/wildu/llama.cpp/models/Meta-Llama-3.1-8B-Instruct-Q4_K_M.gguf" \
        --exclude="/home/wildu/llama.cpp/models/Qwen3.5-4B-Q4_K_M.gguf" \
        --exclude="/home/wildu/llama.cpp/models/Qwen3.5-9B-Q4_K_M.gguf" \
        --exclude="/home/wildu/llama.cpp/vox_verbi/data/*.bak" \
        /home/wildu/llama.cpp \
        /home/wildu/whisper.cpp \
        /home/wildu/voice_models \
        /home/wildu/.cache/supertonic3 \
        /home/wildu/.cache/huggingface \
        /home/wildu/.bashrc \
        /etc/systemd/system/llama-server.service \
        /etc/systemd/system/whisper-server.service \
        /etc/systemd/system/asistente-vivo.service \
        /etc/systemd/system/vox-network-guard.service \
        /etc/systemd/system/vox-usb-import.service \
        /etc/udev/rules.d/99-vox-verbi-usb.rules \
        /etc/sudoers.d/vox-wifi \
        /etc/sudoers.d/vox-usb-mount \
        /etc/vox_verbi_network.json \
        /etc/netplan/50-cloud-init.yaml \
        /usr/local/sbin/vox-wifi-manager \
        /usr/local/bin/agregar-red \
        /usr/local/bin/wifi \
        /usr/local/bin/agregar-contenido \
        /usr/local/bin/importar-usb

    echo "[4/5] Empaquetando imagen del Sistema Operativo Ubuntu 24.04 + Drivers Vulkan/Mesa (os_rootfs_base.tar.zst)..."
    tar -I "zstd -T0 -1" --one-file-system -cf "$DEST/imagen_sistema/os_rootfs_base.tar.zst" \
        --exclude="/home/wildu/*" \
        --exclude="/proc/*" \
        --exclude="/sys/*" \
        --exclude="/dev/*" \
        --exclude="/run/*" \
        --exclude="/tmp/*" \
        --exclude="/mnt/*" \
        --exclude="/media/*" \
        --exclude="/var/cache/apt/archives/*.deb" \
        --exclude="/var/log/journal/*" \
        --exclude="/swap.img" \
        /

    echo "[5/5] Copiando documentos de la Comisión Teológica Internacional y scripts de restauración a la USB..."
    cp -r /home/wildu/llama.cpp/vox_verbi/data/corpus_whitelist/cti_vatican/* "$DEST/VOX_VERBI/fuentes_nuevas/" 2>/dev/null || true
    cp "$0" "$DEST/RESTAURAR_EN_NUEVO_MINIPC.sh"
    chmod 755 "$DEST/RESTAURAR_EN_NUEVO_MINIPC.sh"

    cat > "$DEST/LEEME_COMO_CLONAR_O_INSTALAR.txt" << 'EOF'
========================================================================
  IMAGEN MAESTRA DE CLONACIÓN E INSTALACIÓN - VOX VERBI v3.6
========================================================================

Esta memoria USB contiene la imagen completa del Mini PC Vox Verbi:
- Sistema Operativo Ubuntu 24.04 LTS + Drivers AMD Radeon 680M Vulkan (Mesa RADV).
- LLM Gemma 4 E4B Instruct (Vulkan) + Whisper.cpp STT (Vulkan ~218 ms).
- Voz contemplativa Supertonic 3 M5 (44.1 kHz).
- Base de Conocimiento de 4 Columnas + 33 Documentos de la Comisión Teológica Internacional.
- Servicios automáticos, IP fija (192.168.1.11 / custodia.local), comando "agregar-red"
  y detector automático de memorias USB ("agregar-contenido").

------------------------------------------------------------------------
OPCIÓN A: INSTALAR EN UN MINI PC QUE YA TIENE UBUNTU 24.04 INSTALADO
(Método recomendado y más rápido: toma ~3 minutos)
------------------------------------------------------------------------
1. Conecta esta memoria USB al nuevo Mini PC con Ubuntu 24.04 (usuario: wildu).
2. Abre la terminal, entra a la carpeta VOX_VERBI_IMAGEN_MAESTRA de la USB y ejecuta:

     sudo bash RESTAURAR_EN_NUEVO_MINIPC.sh --install-on-ubuntu

3. Al terminar (3 minutos), el nuevo Mini PC quedará 100% configurado y hablando.

------------------------------------------------------------------------
OPCIÓN B: CLONAR EL DISCO COMPLETO EN UN MINI PC NUEVO EN BLANCO
(Desde un Live USB de Ubuntu 24.04: toma ~6 minutos)
------------------------------------------------------------------------
1. Arranca el nuevo Mini PC con un Live USB de Ubuntu y conecta también esta USB.
2. Abre una terminal en la carpeta VOX_VERBI_IMAGEN_MAESTRA y ejecuta:

     sudo bash RESTAURAR_EN_NUEVO_MINIPC.sh --full-disk /dev/nvme0n1

3. Al finalizar, retira el Live USB y reinicia.
========================================================================
EOF

    sync
    echo ""
    echo "=== RESUMEN DE ARCHIVOS EN LA USB DE 64 GB ==="
    ls -lh "$DEST/imagen_sistema/"
    df -h "$MNT_DIR"
    echo "¡IMAGEN MAESTRA CREADA EXITOSAMENTE EN $DEST!"
    exit 0
fi

if [ "$MODE" = "--install-on-ubuntu" ]; then
    SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
    echo "=================================================================="
    echo "  INSTALANDO STACK COMPLETO VOX VERBI EN ESTE EQUIPO..."
    echo "=================================================================="
    apt-get update -qq || true
    apt-get install -y -qq zstd avahi-daemon avahi-utils alsa-utils curl libportaudio2 libsndfile1 ffmpeg || true
    id -u wildu &>/dev/null || useradd -m -s /bin/bash -G sudo,audio,video,render,input wildu
    usermod -aG sudo,audio,video,render,input wildu

    echo "Extrayendo Vox Verbi Stack Completo..."
    tar -I zstd -xf "$SCRIPT_DIR/imagen_sistema/vox_verbi_stack_completo.tar.zst" -C /
    chown -R wildu:wildu /home/wildu

    systemctl daemon-reload
    udevadm control --reload-rules || true
    systemctl enable --now avahi-daemon.service vox-network-guard.service llama-server.service whisper-server.service asistente-vivo.service
    echo "¡VOX VERBI INSTALADO Y CORRIENDO!"
    exit 0
fi

if [ "$MODE" = "--full-disk" ]; then
    SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
    DISK="${2:-/dev/nvme0n1}"
    echo "=================================================================="
    echo "  RESTAURANDO IMAGEN DE DISCO COMPLETA EN $DISK"
    echo "=================================================================="
    sfdisk "$DISK" < "$SCRIPT_DIR/tabla_particiones/nvme0n1.sfdisk"
    partprobe "$DISK"
    sleep 2
    mkfs.vfat -F 32 "${DISK}p1"
    mkfs.ext4 -F "${DISK}p2"
    pvcreate -ff -y "${DISK}p3"
    vgcreate -y ubuntu-vg "${DISK}p3"
    lvcreate -l 100%FREE -n ubuntu-lv ubuntu-vg
    mkfs.ext4 -F /dev/ubuntu-vg/ubuntu-lv

    mkdir -p /mnt/target_root
    mount /dev/ubuntu-vg/ubuntu-lv /mnt/target_root
    mkdir -p /mnt/target_root/boot
    mount "${DISK}p2" /mnt/target_root/boot
    mkdir -p /mnt/target_root/boot/efi
    mount "${DISK}p1" /mnt/target_root/boot/efi

    echo "1/3 Extrayendo Sistema Operativo base..."
    tar -I zstd -xf "$SCRIPT_DIR/imagen_sistema/os_rootfs_base.tar.zst" -C /mnt/target_root
    echo "2/3 Extrayendo Stack Completo Vox Verbi..."
    tar -I zstd -xf "$SCRIPT_DIR/imagen_sistema/vox_verbi_stack_completo.tar.zst" -C /mnt/target_root
    echo "3/3 Extrayendo particiones de arranque /boot y /boot/efi..."
    tar -I zstd -xf "$SCRIPT_DIR/imagen_sistema/boot_partition.tar.zst" -C /mnt/target_root/boot
    tar -I zstd -xf "$SCRIPT_DIR/imagen_sistema/efi_partition.tar.zst" -C /mnt/target_root/boot/efi

    UUID_P1=$(blkid -s UUID -o value "${DISK}p1")
    UUID_P2=$(blkid -s UUID -o value "${DISK}p2")
    cat > /mnt/target_root/etc/fstab << EOF
/dev/disk/by-id/dm-uuid-LVM-$(lvs --noheadings -o lv_uuid ubuntu-vg/ubuntu-lv | tr -d ' ') / ext4 defaults 0 1
UUID=$UUID_P2 /boot ext4 defaults 0 1
UUID=$UUID_P1 /boot/efi vfat umask=0077 0 1
EOF

    for d in dev proc sys run; do mount --bind "/$d" "/mnt/target_root/$d"; done
    chroot /mnt/target_root grub-install --target=x86_64-efi --efi-directory=/boot/efi --bootloader-id=ubuntu --recheck || true
    chroot /mnt/target_root update-grub || true
    sync
    echo "¡CLONACIÓN DE DISCO COMPLETA FINALIZADA! Ya puedes reiniciar el nuevo Mini PC."
    exit 0
fi

echo "Uso: sudo bash $0 [--backup /dev/sda1 | --install-on-ubuntu | --full-disk /dev/nvme0n1]"
