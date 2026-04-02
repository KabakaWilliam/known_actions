#!/bin/bash
set -e

WORKDIR=/VData/linna4335/known_actions

export APPTAINER_CACHEDIR=$WORKDIR/apptainer_cache
export APPTAINER_TMPDIR=$WORKDIR/apptainer_tmp
export TMPDIR=$WORKDIR/apptainer_tmp

mkdir -p "$APPTAINER_CACHEDIR"
mkdir -p "$APPTAINER_TMPDIR"
mkdir -p "$WORKDIR/data"
mkdir -p "$WORKDIR/container_home"
mkdir -p "$WORKDIR/logs/supervisor"

# Kill any leftover processes from previous runs
echo "[*] Cleaning up any previous runs..."
pkill -f "midscene-desktop.sif" 2>/dev/null || true
pkill -f "webhookd" 2>/dev/null || true
pkill -f "Xvnc" 2>/dev/null || true
sleep 2



# Write the run_inside.sh script
cat > "$WORKDIR/run_inside.sh" << 'EOF'
#!/bin/bash
set -e

export PATH="/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin"
export VNC_OFFSET=${VNC_OFFSET:-10}
export DISPLAY=:$VNC_OFFSET
export HOME=/home/headless
export USER=headless
export SHELL=/bin/bash
export LANG=en_US.UTF-8
export LANGUAGE=en_US:en

PORT_VNC=${PORT_VNC:-10091}

echo "[*] Setting up novnc token config..."
mkdir -p /etc/novnc
port0=$(expr 5900 + $VNC_OFFSET)
echo "display$VNC_OFFSET: 127.0.0.1:$port0" > /etc/novnc/token.conf

echo "[*] Writing /.env..."
echo "export DISPLAY=:$VNC_OFFSET" > /.env
echo "export HOME=/home/headless" >> /.env
echo "export LANG=en_US.UTF-8" >> /.env
echo "export LANGUAGE=en_US:en" >> /.env

echo "[*] Writing custom supervisor config..."
mkdir -p /etc/supervisor/conf.d

cat > /etc/supervisor/conf.d/xvnc_custom.conf << SVEOF
[program:x${VNC_OFFSET}-xvnc]
environment=DISPLAY=:${VNC_OFFSET},HOME=/home/headless
priority=35
command=/xvnc.sh xvnc ${VNC_OFFSET}
autostart=true
autorestart=true
stdout_logfile=/var/log/supervisor/xvnc.log
stdout_logfile_maxbytes=50MB
redirect_stderr=true

[program:x${VNC_OFFSET}-de]
environment=DISPLAY=:${VNC_OFFSET},HOME=/home/headless,USER=headless,SHELL=/bin/bash,TERM=xterm,LANG=en_US.UTF-8,LANGUAGE=en_US:en
priority=45
command=bash -c "source /.env; exec startfluxbox"
autostart=true
autorestart=true
stdout_logfile=/var/log/supervisor/de.log
stdout_logfile_maxbytes=50MB
redirect_stderr=true

[program:novnc]
priority=34
command=bash -c "cd /usr/local/webhookd; exec bash ./run.sh"
autostart=true
autorestart=true
stdout_logfile=/var/log/supervisor/novnc.log
stdout_logfile_maxbytes=50MB
redirect_stderr=true

[program:midscene-pc]
priority=50
environment=DISPLAY=:${VNC_OFFSET},HOME=/home/headless,USER=headless
command=/usr/bin/npx midscene-pc
directory=/home/headless
autostart=true
autorestart=true
startsecs=5
stdout_logfile=/var/log/midscene-pc.log
stderr_logfile=/var/log/midscene-pc-error.log
SVEOF

# Remove old configs with user=root/headless
rm -f /etc/supervisor/conf.d/sv.conf
rm -f /etc/supervisor/conf.d/midscene-pc.conf

echo "[*] Launching supervisord..."
exec supervisord -n
EOF

chmod +x "$WORKDIR/run_inside.sh"

SIF=$WORKDIR/midscene-desktop.sif

if [ ! -f "$SIF" ]; then
  echo "[*] SIF image not found. Pulling from Docker Hub..."
  apptainer pull "$SIF" docker://ppagent/midscene-ubuntu-desktop:latest
else
  echo "[*] SIF image found: $SIF"
fi

# Generate VNC password file using vncpasswd from inside the container
echo "[*] Generating VNC password file..."
apptainer exec \
  --no-home --writable-tmpfs \
  --home "$WORKDIR/container_home":/home/headless \
  --bind "$WORKDIR":/workdir_out \
  "$SIF" \
  bash -c "echo -e 'midscene-pc\nmidscene-pc\ny\nmidscene_pc\nmidscene_pc' | vncpasswd /workdir_out/vnc_pass 2>/dev/null"
chmod 644 "$WORKDIR/vnc_pass"
echo "[*] VNC password file written: $(ls -la $WORKDIR/vnc_pass)"

echo "[*] Starting Midscene desktop container..."
echo "    VNC web interface  -> http://localhost:10091"
echo "    Midscene PC service -> localhost:3333"
echo "    VNC password       -> midscene-pc"
echo "    Logs -> $WORKDIR/logs/"
echo ""
echo "    Press Ctrl+C to stop."
echo ""

apptainer exec \
  --no-home \
  --writable-tmpfs \
  --home "$WORKDIR/container_home":/home/headless \
  --bind "$WORKDIR/data":/mnt/data \
  --bind "$WORKDIR/logs":/var/log \
  --bind "$WORKDIR/run_inside.sh":/run_inside.sh \
  --bind "$WORKDIR/vnc_pass":/etc/xrdp/vnc_pass \
  --env VNC_OFFSET=10 \
  --env PORT_VNC=10091 \
  "$SIF" \
  /run_inside.sh