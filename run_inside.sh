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
