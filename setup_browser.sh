mkdir -p /VData/linna4335/known_actions/scripts/{lib,tasks}
cd /VData/linna4335/known_actions/scripts

# Run npm init inside the container
apptainer exec \
  --no-home --writable-tmpfs \
  --home /VData/linna4335/known_actions/container_home:/home/headless \
  --bind /VData/linna4335/known_actions/scripts:/scripts \
  /VData/linna4335/known_actions/midscene-desktop.sif \
  bash -c "cd /scripts && npm init -y && npm install @midscene/computer tsx dotenv"