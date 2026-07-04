#!/usr/bin/env bash
set -u

SCRIPT_DIR=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
cd "$SCRIPT_DIR" || exit 1

PORT="${1:-${LILO_PORT:-8765}}"
OPEN_FLAG="--open"
if [ "${1:-}" = "--no-browser" ]; then
  PORT="${LILO_PORT:-8765}"
  OPEN_FLAG=""
elif [ "${2:-}" = "--no-browser" ]; then
  OPEN_FLAG=""
fi
LOG_FILE="$SCRIPT_DIR/server.start.log"
export PYTHONUNBUFFERED=1

pause_on_error() {
  status="$1"
  echo
  echo "BlackBoxLab failed to start. Full log:"
  echo "------------------------------------------------------------"
  if [ -f "$LOG_FILE" ]; then
    cat "$LOG_FILE"
  fi
  echo "------------------------------------------------------------"
  echo
  printf "Press Enter to close this window..."
  read -r _ || true
  exit "$status"
}

test_python() {
  exe="$1"
  shift
  if command -v "$exe" >/dev/null 2>&1 || [ -x "$exe" ]; then
    output=$("$exe" "$@" -c "import sys, numpy; print(sys.executable)" 2>/dev/null)
    status=$?
    if [ "$status" -eq 0 ] && [ -n "$output" ]; then
      PYTHON_EXE="$exe"
      PYTHON_ARGS="$*"
      PYTHON_PATH="$output"
      return 0
    fi
  fi
  return 1
}

PYTHON_EXE=""
PYTHON_ARGS=""
PYTHON_PATH=""

if [ -n "${PYTHON:-}" ]; then
  test_python "$PYTHON" || true
fi

if [ -z "$PYTHON_EXE" ]; then test_python python3 || true; fi
if [ -z "$PYTHON_EXE" ]; then test_python python || true; fi
if [ -z "$PYTHON_EXE" ]; then test_python py -3 || true; fi

for candidate in \
  "/c/ProgramData/anaconda3/python.exe" \
  "$HOME/anaconda3/python.exe" \
  "$HOME/miniconda3/python.exe" \
  "${LOCALAPPDATA:-}/Programs/Python/Python312/python.exe" \
  "${LOCALAPPDATA:-}/Programs/Python/Python311/python.exe"
do
  if [ -z "$PYTHON_EXE" ] && [ -n "$candidate" ]; then
    test_python "$candidate" || true
  fi
done

if [ -z "$PYTHON_EXE" ]; then
  echo "Python with NumPy was not found. Install Python/Anaconda or add it to PATH." > "$LOG_FILE"
  pause_on_error 1
fi

: > "$LOG_FILE"
echo "Starting BlackBoxLab..."
echo "Project: $SCRIPT_DIR"
echo "Port: $PORT"
echo "Python: $PYTHON_PATH"
echo
echo "A browser window should open automatically after the backend is ready."
echo "Keep this terminal open while using BlackBoxLab. Press Ctrl+C here to stop."
echo

if [ -n "$PYTHON_ARGS" ]; then
  # shellcheck disable=SC2086
  "$PYTHON_EXE" $PYTHON_ARGS server.py "$PORT" $OPEN_FLAG 2>&1 | tee -a "$LOG_FILE"
else
  "$PYTHON_EXE" server.py "$PORT" $OPEN_FLAG 2>&1 | tee -a "$LOG_FILE"
fi
status=${PIPESTATUS:-$?}

case "$status" in
  *" "*) status=$(printf "%s" "$status" | awk '{print $1}') ;;
esac

if [ "$status" -ne 0 ]; then
  pause_on_error "$status"
fi
