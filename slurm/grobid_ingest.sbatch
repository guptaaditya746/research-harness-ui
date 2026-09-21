#!/usr/bin/env bash
#SBATCH --job-name=harness-grobid
#SBATCH --partition=gammaweb
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --time=04:00:00
#SBATCH --output=slurm/logs/grobid-%j.out
#SBATCH --error=slurm/logs/grobid-%j.err

set -euo pipefail

PROJECT_DIR="${PROJECT_DIR:-$SLURM_SUBMIT_DIR}"
GROBID_VERSION="${GROBID_VERSION:-0.9.0}"
INSTALL_ROOT="${GROBID_INSTALL_ROOT:-$PROJECT_DIR/.services}"
JAVA_HOME="${JAVA_HOME:-$INSTALL_ROOT/jdk-21}"
RUNTIME_ARCHIVE="$INSTALL_ROOT/grobid-$GROBID_VERSION-runtime.tar.gz"
NATIVE_LIB_DIR="$INSTALL_ROOT/native-libs/usr/lib/x86_64-linux-gnu"
CORPUS_ROOT="${CORPUS_ROOT:-$PROJECT_DIR/corpus_data}"
: "${CORPUS:?Set CORPUS to the directory containing PDFs}"

if [[ ! -x "$JAVA_HOME/bin/java" || ! -f "$RUNTIME_ARCHIVE" ]]; then
  echo "Native GROBID is absent; building JDK 21 and GROBID $GROBID_VERSION on node-local scratch"
  (cd "$PROJECT_DIR" && GROBID_INSTALL_ROOT="$INSTALL_ROOT" GROBID_VERSION="$GROBID_VERSION" \
    ./scripts/bootstrap_grobid_native.sh)
fi

if [[ ! -e "$NATIVE_LIB_DIR/libfontconfig.so.1" ]]; then
  (cd "$PROJECT_DIR" && GROBID_INSTALL_ROOT="$INSTALL_ROOT" ./scripts/bootstrap_grobid_native_libs.sh)
fi

runtime_dir="${SLURM_TMPDIR:-/tmp}/harness-grobid-runtime-$SLURM_JOB_ID"
mkdir -p "$runtime_dir"
tar -xzf "$RUNTIME_ARCHIVE" -C "$runtime_dir"
GROBID_HOME="$runtime_dir"
chmod -R 755 "$GROBID_HOME/grobid-home/pdfalto"

export JAVA_HOME
export PATH="$JAVA_HOME/bin:$PATH"
export LD_LIBRARY_PATH="$NATIVE_LIB_DIR:$GROBID_HOME/grobid-home/lib/lin-64${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"

cd "$GROBID_HOME"
./grobid-service/bin/grobid-service >"$PROJECT_DIR/slurm/logs/grobid-service-$SLURM_JOB_ID.log" 2>&1 &
grobid_pid=$!
cleanup() {
  kill "$grobid_pid" 2>/dev/null || true
  wait "$grobid_pid" 2>/dev/null || true
}
trap cleanup EXIT INT TERM

for _ in $(seq 1 90); do
  if curl -fsS http://127.0.0.1:8070/api/isalive | grep -q true; then
    break
  fi
  if ! kill -0 "$grobid_pid" 2>/dev/null; then
    echo "GROBID exited during startup; inspect the service log" >&2
    exit 1
  fi
  sleep 2
done
curl -fsS http://127.0.0.1:8070/api/isalive | grep -q true

cd "$PROJECT_DIR"
GROBID_URL=http://127.0.0.1:8070 CORPUS_ROOT="$CORPUS_ROOT" \
  uv run --python 3.12 python -m harness_corpus "$CORPUS" --root "$CORPUS_ROOT"
