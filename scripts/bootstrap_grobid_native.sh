#!/usr/bin/env bash
set -euo pipefail

GROBID_VERSION="${GROBID_VERSION:-0.9.0}"
INSTALL_ROOT="${GROBID_INSTALL_ROOT:-$PWD/.services}"
JDK_HOME="$INSTALL_ROOT/jdk-21"
RUNTIME_ARCHIVE="$INSTALL_ROOT/grobid-$GROBID_VERSION-runtime.tar.gz"
DOWNLOADS="$INSTALL_ROOT/downloads"
BUILD_ROOT="${SLURM_TMPDIR:-${TMPDIR:-/tmp}}/harness-grobid-build-${SLURM_JOB_ID:-$$}"

case "$(uname -m)" in
  x86_64) java_arch=x64 ;;
  aarch64|arm64) java_arch=aarch64 ;;
  *) echo "Unsupported CPU architecture: $(uname -m)" >&2; exit 1 ;;
esac

mkdir -p "$DOWNLOADS" "$BUILD_ROOT"
cleanup() { rm -rf "$BUILD_ROOT"; }
trap cleanup EXIT

if [[ ! -x "$JDK_HOME/bin/java" ]]; then
  archive="$DOWNLOADS/temurin-jdk-21-$java_arch.tar.gz"
  curl -fL --retry 3 -o "$archive" \
    "https://api.adoptium.net/v3/binary/latest/21/ga/linux/$java_arch/jdk/hotspot/normal/eclipse"
  mkdir -p "$BUILD_ROOT/jdk"
  tar -xzf "$archive" -C "$BUILD_ROOT/jdk"
  extracted_jdk="$(find "$BUILD_ROOT/jdk" -mindepth 1 -maxdepth 1 -type d | head -1)"
  mv "$extracted_jdk" "$JDK_HOME"
fi

export JAVA_HOME="$JDK_HOME"
export PATH="$JAVA_HOME/bin:$PATH"
java -version

if [[ ! -f "$RUNTIME_ARCHIVE" ]]; then
  source_archive="$DOWNLOADS/grobid-$GROBID_VERSION.zip"
  [[ -f "$source_archive" ]] || curl -fL --retry 3 -o "$source_archive" \
    "https://github.com/grobidOrg/grobid/archive/refs/tags/$GROBID_VERSION.zip"
  unzip -q "$source_archive" -d "$BUILD_ROOT"
  source_dir="$BUILD_ROOT/grobid-$GROBID_VERSION"
  (cd "$source_dir" && GRADLE_USER_HOME="$BUILD_ROOT/gradle-cache" ./gradlew clean assemble --no-daemon)

  mkdir -p "$BUILD_ROOT/runtime"
  unzip -q "$source_dir/grobid-service/build/distributions/grobid-service-$GROBID_VERSION.zip" \
    -d "$BUILD_ROOT/runtime"
  unzip -q "$source_dir/grobid-home/build/distributions/grobid-home-$GROBID_VERSION.zip" \
    -d "$BUILD_ROOT/runtime"
  service_dir="$(find "$BUILD_ROOT/runtime" -mindepth 1 -maxdepth 1 -type d -name 'grobid-service*' | head -1)"
  home_dir="$(find "$BUILD_ROOT/runtime" -mindepth 1 -maxdepth 1 -type d -name 'grobid-home*' | head -1)"
  [[ -n "$service_dir" && -n "$home_dir" ]] || { echo "GROBID runtime directories missing" >&2; exit 1; }
  [[ "$service_dir" == "$BUILD_ROOT/runtime/grobid-service" ]] || mv "$service_dir" "$BUILD_ROOT/runtime/grobid-service"
  [[ "$home_dir" == "$BUILD_ROOT/runtime/grobid-home" ]] || mv "$home_dir" "$BUILD_ROOT/runtime/grobid-home"
  temporary_archive="$RUNTIME_ARCHIVE.tmp.${SLURM_JOB_ID:-$$}"
  tar -C "$BUILD_ROOT/runtime" -czf "$temporary_archive" grobid-service grobid-home
  mv "$temporary_archive" "$RUNTIME_ARCHIVE"
fi

printf 'Native GROBID runtime ready at %s\n' "$RUNTIME_ARCHIVE"
