#!/usr/bin/env bash
set -euo pipefail

INSTALL_ROOT="${GROBID_INSTALL_ROOT:-$PWD/.services}"
DEB_DIR="$INSTALL_ROOT/debs"
LIB_ROOT="$INSTALL_ROOT/native-libs"
LIB_DIR="$LIB_ROOT/usr/lib/x86_64-linux-gnu"

[[ -e "$LIB_DIR/libfontconfig.so.1" ]] && exit 0
mkdir -p "$DEB_DIR" "$LIB_ROOT"
(
  cd "$DEB_DIR"
  apt-get download libfontconfig1 libxml2-16
  for package in ./*.deb; do
    dpkg-deb -x "$package" "$LIB_ROOT"
  done
)
test -e "$LIB_DIR/libfontconfig.so.1"
