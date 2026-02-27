#!/usr/bin/env bash
# Decrypt .env.age into tmpfs ramdisk — secrets only live in RAM.
# Usage: ./decrypt_secrets.sh [path-to-env.age]
set -euo pipefail

ENV_AGE="${1:-$(dirname "$0")/../.env.age}"
TARGET="/dev/shm/secrets/.env"

mkdir -p /dev/shm/secrets
chmod 700 /dev/shm/secrets

echo "Decrypting $ENV_AGE → $TARGET"
age -d -o "$TARGET" "$ENV_AGE"
chmod 600 "$TARGET"

echo "✅  Secrets decrypted to tmpfs (RAM only)."
