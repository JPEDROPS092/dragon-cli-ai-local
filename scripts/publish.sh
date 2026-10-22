#!/usr/bin/env bash
# Build dragon-cli and upload it to PyPI using the token stored in .env.
#
#   scripts/publish.sh            build, check and upload to PyPI
#   scripts/publish.sh --dry-run  build and check only, upload nothing
#   scripts/publish.sh --test     upload to TestPyPI instead (needs testpypi_api in .env)
#
# .env must define the token (it is gitignored):
#   pypi_api=pypi-AgEI...
set -euo pipefail

cd "$(dirname "$0")/.."

MODE=pypi
case "${1:-}" in
  "") ;;
  --dry-run) MODE=dry ;;
  --test) MODE=test ;;
  *) echo "usage: $0 [--dry-run|--test]" >&2; exit 2 ;;
esac

if [[ -f .env ]]; then
  set -a
  # shellcheck disable=SC1091
  . ./.env
  set +a
fi

if [[ "$MODE" == test ]]; then
  TOKEN="${testpypi_api:-}"
  REPO_URL=https://test.pypi.org/legacy/
  INDEX=https://test.pypi.org
else
  TOKEN="${pypi_api:-}"
  REPO_URL=https://upload.pypi.org/legacy/
  INDEX=https://pypi.org
fi

if [[ "$MODE" != dry && -z "$TOKEN" ]]; then
  echo "error: no token found. Add 'pypi_api=pypi-...' (or 'testpypi_api=...' for --test) to .env" >&2
  exit 1
fi

PY="${PYTHON:-python3}"
NAME=$("$PY" -c 'import tomllib; print(tomllib.load(open("pyproject.toml","rb"))["project"]["name"])')
VERSION=$("$PY" -c 'import tomllib; print(tomllib.load(open("pyproject.toml","rb"))["project"]["version"])')
echo "==> $NAME $VERSION"

# PyPI rejects re-uploading an existing version, so fail early with a clear message.
STATUS=$(curl -s -o /dev/null -w '%{http_code}' "$INDEX/pypi/$NAME/$VERSION/json" || true)
if [[ "$STATUS" == 200 ]]; then
  echo "error: $NAME $VERSION is already on $INDEX. Bump 'version' in pyproject.toml and src/dragon_cli/__init__.py." >&2
  exit 1
fi

if [[ -n "$(git status --porcelain 2>/dev/null)" ]]; then
  echo "warning: working tree has uncommitted changes; they will be included in the build."
fi

# Isolated toolchain so build/twine never touch the project's own venv.
VENV=.venv-release
if [[ ! -x "$VENV/bin/python" ]]; then
  echo "==> creating $VENV"
  "$PY" -m venv "$VENV"
fi
"$VENV/bin/python" -m pip install -q --upgrade pip build twine

echo "==> building"
rm -rf dist build
"$VENV/bin/python" -m build

echo "==> checking"
"$VENV/bin/twine" check --strict dist/*

if [[ "$MODE" == dry ]]; then
  echo "==> dry run: built dist/ but did not upload"
  exit 0
fi

echo "==> uploading to $REPO_URL"
# Pass the token through the environment, not argv, so it never shows up in `ps`.
TWINE_USERNAME=__token__ TWINE_PASSWORD="$TOKEN" \
  "$VENV/bin/twine" upload --non-interactive --repository-url "$REPO_URL" dist/*

echo "==> done: $INDEX/project/$NAME/$VERSION/"
