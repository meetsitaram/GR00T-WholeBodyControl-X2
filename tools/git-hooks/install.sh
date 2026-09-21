#!/usr/bin/env bash
# Install the security pre-commit hook into this clone (idempotent; chains
# any existing pre-commit hook).
set -euo pipefail
cd "$(git rev-parse --show-toplevel)"
HOOK=.git/hooks/pre-commit
SRC=tools/git-hooks/pre-commit-security.sh

if [ -f "$HOOK" ] && ! grep -q pre-commit-security "$HOOK"; then
  mv "$HOOK" "$HOOK.pre-security"
  echo "existing hook preserved as $HOOK.pre-security (will be chained)"
fi

cat > "$HOOK" <<'EOF'
#!/usr/bin/env bash
REPO_ROOT=$(git rev-parse --show-toplevel)
bash "$REPO_ROOT/tools/git-hooks/pre-commit-security.sh" || exit 1
[ -x "$REPO_ROOT/.git/hooks/pre-commit.pre-security" ] && exec "$REPO_ROOT/.git/hooks/pre-commit.pre-security"
exit 0
EOF
chmod +x "$HOOK" "$SRC"
echo "installed: $HOOK -> $SRC"
