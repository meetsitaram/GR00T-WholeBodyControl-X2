#!/usr/bin/env bash
# Pre-commit security gate: blocks commits whose STAGED changes contain
# emails, credentials, tokens, or account identifiers.
# Install (local, per-clone):  bash tools/git-hooks/install.sh
# Override for a deliberate exception:  SKIP_SECURITY_CHECK=1 git commit ...
#
# Extra private patterns (your own emails/accounts/tenants) go in
# .git/security-deny-patterns — one extended-regex per line. That file lives
# under .git/ so it can never itself be committed.
set -uo pipefail
[ "${SKIP_SECURITY_CHECK:-0}" = "1" ] && exit 0

# Only scan ADDED lines of staged text changes.
# Exclude this hook's own directory: the pattern list below is literally
# made of credential-shaped regexes, so scanning it flags itself and the
# hook could never be committed or updated.
STAGED=$(git diff --cached --unified=0 --no-color \
           -- . ':(exclude)tools/git-hooks/*' \
         | grep -E '^\+' | grep -vE '^\+\+\+' || true)
[ -z "$STAGED" ] && exit 0

PATTERNS=(
  # emails (allowlist below)
  '[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}'
  # generic credentials
  '(password|passwd|pwd)[[:space:]]*[:=][[:space:]]*[^[:space:]]{4,}'
  '(api[_-]?key|apikey|secret[_-]?key|access[_-]?key)[[:space:]]*[:=]'
  'Authorization:[[:space:]]*(Bearer|Basic)[[:space:]]+[A-Za-z0-9._~+/=-]{16,}'
  '-----BEGIN[[:space:]].*PRIVATE KEY-----'
  # provider tokens
  'gh[pousr]_[A-Za-z0-9]{20,}'                 # GitHub
  'AKIA[0-9A-Z]{16}'                           # AWS access key
  'sk-[A-Za-z0-9]{20,}'                        # generic sk- API keys
  'xox[baprs]-[A-Za-z0-9-]{10,}'               # Slack
  'AccountKey=[A-Za-z0-9+/=]{40,}'             # Azure storage key
  '\bsig=[A-Za-z0-9%+/=]{30,}'                 # Azure SAS token
  'eyJ[A-Za-z0-9_-]{20,}\.eyJ[A-Za-z0-9_-]{20,}'  # JWT
  # machine credentials files
  'machine[[:space:]]+[^[:space:]]+[[:space:]]+login[[:space:]]+[^[:space:]]+[[:space:]]+password'  # .netrc
  'x-access-token:[^@[:space:]]+@'             # embedded git credential URL
)

# emails that are fine to commit
EMAIL_ALLOW='noreply@anthropic\.com|users\.noreply\.github\.com|@example\.(com|org)|(support|info|sales)@[a-z]+\.'

HITS=""
for p in "${PATTERNS[@]}"; do
  M=$(echo "$STAGED" | grep -EIn -e "$p" || true)
  if [ -n "$M" ] && [[ "$p" == *'@[A-Za-z0-9.-]'* ]]; then
    M=$(echo "$M" | grep -Ev "$EMAIL_ALLOW" || true)
  fi
  [ -n "$M" ] && HITS+="pattern: $p"$'\n'"$M"$'\n\n'
done

# private per-clone deny patterns (specific emails/accounts/tenants)
DENY_FILE="$(git rev-parse --git-dir)/security-deny-patterns"
if [ -f "$DENY_FILE" ]; then
  while IFS= read -r p; do
    [ -z "$p" ] || [[ "$p" == \#* ]] && continue
    M=$(echo "$STAGED" | grep -EIn -e "$p" || true)
    [ -n "$M" ] && HITS+="private deny-pattern matched (see .git/security-deny-patterns)"$'\n'"$M"$'\n\n'
  done < "$DENY_FILE"
fi

if [ -n "$HITS" ]; then
  echo "COMMIT BLOCKED — staged changes contain sensitive-looking content:" >&2
  echo >&2
  echo "$HITS" | head -40 >&2
  echo "Fix the staged content, or for a deliberate exception:" >&2
  echo "  SKIP_SECURITY_CHECK=1 git commit ..." >&2
  exit 1
fi
exit 0
