#!/usr/bin/env bash
# The CI's quick checks, run locally before a push (the jobs format-cpp, lint-cpp, tools' ruff and mypy, docs and
# license-audit of .github/workflows/ci.yml). The tests (ctest, colcon test) and the sanitizer and coverage builds
# are not run here: they take long, and they are run with the builds.
#
#   scripts/check_before_push.sh                  # check what the commits not yet pushed change
#   scripts/check_before_push.sh --all            # clang-tidy on every source of the core
#   scripts/check_before_push.sh --install-hook   # run it before every `git push` (pre-push hook)
#
# clang-tidy is the slow one (a minute per source of the core): by default it checks the sources the unpushed commits
# (and the working tree) change, and every source when a header of the core changed. Its compile database is a
# configured superbuild in $STRATASLAM_TIDY_BUILD (default ~/.cache/strataslam_tidy), made on the first run; the
# first configure downloads small_gicp, as the build does.
# Needs: clang-format-18, clang-tidy (run-clang-tidy), ruff 0.17.3 and mypy 1.9.0 (strataslam_tools' dev extra).
# A push can skip it with `git push --no-verify`.
set -uo pipefail

repo="$(cd "$(dirname "$0")/.." && pwd)"
cd "$repo"
tidy_build="${STRATASLAM_TIDY_BUILD:-$HOME/.cache/strataslam_tidy}"
all=0
case "${1:-}" in
  --install-hook)
    hook="$(git rev-parse --git-path hooks)/pre-push"
    printf '#!/usr/bin/env bash\nexec "%s/scripts/check_before_push.sh"\n' "$repo" > "$hook"
    chmod +x "$hook"
    echo "pre-push hook installed: $hook (skip it once with git push --no-verify)"
    exit 0 ;;
  --all) all=1 ;;
  "") ;;
  *) sed -n '2,17p' "$0"; exit 2 ;;
esac

missing=()
command -v clang-format-18 >/dev/null || missing+=("clang-format-18 (sudo apt install clang-format-18)")
command -v run-clang-tidy >/dev/null || missing+=("clang-tidy (sudo apt install clang-tidy)")
command -v ruff >/dev/null || missing+=("ruff 0.17.3 (pip install --user --break-system-packages ruff==0.17.3)")
python3 -m mypy --version >/dev/null 2>&1 || missing+=("mypy 1.9.0 (pip install --user --break-system-packages mypy==1.9.0)")
if ((${#missing[@]})); then
  printf 'check_before_push: missing\n'; printf '  %s\n' "${missing[@]}"
  exit 1
fi

failed=()
step() {  # step <name> <command...>: runs it, reports, remembers a failure
  local name="$1"; shift
  printf '== %s\n' "$name"
  if "$@"; then printf '   ok\n'; else failed+=("$name"); printf '   FAILED\n'; fi
}

format_cpp() { git ls-files -z '*.cpp' '*.hpp' | xargs -0 clang-format-18 --dry-run --Werror; }

lint_cpp() {
  # The core's sources to check: those changed since the upstream branch (unpushed commits and the working tree),
  # or all of them when a header of the core changed (or with --all, or without an upstream).
  local base changed files
  base="$(git merge-base HEAD '@{upstream}' 2>/dev/null)" || all=1
  if ((all == 0)); then
    changed="$( { git diff --name-only "$base" HEAD; git diff --name-only HEAD; } | sort -u)"
    if grep -q '^strataslam_core/include/' <<<"$changed"; then
      all=1
    else
      files="$(grep -E '^strataslam_core/src/.*\.cpp$' <<<"$changed" | while read -r f; do [[ -f $f ]] && echo "$f"; done)"
      if [[ -z "$files" ]]; then echo "   no source of the core changed"; return 0; fi
    fi
  fi
  if [[ ! -f "$tidy_build/compile_commands.json" ]]; then
    echo "   configuring the compile database in $tidy_build (once)"
    # ROS Jazzy's GTSAM when there is one (Ubuntu's lacks libCppUnitLite.a, ADR-0007), as scripts/dev_env.sh.
    local gtsam=()
    if [[ -z "${GTSAM_DIR:-}" && -d /opt/ros/jazzy/lib/cmake/GTSAM ]]; then
      gtsam=(-DGTSAM_DIR=/opt/ros/jazzy/lib/cmake/GTSAM)
    fi
    cmake -S superbuild -B "$tidy_build" -DCMAKE_EXPORT_COMPILE_COMMANDS=ON "${gtsam[@]}" > "$tidy_build.log" 2>&1 \
      || { echo "   cmake failed: see $tidy_build.log"; return 1; }
  fi
  if ((all)); then
    run-clang-tidy -p "$tidy_build" -quiet 'strataslam_core/src/.*\.cpp$'
  else
    echo "   $(wc -w <<<"$files") changed source(s)"
    # shellcheck disable=SC2086 # one path per word
    run-clang-tidy -p "$tidy_build" -quiet $(sed 's|^|'"$repo"'/|; s|$|$|' <<<"$files")
  fi
}

ruff_check() { ruff check .; }
mypy_check() { (cd strataslam_tools && python3 -m mypy strataslam_tools); }
docs_links() { python3 scripts/check_links.py; }
license() { python3 scripts/license_audit.py; }

step "format-cpp (clang-format-18)" format_cpp
step "lint-cpp (clang-tidy)" lint_cpp
step "tools: ruff" ruff_check
step "tools: mypy" mypy_check
step "docs: links" docs_links
step "license-audit" license

if ((${#failed[@]})); then
  printf '\ncheck_before_push: %d check(s) failed: %s\n' "${#failed[@]}" "$(IFS=,; echo "${failed[*]}")"
  printf 'The push would fail in CI. Fix them, or push anyway with git push --no-verify.\n'
  exit 1
fi
printf '\ncheck_before_push: all checks passed\n'
