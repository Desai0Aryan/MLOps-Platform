#!/usr/bin/env bash
# Deploys an immutable image tag with docker compose and rolls back automatically if it does not become healthy.
#
#   scripts/deploy.sh sha-3f2c9ab        deploy this tag (previous tag is remembered for rollback)
#   scripts/deploy.sh --rollback         go back to the previously deployed tag
#
# Environment (all optional):
#   COMPOSE_DIR      directory holding docker-compose.yml + docker-compose.prod.yml + .env   (default: parent of this script)
#   HEALTH_URL       readiness endpoint                                                     (default: http://127.0.0.1:8000/ready)
#   HEALTH_TIMEOUT   seconds to wait for health                                             (default: 240)
#   POLL_SECONDS     seconds between health probes                                          (default: 3)
#   HEALTH_CMD       custom health command (used by the tests)
#   DOCKER           docker binary                                                          (default: docker)
#
# NOTE: this rolls back the *code* (images). The *model* is versioned separately in the registry;
#       to revert a bad model use `python -m mlops.cli registry rollback`.
set -euo pipefail

DOCKER="${DOCKER:-docker}"
COMPOSE_DIR="${COMPOSE_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
HEALTH_URL="${HEALTH_URL:-http://127.0.0.1:8000/ready}"
HEALTH_TIMEOUT="${HEALTH_TIMEOUT:-240}"
POLL_SECONDS="${POLL_SECONDS:-3}"

log() { printf '[deploy %s] %s\n' "$(date -u +%H:%M:%S)" "$*"; }

compose() {  # compose <tag> <args...>
  local tag="$1"; shift
  MLOPS_VERSION="$tag" "$DOCKER" compose -f docker-compose.yml -f docker-compose.prod.yml "$@"
}

healthy() {
  if [ -n "${HEALTH_CMD:-}" ]; then bash -c "$HEALTH_CMD"; else curl -fsS --max-time 3 "$HEALTH_URL" >/dev/null; fi
}

wait_healthy() {
  local deadline=$((SECONDS + HEALTH_TIMEOUT))
  while [ "$SECONDS" -lt "$deadline" ]; do
    if healthy; then return 0; fi
    sleep "$POLL_SECONDS"
  done
  return 1
}

release() {  # release <tag>: pull first, so a missing/forbidden image fails BEFORE anything is stopped
  compose "$1" pull api worker mlflow bootstrap
  compose "$1" up -d --remove-orphans
}

main() {
  if [ "$#" -ne 1 ]; then echo "usage: $0 <image-tag> | --rollback" >&2; exit 2; fi
  cd "$COMPOSE_DIR"
  local mode=deploy target current
  current="$(cat .deployed_tag 2>/dev/null || true)"
  if [ "$1" = "--rollback" ]; then
    mode=rollback
    target="$(cat .previous_tag 2>/dev/null || true)"
    if [ -z "$target" ]; then log "no previous tag recorded: nothing to roll back to"; exit 1; fi
  else
    target="$1"
  fi

  log "$mode: ${current:-<none>} -> $target"
  if release "$target" && wait_healthy; then
    if [ -n "$current" ] && [ "$current" != "$target" ]; then echo "$current" > .previous_tag; fi
    echo "$target" > .deployed_tag
    "$DOCKER" image prune -f >/dev/null 2>&1 || true
    log "SUCCESS: $target is live"
    exit 0
  fi

  log "FAILED: $target did not become healthy within ${HEALTH_TIMEOUT}s"
  if [ "$mode" = deploy ] && [ -n "$current" ] && [ "$current" != "$target" ]; then
    log "rolling back to $current"
    if release "$current" && wait_healthy; then
      log "ROLLED BACK: $current is live again (.deployed_tag unchanged)"
    else
      log "ROLLBACK ALSO FAILED: manual intervention required"
    fi
  else
    log "no previous release to fall back to"
  fi
  exit 1
}

main "$@"
