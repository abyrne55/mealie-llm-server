#!/usr/bin/env bash
set -euo pipefail

CONTAINER_NAME="mealie-test"
IMAGE="ghcr.io/abyrne55/mealie-hummingbird:v3"
PORT=9797
MEALIE_URL="http://127.0.0.1:${PORT}"
ENV_FILE=".env.test"

DEFAULT_EMAIL="changeme@example.com"
DEFAULT_PASS="MyPassword"

TOKEN=""

check_deps() {
  local missing=()
  for cmd in podman curl jq; do
    command -v "$cmd" &>/dev/null || missing+=("$cmd")
  done
  if [[ ${#missing[@]} -gt 0 ]]; then
    echo "Missing required tools: ${missing[*]}" >&2
    exit 1
  fi
}

login() {
  curl -sf -X POST "$MEALIE_URL/api/auth/token" \
    --data-urlencode "username=$1" \
    --data-urlencode "password=$2" \
    | jq -r '.access_token'
}

api() {
  curl -sf -H "Authorization: Bearer $TOKEN" -H "Content-Type: application/json" "$@"
}

print_env() {
  cat > "$ENV_FILE" <<EOF
MEALIE_URL=${MEALIE_URL}
MEALIE_API_KEY=${TOKEN}
EOF

  echo ""
  echo "export MEALIE_URL=${MEALIE_URL}"
  echo "export MEALIE_API_KEY=${TOKEN}"
  echo ""
  echo "Cleanup: podman stop $CONTAINER_NAME && podman rm $CONTAINER_NAME"
}

cleanup_hint() {
  echo "" >&2
  echo "Setup failed. Clean up with: podman stop $CONTAINER_NAME && podman rm $CONTAINER_NAME" >&2
}
trap cleanup_hint ERR

handle_existing_container() {
  local state
  state=$(podman inspect --format '{{.State.Status}}' "$CONTAINER_NAME" 2>/dev/null || echo "missing")

  case "$state" in
    running)
      if [[ -f "$ENV_FILE" ]]; then
        local existing_key
        existing_key=$(grep '^MEALIE_API_KEY=' "$ENV_FILE" | cut -d= -f2-)
        if [[ -n "$existing_key" ]]; then
          TOKEN="$existing_key"
          if api "$MEALIE_URL/api/users/self" -o /dev/null 2>/dev/null; then
            echo "Container '$CONTAINER_NAME' is already running with valid credentials."
            print_env
            exit 0
          fi
        fi
      fi
      echo "Container '$CONTAINER_NAME' is running but credentials are invalid. Recreating..."
      podman stop "$CONTAINER_NAME"
      podman rm "$CONTAINER_NAME"
      ;;
    missing)
      ;;
    *)
      echo "Removing stopped container '$CONTAINER_NAME'..."
      podman rm "$CONTAINER_NAME"
      ;;
  esac
}

start_container() {
  echo "Starting Mealie container '$CONTAINER_NAME'..."
  podman run -d \
    --name "$CONTAINER_NAME" \
    -p "${PORT}:9000" \
    -e MAX_WORKERS=1 \
    -e WEB_CONCURRENCY=1 \
    "$IMAGE"
}

wait_for_ready() {
  echo "Waiting for Mealie to become ready..."
  local elapsed=0
  local timeout=120
  while ! curl -sf "$MEALIE_URL/api/app/about" -o /dev/null 2>/dev/null; do
    if [[ $elapsed -ge $timeout ]]; then
      echo "Timed out waiting for Mealie after ${timeout}s" >&2
      exit 1
    fi
    sleep 2
    elapsed=$((elapsed + 2))
  done
  echo "Mealie is ready."
}

seed() {
  echo "Logging in with default credentials..."
  TOKEN=$(login "$DEFAULT_EMAIL" "$DEFAULT_PASS")

  echo "Creating long-lived API token..."
  TOKEN=$(api -X POST "$MEALIE_URL/api/users/api-tokens" \
    -d '{"name": "mealie-llm-server-test"}' \
    | jq -r '.token')

  local count
  count=$(api "$MEALIE_URL/api/foods?per_page=1" | jq '.total')
  if [[ "$count" -eq 0 ]]; then
    echo "Seeding foods..."
    api -o /dev/null -X POST "$MEALIE_URL/api/groups/seeders/foods" -d '{"locale": "en-US"}'
  else
    echo "Foods already populated ($count items); skipping"
  fi

  count=$(api "$MEALIE_URL/api/units?per_page=1" | jq '.total')
  if [[ "$count" -eq 0 ]]; then
    echo "Seeding units..."
    api -o /dev/null -X POST "$MEALIE_URL/api/groups/seeders/units" -d '{"locale": "en-US"}'
  else
    echo "Units already populated ($count items); skipping"
  fi

  count=$(api "$MEALIE_URL/api/groups/labels?per_page=1" | jq '.total')
  if [[ "$count" -eq 0 ]]; then
    echo "Seeding labels..."
    api -o /dev/null -X POST "$MEALIE_URL/api/groups/seeders/labels" -d '{"locale": "en-US"}'
  else
    echo "Labels already populated ($count items); skipping"
  fi
}

check_deps
handle_existing_container
start_container
wait_for_ready
seed
print_env
echo "Done."
