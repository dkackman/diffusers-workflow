#!/usr/bin/env bash
# Switch this host between the container and the native systemd dw.serve.
# They share the port, the GPUs and the data directories, so only one runs.
#
#   bash docker/switch.sh docker     # stop the systemd unit, start the container
#   bash docker/switch.sh native     # stop the container, start the systemd unit
#   bash docker/switch.sh status
#
# The unit is the user unit scripts/dw-serve.service installs; set
# DW_SYSTEMD_UNIT for another name, or DW_SYSTEMCTL="sudo systemctl" for the
# system unit in contrib/systemd/.
set -euo pipefail
cd "$(dirname "$0")"

unit="${DW_SYSTEMD_UNIT:-dw-serve}"
read -r -a systemctl <<< "${DW_SYSTEMCTL:-systemctl --user}"
compose=(docker compose -f compose.yaml)

case "${1:-status}" in
  docker)
    "${systemctl[@]}" stop "$unit" || true
    "${compose[@]}" up -d
    echo "container started; the login link: ${compose[*]} exec dw python /opt/dw/entrypoint.py login-url"
    ;;
  native)
    "${compose[@]}" down
    "${systemctl[@]}" start "$unit"
    "${systemctl[@]}" --no-pager status "$unit" | head -5
    ;;
  status)
    echo "native: $("${systemctl[@]}" is-active "$unit" || true)"
    "${compose[@]}" ps
    ;;
  *)
    echo "usage: $0 docker|native|status" >&2
    exit 2
    ;;
esac
