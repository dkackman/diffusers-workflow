# Running dw-serve as a systemd service

For a Linux GPU box that should serve `dw` to the rest of your network
without anyone keeping a terminal open. The unit file assumes a user `dw`
with the repository checked out and installed (`bash ./install.sh`) at
`/home/dw/diffusers-workflow`; edit `User`, `Group`, `WorkingDirectory` and
`ExecStart` if yours differ.

    sudo cp dw-serve.service /etc/systemd/system/
    sudo cp dw-serve.env.example /etc/dw-serve.env
    sudo chmod 600 /etc/dw-serve.env           # it holds the API token
    sudo $EDITOR /etc/dw-serve.env             # set DW_API_TOKEN at least
    sudo systemctl daemon-reload
    sudo systemctl enable --now dw-serve
    systemctl status dw-serve
    journalctl -u dw-serve -f                  # logs

Then, from another machine on the LAN, `curl http://<box>:8765/api/health
-H "Authorization: Bearer <token>"` should answer with the box's hostname.
The full client-side setup is in [docs/REMOTE.md](../../docs/REMOTE.md).

Updating: `git pull && bash ./install.sh` in the checkout, then
`sudo systemctl restart dw-serve`. (The web UI's Models page can update
`diffusers` alone without a restart.)

macOS as the server: there is no unit here; a `launchd` plist with the same
`ExecStart` and environment does the job, or run it under `tmux`.
