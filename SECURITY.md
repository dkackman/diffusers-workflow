# Security policy

## Reporting a vulnerability

Please don't open a public issue for a security problem: the tracker is
public, and an issue publishes the way in before a fix exists.

Report it privately instead, through GitHub's
[Report a vulnerability](https://github.com/dkackman/diffusers-workflow/security/advisories/new)
form (the repository's **Security** tab, then **Advisories**). Only the
maintainer sees it. Include what you ran, what happened, and the version
(`dw --version`, or `get_server_info` over MCP).

Examples of what counts:
- a workflow that runs code or loads a class outside what an untrusted
  workflow may;
- a path, URL or reference that reads or writes outside the directories the
  server works in;
- a route or tool that discloses a secret, a server path or whether a file
  exists;
- a destructive call that runs without its acknowledgement.

## Supported versions

Fixes go into the next release. The latest release is the one supported.
