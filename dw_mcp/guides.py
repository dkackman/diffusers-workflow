"""The prose guides, read from the engine an agent is about to drive.

These proxy `GET /api/guides` like every other handler proxies its route.
They used to read the markdown shipped in this package, on the theory that
documentation works the same against a remote engine as a local one. It
does not: an MCP at one version against a `dw.serve` at another confidently
indexed sections - `Speech Generation`, `templates/` - the server did not
have, and the guides an agent reads have to describe the engine that will
run what it authors. The one thing given up is answering `list_guides`
while the server is down, which is not a real use.

The `GUIDES` table, section extraction and the checkout-else-packaged file
resolution now live in `dw/server/guides.py`.
"""

from dw_mcp.client import api_path


def list_guides(client):
    """Every guide the engine serves, with what it covers and the sections
    it holds. The section headings are the routing table."""
    return client.get_json("/api/guides")


def get_guide(client, name, section=None):
    """One guide, whole or one section of it. A section name is matched
    loosely on the server, so a heading copied approximately resolves."""
    params = {"section": section} if section is not None else None
    return client.get_json(api_path("api", "guides", name), params=params)
