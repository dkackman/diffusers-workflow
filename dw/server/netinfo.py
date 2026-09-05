"""Which addresses this machine can be reached at.

The Server page answers one question - "what URL do I give the other
machine?" - and the only part of the answer the server itself knows is the
list of addresses its interfaces carry. URL composition stays in the UI;
this module only enumerates.

Two enumeration methods, in order: psutil when it happens to be installed
(it is a transitive dependency here, not a declared one, so it is used
opportunistically and never required), and the stdlib otherwise. Neither
raises: an address list is a convenience, and a machine whose interfaces
cannot be read still has a working server, so failure yields [].
"""

import socket
import ipaddress

__all__ = ["local_addresses"]


def _usable(address):
    """A candidate address, normalized, or None if it is not worth showing.

    Loopback is dropped (it names this machine only - the page's whole
    point is the other machine) and so is link-local (169.254.x / fe80::,
    which needs a scope id to be usable at all). Docker and veth addresses
    survive on purpose: on a box that runs containers they are real
    routes, and guessing which of a machine's networks the user meant is
    not the server's job.
    """
    if not address:
        return None
    # getaddrinfo and psutil both hand back scoped IPv6 ('fe80::1%eth0')
    address = address.split("%")[0].strip()
    try:
        parsed = ipaddress.ip_address(address)
    except ValueError:
        return None
    if parsed.is_loopback or parsed.is_link_local or parsed.is_unspecified:
        return None
    return str(parsed), "IPv4" if parsed.version == 4 else "IPv6"


def _psutil_addresses():
    """(address, family, interface) triples from psutil, which is the only
    method that can name the interface an address belongs to."""
    import psutil

    found = []
    for interface, entries in psutil.net_if_addrs().items():
        for entry in entries:
            if entry.family not in (socket.AF_INET, socket.AF_INET6):
                continue
            usable = _usable(entry.address)
            if usable is not None:
                found.append((usable[0], usable[1], interface))
    return found


def _outbound_address():
    """The address a packet to the outside world would leave from.

    A UDP socket's connect() only sets the peer for later sends; it does
    no handshake and puts nothing on the wire. It exists because
    getaddrinfo(gethostname()) is silent on machines whose hostname
    resolves to 127.0.1.1 - a Debian default - and that is exactly the
    machine a remote-access page is being read on.
    """
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        sock.connect(("192.0.2.1", 9))  # TEST-NET-1, routed nowhere
        return sock.getsockname()[0]
    finally:
        sock.close()


def _stdlib_addresses():
    """The psutil-free fallback: whatever this host's name resolves to,
    plus the primary outbound address. No interface names are available
    this way, so they come back None."""
    found = []
    try:
        infos = socket.getaddrinfo(socket.gethostname(), None)
    except OSError:
        infos = []
    for family, _type, _proto, _canonical, sockaddr in infos:
        if family not in (socket.AF_INET, socket.AF_INET6):
            continue
        usable = _usable(sockaddr[0])
        if usable is not None:
            found.append((usable[0], usable[1], None))
    try:
        usable = _usable(_outbound_address())
    except OSError:
        usable = None
    if usable is not None:
        found.append((usable[0], usable[1], None))
    return found


def local_addresses():
    """This machine's non-loopback addresses, IPv4 first, as
    [{"address", "family", "interface"}] - interface is None when the
    method that found it cannot say. Never raises; [] on failure."""
    found = []
    try:
        found = _psutil_addresses()
    except Exception:
        found = []
    if not found:
        try:
            found = _stdlib_addresses()
        except Exception:
            found = []
    seen = set()
    unique = []
    for address, family, interface in found:
        if address in seen:
            continue
        seen.add(address)
        unique.append({"address": address, "family": family, "interface": interface})
    # IPv4 before IPv6, then by address so the list is stable between calls
    unique.sort(key=lambda entry: (entry["family"] != "IPv4", entry["address"]))
    return unique
