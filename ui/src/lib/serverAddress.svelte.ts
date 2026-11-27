import { browserUrl, isLoopbackBind, reachedOrigin } from './serverinfo'
import type { ServerInfo } from './types'

/** Which address a server is reached at, and the base URL the connect
 * snippets are built on, from what the server says about its bind. */
export function serverAddress(info: () => ServerInfo | null) {
  /** The interface address the connect snippets are written against. */
  let selected = $state('')
  // A wildcard bind answers on every interface; a bind to one address answers
  // on that address only, so it is the only one worth offering
  const addresses = $derived.by(() => {
    const i = info()
    if (!i) return []
    if (i.wildcard_bind || isLoopbackBind(i.bind_host)) return i.addresses
    const named = i.addresses.find((a) => a.address === i.bind_host)
    return [
      named ?? {
        address: i.bind_host,
        family: i.bind_host.includes(':') ? 'IPv6' : 'IPv4',
        interface: '',
      },
    ]
  })
  const address = $derived(
    addresses.find((a) => a.address === selected)?.address ??
      addresses[0]?.address ??
      '',
  )
  const loopback = $derived.by(() => {
    const i = info()
    return i ? isLoopbackBind(i.bind_host) : false
  })
  // A loopback bind read from a page that is not on loopback: the reader got
  // here through a proxy or tunnel, so the page's own origin is the address
  // that works, and rebinding (plaintext on every interface) is never advised
  const origin = reachedOrigin()
  const proxied = $derived(loopback && origin !== null)
  /** The base URL the connect snippets are built on; empty when there is
   * none worth offering (a loopback page on a loopback bind, no address). */
  const base = $derived(
    proxied
      ? (origin ?? '')
      : loopback || !address
        ? ''
        : browserUrl(address, info()?.port ?? 0),
  )

  return {
    get addresses() {
      return addresses
    },
    get selected() {
      return selected
    },
    set selected(value: string) {
      selected = value
    },
    get address() {
      return address
    },
    get loopback() {
      return loopback
    },
    get origin() {
      return origin
    },
    get proxied() {
      return proxied
    },
    get base() {
      return base
    },
  }
}
