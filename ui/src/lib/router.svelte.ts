/** Minimal hash router: '#/jobs/abc' -> ['jobs', 'abc'] */
function decode(part: string): string {
  // A hand-edited or truncated hash can hold a stray '%', which
  // decodeURIComponent throws on - at module load that would blank the whole
  // app, so an undecodable segment is used as it was typed
  try {
    return decodeURIComponent(part)
  } catch {
    return part
  }
}

function parse(): string[] {
  const hash = location.hash.replace(/^#\/?/, '')
  return hash ? hash.split('/').map(decode) : ['workflows']
}

export const route = $state({ parts: parse() })

window.addEventListener('hashchange', () => {
  route.parts = parse()
})

export function go(...parts: string[]) {
  location.hash = '/' + parts.map(encodeURIComponent).join('/')
}
