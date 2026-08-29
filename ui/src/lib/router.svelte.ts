/** Minimal hash router: '#/jobs/abc' -> ['jobs', 'abc'] */
function parse(): string[] {
  const hash = location.hash.replace(/^#\/?/, '')
  return hash ? hash.split('/').map(decodeURIComponent) : ['workflows']
}

export const route = $state({ parts: parse() })

window.addEventListener('hashchange', () => {
  route.parts = parse()
})

export function go(...parts: string[]) {
  location.hash = '/' + parts.map(encodeURIComponent).join('/')
}
