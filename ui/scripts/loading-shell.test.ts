// The loading shell in index.html paints before the bundle arrives, so it
// repeats app.css's colours and the theme script main.ts used to hold. These
// tests keep the copy from drifting.
import { readFileSync } from 'node:fs'
import { dirname, join } from 'node:path'
import { fileURLToPath } from 'node:url'
import { beforeEach, describe, expect, it } from 'vitest'

const UI = join(dirname(fileURLToPath(import.meta.url)), '..')
const read = (p: string) => readFileSync(join(UI, p), 'utf8')
const html = read('index.html')
const css = read('src/app.css')

const inline = (tag: 'style' | 'script'): string => {
  const m = html.match(
    new RegExp(`<${tag}(?![^>]*\\bsrc=)[^>]*>([\\s\\S]*?)</${tag}>`),
  )
  if (!m) throw new Error(`index.html has no inline <${tag}>`)
  return m[1]
}

// The declarations of the first rule whose selector (whitespace-collapsed)
// contains `selector`, inside `scope` when given
const declarations = (source: string, selector: string, scope?: string) => {
  const body = scope ? source.slice(source.indexOf(scope)) : source
  const at = body.indexOf(selector)
  if (at < 0) throw new Error(`no rule for ${selector}`)
  const open = body.indexOf('{', at)
  const block = body.slice(open + 1, body.indexOf('}', open))
  const out: Record<string, string> = {}
  for (const m of block.matchAll(/(--[a-z-]+)\s*:\s*([^;]+);/g))
    out[m[1]] = m[2].trim()
  return out
}

describe('the shell colours', () => {
  const style = inline('style')
  const cases: [string, string, string | undefined, string | undefined][] = [
    ['light :root', ':root {', undefined, undefined],
    [
      'dark by preference',
      ":root:not([data-theme='light'])",
      '@media (prefers-color-scheme: dark)',
      '@media (prefers-color-scheme: dark)',
    ],
    ['dark by choice', ":root[data-theme='dark']", undefined, undefined],
    ['light by choice', ":root[data-theme='light']", undefined, undefined],
  ]
  it("the shell's colours are app.css's, in both themes", () => {
    for (const [, selector, cssScope, shellScope] of cases) {
      const want = declarations(css, selector, cssScope)
      const got = declarations(style, selector, shellScope)
      for (const name of ['--bg', '--ink', '--muted']) {
        expect(got[name], `${selector} ${name}`).toBe(want[name])
      }
    }
  })
})

describe('the theme script', () => {
  const script = inline('script')
  const run = (storage: unknown) => {
    new Function('localStorage', script)(storage)
  }
  beforeEach(() => {
    delete document.documentElement.dataset.theme
  })

  it('applies a stored choice', () => {
    for (const choice of ['light', 'dark']) {
      delete document.documentElement.dataset.theme
      run({ getItem: () => choice })
      expect(document.documentElement.dataset.theme).toBe(choice)
    }
  })

  it('ignores anything else', () => {
    for (const value of [null, '', 'system', 'blue']) {
      run({ getItem: () => value })
      expect(document.documentElement.dataset.theme).toBeUndefined()
    }
  })

  it('survives storage that throws', () => {
    expect(() =>
      run({
        getItem: () => {
          throw new Error('blocked')
        },
      }),
    ).not.toThrow()
    expect(document.documentElement.dataset.theme).toBeUndefined()
  })

  it('reads the key theme.ts writes', () => {
    expect(script).toContain("'dw-theme'")
    expect(read('src/lib/theme.ts')).toContain(
      "localStorage.setItem('dw-theme'",
    )
  })
})

describe('the shell text', () => {
  it('says the app did not load after 10 s', () => {
    const style = inline('style')
    expect(
      html.replace(/\s+/g, ' ').replace(/ ?<\/?span[^>]*> ?/g, ''),
    ).toContain(
      "dw didn't load. Reload the page; if that doesn't help, check that the server is running.",
    )
    expect(html).toContain('>Loading dw…<')
    expect(style).toMatch(/#boot \.failed \{[^}]*animation:[^;}]*\b10s\b/)
    expect(style).toMatch(/#boot \.loading \{[^}]*animation:[^;}]*\b10s\b/)
  })

  it('main clears the shell before mounting', () => {
    const main = read('src/main.ts')
    expect(main.indexOf('replaceChildren')).toBeGreaterThan(-1)
    expect(main.indexOf('replaceChildren')).toBeLessThan(
      main.indexOf('mount(App'),
    )
    expect(main).not.toContain('dw-theme')
  })
})
