// Monaco loads its editor core, its contributions and the JSON language
// only (monaco.ts, monacoContribs.ts): the contribution list follows the
// installed package's own, and nothing else the whole package carries -
// eighty language definitions, the CSS/HTML/TypeScript services, the LSP
// client - is imported
import { describe, expect, it } from 'vitest'
import packageIndex from '../../node_modules/monaco-editor/esm/vs/index.js?raw'
import monacoSource from './monaco.ts?raw'
import contribsSource from './monacoContribs.ts?raw'

const SOURCES: Record<string, string> = {
  'monaco.ts': monacoSource,
  'monacoContribs.ts': contribsSource,
}
const read = (file: string) => SOURCES[file]
// A stylesheet is imported by its file path (the exports map has no route
// to one); it names the same file as its package specifier
const CSS_PATH = '../../node_modules/monaco-editor/esm/vs/'

/** CONTRIBS as monacoContribs.ts spells it: the module itself cannot be
 * imported here, since each contribution needs a browser */
function contribs(): string[] {
  const list = read('monacoContribs.ts').split(
    'CONTRIBS: readonly string[] =',
  )[1]
  return [...list.matchAll(/'([^']+)'/g)].map((m) => m[1])
}

/** Every module specifier `source` imports, statically or dynamically */
function imports(source: string): string[] {
  const found = source.matchAll(
    /(?:^import\s+(?:[^'"]*?\s+from\s+)?|import\()\s*'([^']+)'/gm,
  )
  return [...found].map((m) => m[1].replace(CSS_PATH, 'monaco-editor/'))
}

/** The contributions the whole package imports, as `monaco-editor/...` */
function packageContribs(): string[] {
  return [...packageIndex.matchAll(/^import '\.\/([^']+)';$/gm)]
    .map((m) => m[1])
    .filter((path) => !path.startsWith('languages/'))
    .map((path) => `monaco-editor/${path}`)
}

describe('monaco contributions', () => {
  it('every contribution the package lists is imported', () => {
    const listed = packageContribs()
    const CONTRIBS = contribs()
    expect(listed.length).toBeGreaterThan(50)
    const imported = imports(read('monacoContribs.ts'))
    // a missing or extra specifier is named before the order is compared
    expect({
      missing: listed.filter((s) => !CONTRIBS.includes(s)),
      extra: CONTRIBS.filter((s) => !listed.includes(s)),
    }).toEqual({ missing: [], extra: [] })
    expect(CONTRIBS).toEqual(listed)
    expect(imported).toEqual(listed)
  })

  it('no language definition, LSP client or other language feature is imported', () => {
    const specifiers = [
      ...imports(read('monaco.ts')),
      ...imports(read('monacoContribs.ts')),
      ...contribs(),
    ].filter((s) => s.startsWith('monaco-editor'))
    const forbidden = specifiers.filter(
      (s) =>
        s === 'monaco-editor' ||
        /editor\.main|monaco-lsp-client|basic-languages|languages\/definitions\//.test(
          s,
        ) ||
        (/languages\/features\//.test(s) && !/\/features\/json\//.test(s)),
    )
    expect(forbidden).toEqual([])
    // monaco.ts takes the core, the JSON language and the two workers
    expect(
      imports(read('monaco.ts')).filter((s) => s.startsWith('monaco-editor')),
    ).toEqual([
      'monaco-editor/editor/editor.api.js',
      'monaco-editor/languages/features/json/register.js',
      'monaco-editor/editor/editor.worker.js?worker',
      'monaco-editor/language/json/json.worker.js?worker',
    ])
    expect(imports(read('monaco.ts'))).toContain('./monacoContribs')
  })
})
