/** Monaco, loaded lazily: this module is only imported dynamically from
 * JsonEditor, so the editor lands in its own chunk and the main bundle
 * stays small. The workflow schema is wired into the JSON language service,
 * which is the whole point - validation, completion and hover come from
 * dw/workflow_schema.json's own descriptions. */
import * as monaco from 'monaco-editor'
// monaco 0.56 maps 'monaco-editor/*' onto its esm/vs tree via package
// exports; the JSON language service moved to languages/features/json
import { jsonDefaults } from 'monaco-editor/languages/features/json/register.js'
import editorWorker from 'monaco-editor/editor/editor.worker.js?worker'
import jsonWorker from 'monaco-editor/language/json/json.worker.js?worker'

self.MonacoEnvironment = {
  getWorker(_workerId: string, label: string) {
    return label === 'json' ? new jsonWorker() : new editorWorker()
  },
}

let configured = false

export async function setupMonaco() {
  if (configured) return monaco
  configured = true

  try {
    const { api } = await import('./api')
    const schema = await api.getSchema()
    jsonDefaults.setDiagnosticsOptions({
      validate: true,
      enableSchemaRequest: false,
      schemas: [{ uri: 'dw://workflow-schema', fileMatch: ['*'], schema }],
    })
  } catch {
    /* schema endpoint unreachable - plain JSON editing still works */
  }

  const styles = getComputedStyle(document.documentElement)
  const token = (name: string, fallback: string) =>
    styles.getPropertyValue(name).trim() || fallback

  // Explicit JSON token rules tuned to the app palette - keys quiet,
  // values carrying the color, exactly like the forms. The minifier
  // shortens #ffffff to #fff and Monaco rejects 3-digit hex, so expand
  const hex = (value: string) => {
    const raw = value.replace('#', '').trim()
    return raw.length === 3
      ? raw
          .split('')
          .map((c) => c + c)
          .join('')
      : raw
  }
  const color = (name: string, fallback: string) =>
    '#' + hex(token(name, fallback))
  monaco.editor.defineTheme('dw-dark', {
    base: 'vs-dark',
    inherit: true,
    rules: [
      { token: 'string.key.json', foreground: hex(token('--ink', '#e4eaed')) },
      { token: 'string.value.json', foreground: '84c8a0' },
      { token: 'number.json', foreground: hex(token('--accent', '#4cb8cc')) },
      { token: 'keyword.json', foreground: hex(token('--warn', '#d9a84e')) },
      {
        token: 'delimiter.bracket.json',
        foreground: hex(token('--muted', '#8fa0a8')),
      },
      {
        token: 'delimiter.array.json',
        foreground: hex(token('--muted', '#8fa0a8')),
      },
      {
        token: 'delimiter.colon.json',
        foreground: hex(token('--muted', '#8fa0a8')),
      },
      {
        token: 'delimiter.comma.json',
        foreground: hex(token('--muted', '#8fa0a8')),
      },
    ],
    colors: {
      'editor.background': color('--panel', '#1c2226'),
      'editor.foreground': color('--ink', '#e4eaed'),
      'editorLineNumber.foreground': color('--muted', '#8fa0a8'),
    },
  })
  monaco.editor.defineTheme('dw-light', {
    base: 'vs',
    inherit: true,
    rules: [
      { token: 'string.key.json', foreground: hex(token('--ink', '#1c2428')) },
      { token: 'string.value.json', foreground: '2b7a4b' },
      { token: 'number.json', foreground: hex(token('--accent', '#0b7285')) },
      { token: 'keyword.json', foreground: hex(token('--warn', '#9a6a12')) },
      {
        token: 'delimiter.bracket.json',
        foreground: hex(token('--muted', '#5b6a72')),
      },
      {
        token: 'delimiter.array.json',
        foreground: hex(token('--muted', '#5b6a72')),
      },
      {
        token: 'delimiter.colon.json',
        foreground: hex(token('--muted', '#5b6a72')),
      },
      {
        token: 'delimiter.comma.json',
        foreground: hex(token('--muted', '#5b6a72')),
      },
    ],
    colors: {
      'editor.background': color('--panel', '#ffffff'),
      'editor.foreground': color('--ink', '#1c2428'),
      'editorLineNumber.foreground': color('--muted', '#5b6a72'),
    },
  })
  return monaco
}

export function currentTheme(): string {
  return window.matchMedia('(prefers-color-scheme: light)').matches
    ? 'dw-light'
    : 'dw-dark'
}
