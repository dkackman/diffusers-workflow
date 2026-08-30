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

  // Quiet, near-monochrome JSON matching the pre-Monaco editor: content
  // in the app ink, punctuation muted - the schema tooling is the value
  // Monaco adds, not a syntax rainbow. The minifier shortens #ffffff to
  // #fff and Monaco rejects 3-digit hex, so expand
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
      {
        token: 'string.value.json',
        foreground: hex(token('--ink', '#e4eaed')),
      },
      { token: 'number.json', foreground: hex(token('--ink', '#e4eaed')) },
      { token: 'keyword.json', foreground: hex(token('--ink', '#e4eaed')) },
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
      {
        token: 'string.value.json',
        foreground: hex(token('--ink', '#1c2428')),
      },
      { token: 'number.json', foreground: hex(token('--ink', '#1c2428')) },
      { token: 'keyword.json', foreground: hex(token('--ink', '#1c2428')) },
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
  // The in-app toggle stamps data-theme; only the default "system"
  // setting falls through to the OS preference
  const explicit = document.documentElement.dataset.theme
  const dark = explicit
    ? explicit === 'dark'
    : window.matchMedia('(prefers-color-scheme: dark)').matches
  return dark ? 'dw-dark' : 'dw-light'
}
