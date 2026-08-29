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
    const schema = await (await fetch('/api/schema')).json()
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

  monaco.editor.defineTheme('dw-dark', {
    base: 'vs-dark',
    inherit: true,
    rules: [],
    colors: {
      'editor.background': token('--panel', '#1c2226'),
      'editor.foreground': token('--ink', '#e4eaed'),
      'editorLineNumber.foreground': token('--muted', '#8fa0a8'),
    },
  })
  monaco.editor.defineTheme('dw-light', {
    base: 'vs',
    inherit: true,
    rules: [],
    colors: {
      'editor.background': token('--panel', '#ffffff'),
      'editor.foreground': token('--ink', '#1c2428'),
      'editorLineNumber.foreground': token('--muted', '#5b6a72'),
    },
  })
  return monaco
}

export function currentTheme(): string {
  return window.matchMedia('(prefers-color-scheme: light)').matches
    ? 'dw-light'
    : 'dw-dark'
}
