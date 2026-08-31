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
    // Each editor names its model workflow-*.json or prompt-*.json, which
    // is what routes the right schema to it. A missing prompt schema (an
    // older server) must not cost workflow editing its diagnostics
    const schema = await api.getSchema()
    const schemas = [
      {
        uri: 'dw://workflow-schema',
        fileMatch: ['**/workflow-*.json'],
        schema,
      },
    ]
    try {
      const promptSchema = await api.getPromptSchema()
      schemas.push({
        uri: 'dw://prompt-schema',
        fileMatch: ['**/prompt-*.json'],
        schema: promptSchema,
      })
    } catch {
      /* prompt schema unreachable - prompt JSON stays plain */
    }
    jsonDefaults.setDiagnosticsOptions({
      validate: true,
      enableSchemaRequest: false,
      schemas,
    })
  } catch {
    /* schema endpoint unreachable - plain JSON editing still works */
  }

  // Quiet JSON matching the pre-Monaco editor: values in the app ink,
  // property names in the accent, punctuation muted - enough to read the
  // shape of a document at a glance without a syntax rainbow. The
  // palettes mirror app.css by
  // value rather than reading the live CSS tokens: setup runs once, under
  // whichever theme is active, and the editor re-themes at runtime - a
  // light theme defined while dark was active would carry dark colours
  const PALETTES = {
    'dw-dark': {
      base: 'vs-dark',
      ink: 'e4eaed',
      key: '4cb8cc',
      muted: '8fa0a8',
      panel: '1c2226',
    },
    'dw-light': {
      base: 'vs',
      ink: '1c2428',
      key: '0b7285',
      muted: '5b6a72',
      panel: 'ffffff',
    },
  } as const
  for (const [name, palette] of Object.entries(PALETTES)) {
    monaco.editor.defineTheme(name, {
      base: palette.base,
      inherit: true,
      rules: [
        { token: 'string.key.json', foreground: palette.key },
        { token: 'string.value.json', foreground: palette.ink },
        { token: 'number.json', foreground: palette.ink },
        { token: 'keyword.json', foreground: palette.ink },
        { token: 'delimiter.bracket.json', foreground: palette.muted },
        { token: 'delimiter.array.json', foreground: palette.muted },
        { token: 'delimiter.colon.json', foreground: palette.muted },
        { token: 'delimiter.comma.json', foreground: palette.muted },
      ],
      colors: {
        'editor.background': '#' + palette.panel,
        'editor.foreground': '#' + palette.ink,
        'editorLineNumber.foreground': '#' + palette.muted,
      },
    })
  }
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
