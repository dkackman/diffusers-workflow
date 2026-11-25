import { expect, test, type Locator, type Page } from '@playwright/test'

// The JSON view loads Monaco's editor core, its contributions and the JSON
// language only (src/lib/monaco.ts, src/lib/monacoContribs.ts). Each test
// pins one thing the whole-package editor gave its users, against the
// built UI and the real server's schemas. Every case owns a scratch
// workflow or prompt, made and removed through the API.
const BASE = {
  id: 'E2EJsonView',
  description: 'Scratch workflow for the JSON view specs.',
  variables: { line: 'Title card' },
  steps: [
    {
      name: 'card',
      task: {
        command: 'compose_text',
        arguments: { parts: ['variable:line'] },
      },
      result: { content_type: 'text/plain' },
    },
  ],
}

// Monaco's Format Document chord is the platform's own: Shift+Alt+F on
// macOS and Windows, Ctrl+Shift+I on Linux - where CI runs
const FORMAT =
  process.platform === 'linux' ? 'Control+Shift+KeyI' : 'Shift+Alt+KeyF'

type Request = Page['request']

/** A scratch workflow of its own for one test, removed in `finally` */
async function withWorkflow(
  request: Request,
  run: (name: string) => Promise<void>,
): Promise<void> {
  const name = `e2e-json-${Math.random().toString(36).slice(2, 10)}`
  const saved = await request.put(`/api/workflows/${name}`, {
    data: { workflow: BASE },
  })
  expect(saved.status()).toBe(200)
  try {
    await run(name)
  } finally {
    await request.delete(`/api/workflows/${name}`)
  }
}

async function jsonView(page: Page): Promise<Locator> {
  await page.getByRole('button', { name: 'JSON', exact: true }).click()
  const editor = page.locator('.monaco-editor').first()
  await expect(editor).toBeVisible({ timeout: 20_000 })
  return editor
}

/** Replace the Monaco document with `text`, typed the way a person would.
 * One line only: typed newlines pick up Monaco's auto-indent and
 * auto-closing, so a multi-line document is typed flat and formatted */
async function typeJson(page: Page, text: string) {
  await page.locator('.view-lines').first().click()
  await page.keyboard.press('ControlOrMeta+a')
  await page.keyboard.press('Backspace')
  await page.keyboard.insertText(text)
}

/** Open `name` in the editor's JSON view with `text` typed in */
async function editJson(page: Page, name: string, text: string) {
  await page.goto(`/#/ws/default/edit/${name}`)
  const editor = await jsonView(page)
  await typeJson(page, text)
  return editor
}

/** The rendered token spelling `word` (quotes included) */
function token(editor: Locator, word: string): Locator {
  return editor.locator('.view-line span span', { hasText: word }).first()
}

/** Whether a schema squiggle sits under `target`, horizontally and on its
 * line. The JSON language reports a schema violation as a warning (its
 * `schemaValidation` default), a syntax error as an error */
async function squiggleUnder(editor: Locator, target: Locator) {
  const box = await target.boundingBox()
  if (!box) return false
  for (const squiggle of await editor.locator('.squiggly-warning').all()) {
    const s = await squiggle.boundingBox()
    if (!s) continue
    const sameLine = Math.abs(s.y - box.y) < box.height
    const overlaps = s.x < box.x + box.width && s.x + s.width > box.x
    if (sameLine && overlaps) return true
  }
  return false
}

const flat = (doc: object) => JSON.stringify(doc)

/** Open `name` with `doc` typed flat, then formatted over several lines */
async function editPretty(page: Page, name: string, doc: object) {
  const editor = await editJson(page, name, flat(doc))
  await page.keyboard.press(FORMAT)
  await expect
    .poll(() => editor.locator('.view-line').count())
    .toBeGreaterThan(5)
  return editor
}

test('a key the workflow schema does not know is marked', async ({
  page,
  request,
}) => {
  await withWorkflow(request, async (name) => {
    const editor = await editJson(
      page,
      name,
      flat({ id: 'X', stepz: [], steps: [] }),
    )
    const stepz = token(editor, '"stepz"')
    await expect(stepz).toBeVisible()
    await expect
      .poll(() => squiggleUnder(editor, stepz), { timeout: 15_000 })
      .toBe(true)
  })
})

test('completion offers the schema keys', async ({ page, request }) => {
  await withWorkflow(request, async (name) => {
    const editor = await editJson(page, name, '{\n  "id": "X",\n  ')
    await page.keyboard.press('Control+Space')
    const suggest = editor.locator('.suggest-widget')
    await expect(suggest).toBeVisible()
    await expect(
      suggest.locator('.monaco-list-row', { hasText: 'steps' }).first(),
    ).toBeVisible()
  })
})

test('hover shows the schema description', async ({ page, request }) => {
  await withWorkflow(request, async (name) => {
    const editor = await editJson(
      page,
      name,
      flat({ id: 'X', seed: 7, steps: [] }),
    )
    const seed = token(editor, '"seed"')
    const hover = page.locator('.monaco-hover', {
      hasText: 'Default seed for the entire workflow.',
    })
    // a hover before the JSON worker is up shows nothing: move off and
    // back until the schema's description arrives
    await expect(async () => {
      await page.mouse.move(0, 0)
      await seed.hover()
      await expect(hover.first()).toBeVisible({ timeout: 2_000 })
    }).toPass({ timeout: 20_000 })
  })
})

test('Ctrl+F opens find', async ({ page, request }) => {
  await withWorkflow(request, async (name) => {
    const editor = await editJson(page, name, flat(BASE))
    await page.keyboard.press('ControlOrMeta+f')
    const find = editor.locator('.find-widget')
    await expect(find).toHaveClass(/\bvisible\b/)
    await page.keyboard.insertText('card')
    await expect(find.locator('.matchesCount')).toContainText('of')
  })
})

test('steps folds', async ({ page, request }) => {
  await withWorkflow(request, async (name) => {
    const editor = await editPretty(page, name, BASE)
    const steps = editor.locator('.view-line', { hasText: '"steps"' })
    await expect(steps).toBeVisible()
    const line = await steps.boundingBox()
    // the margin's fold control on the steps line
    const controls = editor.locator(
      '.margin-view-overlays .codicon-folding-expanded',
    )
    // the controls are drawn as the folding ranges arrive: look until the
    // steps line has its own
    let onSteps: Locator | null = null
    await expect(async () => {
      for (const control of await controls.all()) {
        const box = await control.boundingBox()
        if (box && line && Math.abs(box.y - line.y) < line.height)
          onSteps = control
      }
      expect(onSteps).not.toBeNull()
    }).toPass()
    await onSteps!.click()
    await expect(
      editor.locator('.margin-view-overlays .codicon-folding-collapsed'),
    ).toHaveCount(1)
    await expect(
      editor.locator('.view-line', { hasText: '"card"' }),
    ).toHaveCount(0)
  })
})

test('the editor follows the header theme toggle into dw-dark', async ({
  page,
  request,
}) => {
  await withWorkflow(request, async (name) => {
    await page.goto(`/#/ws/default/edit/${name}`)
    const editor = await jsonView(page)
    const background = editor.locator('.monaco-editor-background').first()
    // Playwright's colour scheme is light, so "system" starts on dw-light
    await expect(background).toHaveCSS('background-color', 'rgb(255, 255, 255)')
    const toggle = page.getByRole('button', { name: /^theme:/i })
    while (
      !/theme: dark/i.test((await toggle.getAttribute('aria-label')) ?? '')
    )
      await toggle.click()
    // dw-dark's panel colour (src/lib/monaco.ts PALETTES)
    await expect(editor).toHaveClass(/vs-dark/)
    await expect(background).toHaveCSS('background-color', 'rgb(28, 34, 38)')
  })
})

test('the prompt editor validates against the prompt schema', async ({
  page,
  request,
}) => {
  const name = `e2e-json-prompt-${Math.random().toString(36).slice(2, 10)}`
  const saved = await request.put(`/api/prompts/${name}`, {
    data: { prompt: { text: 'first' } },
  })
  expect(saved.status()).toBe(200)
  try {
    await page.goto(`/#/shared/prompt-edit/${name}`)
    const editor = await jsonView(page)
    // "steps" is a workflow key: only the prompt schema refuses it
    await typeJson(page, flat({ text: 'x', steps: [] }))
    const steps = token(editor, '"steps"')
    await expect(steps).toBeVisible()
    await expect
      .poll(() => squiggleUnder(editor, steps), { timeout: 15_000 })
      .toBe(true)
    await expect(token(editor, '"text"')).toBeVisible()
    expect(await squiggleUnder(editor, token(editor, '"text"'))).toBe(false)
  } finally {
    await request.delete(`/api/prompts/${name}`)
  }
})

test('Ctrl+S saves the JSON edit', async ({ page, request }) => {
  await withWorkflow(request, async (name) => {
    await editJson(page, name, flat({ ...BASE, id: 'SavedFromJson' }))
    await page.keyboard.press('ControlOrMeta+s')
    await expect(page.getByText(`Saved to ${name}`)).toBeVisible()
    const read = await request.get(`/api/workflows/${name}`)
    expect((await read.json()).id).toBe('SavedFromJson')
  })
})

// Beyond the eight: what else the whole-package editor gave the JSON view

test('the Format Document chord formats the document', async ({
  page,
  request,
}) => {
  await withWorkflow(request, async (name) => {
    const editor = await editJson(page, name, JSON.stringify(BASE))
    await expect(editor.locator('.view-line')).not.toHaveCount(0)
    const before = await editor.locator('.view-line').count()
    await page.keyboard.press(FORMAT)
    await expect
      .poll(() => editor.locator('.view-line').count())
      .toBeGreaterThan(before + 5)
    await expect(
      editor.locator('.view-line', { hasText: '"steps": [' }),
    ).toBeVisible()
  })
})

test('the bracket under the cursor is matched', async ({ page, request }) => {
  await withWorkflow(request, async (name) => {
    // typing leaves the cursor just after the document's closing brace
    const editor = await editJson(page, name, flat(BASE))
    await expect(editor.locator('.bracket-match')).toHaveCount(2)
  })
})

test('the codicon font loads for the editor widgets', async ({
  page,
  request,
}) => {
  await withWorkflow(request, async (name) => {
    await editJson(page, name, flat(BASE))
    const faces = await page.evaluate(async () => {
      const loaded = await document.fonts.load('16px codicon')
      return loaded.map((face) => face.status)
    })
    expect(faces).toEqual(['loaded'])
  })
})

test('a Monaco chunk that cannot load says so, and the form view still works', async ({
  page,
  request,
}) => {
  await withWorkflow(request, async (name) => {
    // a tab older than the server's build asks for a chunk it no longer has
    await page.route('**/assets/monaco-*.js', (route) =>
      route.fulfill({ status: 404, body: 'not found' }),
    )
    await page.goto(`/#/ws/default/edit/${name}`)
    await page.getByRole('button', { name: 'JSON', exact: true }).click()
    await expect(
      page.getByRole('alert').filter({ hasText: "The JSON editor couldn't" }),
    ).toContainText(
      "The JSON editor couldn't load. Reload the page; if that doesn't help, check that the server is running.",
    )
    await expect(page.getByText('What failed')).toBeVisible()
    await page.getByRole('button', { name: 'form', exact: true }).click()
    await expect(page.locator('input.wfid')).toHaveValue('E2EJsonView')
  })
})
