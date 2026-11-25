import { mkdirSync, writeFileSync } from 'node:fs'
import { join } from 'node:path'
import { expect, test, type Page } from '@playwright/test'

// What an agent writes through the HTTP API shows up on every page that
// reads it, without a reload: one case per kind the change feed carries
// (src/lib/changes.svelte.ts). The real server and the real routes, so the
// counter each write moves is the server's own. The feed polls every 5 s
// (CHANGES_POLL_MS), so "appears" is allowed two polls: 15 s.
//
// Each case makes what it writes under a name of its own and removes it, so
// the seeded fixture the other specs read stays as it was. The one thing the
// API cannot give back is the seeded hub repo, which only the models case
// deletes: it puts the repo back in `finally` (`seedTinyRepo`).
const AGENT = 'e2e-agent'
const FEED = { timeout: 15_000 }
const PNG = Buffer.from(
  'iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mP8z8BQDwAEhQGAhKmM' +
    'IQAAAABJRU5ErkJggg==',
  'base64',
)
const REMOTE = 'Updated by an agent or another tab'

type Request = Page['request']

/** Count the page's navigations from here on: a reload would be one */
function countNavigations(page: Page) {
  const seen = { count: 0 }
  page.on('framenavigated', () => seen.count++)
  return seen
}

function workflow(line = 'Title card') {
  return {
    id: 'E2EAgent',
    description: 'Scratch workflow an agent edits.',
    variables: { line },
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
}

async function putWorkflow(request: Request, name: string, body: object) {
  const saved = await request.put(`/api/workflows/${name}`, {
    data: { workflow: body },
  })
  expect(saved.status()).toBe(200)
}

test('workflows: a workflow an agent saves appears as a changed card', async ({
  page,
  request,
}) => {
  await page.goto('/#/ws/default/workflows')
  await expect(page.locator('a.card').first()).toBeVisible()
  const card = page.locator('a.card', { hasText: AGENT })
  await expect(card).toHaveCount(0)
  const navigated = countNavigations(page)
  try {
    await putWorkflow(request, AGENT, workflow())
    await expect(card).toHaveCount(1, FEED)
    await expect(card).toHaveClass(/row-changed/)
    expect(navigated.count).toBe(0)
  } finally {
    await request.delete(`/api/workflows/${AGENT}`)
  }
  await expect(card).toHaveCount(0, FEED)
})

test('editor: an unedited document reloads with a toast; an edited one asks', async ({
  page,
  request,
}) => {
  await putWorkflow(request, AGENT, workflow())
  try {
    await page.goto(`/#/ws/default/edit/${AGENT}`)
    const line = page.locator('#wfvar-line')
    await expect(line).toHaveValue('Title card')
    const navigated = countNavigations(page)

    // unedited: the agent's change loads, and the toast says so
    let patched = await request.patch(`/api/workflows/${AGENT}`, {
      data: { variables: { line: 'From the agent' } },
    })
    expect(patched.status()).toBe(200)
    await expect(page.getByText(REMOTE)).toBeVisible(FEED)
    await expect(line).toHaveValue('From the agent')
    await expect(page.locator('.chip.unsaved')).toHaveCount(0)

    // edited: nothing is overwritten, and both ways on are offered
    await line.fill('My edit')
    patched = await request.patch(`/api/workflows/${AGENT}`, {
      data: { variables: { line: 'Second agent edit' } },
    })
    expect(patched.status()).toBe(200)
    const banner = page.locator('.remotechanged')
    await expect(banner).toBeVisible(FEED)
    await expect(banner).toContainText(
      'changed since you opened it, in another tab or by an agent',
    )
    await expect(line).toHaveValue('My edit')
    await expect(
      banner.getByRole('button', { name: 'Load theirs' }),
    ).toBeVisible()

    // Keep mine, then Save: the stale write is asked about, not made
    await banner.getByRole('button', { name: 'Keep mine' }).click()
    await expect(banner).toHaveCount(0)
    await page.getByRole('button', { name: /^Save/ }).click()
    const dialog = page.getByRole('alertdialog')
    await expect(dialog).toContainText(
      `${AGENT} changed since you opened it, in another tab or by an agent. Replace that version with yours?`,
    )
    await dialog.getByRole('button', { name: 'Cancel' }).click()
    const stored = await (await request.get(`/api/workflows/${AGENT}`)).json()
    expect(stored.variables.line).toBe('Second agent edit')
    expect(navigated.count).toBe(0)
  } finally {
    await request.delete(`/api/workflows/${AGENT}`)
  }
})

// The JSON editor leaves its text alone while it has the caret. A reload
// still reaches it because the re-read's flush blurs the focused field first
// (DocumentEditor.flush), and only Monaco itself can show that (#1655)
test('editor: a reload replaces the JSON text under an idle caret', async ({
  page,
  request,
}) => {
  const name = `${AGENT}-json`
  await putWorkflow(request, name, workflow())
  try {
    await page.goto(`/#/ws/default/edit/${name}`)
    await page.getByRole('button', { name: 'JSON', exact: true }).click()
    const editor = page.locator('.monaco-editor').first()
    await expect(editor).toBeVisible({ timeout: 20_000 })
    const lines = editor.locator('.view-lines')
    await lines.click()
    await expect(
      editor.getByRole('textbox', { name: 'Editor content' }),
    ).toBeFocused()

    const patched = await request.patch(`/api/workflows/${name}`, {
      data: { variables: { line: 'FromTheAgent' } },
    })
    expect(patched.status()).toBe(200)
    await expect(page.getByText(REMOTE)).toBeVisible(FEED)
    await expect(lines).toContainText('FromTheAgent')

    // typing goes onto their text: whitespace alone leaves nothing unsaved,
    // where the text from before the reload would undo their change
    await page.keyboard.press('End')
    await page.keyboard.type(' ')
    await page.waitForTimeout(600)
    await expect(page.locator('.chip.unsaved')).toHaveCount(0)
    await page.getByRole('button', { name: 'form', exact: true }).click()
    await expect(page.locator('#wfvar-line')).toHaveValue('FromTheAgent')
  } finally {
    await request.delete(`/api/workflows/${name}`)
  }
})

test('prompts: a prompt an agent saves appears as a changed card', async ({
  page,
  request,
}) => {
  await page.goto('/#/prompts')
  await expect(page.locator('.card').first()).toBeVisible()
  const card = page.locator('.card', { hasText: AGENT })
  await expect(card).toHaveCount(0)
  const navigated = countNavigations(page)
  try {
    const saved = await request.put(`/api/prompts/${AGENT}`, {
      data: { prompt: { text: 'written by an agent' } },
    })
    expect(saved.status()).toBe(200)
    await expect(card).toHaveCount(1, FEED)
    await expect(card).toHaveClass(/row-changed/)
    expect(navigated.count).toBe(0)
  } finally {
    await request.delete(`/api/prompts/${AGENT}`)
  }
  await expect(card).toHaveCount(0, FEED)
})

test('assets: an upload an agent makes appears as a changed tile', async ({
  page,
  request,
}) => {
  await page.goto('/#/ws/default/assets')
  await expect(page.getByRole('heading', { name: 'Assets' })).toBeVisible()
  const tile = page.locator('.cellwrap', { hasText: `${AGENT}.png` })
  await expect(tile).toHaveCount(0)
  const navigated = countNavigations(page)
  try {
    const sent = await request.post(
      `/api/uploads?filename=${AGENT}.png&asset_name=${AGENT}.png`,
      { data: PNG },
    )
    expect(sent.status()).toBe(201)
    await expect(tile).toHaveCount(1, FEED)
    await expect(tile).toHaveClass(/row-changed/)
    expect(navigated.count).toBe(0)
  } finally {
    await request.delete(`/api/assets/uploads/${AGENT}.png`)
  }
  await expect(tile).toHaveCount(0, FEED)
})

test('requests: a request an agent writes appears on the open list', async ({
  page,
  request,
}) => {
  const id = 'e2e-fresh'
  const api = `/api/workspaces/default/requests/${id}`
  await page.goto('/#/ws/default/requests')
  await expect(page.getByRole('heading', { name: 'Requests' })).toBeVisible()
  // the seeded request is listed: the page has made its first read
  await expect(page.locator('a.row', { hasText: 'e2e-launch' })).toHaveCount(1)
  await expect(page.locator('a.row', { hasText: id })).toHaveCount(0)
  const navigated = countNavigations(page)
  const row = page.locator('a.row', { hasText: id })
  try {
    const created = await request.put(api, {
      data: {
        title: 'Made by an agent',
        goal: 'Shows up without a reload',
        revision: 0,
        why: 'e2e: written from outside the page',
        by: 'agent',
        items: [{ key: 'one', title: 'One thing' }],
      },
    })
    expect(created.status()).toBe(200)
    await expect(row).toHaveCount(1, FEED)
    await expect(row).toContainText('Made by an agent')
    await expect(row).toHaveClass(/row-changed/)
    expect(navigated.count).toBe(0)
  } finally {
    const read = await request.get(api)
    if (read.ok()) {
      const { revision } = await read.json()
      await request.delete(`${api}?revision=${revision}`)
    }
  }
  await expect(row).toHaveCount(0, FEED)
})

test("job page: an agent's note shows, and says it was updated; the user's own does not", async ({
  page,
  request,
}) => {
  const run = 'e2e-request/20260907-090000-0badf00d'
  await page.goto('/#/ws/default/jobs/e2ereqdone01')
  const note = page.getByText('Your note on this job', { exact: true })
  await expect(note).toBeVisible()
  const notice = page.getByText(REMOTE)
  const navigated = countNavigations(page)
  try {
    // the user's own save is adopted: no notice, no mark, even after a poll
    await page.getByRole('button', { name: 'Edit note' }).click()
    await page.getByLabel('Your note on this job').fill('A note of my own')
    await page.getByRole('button', { name: 'Save', exact: true }).click()
    await expect(page.getByText('A note of my own')).toBeVisible()
    await page.waitForTimeout(6_000)
    await expect(notice).toHaveCount(0)
    await expect(page.locator('.row-changed')).toHaveCount(0)

    // an agent's is shown, with the notice beside it
    const put = await request.put('/api/notes', {
      data: { target: run, note: 'A note from the agent' },
    })
    expect(put.status()).toBe(200)
    await expect(page.getByText('A note from the agent')).toBeVisible(FEED)
    await expect(notice).toBeVisible()
    expect(navigated.count).toBe(0)
  } finally {
    await request.put('/api/notes', { data: { target: run, note: null } })
  }
})

test('sidebar: a workspace an agent makes appears, and goes when deleted', async ({
  page,
  request,
}) => {
  await page.goto('/#/ws/default/workflows')
  const group = page.getByRole('group', { name: 'workspaces' })
  await expect(group).toBeVisible()
  const entry = group.getByText(AGENT, { exact: true })
  await expect(entry).toHaveCount(0)
  const navigated = countNavigations(page)
  try {
    const made = await request.post('/api/workspaces', {
      data: { name: AGENT },
    })
    expect(made.status()).toBe(201)
    await expect(entry).toHaveCount(1, FEED)
    expect(navigated.count).toBe(0)
  } finally {
    await request.delete(`/api/workspaces/${AGENT}?acknowledged=true`)
  }
  await expect(entry).toHaveCount(0, FEED)
})

/** Put the fixture's made-up hub repo back (what serve_fixture.py seeds), so
 * the models case can run again on the same server: --repeat-each, a retry */
function seedTinyRepo() {
  const scratch = process.env.DW_E2E_SCRATCH
  if (!scratch) return false
  const revision = '0123456789abcdef0123456789abcdef01234567'
  const repo = join(scratch, 'hub-cache', 'models--e2e--tiny')
  mkdirSync(join(repo, 'snapshots', revision), { recursive: true })
  mkdirSync(join(repo, 'refs'), { recursive: true })
  writeFileSync(join(repo, 'snapshots', revision, 'config.json'), '{}')
  writeFileSync(join(repo, 'refs', 'main'), revision)
  return true
}

test('models: a repo an agent deletes leaves the open list', async ({
  page,
  request,
}) => {
  await page.goto('/#/models')
  const row = page.getByRole('row', { name: /e2e\/tiny/ })
  test.skip((await row.count()) === 0, 'e2e/tiny is not seeded')
  const navigated = countNavigations(page)
  try {
    const removed = await request.delete('/api/models?repo=e2e/tiny')
    expect(removed.status()).toBe(200)
    await expect(row).toHaveCount(0, FEED)
    expect(navigated.count).toBe(0)
  } finally {
    seedTinyRepo()
  }
})

test('loras: an entry an agent saves is offered in the open editor', async ({
  page,
  request,
}) => {
  await putWorkflow(request, AGENT, {
    id: 'E2EAgentLora',
    variables: {},
    steps: [
      {
        name: 'gen',
        pipeline: {
          configuration: { component_type: 'ZImagePipeline' },
          from_pretrained_arguments: { model_name: 'e2e-org/e2e-base' },
          arguments: { prompt: 'p' },
        },
      },
    ],
  })
  try {
    await page.goto(`/#/ws/default/edit/${AGENT}`)
    const step = page.locator('.panel.step').first()
    await step.getByRole('button', { name: 'full' }).click()
    await step.locator('summary', { hasText: 'LoRAs' }).click()
    const catalog = step.getByRole('region', { name: 'From the catalog' })
    await expect(catalog.locator('li.suggestion')).toHaveCount(1)
    const navigated = countNavigations(page)
    try {
      const saved = await request.put(`/api/loras/${AGENT}`, {
        data: {
          entry: {
            model_name: 'e2e-org/e2e-agent-lora',
            weight_name: 'e2e_agent.safetensors',
            base_models: ['e2e-org/e2e-base'],
            workflow: null,
            description: 'Saved by an agent',
            use_when: 'the e2e journey adds one',
            scale: { default: 0.8 },
            status: 'trial',
          },
        },
      })
      expect(saved.status()).toBe(200)
      await expect(catalog.locator('li.suggestion')).toHaveCount(2, FEED)
      await expect(
        catalog.getByRole('button', { name: `add ${AGENT}` }),
      ).toBeVisible()
      expect(navigated.count).toBe(0)
    } finally {
      await request.delete(`/api/loras/${AGENT}`)
    }
  } finally {
    await request.delete(`/api/workflows/${AGENT}`)
  }
})
