import { expect, test, type Response } from '@playwright/test'

// The journey of the UI quality program's last phase, end to end against the
// real server: a newcomer's first load, the card that leads to a workflow
// that runs, the shortcuts, a workspace made and switched to from the
// sidebar (open and collapsed), what the Status page says about a
// loopback-only server, a phone-width Overview, and how the entry bundle is
// served. The card's dismissal is per browser, so the journey starts from
// empty storage; the workspace it makes has a name of its own and is deleted
// whatever happens.
test.use({ storageState: { cookies: [], origins: [] } })

const ENTRY_JS = '**/assets/index-*.js'
const isEntryJs = (r: Response) =>
  /\/assets\/index-[^/]+\.js$/.test(new URL(r.url()).pathname)
/** A `http://<host>:<port>` that is not this machine's loopback */
const REMOTE_URL =
  /https?:\/\/(?!127\.0\.0\.1[:/]|localhost[:/]|\[::1\][:/])[^\s/]+:\d+/

test('a newcomer finds a workflow that runs, makes a workspace and switches to it', async ({
  page,
  request,
}) => {
  test.setTimeout(90_000)
  // The card's link names one template of the default workspace: without it
  // there is no journey to make
  const template = await request.get(
    '/api/workflows/templates/text-to-image?workspace=default',
  )
  test.skip(
    template.status() === 404,
    'the default workspace holds no templates/text-to-image workflow',
  )
  const name = `e2e-first-${Math.random().toString(36).slice(2, 8)}`
  const nav = page.getByRole('complementary', { name: 'navigation' })
  const card = page.getByRole('region', {
    name: 'New here? Make your first picture',
  })
  try {
    // 1. A slow bundle shows the loading shell first, then the Overview
    const seen: Response[] = []
    page.on('response', (response) => seen.push(response))
    await page.route(ENTRY_JS, async (route) => {
      await new Promise((resolve) => setTimeout(resolve, 1000))
      await route.continue()
    })
    await page.goto('/', { waitUntil: 'commit' })
    await expect(page.locator('#boot')).toContainText('Loading dw…')
    await expect(page).toHaveURL(/#\/ws\/default\/overview$/)
    await expect(
      page.getByRole('heading', { level: 1, name: 'default' }),
    ).toBeVisible()
    await page.unroute(ENTRY_JS)
    await expect(card).toBeVisible()

    // 2. The card's link opens the template, and it is ready to run
    await card.getByRole('link', { name: 'Open text-to-image' }).click()
    await expect(page).toHaveURL(/\/workflows\/templates\/text-to-image$/)
    const check = page.getByRole('region', { name: 'run check' })
    await expect(check.getByText('Ready to run.', { exact: true })).toBeVisible(
      { timeout: 20_000 },
    )
    await page.goBack()
    await expect(page).toHaveURL(/#\/ws\/default\/overview$/)

    // 3. Got it dismisses the card, for this browser: a reload keeps it gone
    await card
      .getByRole('button', { name: 'Got it, dismiss this introduction' })
      .click()
    await expect(card).toHaveCount(0)
    await page.reload()
    await expect(
      page.getByRole('heading', { level: 1, name: 'default' }),
    ).toBeVisible()
    await expect(card).toHaveCount(0)

    // 4. The header's shortcuts list names /, and / finds the Jobs search
    await page.getByRole('button', { name: 'Keyboard shortcuts' }).click()
    const help = page.getByRole('dialog', { name: 'keyboard shortcuts' })
    await expect(help.locator('dt', { hasText: /^\/$/ })).toBeVisible()
    await help.getByRole('button', { name: 'Close' }).click()
    await expect(help).toBeHidden()
    await nav.getByRole('link', { name: 'Jobs', exact: true }).click()
    await expect(page).toHaveURL(/#\/ws\/default\/jobs$/)
    const search = page.getByPlaceholder(
      'search names, prompts, notes and ids…',
    )
    await expect(search).toBeVisible()
    await page.keyboard.press('/')
    await expect(search).toBeFocused()

    // 5. A new workspace from the sidebar opens on its Overview, whose empty
    // recent outputs lead to its Workflows page
    await nav.getByRole('button', { name: 'new workspace' }).click()
    await nav.getByPlaceholder('name').fill(name)
    await nav.getByRole('button', { name: 'Create' }).click()
    await expect(page).toHaveURL(new RegExp(`#/ws/${name}/overview$`))
    await expect(page.getByRole('heading', { level: 1, name })).toBeVisible()
    const recent = page.locator('section.panel', {
      has: page.getByRole('heading', { name: 'Recent outputs' }),
    })
    await expect(recent).toContainText('Nothing generated yet.')
    await expect(
      recent.getByRole('link', { name: 'Run a workflow' }),
    ).toHaveAttribute('href', `#/ws/${name}/workflows`)

    // 6. From its Gallery, another workspace's name lands on that Gallery
    await expect(nav.getByRole('link', { name: 'Gallery' })).toHaveCount(1)
    await nav.getByRole('link', { name: 'Gallery' }).click()
    await expect(page).toHaveURL(new RegExp(`#/ws/${name}/gallery$`))
    await nav.getByRole('link', { name: /^default/ }).click()
    await expect(page).toHaveURL(/#\/ws\/default\/gallery$/)

    // 7. Collapsed, a workspace is a dot named for it, and switches the same
    await nav.getByRole('button', { name: 'collapse sidebar' }).click()
    await nav.getByRole('link', { name: `${name} workspace` }).click()
    await expect(page).toHaveURL(new RegExp(`#/ws/${name}/gallery$`))
    await expect(page.getByRole('heading', { name: 'Gallery' })).toBeVisible()
    await nav.getByRole('button', { name: 'expand sidebar' }).click()
    await expect(
      nav.getByRole('button', { name: 'new workspace' }),
    ).toBeVisible()

    // 8. A loopback-only server says so, and offers no remote URL; MCP not
    // mounted names the flag that mounts it
    await nav.getByRole('link', { name: 'Status' }).click()
    await expect(page).toHaveURL(/#\/server\/status$/)
    const connect = page.locator('.panel', {
      has: page.getByRole('heading', { name: 'Connect from another machine' }),
    })
    await expect(connect).toContainText(
      'Only this machine can reach this server.',
    )
    expect(await page.locator('body').innerText()).not.toMatch(REMOTE_URL)
    const mcp = page.locator('.panel', {
      has: page.getByRole('heading', { name: 'MCP', exact: true }),
    })
    await expect(mcp).toContainText('python -m dw.serve --mcp')

    // 9. At phone width the Overview is no wider than the screen
    await page.setViewportSize({ width: 390, height: 844 })
    await page.goto('/#/ws/default/overview')
    await expect(
      page.getByRole('heading', { level: 1, name: 'default' }),
    ).toBeVisible()
    const widths = await page.evaluate(() => ({
      doc: document.documentElement.scrollWidth,
      view: window.innerWidth,
    }))
    expect(widths.doc).toBeLessThanOrEqual(widths.view)

    // 10. The entry bundle came compressed, and cached for good
    const js = seen.find(isEntryJs)
    expect(js, 'the entry JS was loaded').toBeDefined()
    expect(js?.headers()['content-encoding']).toBe('gzip')
    expect(js?.headers()['cache-control']).toContain('immutable')
  } finally {
    await request.delete(
      `/api/workspaces/${encodeURIComponent(name)}?acknowledged=true`,
    )
  }
})
