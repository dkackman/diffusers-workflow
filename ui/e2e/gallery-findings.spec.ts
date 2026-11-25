import { expect, test, type Page } from '@playwright/test'

// The Findings and Assess beside the Gallery's open file (#1477), on a
// mocked gallery: one video whose metadata carries a level finding, and an
// Assess reply with one warn and one info finding and a probe that did not
// apply. Screenshots of the viewer in both themes land in this spec's
// test-results folder.

const NAME = 'clip/run-1/final/clip.mp4'

const video = {
  name: NAME,
  folder: 'clip',
  subfolder: 'final',
  run_id: 'run-1',
  version: 1,
  url: `/outputs/${NAME}`,
  kind: 'video',
  size: 2048,
  mtime: 1_700_000_000,
  label: 'clip.mp4',
  role: 'final',
}

const nearSilent = {
  rule: 'near_silent',
  severity: 'warn',
  at: null,
  value: -62.1,
  threshold: -50,
  says: 'near-silent for a track meant to be heard; check the step that made it, or raise it with normalize_audio',
}

const assessReply = {
  name: NAME,
  source: 'output',
  kind: 'video',
  shots_source: 'none',
  findings: [
    nearSilent,
    {
      rule: 'seam_frame_jump',
      severity: 'info',
      at: 64.25,
      value: 0.4,
      threshold: 0.3,
      says: 'the picture jumps at the cut',
    },
  ],
  rules_applied: ['full_scale', 'near_silent', 'shot_dead_air'],
  rules_skipped: [],
  not_applicable: {
    analyze_seams:
      'no shot boundaries recorded for this file - analyze_seams needs two or more shots',
  },
}

async function mockGallery(page: Page) {
  const assessCalls: string[] = []
  await page.route('**/api/gallery?**', (route) =>
    route.fulfill({
      json: {
        files: [video],
        folders: ['clip'],
        subfolders: ['final'],
        limit: 200,
        offset: 0,
        total: 1,
        workspace: 'default',
      },
    }),
  )
  await page.route('**/api/gallery/**/metadata**', (route) =>
    route.fulfill({
      json: {
        name: NAME,
        source: 'output',
        run_id: 'run-1',
        version: 1,
        role: 'final',
        job: null,
        metadata: null,
        media: null,
        inputs: [],
        findings: [nearSilent],
      },
    }),
  )
  await page.route('**/api/gallery/**/assess**', (route) => {
    assessCalls.push(route.request().method())
    return route.fulfill({ json: assessReply })
  })
  // No media is served (the aside is what is being looked at), but the file
  // is not gone: a HEAD finds it, so a failed draw is not called missing
  await page.route('**/outputs/**', (route) =>
    route.fulfill({
      status: route.request().method() === 'HEAD' ? 200 : 404,
    }),
  )
  return assessCalls
}

for (const theme of ['light', 'dark'] as const) {
  test(`findings and Assess in the gallery viewer (${theme})`, async ({
    page,
  }, testInfo) => {
    await page.emulateMedia({ colorScheme: theme })
    const assessCalls = await mockGallery(page)
    await page.goto('/#/ws/default/gallery')
    await page.locator('button.filetile').first().click()

    const findings = page.getByRole('region', { name: 'Findings' })
    // The metadata's level finding shows without a click
    await expect(findings).toContainText('near-silent')

    const assess = page.getByRole('button', { name: 'Assess' })
    await expect(assess).toBeEnabled()
    await assess.click()
    await expect(findings.locator('li')).toHaveCount(2)
    await expect(findings.locator('li').first()).toContainText('warn')
    await expect(findings.locator('li').nth(1)).toContainText('1:04.3')
    await expect(findings).toContainText('analyze_seams not applicable')
    expect(assessCalls).toEqual(['GET'])

    await page.getByRole('dialog', { name: 'clip.mp4' }).screenshot({
      path: testInfo.outputPath(`gallery-findings-${theme}.png`),
    })
  })
}
