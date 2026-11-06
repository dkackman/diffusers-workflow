// Captures the README's web UI screenshots into docs/img/ from a running
// server that has real work in it - the e2e fixture's one-pixel outputs make
// a poor gallery. Read-only: it only navigates, it never queues a job.
//
//   DW_URL=http://gpu-box:8765 DW_API_TOKEN=... node scripts/readme-screenshots.mjs
//
// DW_SHOT_WORKSPACE, DW_SHOT_WORKFLOW and DW_SHOT_JOB pick what is shown;
// DW_SHOT_KEEP is a comma list of the workspaces left in the sidebar (the
// rest - test and probe workspaces - are hidden from the picture only).
// Shots land as PNG in DW_SHOT_OUT (default: a temp dir) for review before
// they are converted and copied into docs/img/.
import { mkdtempSync } from 'node:fs'
import { tmpdir } from 'node:os'
import { join } from 'node:path'
import { chromium } from '@playwright/test'

const base = process.env.DW_URL ?? 'http://127.0.0.1:8765'
const token = process.env.DW_API_TOKEN ?? ''
const ws = process.env.DW_SHOT_WORKSPACE ?? 'shedding-skin-quartet'
const workflow = process.env.DW_SHOT_WORKFLOW ?? 'tool-musical-trailer'
// The job shot is there for its flow graph, so it defaults to a job whose
// workflow has one worth showing, in a workspace of its own
const jobWs = process.env.DW_SHOT_JOB_WORKSPACE ?? 'acorn-wars'
const job = process.env.DW_SHOT_JOB ?? '8e31d14dd7b1'
const keep = (process.env.DW_SHOT_KEEP ?? `default,${ws},${jobWs}`).split(',')
const out = process.env.DW_SHOT_OUT ?? mkdtempSync(join(tmpdir(), 'dw-shots-'))

// [file, hash route, color schemes, options]. `view` is the editor view to
// click; `flow` scales the data-flow graph to its card instead of scrolling,
// and crops the picture just below it.
const shots = [
  ['ui-overview', `ws/${ws}/overview`, ['light', 'dark']],
  ['ui-workflow', `ws/${ws}/workflows/${workflow}`, ['light']],
  ['ui-job', `ws/${jobWs}/jobs/${job}`, ['light'], { flow: true }],
  ['ui-gallery', `ws/${ws}/gallery`, ['light', 'dark']],
  ['ui-editor', `ws/${ws}/edit/${workflow}`, ['light'], { view: 'split' }],
  ['ui-models', 'server/models', ['light']],
]

const browser = await chromium.launch()
for (const [name, route, schemes, { view, flow } = {}] of shots) {
  for (const colorScheme of schemes) {
    const context = await browser.newContext({
      viewport: { width: 1440, height: 900 },
      deviceScaleFactor: 2,
      colorScheme,
    })
    const page = await context.newPage()
    const login = token ? `?token=${encodeURIComponent(token)}` : ''
    await page.goto(`${base}/${login}#/${route}`)
    await page.waitForLoadState('networkidle').catch(() => {})
    await page.waitForTimeout(3000)
    if (view) {
      await page
        .getByRole('button', { name: view, exact: true })
        .first()
        .click()
      await page.waitForTimeout(2500)
    }
    await page.evaluate((keep) => {
      for (const el of document.querySelectorAll('.wsgroup, .hiddentoggle'))
        el.style.display = 'none'
      for (const el of document.querySelectorAll('.ws.shut')) {
        const label = el.querySelector('.name')?.textContent?.trim()
        if (!keep.includes(label)) el.style.display = 'none'
      }
    }, keep)
    let clip
    if (flow) {
      clip = await page.evaluate(() => {
        const area = document.querySelector('.scrollarea')
        const svg = area.querySelector('svg')
        area.style.width = '100%'
        svg.style.width = '100%'
        svg.style.height = 'auto'
        return {
          x: 0,
          y: 0,
          width: 1440,
          height: area.getBoundingClientRect().bottom + 24,
        }
      })
    }
    const file = join(
      out,
      `${name}${colorScheme === 'dark' ? '-dark' : ''}.png`,
    )
    await page.screenshot({ path: file, clip })
    console.log(file)
    await context.close()
  }
}
await browser.close()
