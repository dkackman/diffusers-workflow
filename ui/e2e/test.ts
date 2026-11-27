import { test as base } from '@playwright/test'
import { SERVERS, serverPort, serverScratch } from './serverPool'

export { expect } from '@playwright/test'
export type {
  APIRequestContext,
  Locator,
  Page,
  Response,
  Route,
} from '@playwright/test'

/** Every spec's `test`: Playwright's, with `baseURL` (so `page.goto('/')` and
 * the `request` fixture) on this worker's own fixture server (serverPool.ts) */
export const test = base.extend({
  // Playwright reads a fixture's dependencies off its first parameter's
  // destructuring, so one with none is the empty pattern
  // eslint-disable-next-line no-empty-pattern
  baseURL: async ({}, use, testInfo) => {
    const index = testInfo.parallelIndex
    if (index >= SERVERS)
      throw new Error(
        `worker ${index} has no fixture server: ${SERVERS} started` +
          ' - set DW_E2E_WORKERS rather than --workers',
      )
    await use(`http://127.0.0.1:${serverPort(index)}`)
  },
})

/** This worker's fixture server's scratch directory (serve_fixture.py) */
export function scratchDir(): string {
  return serverScratch(test.info().parallelIndex)
}
