import { mkdtempSync, rmSync } from 'node:fs'
import { tmpdir } from 'node:os'
import { join } from 'node:path'

/** The e2e fixture servers: one per Playwright worker, each with its own
 * port and scratch directory, so specs on different workers never share a
 * workspace. A spec file runs whole on one worker, in order, against that
 * worker's server - the one-server sequence the specs were written for.
 *
 * Read by playwright.config.ts (to start the servers) and by e2e/test.ts
 * (to point a worker's `baseURL` at its own). Both re-read this module in
 * every worker process; the scratch root is made once, by the first process
 * to load it, and inherited by the workers through the environment. */

export const BASE_PORT = 8971

/** How many servers, and so workers: DW_E2E_WORKERS when set, else two on
 * CI (a four-core runner) and four locally */
export const SERVERS = Math.max(
  1,
  Number(process.env.DW_E2E_WORKERS) || (process.env.CI ? 2 : 4),
)

if (!process.env.DW_E2E_SCRATCH) {
  const scratch = mkdtempSync(join(tmpdir(), 'dw-e2e-'))
  process.env.DW_E2E_SCRATCH = scratch
  // the process that made it removes it on exit, after the servers stopped;
  // one the caller named is left alone
  process.on('exit', () => rmSync(scratch, { recursive: true, force: true }))
}

/** Server `index`'s scratch directory: its workspace, outputs and hub cache */
export function serverScratch(index: number): string {
  return join(process.env.DW_E2E_SCRATCH!, `server-${index}`)
}

export function serverPort(index: number): number {
  return BASE_PORT + index
}
