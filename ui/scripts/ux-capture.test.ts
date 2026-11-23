// @vitest-environment node
import { describe, expect, it } from 'vitest'
import {
  SERVER_SECTIONS,
  SHARED_SECTIONS,
  WS_SECTIONS,
} from '../src/lib/routes'
import { captureRoutes } from './ux-capture.mjs'

describe('captureRoutes', () => {
  it('names every workspace section once and each fixture item route', () => {
    const routes = captureRoutes()
    const hashes = routes.map((r) => r.hash)
    for (const s of WS_SECTIONS)
      expect(hashes.filter((h) => h === `ws/default/${s}`)).toHaveLength(1)
    for (const h of [
      'ws/default/workflows/e2e-card',
      'ws/default/edit/e2e-card',
      'ws/default/requests/e2e-launch',
      'ws/default/jobs/e2erestart01',
    ])
      expect(hashes).toContain(h)
    for (const s of SHARED_SECTIONS) expect(hashes).toContain(`shared/${s}`)
    for (const s of SERVER_SECTIONS) expect(hashes).toContain(`server/${s}`)
    expect(new Set(routes.map((r) => r.name)).size).toBe(routes.length)
    expect(routes).toHaveLength(
      WS_SECTIONS.length + 4 + SHARED_SECTIONS.length + SERVER_SECTIONS.length,
    )
  })
})
