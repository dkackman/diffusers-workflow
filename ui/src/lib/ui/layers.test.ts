import { expect, it } from 'vitest'
import { holdLayer, overlayOpen } from './layers.svelte'

it('counts open overlays and releases each once', () => {
  expect(overlayOpen()).toBe(false)
  const releaseA = holdLayer()
  const releaseB = holdLayer()
  releaseA()
  releaseA()
  expect(overlayOpen()).toBe(true)
  releaseB()
  expect(overlayOpen()).toBe(false)
})
