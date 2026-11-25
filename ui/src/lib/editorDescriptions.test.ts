import { describe, expect, it, vi } from 'vitest'

import { classDescription } from './editor'
import { api } from './test/apiMock'
import type { PipelineDescription } from './types'

vi.mock('./api', async () => (await import('./test/apiMock')).apiModule())

describe('classDescription', () => {
  it('asks again after a failed lookup, and caches the answer', async () => {
    const description = {
      name: 'X',
      parameters: [],
    } as unknown as PipelineDescription
    api.describePipeline
      .mockRejectedValueOnce(new Error('404'))
      .mockResolvedValue(description)

    expect(await classDescription('X')).toBeNull()
    expect(await classDescription('X')).toBe(description)
    expect(await classDescription('X')).toBe(description)
    expect(api.describePipeline).toHaveBeenCalledTimes(2)
  })
})
