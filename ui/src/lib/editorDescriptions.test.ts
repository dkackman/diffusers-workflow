import { describe, expect, it, vi } from 'vitest'

vi.mock('./api', async () => (await import('./test/apiMock')).apiMock())

import { api } from './api'
import { classDescription } from './editor'
import type { PipelineDescription } from './types'

describe('classDescription', () => {
  it('asks again after a failed lookup, and caches the answer', async () => {
    const description = { name: 'X', parameters: [] } as unknown as PipelineDescription
    const describePipeline = vi.mocked(api.describePipeline)
    describePipeline
      .mockRejectedValueOnce(new Error('404'))
      .mockResolvedValue(description)

    expect(await classDescription('X')).toBeNull()
    expect(await classDescription('X')).toBe(description)
    expect(await classDescription('X')).toBe(description)
    expect(describePipeline).toHaveBeenCalledTimes(2)
  })
})
