import { describe, expect, it } from 'vitest'
import { chainSummary } from './flowChain'

describe('chained segments', () => {
  const chainStep = (chain: Record<string, unknown>) => ({
    name: 'clip',
    pipeline: { configuration: { component_type: 'ModularPipeline' }, chain },
  })
  const summary = (
    chain: Record<string, unknown>,
    variables: Record<string, unknown> = {},
  ) => chainSummary({ variables }, chainStep(chain))

  it("reads a declared segment count and the seam's last-frame bookkeeping", () => {
    // The chained-segments template's shape: the count and continuity are
    // variables, which a realized workflow always carries
    expect(
      summary(
        {
          segments: 'variable:segments',
          continuity: 'variable:continuity',
          anchor: 'variable:anchor',
          trim_frames: 2,
          crossfade_ms: 80,
        },
        { segments: 3, continuity: 'last_frame', anchor: 'segment_1' },
      ),
    ).toEqual({
      segments: 3,
      fromAudio: false,
      seam: 'last frame · trim 2 · 80ms',
      anchor: true,
    })
  })

  it('falls back to the engine defaults the block leaves out', () => {
    expect(chainSummary({}, chainStep({ segments: 2 }))).toEqual({
      segments: 2,
      fromAudio: false,
      seam: 'last frame · trim 1 · 75ms',
      anchor: false,
    })
  })

  it('names what a guide or a video reference carries across the seam', () => {
    const seam = (chain: Record<string, unknown>) =>
      summary({ segments: 2, ...chain })!.seam
    expect(seam({ continuity: 'guide' })).toBe('guide · 22 frames')
    expect(seam({ continuity: 'guide', guide_frames: 39 })).toBe(
      'guide · 39 frames',
    )
    expect(seam({ continuity: 'last_segment' })).toBe('video reference')
    expect(
      seam({
        continuity: 'last_segment',
        carry_frames: 24,
        carry_audio: false,
      }),
    ).toBe('video reference · 24 frames · no audio')
    expect(seam({ continuity: 'variable:unset' })).toBe('carry')
  })

  it('keeps the audio under a last-frame seam, whatever carry_audio says', () => {
    expect(summary({ segments: 2, carry_audio: false })!.seam).toBe(
      'last frame · trim 1 · 75ms',
    )
  })

  it('leaves the count unknown when the length comes from audio or an unset variable', () => {
    const fromAudio = summary({ match_audio: true })!
    expect(fromAudio.segments).toBeNull()
    expect(fromAudio.fromAudio).toBe(true)
    const unset = summary({ segments: 'variable:missing' })!
    expect(unset.segments).toBeNull()
    expect(unset.fromAudio).toBe(false)
  })

  it('is null on a step with no chain', () => {
    expect(
      chainSummary({}, { name: 'plain', task: { command: 'noop' } }),
    ).toBeNull()
    expect(
      chainSummary({}, { name: 'plain', pipeline: { chain: null } }),
    ).toBeNull()
  })
})
