import { expect, it } from 'vitest'
import { resultProof, statusCounts } from './requestResult'
import type { JobDetail } from './types'

const job = (
  manifest: Array<{ step: string; files: string[]; role?: string }>,
  output_kinds?: Record<string, string | null>,
): JobDetail => ({ id: 'j1', manifest, output_kinds }) as unknown as JobDetail

it('stands a job by the first image of its final step', () => {
  expect(
    resultProof(
      job(
        [
          { step: 'draft', files: ['draft.png'], role: 'intermediate' },
          { step: 'out', files: ['out.txt', 'out.png'], role: 'final' },
        ],
        { 'draft.png': 'image', 'out.txt': 'text', 'out.png': 'image' },
      ),
    ),
  ).toEqual({ file: 'out.png', kind: 'image' })
})

it('stands a job whose result is a video by that video', () => {
  expect(
    resultProof(
      job([{ step: 'clip', files: ['shot/clip.mp4'], role: 'final' }], {
        'shot/clip.mp4': 'video',
      }),
    ),
  ).toEqual({ file: 'shot/clip.mp4', kind: 'video' })
})

it("prefers the final step's video to a keyframe an earlier step made", () => {
  expect(
    resultProof(
      job(
        [
          { step: 'keyframe', files: ['first.png'], role: 'intermediate' },
          { step: 'clip', files: ['clip.mp4'], role: 'final' },
        ],
        { 'first.png': 'image', 'clip.mp4': 'video' },
      ),
    ),
  ).toEqual({ file: 'clip.mp4', kind: 'video' })
})

it('prefers an image to a video inside the same tier', () => {
  expect(
    resultProof(
      job(
        [
          { step: 'clip', files: ['clip.mp4'], role: 'final' },
          { step: 'poster', files: ['poster.png'], role: 'final' },
        ],
        { 'clip.mp4': 'video', 'poster.png': 'image' },
      ),
    ),
  ).toEqual({ file: 'poster.png', kind: 'image' })
})

it('reads the kind off the extension when the job does not say', () => {
  expect(resultProof(job([{ step: 's', files: ['a.webm'] }]))).toEqual({
    file: 'a.webm',
    kind: 'video',
  })
  expect(resultProof(job([{ step: 's', files: ['a.JPG'] }]))).toEqual({
    file: 'a.JPG',
    kind: 'image',
  })
})

it('has nothing to show for audio, text or no output', () => {
  expect(
    resultProof(
      job([{ step: 's', files: ['a.wav', 'a.txt'], role: 'final' }], {
        'a.wav': 'audio',
        'a.txt': 'text',
      }),
    ),
  ).toBeNull()
  expect(resultProof(job([]))).toBeNull()
})

it('counts statuses in the order a plan reads', () => {
  expect(statusCounts({ planned: 1, done: 2, odd: 3, failed: 0 })).toBe(
    '2 done · 1 planned · 3 odd',
  )
})
