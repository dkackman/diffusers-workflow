/** What a failed job's engine text means, in the words a person acts on: the
 * lead, what to do next, and the one button that does it. Copy is the
 * audit's Tier 2 section A; a text that matches no case gets the plain lead
 * and no remedy, and the caller still shows the engine's own text. */

export type JobErrorAction = 'rerun' | 'models' | null

export interface JobErrorExplained {
  title: string
  remedy: string | null
  action: JobErrorAction
}

const CASES: Array<[RegExp, JobErrorExplained]> = [
  [
    /CUDA.*out of memory|MPS.*out of memory|OutOfMemoryError/is,
    {
      title: 'The card ran out of memory.',
      remedy: 'Run it again with a smaller size or fewer frames.',
      action: 'rerun',
    },
  ],
  [
    // The host's RAM (a failed allocation, a killed process, ffmpeg's
    // own) is not the card's: after the card's, since OutOfMemoryError contains MemoryError
    /Cannot allocate memory|MemoryError|\bKilled\b/,
    {
      title: 'The machine ran out of memory.',
      remedy: 'Run it again with a smaller size or fewer frames.',
      action: 'rerun',
    },
  ],
  [
    /divisible by 8\b|multiple of 8\b/i,
    {
      title: 'The width and height must be multiples of 8.',
      remedy: 'Run it again with a size divisible by 8.',
      action: 'rerun',
    },
  ],
  [
    /GatedRepoError|gated repo/i,
    {
      title: 'This model needs Hugging Face access.',
      remedy:
        'Accept its terms on Hugging Face, then sign in on the server with hf auth login.',
      action: 'models',
    },
  ],
  [
    // A 403 alone is also a private or removed repo: gating is not claimed
    /403 Client Error.*huggingface/i,
    {
      title: 'The model repo refused access.',
      remedy: 'Check that the repo exists and that this server may read it.',
      action: 'models',
    },
  ],
]

const FALLBACK: JobErrorExplained = {
  title: 'The job stopped with an error.',
  remedy: null,
  action: null,
}

export function explainJobError(error: string): JobErrorExplained {
  for (const [pattern, explained] of CASES)
    if (pattern.test(error)) return { ...explained }
  return { ...FALLBACK }
}
