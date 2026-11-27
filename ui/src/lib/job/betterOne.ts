// "Make a better one" (#1762 stage D): a finished first picture from the
// SD 1.5 template leads to the same prompt and seed on the model this
// server prefers. The preferred template is opened pre-filled, never run:
// its run check shows any download before anything is spent.
import { FIRST_RUN_WORKFLOW } from '../firstRun'
import type { JobDetail } from '../types'

/** The template a better one is made with: it composes
 * preference:text_to_image and picks the model by tier */
export const BETTER_WORKFLOW = 'templates/preferred/text-to-image'

/** What the preferred template opens with: the job's prompt and seed, and
 * nothing else. No size, so the tier's size applies (an explicit one beats
 * it, and SD 1.5's 512x512 would hold a better model at 512), and no tier,
 * so the server's default tier applies. Null when the job is not a
 * succeeded run of the first-run template. `declared` is the definition's
 * variables: a prompt the job did not override is the default it ran. */
export function betterOneArguments(
  job: Pick<JobDetail, 'status' | 'workflow' | 'workflow_name' | 'arguments'>,
  declared: Record<string, unknown> | null,
  seed: number | undefined,
): Record<string, string | number> | null {
  if (job.status !== 'succeeded') return null
  if ((job.workflow_name || job.workflow) !== FIRST_RUN_WORKFLOW) return null
  const args: Record<string, string | number> = {}
  const prompt = job.arguments?.prompt ?? declared?.prompt
  if (typeof prompt === 'string' && prompt) args.prompt = prompt
  if (seed !== undefined) args.seed = seed
  return args
}
