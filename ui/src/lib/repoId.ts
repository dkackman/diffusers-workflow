// The server checks a download with huggingface_hub's validate_repo_id
// (dw/hub_cache.py); this is that rule, so the field never refuses a repo
// the server takes nor sends one it refuses. Pinned by
// tests/test_ui_twins.py. A part is 1-96 word, "-" or "." characters,
// opening and closing on a word character; the namespace is optional
const PART = String.raw`[A-Za-z0-9_](?:[A-Za-z0-9_.-]{0,94}[A-Za-z0-9_])?`
const REPO_ID = new RegExp(`^(?:${PART}/)?${PART}$`)

const REPO_ID_HINT =
  'A repo id is a name or org/name, like gpt2 or black-forest-labs/FLUX.1-dev.'

/** Why a download field's value cannot be sent, or null when it is empty
 * or a repo id the server accepts */
export function repoIdHint(value: string): string | null {
  const repo = value.trim()
  if (!repo) return null
  const valid =
    REPO_ID.test(repo) &&
    !repo.includes('--') &&
    !repo.includes('..') &&
    !repo.endsWith('.git')
  return valid ? null : REPO_ID_HINT
}
