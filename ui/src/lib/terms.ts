// The words the UI does not use, each with the one it uses instead
// (ui/GLOSSARY.md explains every term). `banned_terms` in
// scripts/arch-metrics.mjs counts each match in user-facing copy, and
// scripts/copy-rules.test.ts pins what each pattern catches and spares.
//
// Node imports this file without a build (arch-metrics.mjs), so it stays
// dependency-free and uses only erasable TypeScript.
//
// "attempt" is deliberately not banned: in the UI it names only a chain
// segment's retry inside one job (src/lib/chainRows.ts).

export type Banned = { id: string; pattern: RegExp; use: string; why: string }

export const BANNED: Banned[] = [
  {
    id: 'tilde',
    pattern: /(^|[\s(·])~(\s?\d|$)/,
    use: 'about 3 min',
    why: 'A leading tilde reads as a minus sign in the monospace font.',
  },
  {
    id: 'keep-asset',
    pattern: /\bkeep(s|ing)? (it |this file |\S+ )?as (an? )?(input )?asset\b/i,
    use: 'Save as asset',
    why: '"Keep" already means keeping your own copy in a conflict.',
  },
  {
    id: 'star-keep',
    pattern: /\bstar to keep\b/i,
    use: 'Star',
    why: 'Starring favourites a file; "keep" means something else.',
  },
  {
    id: 'latest-result',
    pattern: /\bresult \(latest\)/i,
    use: 'latest take',
    why: 'A job made for a request item is a take.',
  },
  {
    id: 'run-noun',
    pattern:
      /\b(this|the|a|each|every|earlier|last|latest|first) runs?\b(?! (it|them|again|now|here|the|a|this|once|only)\b)|\bRun note\b/i,
    use: 'job',
    why: 'One execution of a workflow is a job (a take inside a request); "run" is the verb.',
  },
  {
    id: 'session',
    pattern: /\bsession\b/i,
    use: 'the agent is connected',
    why: 'The person sees an agent, not a session.',
  },
  {
    id: 'held-review',
    pattern: /held for your review/i,
    use: 'waiting for your OK',
    why: 'The hold waits for the person, in their words.',
  },
  {
    id: 'starter',
    pattern: /\bstarters?\b|\b(a|the) template\b/i,
    use: 'an example',
    why: 'Starter workflows are Examples everywhere.',
  },
]

/** The files allowed to spell a banned term: this list's own */
export const COPY_OWNERS = new Set(['src/lib/terms.ts'])
