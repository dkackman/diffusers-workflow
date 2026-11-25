/** One name segment - a workspace, a folder, a stored workflow or prompt:
 * dw/security.py's WORKSPACE_NAME_PATTERN, `^[\w][\w.-]*\Z`, where Python's
 * \w is Unicode letters and digits plus underscore. Length is in code
 * points, as Python's len counts. tests/fixtures/workspace_names.json is
 * read by both sides' tests. */
const SEGMENT = /^[\p{L}\p{N}_][\p{L}\p{N}_.-]*$/u
export const MAX_NAME_LENGTH = 100

export function isNameSegment(name: string): boolean {
  return [...name].length <= MAX_NAME_LENGTH && SEGMENT.test(name)
}

/** The last segment of a slash path - a file's name without its folders. */
export const leafName = (name: string) => name.split('/').pop() ?? name

/** Each path's shortest tail of whole segments that no other path in
 * `paths` ends with: a leaf no other path shares is the leaf, and two
 * same-leaf files (`people/edit`, `pets/edit`) keep as many folders as tell
 * them apart. A path given twice is its leaf. One count of every path's
 * tail per depth, so a gallery of thousands stays cheap. */
export function distinctLeaves(paths: string[]): Map<string, string> {
  const split = new Map(paths.map((path) => [path, path.split('/')]))
  const tail = (parts: string[], n: number) => parts.slice(-n).join('/')
  const shown = new Map([...split.keys()].map((path) => [path, path]))
  let open = [...split.keys()]
  for (let n = 1; open.length; n++) {
    const count = new Map<string, number>()
    for (const parts of split.values()) {
      const key = tail(parts, n)
      count.set(key, (count.get(key) ?? 0) + 1)
    }
    open = open.filter((path) => {
      const parts = split.get(path)!
      const key = tail(parts, n)
      if (count.get(key) !== 1 && n < parts.length) return true
      shown.set(path, key)
      return false
    })
  }
  return shown
}
