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
