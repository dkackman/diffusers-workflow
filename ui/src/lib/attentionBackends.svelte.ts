/** The attention backends this server can run, from its last good health
 * poll. The server lists only what its device supports ('sparse' only on a
 * CUDA server with the kernels), so a dropdown built from it offers nothing
 * that would fall back. Empty until a poll answers - and after an older
 * server that does not report the field - which leaves the text inputs. */
export const attentionBackends = $state({ list: [] as string[] })

/** Take a health poll's answer. A failed poll (null) or a server that does
 * not report backends keeps the last known list rather than blanking it. */
export function setAttentionBackends(list: string[] | null | undefined): void {
  if (list) attentionBackends.list = list
}
