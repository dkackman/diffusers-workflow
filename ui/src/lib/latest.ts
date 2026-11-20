/** A stale-response guard for a call a page makes again and again - a
 * listing reloaded when its filter or workspace changes. Each call takes a
 * ticket as it starts and asks it before using the answer:
 *
 *   const begin = latestOnly()
 *   async function load() {
 *     const current = begin()
 *     const reply = await api.list(filter)
 *     if (current()) rows = reply.rows
 *   }
 *
 * An answer that lands after a later call started is dropped, so the page
 * never shows the reply to a question it has stopped asking. One guard per
 * kind of call. */
export function latestOnly(): () => () => boolean {
  let issued = 0
  return () => {
    const mine = ++issued
    return () => mine === issued
  }
}
