/** Call `fn` every `ms` - at once too, unless `immediate` is false - until
 * the returned function stops it. Written to be returned from an $effect,
 * which then stops the polling when the effect reruns or its owner goes. */
export function poll(
  fn: () => unknown,
  ms: number,
  { immediate = true }: { immediate?: boolean } = {},
): () => void {
  if (immediate) void fn()
  const timer = setInterval(() => void fn(), ms)
  return () => clearInterval(timer)
}

export const sleep = (ms: number) =>
  new Promise<void>((resolve) => setTimeout(resolve, ms))
