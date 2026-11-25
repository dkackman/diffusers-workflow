// Run `make` inside an effect root, as a component's init would, and settle
// its first effects. A test that resets its modules loads a fresh copy of
// Svelte with the module under test, so it imports this one after the reset
// too: a root from the test's own copy would not hold the module's effects.
import { flushSync } from 'svelte'

export function inEffectRoot<T>(make: () => T): {
  made: T
  stop: () => void
  flush: () => void
} {
  let made!: T
  const stop = $effect.root(() => {
    made = make()
  })
  flushSync()
  return { made, stop, flush: () => flushSync() }
}
