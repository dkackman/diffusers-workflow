// A stand-in for the `api` object (api.ts) in a component test. Every key
// the real `api` has is a `vi.fn` made on first use, so a test stubs only
// what its page calls - and a stub for a key `api` no longer has does not
// compile, where a hand-written mock object kept it for ever. A method the
// test did not stub throws its own name, not "is not a function".
//
// The one a test file mocks the module with:
//   vi.mock('../api', async () =>
//     (await import('../test/apiMock')).apiModule(),
//   )
// and imports itself to stub and to assert on:
//   import { api } from '../test/apiMock'
//   api.listPrompts.mockResolvedValue({ ... })
//   expect(api.savePrompt).toHaveBeenCalledWith('mine', ...)
// The module's other exports (`ApiError`, `TERMINAL_STATUSES`, `outputUrl`)
// stay the real ones; `apiModule({ streamJobEvents })` replaces one that
// would reach the network.
import { vi, type Mock } from 'vitest'
import type { api as realApi } from '../api'

type Api = typeof realApi

/** Each `api` method as a mock taking the real arguments. What it answers
 * is left open: a test hands back only the fields its page reads. */
export type ApiMock = {
  [K in keyof Api]: Mock<(...args: Parameters<Api[K]>) => any>
}

// Kept on the global object, not in this module: a test that resets the
// module registry (`vi.resetModules`) gets this file again, and the `api`
// its page then imports must still be the one the test stubbed
const STORE = Symbol.for('dw.test.apiMock')
const made = ((globalThis as Record<symbol, unknown>)[STORE] ??= new Map<
  string,
  Mock
>()) as Map<string, Mock>

function unstubbed(key: string): Mock {
  return vi.fn(() => {
    throw new Error(`api.${key} was called, and this test did not stub it`)
  })
}

export const api = new Proxy({} as ApiMock, {
  get(_target, key) {
    if (typeof key !== 'string') return undefined
    let fn = made.get(key)
    if (!fn) made.set(key, (fn = unstubbed(key)))
    return fn
  },
  // `'key' in api` and a spread see what has been stubbed or called so far
  has: (_target, key) => typeof key === 'string' && made.has(key),
})

/** Put every method back to unstubbed and forget its calls - for a file
 * whose tests must not see each other's stubs:
 *   beforeEach(resetApi) */
export function resetApi(): void {
  for (const fn of made.values()) fn.mockReset()
}

/** The mocked `../api` module: the real one with `api` replaced, and any
 * other export named in `over`. */
export async function apiModule(
  over: Record<string, unknown> = {},
): Promise<Record<string, unknown>> {
  const actual = await vi.importActual<Record<string, unknown>>('../api')
  return { ...actual, api, ...over }
}
