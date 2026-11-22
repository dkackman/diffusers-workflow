import { render } from '@testing-library/svelte'
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'

const fake = vi.hoisted(() => {
  const state = {
    value: '',
    content: [] as Array<() => void>,
    blur: [] as Array<() => void>,
  }
  const editor = {
    onDidChangeModelContent: (fn: () => void) => {
      state.content.push(fn)
      return { dispose() {} }
    },
    onDidBlurEditorWidget: (fn: () => void) => {
      state.blur.push(fn)
      return { dispose() {} }
    },
    getValue: () => state.value,
    setValue: (v: string) => (state.value = v),
    hasTextFocus: () => false,
    saveViewState: () => null,
    restoreViewState: () => {},
    dispose() {},
  }
  const monaco = {
    editor: {
      createModel: (v: string) => {
        state.value = v
        return { dispose() {} }
      },
      create: () => editor,
      setTheme: () => {},
    },
    Uri: { parse: (s: string) => s },
  }
  return { state, monaco }
})

vi.mock('../monaco', () => ({
  setupMonaco: async () => fake.monaco,
  currentTheme: () => 'vs',
}))

import JsonEditor from './JsonEditor.svelte'

function type(text: string) {
  fake.state.value = text
  fake.state.content.forEach((f) => f())
}

type Change = (v: string, o?: { quiet?: boolean }) => void

async function mount(onchange: Change) {
  let flush: () => void = () => {}
  const unregister = vi.fn()
  const view = render(JsonEditor, {
    value: '{}',
    onchange,
    register: (f: () => void) => {
      flush = f
      return unregister
    },
  })
  await vi.waitFor(() => expect(fake.state.content.length).toBe(1))
  return { flush: () => flush(), unregister, view }
}

describe('JsonEditor apply as you type', () => {
  beforeEach(() => {
    fake.state.content = []
    fake.state.blur = []
    window.matchMedia ??= (() => ({
      addEventListener() {},
      removeEventListener() {},
    })) as never
  })
  afterEach(() => vi.useRealTimers())

  it('applies quietly 300 ms after typing stops', async () => {
    const onchange = vi.fn<Change>()
    await mount(onchange)
    vi.useFakeTimers()
    type('{"a":1}')
    vi.advanceTimersByTime(299)
    expect(onchange).not.toHaveBeenCalled()
    type('{"a":12}')
    vi.advanceTimersByTime(299)
    expect(onchange).not.toHaveBeenCalled()
    vi.advanceTimersByTime(1)
    expect(onchange).toHaveBeenCalledExactlyOnceWith('{"a":12}', {
      quiet: true,
    })
  })

  it('flushes a pending JSON edit before saving', async () => {
    const onchange = vi.fn<Change>()
    const { flush } = await mount(onchange)
    vi.useFakeTimers()
    type('{"b":2}')
    flush()
    expect(onchange).toHaveBeenCalledExactlyOnceWith('{"b":2}')
    vi.advanceTimersByTime(1000)
    expect(onchange).toHaveBeenCalledTimes(1)
  })

  it('does not re-apply an unchanged value', async () => {
    const onchange = vi.fn<Change>()
    const { flush } = await mount(onchange)
    flush()
    fake.state.blur.forEach((f) => f())
    expect(onchange).not.toHaveBeenCalled()
    vi.useFakeTimers()
    type('{"c":3}')
    vi.advanceTimersByTime(300)
    fake.state.blur.forEach((f) => f())
    flush()
    expect(onchange).toHaveBeenCalledTimes(1)
  })

  it('unregisters the flush on teardown', async () => {
    const { unregister, view } = await mount(vi.fn<Change>())
    view.unmount()
    expect(unregister).toHaveBeenCalled()
  })
})
