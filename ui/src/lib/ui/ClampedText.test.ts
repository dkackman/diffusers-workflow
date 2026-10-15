import { cleanup, fireEvent, render, screen } from '@testing-library/svelte'
import { afterEach, describe, expect, it, vi } from 'vitest'
import ClampedText from './ClampedText.svelte'

// jsdom lays nothing out, so overflow is faked on the element's heights
function fakeHeights(scroll: number, client: number) {
  vi.spyOn(HTMLElement.prototype, 'scrollHeight', 'get').mockReturnValue(scroll)
  vi.spyOn(HTMLElement.prototype, 'clientHeight', 'get').mockReturnValue(client)
}

describe('ClampedText', () => {
  afterEach(() => {
    cleanup()
    vi.restoreAllMocks()
  })

  it('offers no toggle when the text fits', () => {
    fakeHeights(40, 40)
    render(ClampedText, { props: { text: 'short' } })
    expect(screen.getByText('short')).toBeTruthy()
    expect(screen.queryByRole('button')).toBeNull()
  })

  it('folds long text and expands and collapses it', async () => {
    fakeHeights(400, 80)
    render(ClampedText, { props: { text: 'long text', lines: 3 } })
    const text = screen.getByText('long text')
    expect(text.classList.contains('clamped')).toBe(true)
    const toggle = screen.getByRole('button', { name: 'Show more' })
    expect(toggle.getAttribute('aria-expanded')).toBe('false')
    await fireEvent.click(toggle)
    expect(text.classList.contains('clamped')).toBe(false)
    await fireEvent.click(screen.getByRole('button', { name: 'Show less' }))
    expect(text.classList.contains('clamped')).toBe(true)
  })

  it('opens already expanded when its key was left expanded', async () => {
    localStorage.clear()
    fakeHeights(400, 80)
    const first = render(ClampedText, {
      props: { text: 'long text', rememberKey: 'k' },
    })
    await fireEvent.click(screen.getByRole('button', { name: 'Show more' }))
    first.unmount()
    render(ClampedText, { props: { text: 'long text', rememberKey: 'k' } })
    expect(screen.getByText('long text').classList.contains('clamped')).toBe(
      false,
    )
  })

  it('renders markdown as elements when asked', () => {
    fakeHeights(40, 40)
    render(ClampedText, { props: { text: '- **one**', markdown: true } })
    expect(screen.getByText('one').tagName).toBe('STRONG')
  })
})
