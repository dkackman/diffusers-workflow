import { cleanup, render, screen } from '@testing-library/svelte'
import { afterEach, describe, expect, it } from 'vitest'
import Markdown from './Markdown.svelte'

describe('Markdown', () => {
  afterEach(cleanup)
  it('renders the subset as elements', () => {
    render(Markdown, {
      props: { source: '## Rule\n\n- **one**\n\n[docs](https://x.dev)' },
    })
    expect(screen.getByRole('heading', { name: 'Rule' })).toBeTruthy()
    expect(screen.getByText('one').tagName).toBe('STRONG')
    const link = screen.getByRole('link', { name: 'docs' })
    expect(link.getAttribute('href')).toBe('https://x.dev/')
    expect(link.getAttribute('rel')).toBe('noopener noreferrer')
  })
  it('never renders markup in the source', () => {
    render(Markdown, { props: { source: '<img src=x onerror=alert(1)>' } })
    expect(document.querySelector('img')).toBeNull()
    expect(screen.getByText('<img src=x onerror=alert(1)>')).toBeTruthy()
  })
  it('never links a javascript: url', () => {
    render(Markdown, { props: { source: '[x](javascript:alert(1))' } })
    expect(screen.queryByRole('link')).toBeNull()
  })
})
