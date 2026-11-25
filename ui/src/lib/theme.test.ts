import { expect, it } from 'vitest'
import { nextTheme, themeLabel, type Theme } from './theme'

it('the theme button says what a press does', () => {
  const labels: Record<Theme, string> = {
    system: 'Theme: system. Switch to light.',
    light: 'Theme: light. Switch to dark.',
    dark: 'Theme: dark. Switch to system.',
  }
  for (const t of Object.keys(labels) as Theme[])
    expect(themeLabel(t)).toBe(labels[t])
})

it('nextTheme walks system, light, dark and back', () => {
  expect(nextTheme('system')).toBe('light')
  expect(nextTheme('light')).toBe('dark')
  expect(nextTheme('dark')).toBe('system')
})
