// The theme choice: 'system' follows the OS and leaves no mark, the other two
// set data-theme on the root and are remembered in localStorage 'dw-theme'
export type Theme = 'system' | 'light' | 'dark'

export function initialTheme(): Theme {
  try {
    const stored = localStorage.getItem('dw-theme')
    return stored === 'light' || stored === 'dark' ? stored : 'system'
  } catch {
    return 'system'
  }
}

/** What a press on the theme button switches to. */
export function nextTheme(t: Theme): Theme {
  return t === 'system' ? 'light' : t === 'light' ? 'dark' : 'system'
}

/** The theme button's name and title: where it is, and what a press does. */
export function themeLabel(t: Theme): string {
  return `Theme: ${t}. Switch to ${nextTheme(t)}.`
}

// Applies the next theme in the cycle and returns it
export function cycleTheme(current: Theme): Theme {
  const next = nextTheme(current)
  if (next === 'system') delete document.documentElement.dataset.theme
  else document.documentElement.dataset.theme = next
  try {
    if (next === 'system') localStorage.removeItem('dw-theme')
    else localStorage.setItem('dw-theme', next)
  } catch {
    /* fine - applies for this session */
  }
  return next
}
