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

// Applies the next theme in the cycle and returns it
export function cycleTheme(current: Theme): Theme {
  const next: Theme =
    current === 'system' ? 'light' : current === 'light' ? 'dark' : 'system'
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
