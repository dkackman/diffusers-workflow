import { mount } from 'svelte'
import './app.css'

// Apply a manual theme choice before first paint; absent = follow the system
try {
  const theme = localStorage.getItem('dw-theme')
  if (theme === 'light' || theme === 'dark') {
    document.documentElement.dataset.theme = theme
  }
} catch {
  /* storage unavailable - system theme applies */
}
import App from './App.svelte'
import { restoreWorkspace } from './lib/workspace.svelte'

// Before the first request: every call is scoped to the selected workspace,
// and a page that loaded against the default first would flash the wrong one
restoreWorkspace()

const app = mount(App, { target: document.getElementById('app')! })

export default app
