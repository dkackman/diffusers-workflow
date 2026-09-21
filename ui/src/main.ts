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

const app = mount(App, { target: document.getElementById('app')! })

export default app
