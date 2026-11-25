import { mount } from 'svelte'
import './app.css'
import './frames.css'
import './patterns.css'

import App from './App.svelte'
import { adoptTokenFromUrl } from './lib/token'

adoptTokenFromUrl()

const target = document.getElementById('app')!
target.replaceChildren()
const app = mount(App, { target })

export default app
