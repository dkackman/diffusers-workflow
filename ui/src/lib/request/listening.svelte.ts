// Whether an agent session will hear a note now: its notes channel polls
// the workspace's request list, and the server says when it last did
// (GET .../notes-channel). One shared answer per workspace, asked every
// LISTENING_POLL_MS while a note list is on screen and the tab is visible,
// and once more just before a save. A failed read counts as no session.
import { api } from '../api'

export const LISTENING_POLL_MS = 15_000

const visible = () =>
  typeof document === 'undefined' || document.visibilityState !== 'hidden'

export class NotesChannel {
  listening = $state(false)
  #workspace: string
  #holders = 0
  #timer: ReturnType<typeof setInterval> | null = null
  #asked = 0

  constructor(workspace: string) {
    this.#workspace = workspace
  }

  /** Ask now; the answer is also the shared state. A reply overtaken by a
   * later question is dropped */
  async refresh(): Promise<boolean> {
    const asked = ++this.#asked
    let answer: boolean
    try {
      answer = (await api.notesChannel(this.#workspace)).listening
    } catch {
      answer = false
    }
    if (asked === this.#asked) this.listening = answer
    return this.listening
  }

  /** A note list is on screen: keep the answer fresh until the returned
   * function says it has gone. Written to be returned from an $effect */
  hold(): () => void {
    if (++this.#holders === 1) {
      document.addEventListener('visibilitychange', this.#onVisibility)
      this.#start()
    }
    let held = true
    return () => {
      if (!held) return
      held = false
      if (--this.#holders > 0) return
      this.#stop()
      document.removeEventListener('visibilitychange', this.#onVisibility)
    }
  }

  #start() {
    if (this.#timer !== null || !visible()) return
    void this.refresh()
    this.#timer = setInterval(() => void this.refresh(), LISTENING_POLL_MS)
  }

  #stop() {
    if (this.#timer !== null) clearInterval(this.#timer)
    this.#timer = null
  }

  #onVisibility = () => (visible() ? this.#start() : this.#stop())
}

// Not reactive: a channel is made on first ask, inside a component's
// $derived, and its `listening` is the state
const channels: Record<string, NotesChannel> = {}

/** The shared answer for `workspace` */
export function listening(workspace: string): NotesChannel {
  return (channels[workspace] ??= new NotesChannel(workspace))
}
