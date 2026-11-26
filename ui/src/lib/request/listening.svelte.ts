// Whether an agent session will hear a note now: its notes channel polls
// the workspace's request list, and the server says when it last did
// (GET .../notes-channel). One shared answer per workspace, asked every
// LISTENING_POLL_MS while a note list is on screen and the tab is visible,
// and once more just before a save. A failed read counts as no session.
import { api } from '../api'
import { poll } from '../poll'

export const LISTENING_POLL_MS = 15_000

export class NotesChannel {
  listening = $state(false)
  #workspace: string
  #holders = 0
  #stop: (() => void) | null = null
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
    if (++this.#holders === 1)
      this.#stop = poll(() => this.refresh(), LISTENING_POLL_MS)
    let held = true
    return () => {
      if (!held) return
      held = false
      if (--this.#holders > 0) return
      this.#stop?.()
      this.#stop = null
    }
  }
}

// Not reactive: a channel is made on first ask, inside a component's
// $derived, and its `listening` is the state
const channels: Record<string, NotesChannel> = {}

/** The shared answer for `workspace` */
export function listening(workspace: string): NotesChannel {
  return (channels[workspace] ??= new NotesChannel(workspace))
}
