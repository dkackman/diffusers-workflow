/** The alert a finished job raises in a tab nobody is looking at.
 *
 * A desktop notification needs a secure context - https, or localhost - and
 * this UI is routinely opened over plain http from another machine
 * (http://<gpu-box>:8765), where the Notification API is not offered. There
 * the alert is a short chime instead. The browser only lets a page make
 * sound after the person has interacted with it, so the chime is unlocked on
 * the click that turns alerts on, and again on the first click after a
 * reload. */
export type AlertMode = 'notification' | 'chime'

export function alertMode(): AlertMode {
  return typeof Notification === 'function' && globalThis.isSecureContext
    ? 'notification'
    : 'chime'
}

let audio: AudioContext | null = null

export function unlockChime(): void {
  try {
    audio ??= new AudioContext()
    void audio.resume()
  } catch {
    /* no Web Audio - the tab title still carries the news */
  }
}

/** Asks for permission where that applies; false when it was refused. */
export async function enableAlerts(): Promise<boolean> {
  if (alertMode() === 'notification') {
    if (Notification.permission === 'granted') return true
    return (await Notification.requestPermission()) === 'granted'
  }
  unlockChime()
  return true
}

function chime(): void {
  if (!audio) return
  const now = audio.currentTime
  const tone = audio.createOscillator()
  const gain = audio.createGain()
  tone.frequency.value = 880
  gain.gain.setValueAtTime(0.15, now)
  gain.gain.exponentialRampToValueAtTime(0.0001, now + 0.6)
  tone.connect(gain).connect(audio.destination)
  tone.start(now)
  tone.stop(now + 0.6)
}

export function alertFinished(text: string, href: string): void {
  if (alertMode() === 'notification' && Notification.permission === 'granted') {
    const note = new Notification('diffusers-workflow', { body: text })
    note.onclick = () => {
      window.focus()
      location.hash = href.replace(/^#/, '')
      note.close()
    }
    return
  }
  chime()
}
