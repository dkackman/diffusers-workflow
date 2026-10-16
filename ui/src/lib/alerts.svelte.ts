/** Whether a job finishing in a background tab raises an alert
 * (finishAlert.ts) - a per-browser choice, so it is kept locally. Shared
 * state: the toggle in the header sets it, and App's announce and its
 * chime-unlock listener read it. */
import { enableAlerts } from './finishAlert'
import { notify } from './toast'
import { storageGet, storageSet } from './storage'

export const alerts = $state({
  on: storageGet<boolean>('alert-finished', false),
})

export async function toggleAlerts(): Promise<void> {
  if (alerts.on) {
    alerts.on = false
  } else if (await enableAlerts()) {
    alerts.on = true
  } else {
    notify.error(
      'The browser blocked notifications for this site - allow them in its site settings',
    )
    return
  }
  storageSet('alert-finished', alerts.on)
}
