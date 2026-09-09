// A window.confirm replacement that renders through the app's own modal
// styling instead of the browser chrome. One state object plus a single
// <ConfirmDialog> mounted in App.svelte, mirroring the notify()/Toaster
// pair in toast.ts - callers just `await confirmDialog(...)` in place of
// `window.confirm(...)`.

type ConfirmState = {
  open: boolean
  message: string
  confirmLabel: string
  cancelLabel: string
}

export const confirmState = $state<ConfirmState>({
  open: false,
  message: '',
  confirmLabel: 'OK',
  cancelLabel: 'Cancel',
})

// At most one confirm is ever in flight - a second call while one is open
// would have no dialog to land in anyway, so it simply replaces the first.
let resolver: ((value: boolean) => void) | null = null

export function confirmDialog(
  message: string,
  options?: { confirmLabel?: string; cancelLabel?: string },
): Promise<boolean> {
  confirmState.open = true
  confirmState.message = message
  confirmState.confirmLabel = options?.confirmLabel ?? 'OK'
  confirmState.cancelLabel = options?.cancelLabel ?? 'Cancel'
  return new Promise((resolve) => {
    resolver = resolve
  })
}

export function resolveConfirm(value: boolean): void {
  confirmState.open = false
  resolver?.(value)
  resolver = null
}
