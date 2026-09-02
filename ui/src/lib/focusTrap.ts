// Svelte action implementing the WAI-ARIA modal dialog focus contract:
// on mount, move focus into the node; while mounted, Tab/Shift+Tab cycles
// only among the node's own focusable elements; on destroy, focus returns
// to whatever had focus when the dialog opened (its trigger). One place for
// this rather than duplicating it per dialog component.

const FOCUSABLE_SELECTOR = [
  'a[href]',
  'button:not([disabled])',
  'textarea:not([disabled])',
  'input:not([disabled])',
  'select:not([disabled])',
  '[tabindex]:not([tabindex="-1"])',
].join(', ')

function focusables(node: HTMLElement): HTMLElement[] {
  return Array.from(
    node.querySelectorAll<HTMLElement>(FOCUSABLE_SELECTOR),
  ).filter((el) => el.getClientRects().length > 0)
}

export function focusTrap(node: HTMLElement) {
  const trigger = document.activeElement as HTMLElement | null

  // Prefer the dialog's first focusable element; fall back to the dialog
  // node itself, which carries tabindex="-1" for exactly this case.
  const first = focusables(node)[0]
  ;(first ?? node).focus()

  function onKeydown(event: KeyboardEvent) {
    if (event.key !== 'Tab') return
    const items = focusables(node)
    if (items.length === 0) {
      event.preventDefault()
      node.focus()
      return
    }
    const firstEl = items[0]
    const lastEl = items[items.length - 1]
    const active = document.activeElement as HTMLElement | null
    const atEdge = !active || !items.includes(active)

    if (event.shiftKey) {
      if (atEdge || active === firstEl) {
        event.preventDefault()
        lastEl.focus()
      }
    } else if (atEdge || active === lastEl) {
      event.preventDefault()
      firstEl.focus()
    }
  }

  node.addEventListener('keydown', onKeydown)

  return {
    destroy() {
      node.removeEventListener('keydown', onKeydown)
      trigger?.focus()
    },
  }
}
