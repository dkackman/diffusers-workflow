// jsdom never fires `change` when a field loses focus; a browser does, when
// the value moved since the field took focus. A component test that leans
// on that - a form field that commits on change, flushed by a blur - installs
// this for its duration. Returns the uninstall.
type Field = HTMLInputElement | HTMLTextAreaElement

const isField = (target: EventTarget | null): target is Field =>
  target instanceof HTMLInputElement || target instanceof HTMLTextAreaElement

export function changeOnBlur(): () => void {
  const atFocus = new WeakMap<Field, string>()
  const onFocus = (event: FocusEvent) => {
    if (isField(event.target)) atFocus.set(event.target, event.target.value)
  }
  const onBlur = (event: FocusEvent) => {
    const field = event.target
    if (!isField(field) || atFocus.get(field) === field.value) return
    atFocus.delete(field)
    field.dispatchEvent(new Event('change', { bubbles: true }))
  }
  document.addEventListener('focusin', onFocus)
  document.addEventListener('focusout', onBlur)
  return () => {
    document.removeEventListener('focusin', onFocus)
    document.removeEventListener('focusout', onBlur)
  }
}
