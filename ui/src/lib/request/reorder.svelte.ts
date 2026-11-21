// Reordering the plan from an item's handle: with the keys while it has
// focus, or by dragging it with a pointer. Pointer events with capture, not
// HTML5 drag: that does not work on touch. Every move is handed up as
// "put this before that"; the write goes out once, 800 ms after the last
// move or when the handle loses focus.
import { tick } from 'svelte'
import {
  dropIndex,
  KEY_MOVES,
  moveTarget,
  moveTo,
  type Move,
} from './reorderMath'

/** How long after the last move the order is written */
export const COMMIT_MS = 800
/** A press that travels less than this is a click, not a drag */
const SLOP_PX = 4
const FLIP_MS = 150

export const handleId = (key: string) => `req-item-${key}-handle`

export type ReorderOptions = {
  /** The live items' keys, in the order shown */
  keys: () => string[]
  /** The list whose direct children are the rows, each with `data-key` */
  list: () => HTMLElement | undefined
  onmove: (key: string, move: Move) => void
  /** The moving is over: write the order */
  oncommit: () => void
}

type Drag = {
  key: string
  pointer: number
  startY: number
  from: number
  to: number
  rows: HTMLElement[]
  boxes: { top: number; height: number }[]
  active: boolean
}

const still = () =>
  typeof window !== 'undefined' &&
  !!window.matchMedia?.('(prefers-reduced-motion: reduce)').matches

export class Reorder {
  /** The key of the row being dragged */
  dragging = $state<string | null>(null)
  #options: ReorderOptions
  #timer: ReturnType<typeof setTimeout> | null = null
  #drag: Drag | null = null
  /** Focus is being put back after the rows moved: that blur is not the
   * user leaving */
  #refocusing = false

  constructor(options: ReorderOptions) {
    this.#options = options
  }

  #rows(): HTMLElement[] {
    const list = this.#options.list()
    return list ? ([...list.children] as HTMLElement[]) : []
  }

  /** Make the move, and let the rows slide to their new places: first
   * tops, the change, then each row animated from where it was. Only ever
   * called for the user's own move, so a refresh never animates. */
  async #move(key: string, move: Move) {
    const before = this.#rows().map(
      (row) => [row, row.getBoundingClientRect().top] as const,
    )
    this.#options.onmove(key, move)
    this.#schedule()
    await tick()
    if (still()) return
    for (const [row, top] of before) {
      const dy = top - row.getBoundingClientRect().top
      if (dy && row.animate)
        row.animate(
          [{ transform: `translateY(${dy}px)` }, { transform: 'none' }],
          { duration: FLIP_MS, easing: 'ease-out' },
        )
    }
  }

  #schedule() {
    if (this.#timer) clearTimeout(this.#timer)
    this.#timer = setTimeout(() => this.commit(), COMMIT_MS)
  }

  /** Write the order now, if a move is waiting */
  commit() {
    if (!this.#timer) return
    clearTimeout(this.#timer)
    this.#timer = null
    this.#options.oncommit()
  }

  /** ArrowUp and ArrowDown move one place, Home and End to the ends;
   * focus stays on the handle */
  keydown = async (event: KeyboardEvent, key: string) => {
    const direction = KEY_MOVES[event.key]
    if (!direction || event.altKey || event.ctrlKey || event.metaKey) return
    event.preventDefault()
    const move = moveTarget(this.#options.keys(), key, direction)
    if (!move) return
    this.#refocusing = true
    await this.#move(key, move)
    document.getElementById(handleId(key))?.focus()
    this.#refocusing = false
  }

  blur = () => {
    if (!this.#refocusing) this.commit()
  }

  pointerdown = (event: PointerEvent, key: string) => {
    if (event.button !== 0 || this.#drag) return
    const rows = this.#rows()
    const from = rows.findIndex((row) => row.dataset.key === key)
    if (from < 0) return
    const handle = event.currentTarget as HTMLElement
    handle.setPointerCapture?.(event.pointerId)
    this.#drag = {
      key,
      pointer: event.pointerId,
      startY: event.clientY,
      from,
      to: from,
      rows,
      boxes: rows.map((row) => {
        const { top, height } = row.getBoundingClientRect()
        return { top, height }
      }),
      active: false,
    }
  }

  pointermove = (event: PointerEvent) => {
    const drag = this.#drag
    if (!drag || event.pointerId !== drag.pointer) return
    const dy = event.clientY - drag.startY
    if (!drag.active && Math.abs(dy) < SLOP_PX) return
    drag.active = true
    this.dragging = drag.key
    event.preventDefault()
    drag.rows[drag.from].style.transform = `translateY(${dy}px)`
    drag.rows[drag.from].classList.add('dragging')
    drag.to = dropIndex(drag.boxes, drag.from, event.clientY)
    this.#mark(drag)
  }

  /** The insertion line: on the row the dragged one would land beside */
  #mark(drag: Drag) {
    for (const row of drag.rows)
      row.classList.remove('insert-before', 'insert-after')
    if (drag.to === drag.from) return
    const below = drag.to > drag.from
    drag.rows[drag.to].classList.add(below ? 'insert-after' : 'insert-before')
  }

  #clear(): Drag | null {
    const drag = this.#drag
    this.#drag = null
    this.dragging = null
    if (!drag) return null
    drag.rows[drag.from].style.transform = ''
    drag.rows[drag.from].classList.remove('dragging')
    for (const row of drag.rows)
      row.classList.remove('insert-before', 'insert-after')
    return drag
  }

  pointerup = (event: PointerEvent) => {
    if (event.pointerId !== this.#drag?.pointer) return
    const drag = this.#clear()
    if (!drag?.active) return
    const move = moveTo(this.#options.keys(), drag.key, drag.to)
    if (move) void this.#move(drag.key, move)
  }

  /** The drag was taken away (a scroll, a lost capture) or Escape: no move */
  cancel = () => {
    this.#clear()
  }

  /** The page is going: write what is waiting */
  destroy() {
    this.#clear()
    this.commit()
  }
}
