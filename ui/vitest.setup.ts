import '@testing-library/jest-dom/vitest'

// jsdom has no Web Animations API; Svelte transitions call element.animate
// and would throw on mount. A stub that finishes at once keeps the DOM
// assertions about what mounted, without animating anything.
if (typeof Element !== 'undefined' && !Element.prototype.animate) {
  Element.prototype.animate = function () {
    const animation = {
      finished: Promise.resolve(),
      cancel() {},
      finish() {},
      onfinish: null as null | (() => void),
      oncancel: null as null | (() => void),
      currentTime: 0,
      playState: 'finished',
    }
    queueMicrotask(() => animation.onfinish?.())
    return animation as unknown as Animation
  }
}

// jsdom has no matchMedia either; `prefersReducedMotion` (svelte/motion)
// reads it at import. Nothing in the suite asserts on motion, so every
// query answers "does not match".
if (typeof window !== 'undefined' && !window.matchMedia) {
  window.matchMedia = (query: string) =>
    ({
      matches: false,
      media: query,
      onchange: null,
      addEventListener() {},
      removeEventListener() {},
      addListener() {},
      removeListener() {},
      dispatchEvent: () => false,
    }) as MediaQueryList
}
