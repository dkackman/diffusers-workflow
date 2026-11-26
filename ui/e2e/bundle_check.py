"""Whether the built UI the fixture server serves is behind its sources.

The e2e specs drive ui/dist, not ui/src, and Playwright does not build it: a
checkout that had pulled or edited the UI since its last `npm run build`
tested the old bundle, passing or failing for code that was no longer there
(#1686).
"""

import os

# What `vite build` reads from ui/. The unit tests beside the sources are not
# in the bundle, so editing one leaves it current
SOURCES = ("src", "index.html", "vite.config.ts", "svelte.config.js")


def source_files(ui):
    for name in SOURCES:
        path = os.path.join(ui, name)
        if os.path.isfile(path):
            yield path
        for folder, _, files in os.walk(path):
            for file in files:
                if not file.endswith(".test.ts"):
                    yield os.path.join(folder, file)


def bundle_problem(ui):
    """Why ui/dist cannot be tested, or None when it is current."""
    built = os.path.join(ui, "dist", "index.html")
    if not os.path.isfile(built):
        return "ui/dist is not built: run `npm run build` in ui/ first"
    newest = max(source_files(ui), key=os.path.getmtime, default=None)
    if newest and os.path.getmtime(newest) > os.path.getmtime(built):
        return (
            f"ui/dist is older than ui/{os.path.relpath(newest, ui)}: "
            "run `npm run build` in ui/ first"
        )
    return None
