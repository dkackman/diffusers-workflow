"""
Reduce a generated block of text to a known set of labelled sections.

A language model asked for a rigid format usually produces it and then keeps
going - restating the description, appending a summary, or looping until it
runs out of tokens. Downstream that trailing text is not free: a prompt is
conditioning, and a pipeline that does not truncate spends memory and attention
on whatever arrived. Rather than trying to talk the model out of it, keep the
parts that were asked for and drop the rest.
"""

import logging
import re

logger = logging.getLogger("dw")


def extract_sections(text, sections, keep_preamble=True):
    """Keep only the named sections, once each, in the order they are declared.

    A section runs from its `label:` to the end of that paragraph - the format
    these prompts use puts each field in one continuous paragraph, so a blank
    line ends it. Anything outside a named section is dropped, which is what
    removes a trailing restatement whether or not it repeats verbatim.

    Args:
        text: The generated text.
        sections: Section labels to keep, in the order they should appear.
        keep_preamble: Keep any text before the first label. These formats put
            an instruction line above the fields, which is part of the output.

    Returns:
        The reassembled text.
    """
    if not sections:
        return text.strip()

    labels = "|".join(re.escape(name) for name in sections)
    matches = list(re.finditer(rf"^({labels}):", text, re.M))

    if not matches:
        logger.warning(
            "No sections of %s found - leaving the text as it is", list(sections)
        )
        return text.strip()

    bodies = {}
    for index, match in enumerate(matches):
        name = match.group(1)
        if name in bodies:
            # A repeat is the model starting over, not new content
            continue
        # The body ends at the next label or the end of its paragraph, whichever
        # comes first. A label may sit on its own line, so leading blank space
        # is skipped before looking for the break that ends it
        end = matches[index + 1].start() if index + 1 < len(matches) else len(text)
        body = text[match.end() : end]
        stripped = body.lstrip()
        offset = len(body) - len(stripped)
        paragraph = re.search(r"\n\s*\n", body[offset:])
        if paragraph:
            body = body[offset : offset + paragraph.start()]
        bodies[name] = body.strip()

    kept = [f"{name}: {bodies[name]}" for name in sections if name in bodies]

    preamble = text[: matches[0].start()].strip()
    if keep_preamble and preamble:
        kept.insert(0, preamble)

    result = "\n\n".join(kept)
    dropped = len(text.strip()) - len(result)
    if dropped > 0:
        logger.info(f"Trimmed {dropped} characters outside the requested sections")
    return result
