# Layout boxes for `templates/flux3/layout-boxes`

Read this when a request places elements at given positions. The format is
BFL's, from the Klein model card and the box captions in their reference
repository (read 2026-10-15); the stored caption `flux3/balloons_layout`
(`get_prompt`) is a complete worked example, and the clearest one to copy.

A layout caption is one prompt with two parts:

1. **Prose with tags.** The usual long caption, with each placed element
   followed by an id in angle brackets where it is described, such as
   `<obj_1>` or `<sky_1>`. Ids are short and unique; BFL's examples use a
   kind and a number.
2. **A JSON table after the prose.** A list with one object per id:
   `{"id": "obj_1", "bbox": [top, left, bottom, right], "desc": "..."}`.
   The box is on a 0 to 1000 grid over the image, whatever its pixel size,
   with `top < bottom` and `left < right`. `desc` describes that element
   alone, in the same detail as the prose.

Every tag in the prose has a row, and every row's id is tagged in the prose.
Background regions (sky, ground) can be boxed too; boxes may overlap.

Boxes are hints, not masks. A small boxed element may not appear at all, and
an element can run past its box. When one is missing, enlarge its box or give
it more words in both the prose and `desc`, and try another seed before
rewriting the layout.

Writing the boxes from a short request (BFL's `t2i_bbox` upsampler mode)
needs BFL's box planning code and is not in the catalog yet, so a layout
caption is written by hand or adapted from the stored example. Keep the
width and height on the 16-pixel grid as for any Klein run.
