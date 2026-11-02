# The six-cut coverage set

Part of the `storyboard` skill; step 2 says when to read this. Shot Board
(MIT) covers a moment with six cuts. They are a checklist of what an editor
would want, not a rule that every moment needs all six.

## The six

| Cut | Frames | Use it to |
| --- | --- | --- |
| establishing | the whole place, people small | place the scene before it begins |
| medium | the character from the waist up, in the place | carry action and dialogue |
| face close-up | the face fills the frame | show what the character feels |
| over-the-shoulder | one character's shoulder in the foreground, the other facing us | tie two people into one exchange |
| insert | one object, hand or detail | give the plot a thing to look at |
| low angle | the camera below, looking up | give a character or a place weight |

Each cut has a `shot_size` or camera value in `cinematography`'s
vocabulary; use that word in the instruction rather than a new one.

## Procedure

1. Draw the hero: the cut that best shows the moment, usually the medium, with
   `templates/flux3/multi-reference` (the character and location references).
2. Keep the hero. Redraw each other cut with the hero as "image one" and the
   character reference as "image two", so the face and the place are
   the hero's.
3. Per redraw the instruction names the cut and changes nothing else:
   "image one is the scene. Redraw it as a low angle shot, the camera below
   the character looking up. Keep the character, clothing and light."
4. Review each against the hero: same face, same clothes, same light. A cut
   whose face changed is redrawn once with the close-up reference first; a
   second miss is asked about rather than rerolled again.

An over-the-shoulder needs both characters in the references. Two people in
one still is the pipeline's weakest case: faces can swap or merge, so check
both.

For `templates/consistent-set` (FLUX.1 Kontext, not Klein), the same idea
applies as edits of one base picture; it is the older route and its look
differs from Klein's.
