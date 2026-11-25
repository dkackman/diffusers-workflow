// Which of a step's arguments take an image or a video rather than plain
// text, and the two shapes such a value is written in.
import type { PipelineParameter } from './types'

export type MediaKind = 'image' | 'video' | 'audio'

// A pipeline's __call__ signature types media arguments as a union like
// PipelineImageInput (PIL.Image.Image | np.ndarray | torch.Tensor | lists of
// those) - introspection stringifies whatever the annotation or the
// docstring's type column says, and both carry the word "image"/"video".
// No \b word boundary: PyTorch/diffusers type names are often one run of
// camelCase (PipelineImageInput, VaeImageProcessor's output types, ...)
// with no separator before "Image"/"Video" for \b to catch.
const IMAGE_TYPE_PATTERN = /image/i
const VIDEO_TYPE_PATTERN = /video/i

// **kwargs-only callables (from_pretrained) and dw's own tasks (see
// dw/introspection.py's describe_task, e.g. depth_estimator's "image") carry
// no annotation at all - the same argument names workflows/*.json already
// use (img2img's `image`, ControlNet's `control_image`/`conditioning_image`,
// inpainting's `mask_image`) are the only signal left.
const IMAGE_NAME_PATTERN =
  /^(image|mask_image|control_image|conditioning_image|init_image|input_image|ip_adapter_image|reference_image|style_image|depth_map)(_\d+)?$/i
const VIDEO_NAME_PATTERN =
  /^(video|control_video|conditioning_video|input_video)(_\d+)?$/i

/** Whether an argument takes an image/video rather than plain text - and
 * which kind. Reference values (variable:, previous_result:, ...) are the
 * caller's job to exclude first; they resolve to media at run time but are
 * always edited as the reference string itself. */
export function mediaKindFor(
  parameter: PipelineParameter | undefined,
  key: string,
): MediaKind | null {
  const annotation = parameter?.annotation ?? ''
  const docType = parameter?.doc_type ?? ''
  const typeText = `${annotation} ${docType}`
  if (IMAGE_TYPE_PATTERN.test(typeText)) return 'image'
  if (VIDEO_TYPE_PATTERN.test(typeText)) return 'video'
  return annotation || docType ? null : mediaKindByName(key)
}

export function mediaKindByName(key: string): MediaKind | null {
  if (IMAGE_NAME_PATTERN.test(key)) return 'image'
  if (VIDEO_NAME_PATTERN.test(key)) return 'video'
  return null
}

/** The file path or URL a media argument's value names, whichever shape it
 * is in: a bare string, or the `{ location: ... }` object the engine also
 * accepts (see dw/references.py's FROM_FILE_KEY) - optionally alongside a
 * `media_type`. */
export function mediaLocation(value: unknown): string {
  if (typeof value === 'string') return value
  if (
    value !== null &&
    typeof value === 'object' &&
    !Array.isArray(value) &&
    typeof (value as Record<string, unknown>).location === 'string'
  ) {
    return (value as Record<string, unknown>).location as string
  }
  return ''
}

/** Write a new location into a media argument's value, preserving its shape:
 * an object value keeps its other keys (e.g. `media_type`), anything else
 * becomes a plain path string - both are forms the engine accepts. */
export function withMediaLocation(value: unknown, location: string): unknown {
  if (value !== null && typeof value === 'object' && !Array.isArray(value)) {
    return { ...(value as Record<string, unknown>), location }
  }
  return location
}
