// The timeline summary the gallery metadata route returns for an .otioz
// (GalleryMetadata.media, kind "timeline"): tracks as rows of clips and
// gaps. The server types `media` as a free dict, so the shape is held here.

export interface TimelineItem {
  type: string
  name: string
  start_frame: number | null
  frames: number
  rate: number
  seconds: number | null
  media?: string
  /** The `output:` reference of the clip's file; absent in older bundles. */
  source?: string | null
}

export interface TimelineTrack {
  kind: string
  name: string
  clip_count: number
  seconds: number
  items: TimelineItem[]
}

export interface TimelineSummary {
  kind: 'timeline'
  name: string
  otio_version: string | null
  has_audio_track: boolean
  tracks: TimelineTrack[]
  members: string[]
}

export function isTimeline(media: unknown): media is TimelineSummary {
  if (!media || typeof media !== 'object') return false
  const m = media as Record<string, unknown>
  return m.kind === 'timeline' && Array.isArray(m.tracks)
}

/** Each item's share of the track (gaps included), in order. */
export function shares(track: TimelineTrack): number[] {
  const total = track.items.reduce((sum, item) => sum + (item.seconds ?? 0), 0)
  if (total <= 0) return track.items.map(() => 0)
  return track.items.map((item) => (item.seconds ?? 0) / total)
}

/** "name · 4.2 s" */
export function label(item: TimelineItem): string {
  const seconds =
    item.seconds === null ? '' : ` · ${Number(item.seconds.toFixed(1))} s`
  return `${item.name}${seconds}`
}
