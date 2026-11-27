// The sidebar's nav rows: each section's route, label and icon
import {
  BookCopy,
  ClipboardList,
  Database,
  FolderOpen,
  Images,
  LayoutDashboard,
  Layers,
  ListTodo,
  MessageSquareText,
  Server,
  SlidersHorizontal,
  SquarePen,
} from '@lucide/svelte'
import {
  SECTION_LABELS,
  type ServerSection,
  type SharedSection,
  type WsSection,
} from './routes'

export const WS_ITEMS: {
  section: WsSection
  label: string
  icon: typeof Layers
}[] = [
  {
    section: 'overview',
    label: SECTION_LABELS.ws.overview,
    icon: LayoutDashboard,
  },
  {
    section: 'workflows',
    label: SECTION_LABELS.ws.workflows,
    icon: Layers,
  },
  { section: 'jobs', label: SECTION_LABELS.ws.jobs, icon: ListTodo },
  {
    section: 'requests',
    label: SECTION_LABELS.ws.requests,
    icon: ClipboardList,
  },
  { section: 'gallery', label: SECTION_LABELS.ws.gallery, icon: Images },
  { section: 'assets', label: SECTION_LABELS.ws.assets, icon: FolderOpen },
  { section: 'edit', label: SECTION_LABELS.ws.edit, icon: SquarePen },
]
export const SHARED_ITEMS: {
  section: SharedSection
  label: string
  icon: typeof Layers
}[] = [
  {
    section: 'prompts',
    label: SECTION_LABELS.shared.prompts,
    icon: MessageSquareText,
  },
  {
    section: 'assets',
    label: SECTION_LABELS.shared.assets,
    icon: FolderOpen,
  },
  {
    section: 'examples',
    label: SECTION_LABELS.shared.examples,
    icon: BookCopy,
  },
]
export const SERVER_ITEMS: {
  section: ServerSection
  label: string
  icon: typeof Layers
}[] = [
  { section: 'models', label: SECTION_LABELS.server.models, icon: Database },
  {
    section: 'preferences',
    label: SECTION_LABELS.server.preferences,
    icon: SlidersHorizontal,
  },
  { section: 'status', label: SECTION_LABELS.server.status, icon: Server },
]
