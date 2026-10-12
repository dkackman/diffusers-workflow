import { cleanup, fireEvent, render, screen } from '@testing-library/svelte'
import { afterEach, beforeEach, expect, it, vi } from 'vitest'
import type { AssetFile, GalleryFile } from '../types'

vi.mock('../api', () => ({
  api: {
    listAssets: () => new Promise(() => {}),
    gallery: () => new Promise(() => {}),
  },
}))

import BrowseButton from './BrowseButton.svelte'
import MediaPicker from './MediaPicker.svelte'
import { assetLibrary } from '../assetlib.svelte'
import { outputLibrary } from '../outputlib.svelte'
import { workspace } from '../workspace.svelte'

const asset = (
  name: string,
  kind: AssetFile['kind'] = 'image',
  origin: AssetFile['origin'] = 'workspace',
): AssetFile => ({
  name,
  reference: `asset:${name}`,
  folder: name.includes('/') ? name.split('/')[0] : '',
  kind,
  size: 1,
  mtime: 1,
  origin,
  writable: origin !== 'examples',
  url: `/inputs/${name}`,
})

const output = (
  folder: string,
  runId: string,
  version: number,
  file: string,
  kind: GalleryFile['kind'] = 'image',
): GalleryFile => ({
  name: `${folder}/${runId}/${file}`,
  folder,
  subfolder: '',
  run_id: runId,
  version,
  url: `/outputs/${folder}/${runId}/${file}`,
  kind,
  size: 1,
  mtime: version,
  label: file,
})

const RUN1 = '20261001-000000-aaaaaaaa'
const RUN2 = '20261002-000000-bbbbbbbb'

beforeEach(() => {
  assetLibrary.workspace = workspace.current
  assetLibrary.items = [
    asset('cast/priya.png', 'image', 'common'),
    asset('iris.png'),
    asset('clips/intro.mp4', 'video', 'examples'),
    asset('cast/theme.wav', 'audio'),
  ]
  outputLibrary.workspace = workspace.current
  // newest first, as the gallery lists them
  outputLibrary.items = [
    output('portrait', RUN2, 2, 'out.png'),
    output('clip', RUN1, 1, 'shot.mp4', 'video'),
    output('portrait', RUN1, 1, 'out.png'),
  ]
})
afterEach(() => cleanup())

const tiles = () =>
  screen
    .queryAllByRole('button', { name: /^choose / })
    .map((b) => b.getAttribute('aria-label'))
    .filter((label) => label !== 'choose another file')

const showRuns = () =>
  fireEvent.click(screen.getByRole('button', { name: 'Earlier runs' }))

it('shows the asset library by section, this workspace first, as thumbnails', () => {
  render(MediaPicker, { open: true, onpick: () => {} })
  expect(tiles()).toEqual([
    'choose cast/theme.wav',
    'choose iris.png',
    'choose cast/priya.png',
    'choose clips/intro.mp4',
  ])
  const headings = screen.getAllByRole('heading', { level: 3 })
  expect(
    headings.map((h) => h.textContent?.replace(/\s+/g, ' ').trim()),
  ).toEqual(['This workspace (2)', 'Shared library (1)', 'Examples (1)'])
  const priya = screen.getByRole('button', { name: 'choose cast/priya.png' })
  expect(priya.querySelector('img')?.getAttribute('src')).toContain(
    '/api/gallery/asset%3Acast/priya.png/thumbnail',
  )
})

it('keeps to the kind a field takes', () => {
  render(MediaPicker, { open: true, kind: 'video', onpick: () => {} })
  expect(tiles()).toEqual(['choose clips/intro.mp4'])
})

it('searches by any part of the name', async () => {
  render(MediaPicker, { open: true, onpick: () => {} })
  await fireEvent.input(screen.getByLabelText('search the asset library'), {
    target: { value: 'pri' },
  })
  expect(tiles()).toEqual(['choose cast/priya.png'])
})

it('hands back an asset as the reference a field takes', async () => {
  const onpick = vi.fn()
  render(MediaPicker, { open: true, onpick })
  await fireEvent.click(screen.getByRole('button', { name: 'choose iris.png' }))
  expect(onpick).toHaveBeenCalledWith('asset:iris.png')
})

it('marks the asset the field holds now', () => {
  render(MediaPicker, {
    open: true,
    current: 'asset:iris.png',
    onpick: () => {},
  })
  expect(
    screen
      .getByRole('button', { name: 'choose iris.png' })
      .getAttribute('aria-current'),
  ).toBe('true')
})

it('lists earlier runs by workflow, newest first', async () => {
  render(MediaPicker, { open: true, onpick: () => {} })
  await showRuns()
  expect(tiles()).toEqual([
    `choose portrait/${RUN2}/out.png`,
    `choose portrait/${RUN1}/out.png`,
    `choose clip/${RUN1}/shot.mp4`,
  ])
  const v2 = screen.getByRole('button', {
    name: `choose portrait/${RUN2}/out.png`,
  })
  expect(v2.textContent).toContain('v2')
})

it('hands back a run file as written, or as latest when asked', async () => {
  const onpick = vi.fn()
  render(MediaPicker, { open: true, onpick })
  await showRuns()
  await fireEvent.click(
    screen.getByRole('button', { name: `choose portrait/${RUN1}/out.png` }),
  )
  expect(onpick).toHaveBeenLastCalledWith(`output:portrait/${RUN1}/out.png`)
})

it('picks the latest form with the box ticked', async () => {
  const onpick = vi.fn()
  render(MediaPicker, { open: true, onpick })
  await showRuns()
  await fireEvent.click(screen.getByRole('checkbox'))
  await fireEvent.click(
    screen.getByRole('button', { name: `choose portrait/${RUN1}/out.png` }),
  )
  expect(onpick).toHaveBeenLastCalledWith('output:portrait/latest/out.png')
})

it('opens on earlier runs for an output: value, latest ticked and marked', () => {
  render(MediaPicker, {
    open: true,
    current: 'output:portrait/latest/out.png',
    onpick: () => {},
  })
  expect((screen.getByRole('checkbox') as HTMLInputElement).checked).toBe(true)
  // latest resolves to the newest run that has the file
  expect(
    screen
      .getByRole('button', { name: `choose portrait/${RUN2}/out.png` })
      .getAttribute('aria-current'),
  ).toBe('true')
})

it('offers browsing only beside a file reference, as that kind', async () => {
  const onpick = vi.fn()
  const { unmount } = render(BrowseButton, { value: 'a cat', onpick })
  expect(
    screen.queryByRole('button', { name: 'choose another file' }),
  ).toBeNull()
  unmount()

  render(BrowseButton, { value: 'asset:clips/intro.mp4', onpick })
  await fireEvent.click(
    screen.getByRole('button', { name: 'choose another file' }),
  )
  expect(tiles()).toEqual(['choose clips/intro.mp4'])
  await fireEvent.click(
    screen.getByRole('button', { name: 'choose clips/intro.mp4' }),
  )
  expect(onpick).toHaveBeenCalledWith('asset:clips/intro.mp4')
})
