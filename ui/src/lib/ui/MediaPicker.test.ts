import { cleanup, fireEvent, render, screen } from '@testing-library/svelte'
import { afterEach, beforeEach, expect, it, vi } from 'vitest'
import type { AssetFile, GalleryFile } from '../types'

vi.mock('../api', async () => (await import('../test/apiMock')).apiModule())

import MediaPicker from './MediaPicker.svelte'
import { assetLibrary } from '../assetlib.svelte'
import { outputLibrary } from '../outputlib.svelte'
import { workspace } from '../workspace.svelte'
import { api } from '../test/apiMock'

api.listAssets.mockImplementation(() => new Promise(() => {}))
api.gallery.mockImplementation(() => new Promise(() => {}))

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
afterEach(() => {
  cleanup()
  localStorage.clear()
})

const tiles = () =>
  screen
    .queryAllByRole('button', { name: /^choose / })
    .map((b) => b.getAttribute('aria-label'))
    .filter((label) => label !== 'choose another file')

const showRuns = () =>
  fireEvent.click(screen.getByRole('button', { name: 'Earlier jobs' }))

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

it("orders a library's assets naturally: take2 before take10", () => {
  assetLibrary.items = [asset('take10.png'), asset('take2.png')]
  render(MediaPicker, { open: true, onpick: () => {} })
  expect(tiles()).toEqual(['choose take2.png', 'choose take10.png'])
})

it('names the kind it chooses with its article', () => {
  render(MediaPicker, { open: true, kind: 'audio', onpick: () => {} })
  expect(screen.getByRole('heading', { level: 2 })).toHaveTextContent(
    'Choose an audio',
  )
})

it('a clip shows its play badge', async () => {
  render(MediaPicker, { open: true, onpick: () => {} })
  const clip = screen.getByRole('button', { name: 'choose clips/intro.mp4' })
  expect(clip.querySelector('.videobadge')).toBeTruthy()
  expect(clip.classList).toContain('filetile')
  await showRuns()
  const shot = screen.getByRole('button', {
    name: `choose clip/${RUN1}/shot.mp4`,
  })
  expect(shot.querySelector('.videobadge')).toBeTruthy()
  // a picture is never a badge's place
  expect(
    screen
      .getByRole('button', { name: `choose portrait/${RUN1}/out.png` })
      .querySelector('.videobadge'),
  ).toBeNull()
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

const toggle = (name: RegExp) => screen.getByRole('button', { name })

it('group toggles are disabled while searching', async () => {
  render(MediaPicker, { open: true, onpick: () => {} })
  const heading = toggle(/Shared library/)
  expect((heading as HTMLButtonElement).disabled).toBe(false)
  await fireEvent.click(heading)
  expect(tiles()).not.toContain('choose cast/priya.png')
  await fireEvent.input(screen.getByLabelText('search the asset library'), {
    target: { value: 'pri' },
  })
  const searching = toggle(/Shared library/) as HTMLButtonElement
  expect(searching.disabled).toBe(true)
  expect(searching.title).toBe('Showing every match while the search is on')
  expect(tiles()).toEqual(['choose cast/priya.png'])
})

it('the collapse state is per workspace and tab', async () => {
  render(MediaPicker, { open: true, onpick: () => {} })
  await fireEvent.click(toggle(/Shared library/))
  expect(
    localStorage.getItem(
      `dw-collapsed-media-picker:${workspace.current}:assets`,
    ),
  ).toContain('common')
  await showRuns()
  expect(toggle(/^portrait/).getAttribute('aria-expanded')).toBe('true')
  expect(
    localStorage.getItem(
      `dw-collapsed-media-picker:${workspace.current}:outputs`,
    ),
  ).toBeNull()
})

it('the current tile says current', () => {
  render(MediaPicker, {
    open: true,
    current: 'asset:iris.png',
    onpick: () => {},
  })
  const tile = screen.getByRole('button', { name: 'choose iris.png' })
  expect(tile.textContent).toContain('current')
  expect(
    screen.getByRole('button', { name: 'choose cast/priya.png' }).textContent,
  ).not.toContain('current')
})
