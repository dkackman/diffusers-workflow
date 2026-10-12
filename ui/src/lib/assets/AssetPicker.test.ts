import { cleanup, fireEvent, render, screen } from '@testing-library/svelte'
import { afterEach, beforeEach, expect, it, vi } from 'vitest'
import type { AssetFile } from '../types'

vi.mock('../api', () => ({ api: { listAssets: () => new Promise(() => {}) } }))

import AssetBrowseButton from './AssetBrowseButton.svelte'
import AssetPicker from './AssetPicker.svelte'
import { assetLibrary } from '../assetlib.svelte'
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

beforeEach(() => {
  assetLibrary.workspace = workspace.current
  assetLibrary.assets = [
    asset('cast/priya.png', 'image', 'common'),
    asset('iris.png'),
    asset('clips/intro.mp4', 'video', 'examples'),
    asset('cast/theme.wav', 'audio'),
  ]
})
afterEach(() => cleanup())

const tiles = () =>
  screen
    .queryAllByRole('button', { name: /^choose / })
    .map((b) => b.getAttribute('aria-label'))
    .filter((label) => label !== 'choose from the asset library')

it('shows the library by section, this workspace first, as thumbnails', () => {
  render(AssetPicker, { open: true, onpick: () => {} })
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
  render(AssetPicker, { open: true, kind: 'video', onpick: () => {} })
  expect(tiles()).toEqual(['choose clips/intro.mp4'])
})

it('searches by any part of the name', async () => {
  render(AssetPicker, { open: true, onpick: () => {} })
  await fireEvent.input(screen.getByLabelText('search the asset library'), {
    target: { value: 'pri' },
  })
  expect(tiles()).toEqual(['choose cast/priya.png'])
})

it('hands back the chosen name', async () => {
  const onpick = vi.fn()
  render(AssetPicker, { open: true, onpick })
  await fireEvent.click(screen.getByRole('button', { name: 'choose iris.png' }))
  expect(onpick).toHaveBeenCalledWith('iris.png')
})

it('offers browsing only beside an asset: value, as that kind', async () => {
  const onpick = vi.fn()
  const { unmount } = render(AssetBrowseButton, { value: 'a cat', onpick })
  expect(
    screen.queryByRole('button', { name: 'choose from the asset library' }),
  ).toBeNull()
  unmount()

  render(AssetBrowseButton, { value: 'asset:clips/intro.mp4', onpick })
  await fireEvent.click(
    screen.getByRole('button', { name: 'choose from the asset library' }),
  )
  expect(tiles()).toEqual(['choose clips/intro.mp4'])
  await fireEvent.click(
    screen.getByRole('button', { name: 'choose clips/intro.mp4' }),
  )
  expect(onpick).toHaveBeenCalledWith('asset:clips/intro.mp4')
})

it('marks the asset the field holds now', () => {
  render(AssetPicker, { open: true, current: 'iris.png', onpick: () => {} })
  expect(
    screen
      .getByRole('button', { name: 'choose iris.png' })
      .getAttribute('aria-current'),
  ).toBe('true')
})
