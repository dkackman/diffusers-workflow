import { expect, it } from 'vitest'
import type { AssetFile, AssetLibrary, ShadowedAsset } from '../types'
import {
  assetSections,
  folderGroup,
  matchedAssets,
  onPage,
  pickableAssets,
  viewerNames,
  visibleAssets,
} from './sections'

const asset = (
  name: string,
  origin: AssetFile['origin'] = 'workspace',
  mtime = 1,
  used = true,
): AssetFile => ({
  name,
  reference: `asset:${name}`,
  folder: name.includes('/') ? name.split('/')[0] : '',
  kind: 'image',
  size: 1,
  mtime,
  origin,
  writable: origin !== 'examples',
  url: `/inputs/${name}`,
  used_by: used ? ['portrait'] : [],
})
const shadow = (name: string, origin: AssetFile['origin']): ShadowedAsset => {
  const { url, ...rest } = asset(name, origin)
  void url
  return { ...rest, shadowed_by: 'workspace' }
}
const lib = (origin: AssetFile['origin']): AssetLibrary => ({
  origin,
  root: `/${origin}`,
  writable: origin !== 'examples',
})
const LIBS = [lib('workspace'), lib('common'), lib('examples')]
const names = (files: { name: string }[]) => files.map((f) => f.name)
const sectionsOf = (
  assets: AssetFile[],
  opts: Partial<Parameters<typeof assetSections>[3]> = {},
  shadowed: ShadowedAsset[] = [],
) =>
  assetSections(LIBS, assets, shadowed, {
    shared: false,
    filter: '',
    usedNames: new Set(
      assets.filter((a) => a.used_by?.length).map((a) => a.name),
    ),
    ...opts,
  })

it('onPage: a workspace page shows its own and the shared ones it uses', () => {
  expect(onPage(asset('a'), false)).toBe(true)
  expect(onPage(asset('b', 'common'), false)).toBe(true)
  expect(onPage(asset('c', 'common', 1, false), false)).toBe(false)
  // the shared page shows every shared one, and none of the workspace's
  expect(onPage(asset('c', 'common', 1, false), true)).toBe(true)
  expect(onPage(asset('a'), true)).toBe(false)
})

it('matchedAssets keeps names with every word, in any order and case', () => {
  const all = [asset('red/car.png'), asset('car.png'), asset('RED-car.png')]
  expect(names(matchedAssets(all, 'car red'))).toEqual([
    'red/car.png',
    'RED-car.png',
  ])
  expect(matchedAssets(all, '')).toEqual(all)
})

it('visibleAssets: newest first across folders, else the server order', () => {
  const all = [asset('x/old', 'workspace', 1), asset('y/new', 'workspace', 3)]
  expect(names(visibleAssets(all, false))).toEqual(['x/old', 'y/new'])
  expect(names(visibleAssets(all, true))).toEqual(['y/new', 'x/old'])
})

it('sections follow the search path, labelled, holding their own assets', () => {
  const all = [asset('a'), asset('s', 'common'), asset('e', 'examples')]
  const sections = sectionsOf(all)
  expect(sections.map((s) => [s.origin, s.label, names(s.assets)])).toEqual([
    ['workspace', 'This workspace', ['a']],
    ['common', 'Shared library, used here', ['s']],
    ['examples', 'Examples, used here', ['e']],
  ])
})

it('an empty workspace keeps its header; an unused shared library is left out', () => {
  const sections = sectionsOf([asset('s', 'common', 1, false)])
  expect(sections.map((s) => s.origin)).toEqual(['workspace'])
})

it('the shared page has no workspace section and keeps empty headers', () => {
  const sections = sectionsOf([asset('a')], { shared: true })
  expect(sections.map((s) => [s.origin, s.label])).toEqual([
    ['common', 'Shared library'],
    ['examples', 'Examples'],
  ])
})

it('while filtering, a section with nothing to show is skipped', () => {
  const all = matchedAssets([asset('a'), asset('s', 'common')], 's')
  expect(sectionsOf(all, { filter: 's' }).map((s) => s.origin)).toEqual([
    'common',
  ])
})

it('hidden entries are matched by the filter, and on a workspace page by use', () => {
  const shadowed = [shadow('used', 'common'), shadow('other', 'common')]
  const sections = sectionsOf([asset('used')], {}, shadowed)
  const common = sections.find((s) => s.origin === 'common')!
  expect(names(common.shadowed)).toEqual(['used'])
  const shared = sectionsOf([], { shared: true, filter: 'oth' }, shadowed)
  expect(shared.map((s) => [s.origin, names(s.shadowed)])).toEqual([
    ['common', ['other']],
  ])
})

it('pickableAssets is open, writable sections in page order', () => {
  const all = [asset('a'), asset('s', 'common'), asset('e', 'examples')]
  const sections = sectionsOf(all)
  expect(names(pickableAssets(sections, () => true))).toEqual(['a', 's'])
  expect(
    names(pickableAssets(sections, (origin) => origin !== 'workspace')),
  ).toEqual(['s'])
})

it('viewerNames steps through folders in grid order, minus a shut one', () => {
  const all = [asset('b/2'), asset('a/1'), asset('b/1')]
  const [section] = sectionsOf(all)
  const byFolder = folderGroup(
    (n) => all.find((a) => a.name === n)?.folder,
    false,
  )
  const open = viewerNames(section, byFolder, () => false, false)
  expect(open).toEqual(['a/1', 'b/2', 'b/1'])
  expect(viewerNames(section, byFolder, (g) => g === 'a', false)).toEqual([
    'b/2',
    'b/1',
  ])
  // while filtering every folder is open
  expect(viewerNames(section, byFolder, (g) => g === 'a', true)).toEqual(open)
  // newest first is one list: no folders to shut
  const flat = folderGroup((n) => all.find((a) => a.name === n)?.folder, true)
  expect(viewerNames(section, flat, (g) => g === 'a', false)).toEqual([
    'b/2',
    'a/1',
    'b/1',
  ])
})
