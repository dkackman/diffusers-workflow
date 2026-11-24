import { expect, it } from 'vitest'
import { catalogMatches } from './catalogSearch'
import type { WorkflowCard } from '../types'

const card = (extra: Partial<WorkflowCard>): WorkflowCard =>
  ({ kinds: [], steps: 1, variables: 0, ...extra }) as WorkflowCard

const names = ['clip', 'flux-image', 'photo', 'song']
const details = {
  clip: card({ kinds: ['video'], shape: 'shot', description: 'A cat walks.' }),
  'flux-image': card({ kinds: ['image'], shape: 'image' }),
  photo: card({
    kinds: ['image'],
    shape: 'image',
    description: 'A flux still.',
  }),
  song: card({ kinds: ['audio'], shape: 'audio', traits: ['has-audio'] }),
}

it('video of a cat finds the video workflows', () => {
  expect(catalogMatches(names, details, 'video of a cat')).toEqual(['clip'])
  expect(catalogMatches(names, details, 'video')).toEqual(['clip'])
})

it('every matched word ranks higher', () => {
  // photo holds image and flux, flux-image holds both too: a tie, by name
  expect(catalogMatches(names, details, 'flux image')).toEqual([
    'flux-image',
    'photo',
  ])
  // one more word in common beats the rest
  expect(catalogMatches(names, details, 'cat shot flux')).toEqual(['clip'])
  expect(catalogMatches(names, details, 'flux')).toEqual([
    'flux-image',
    'photo',
  ])
})

it('stop words alone match everything', () => {
  expect(catalogMatches(names, details, 'a the of')).toEqual(names)
  expect(catalogMatches(names, details, '')).toEqual(names)
})

it('matches traits and kinds', () => {
  expect(catalogMatches(names, details, 'has-audio')).toEqual(['song'])
  expect(catalogMatches(names, details, 'audio')).toEqual(['song'])
  expect(catalogMatches(names, details, 'nothing here')).toEqual([])
})

it('a name that holds the words beats a description that mentions them', () => {
  const list = ['animatic', 'text-to-image']
  const info = {
    animatic: card({ description: 'Text and image boards.' }),
    'text-to-image': card({}),
  }
  expect(catalogMatches(list, info, 'text to image')).toEqual([
    'text-to-image',
    'animatic',
  ])
})

it('a shallower workflow ranks ahead of a same-named one in a folder', () => {
  const list = [
    'templates/flux3/text-to-image',
    'templates/ltx/text-to-image',
    'templates/text-to-image',
  ]
  expect(catalogMatches(list, {}, 'text to image')).toEqual([
    'templates/text-to-image',
    'templates/flux3/text-to-image',
    'templates/ltx/text-to-image',
  ])
})
