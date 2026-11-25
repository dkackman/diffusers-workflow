import { expect, it } from 'vitest'
import { findingPlace } from './findingPlace'

const doc = {
  variables: { prompt: 'a cat', 'seed-2': 7 },
  steps: [{ name: 'generate' }, { name: 'upscale' }],
}

it('places a step path', () => {
  expect(findingPlace('steps[1]', doc)).toEqual({
    kind: 'step',
    index: 1,
    name: 'upscale',
  })
  expect(findingPlace('steps[0].pipeline.arguments.prompt', doc)).toEqual({
    kind: 'step',
    index: 0,
    name: 'generate',
  })
})

it('places a variable path', () => {
  expect(findingPlace('variables.prompt', doc)).toEqual({
    kind: 'variable',
    name: 'prompt',
  })
  expect(findingPlace('arguments.seed-2[0]', doc)).toEqual({
    kind: 'variable',
    name: 'seed-2',
  })
  // a name the workflow does not declare has no field to go to
  expect(findingPlace('variables.missing', doc)).toBeNull()
  expect(findingPlace('variables.prompts', doc)).toBeNull()
})

it('an index past the end places nothing', () => {
  expect(findingPlace('steps[2]', doc)).toBeNull()
  expect(findingPlace('steps[0]', {})).toBeNull()
  expect(findingPlace(null, doc)).toBeNull()
  expect(findingPlace('model_name', doc)).toBeNull()
})
