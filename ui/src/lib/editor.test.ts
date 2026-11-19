import { describe, expect, it } from 'vitest'
import {
  CONTENT_TYPES,
  attentionBackendVariables,
  requiredVariables,
  constraintHints,
  CONSTRAINT_LABELLED_KEYS,
  coerce,
  contentTypeOptions,
  optionsWith,
  danglingReferences,
  isLongText,
  isReference,
  mediaKindFor,
  mediaLocation,
  referenceSuggestions,
  widgetFor,
  withMediaLocation,
  QUANT_PRESETS,
  presetPlaceholder,
  presetView,
} from './editor'
import type { PipelineParameter } from './types'
import schema from '../../../dw/workflow_schema.json'
import pikaTextToVideo from '../../../workflows/templates/pika-h3/text-to-video.json'

const param = (overrides: Partial<PipelineParameter>): PipelineParameter => ({
  name: 'x',
  required: false,
  default: null,
  annotation: null,
  ...overrides,
})

describe('widgetFor', () => {
  it('follows the existing value type first', () => {
    expect(widgetFor(param({}), 3)).toBe('number')
    expect(widgetFor(param({}), true)).toBe('boolean')
    expect(widgetFor(param({}), { a: 1 })).toBe('json')
    expect(widgetFor(param({}), [1, 2])).toBe('json')
  })

  it('engine references are always text, whatever the annotation says', () => {
    expect(widgetFor(param({ annotation: 'int' }), 'variable:steps')).toBe(
      'text',
    )
    expect(
      widgetFor(param({ annotation: 'bool' }), 'previous_result:gen'),
    ).toBe('text')
  })

  it('falls back to the annotation for empty values', () => {
    expect(widgetFor(param({ annotation: 'int | None' }), '')).toBe('number')
    expect(widgetFor(param({ annotation: 'float' }), '')).toBe('number')
    expect(widgetFor(param({ annotation: 'bool' }), '')).toBe('boolean')
  })
})

describe('coerce', () => {
  it('numbers become numbers, but references and non-numbers stay strings', () => {
    expect(coerce('number', '25')).toBe(25)
    expect(coerce('number', '0.5')).toBe(0.5)
    expect(coerce('number', 'variable:steps')).toBe('variable:steps')
    expect(coerce('number', 'not-a-number')).toBe('not-a-number')
  })

  it('json fields parse when valid and stay raw when not', () => {
    expect(coerce('json', '{"a": 1}')).toEqual({ a: 1 })
    expect(coerce('json', 'broken{')).toBe('broken{')
  })
})

describe('isReference', () => {
  it('recognizes every engine reference prefix', () => {
    for (const prefix of [
      'variable:',
      'previous_result:',
      'constant:',
      'prompt:',
    ]) {
      expect(isReference(prefix + 'x')).toBe(true)
    }
    expect(isReference('plain text')).toBe(false)
    expect(isReference(42)).toBe(false)
  })
})

describe('mediaKindFor', () => {
  it('reads image/video off the signature annotation when introspection has one', () => {
    expect(mediaKindFor(param({ annotation: 'PipelineImageInput' }), 'x')).toBe(
      'image',
    )
    expect(
      mediaKindFor(param({ doc_type: '`PIL.Image.Image`, *optional*' }), 'x'),
    ).toBe('image')
    expect(mediaKindFor(param({ annotation: 'VideoInput' }), 'x')).toBe('video')
    expect(mediaKindFor(param({ annotation: 'int' }), 'x')).toBeNull()
  })

  it('falls back to the naming convention when introspection carries no type at all', () => {
    // dw/introspection.py's describe_task hands back annotation: None for
    // every task argument (e.g. depth_estimator's "image")
    const untyped = param({ annotation: null })
    expect(mediaKindFor(untyped, 'image')).toBe('image')
    expect(mediaKindFor(untyped, 'mask_image')).toBe('image')
    expect(mediaKindFor(untyped, 'control_image')).toBe('image')
    expect(mediaKindFor(untyped, 'video')).toBe('video')
    expect(mediaKindFor(undefined, 'image')).toBe('image')
    expect(mediaKindFor(untyped, 'prompt')).toBeNull()
    expect(mediaKindFor(untyped, 'num_inference_steps')).toBeNull()
  })

  it('trusts a real, non-media annotation over the name heuristic', () => {
    // the name-fallback only fires when introspection gave no type at all -
    // an "image" parameter that is genuinely typed as something else (e.g.
    // a scale factor) must not be swept into the media widget
    expect(mediaKindFor(param({ annotation: 'float' }), 'image')).toBeNull()
  })
})

describe('mediaLocation / withMediaLocation', () => {
  it('reads the location out of either value shape the engine accepts', () => {
    expect(mediaLocation('photo.png')).toBe('photo.png')
    expect(mediaLocation({ location: 'photo.png' })).toBe('photo.png')
    expect(mediaLocation({ media_type: 'image', location: 'photo.png' })).toBe(
      'photo.png',
    )
    expect(mediaLocation(undefined)).toBe('')
    expect(mediaLocation({})).toBe('')
  })

  it('writes a new location back preserving the value shape it started as', () => {
    expect(withMediaLocation('old.png', 'new.png')).toBe('new.png')
    expect(withMediaLocation('', 'new.png')).toBe('new.png')
    expect(
      withMediaLocation(
        { media_type: 'image', location: 'old.png' },
        'new.png',
      ),
    ).toEqual({ media_type: 'image', location: 'new.png' })
  })
})

describe('quantization presets', () => {
  it('every preset carries a config_type and constructor arguments', () => {
    for (const [name, preset] of Object.entries(QUANT_PRESETS)) {
      if (name === 'none') {
        expect(preset).toBeNull()
      } else {
        expect(preset!.config_type).toBeTruthy()
        expect(Object.keys(preset!.arguments).length).toBeGreaterThan(0)
      }
    }
  })
})

describe('referenceSuggestions', () => {
  const workflow = {
    variables: { prompt: 'x', steps: 9 },
    steps: [
      { name: 'gen', result: { content_type: 'video/mp4' } },
      { name: 'refine' },
      { name: 'last' },
    ],
  }

  it('offers variables plus only earlier steps', () => {
    const forLast = referenceSuggestions(workflow, 2)
    expect(forLast).toContain('variable:prompt')
    expect(forLast).toContain('previous_result:gen')
    expect(forLast).toContain('previous_result:refine')
    expect(forLast).not.toContain('previous_result:last')

    const forFirst = referenceSuggestions(workflow, 0)
    expect(forFirst.filter((s) => s.startsWith('previous_result'))).toEqual([])
  })

  it('adds media suffixes for video-producing steps', () => {
    const forSecond = referenceSuggestions(workflow, 1)
    expect(forSecond).toContain('previous_result:gen.frames')
    expect(forSecond).toContain('previous_result:gen.audio')
    expect(forSecond).not.toContain('previous_result:refine.frames')
  })

  it('offers every stored prompt when the library is supplied', () => {
    const suggestions = referenceSuggestions(workflow, 0, [
      'scenic',
      'minimax/fox',
    ])
    expect(suggestions).toContain('prompt:scenic')
    expect(suggestions).toContain('prompt:minimax/fox')
    expect(referenceSuggestions(workflow, 0)).not.toContain('prompt:scenic')
  })
})

describe('referenceSuggestions for for_each steps', () => {
  const shots = [
    { name: 'open', prompt: 'a', references: [] },
    { name: 'close', prompt: 'b', seconds: 4 },
  ]
  const workflow = {
    variables: { shots },
    steps: [
      { name: 'shot', for_each: 'variable:shots', pipeline: {} },
      { name: 'grade', for_each: 'variable:shots', task: {} },
      { name: 'join', task: {} },
    ],
  }

  it('offers item: and every field an entry carries inside a for_each', () => {
    const inShot = referenceSuggestions(workflow, 0)
    expect(inShot).toEqual(
      expect.arrayContaining([
        'item:',
        'item:name',
        'item:prompt',
        'item:references',
        'item:seconds',
      ]),
    )
    expect(referenceSuggestions(workflow, 2)).not.toContain('item:')
  })

  it('offers gather: and members of an earlier for_each step', () => {
    const inJoin = referenceSuggestions(workflow, 2)
    expect(inJoin).toContain('gather:shot')
    expect(inJoin).toContain('previous_result:shot@open')
    expect(inJoin).toContain('previous_result:shot@close')
  })

  it('offers a plain previous_result: to it only from a for_each over the same list', () => {
    expect(referenceSuggestions(workflow, 1)).toContain('previous_result:shot')
    expect(referenceSuggestions(workflow, 2)).not.toContain(
      'previous_result:shot',
    )
  })

  it('suggests nothing the dangling check would flag', () => {
    for (const index of [1, 2]) {
      const offered = referenceSuggestions(workflow, index).filter(
        (s) => s.startsWith('previous_result:') || s.startsWith('gather:'),
      )
      const probe = structuredClone(workflow) as Record<string, any>
      probe.steps[index].task = { arguments: { value: offered } }
      expect(danglingReferences(probe)).toEqual([])
    }
  })
})

describe('danglingReferences', () => {
  it('flags undeclared variables and unknown or later steps', () => {
    const workflow = {
      variables: { prompt: 'x' },
      steps: [
        {
          name: 'gen',
          pipeline: {
            arguments: {
              prompt: 'variable:prompt',
              image: 'previous_result:later',
            },
          },
        },
        {
          name: 'later',
          pipeline: { arguments: { size: 'variable:missing' } },
        },
      ],
    }
    const problems = danglingReferences(workflow)
    expect(problems).toHaveLength(2)
    expect(problems[0]).toContain('previous_result:later')
    expect(problems[1]).toContain('variable:missing')
  })

  it('accepts valid references, property paths and nested values', () => {
    const workflow = {
      variables: { prompt: 'x' },
      steps: [
        { name: 'gen', result: { content_type: 'video/mp4' } },
        {
          name: 'use',
          task: {
            arguments: {
              frames: 'previous_result:gen.frames',
              nested: [{ deep: 'variable:prompt' }],
            },
          },
        },
      ],
    }
    expect(danglingReferences(workflow)).toEqual([])
  })

  it('checks prompt references only against a supplied library', () => {
    const workflow = {
      steps: [
        { name: 'gen', pipeline: { arguments: { prompt: 'prompt:scenic' } } },
      ],
    }
    // no listing - the server resolves these at run time, nothing to flag
    expect(danglingReferences(workflow)).toEqual([])
    expect(danglingReferences(workflow, ['scenic'])).toEqual([])
    const problems = danglingReferences(workflow, ['other'])
    expect(problems).toHaveLength(1)
    expect(problems[0]).toContain('prompt:scenic')
  })
})

describe('isLongText', () => {
  it('long strings and multi-line strings are long', () => {
    expect(isLongText('x'.repeat(61))).toBe(true)
    expect(isLongText('one\ntwo')).toBe(true)
  })

  it('short single-line strings and non-strings are not', () => {
    expect(isLongText('a cat')).toBe(false)
    expect(isLongText('')).toBe(false)
    expect(isLongText(25)).toBe(false)
    expect(isLongText(null)).toBe(false)
    expect(isLongText({ a: 1 })).toBe(false)
  })
})

describe('a reference is always edited as text', () => {
  const boolParam = param({ annotation: 'bool' })
  const intParam = param({ annotation: 'int' })
  it.each([
    'item:flag',
    'gather:shots',
    'asset:cast/priya.png',
    'output:run/v2/final.mp4',
  ])('%s', (value) => {
    expect(isReference(value)).toBe(true)
    expect(widgetFor(boolParam, value)).toBe('text')
    expect(widgetFor(intParam, value)).toBe('text')
  })
})

describe('contentTypeOptions', () => {
  it('keeps a value the list does not hold, so the select shows it', () => {
    expect(contentTypeOptions('audio/x-flac')).toEqual([
      ...CONTENT_TYPES,
      'audio/x-flac',
    ])
  })
  it('adds nothing for a listed value or none', () => {
    expect(contentTypeOptions('image/png')).toEqual([...CONTENT_TYPES])
    expect(contentTypeOptions(undefined)).toEqual([...CONTENT_TYPES])
  })
})

describe('optionsWith', () => {
  it('appends a current value the list lacks', () => {
    expect(optionsWith(['a', 'b'], 'c')).toEqual(['a', 'b', 'c'])
  })
  it('leaves the list alone for a listed or empty value', () => {
    expect(optionsWith(['a', 'b'], 'a')).toEqual(['a', 'b'])
    expect(optionsWith(['a', 'b'], null)).toEqual(['a', 'b'])
  })
})

describe('attentionBackendVariables', () => {
  it('finds variables feeding attention_backend at component level', () => {
    const def = {
      steps: [
        {
          pipeline: {
            configuration: {
              components: {
                transformer_ref: { attention_backend: 'variable:attn' },
              },
            },
          },
        },
      ],
    }
    expect(attentionBackendVariables(def)).toEqual(new Set(['attn']))
  })

  it('finds pipeline-level and nested workflow-step variables', () => {
    const def = {
      steps: [
        { pipeline: { configuration: { attention_backend: 'variable:a' } } },
        {
          workflow: {
            steps: [
              {
                pipeline: {
                  configuration: { attention_backend: 'variable:b' },
                },
              },
            ],
          },
        },
      ],
    }
    expect(attentionBackendVariables(def)).toEqual(new Set(['a', 'b']))
  })

  it('ignores literals and other keys', () => {
    const def = {
      variables: { attn: 'sparse' },
      steps: [
        {
          pipeline: {
            configuration: { attention_backend: 'sage', other: 'variable:x' },
          },
        },
      ],
    }
    expect(attentionBackendVariables(def)).toEqual(new Set())
    expect(attentionBackendVariables(null)).toEqual(new Set())
  })
})

describe('requiredVariables', () => {
  it('names the variables a constraint marks required', () => {
    const def = {
      variables: { reference_audio: null, seed: null },
      variable_constraints: {
        reference_audio: { required: true, min_length: 1 },
        seed: { min_length: 1 },
      },
    }
    expect(requiredVariables(def)).toEqual(new Set(['reference_audio']))
    expect(requiredVariables({ variables: {} })).toEqual(new Set())
  })
})

describe('constraintHints', () => {
  const hints = (
    rule: Record<string, unknown>,
    value: unknown = 1,
  ): string[] | undefined =>
    constraintHints({
      variables: { v: value },
      variable_constraints: { v: rule },
    }).v

  it('states min_frames and max_frames', () => {
    expect(hints({ min_frames: 9, max_frames: 97 })).toEqual(['9 to 97'])
    expect(hints({ min_frames: 9 })).toEqual(['9 or more'])
    expect(hints({ max_frames: 97 })).toEqual(['97 or less'])
  })

  it('states modulus and remainder in the engine form, with the reason', () => {
    expect(hints({ modulus: 32, remainder: 0, reason: 'the VAE' })).toEqual([
      '32 * n + 0 - the VAE',
    ])
  })

  it('states snap up after the grid', () => {
    expect(
      hints({ modulus: 8, remainder: 1, min_frames: 9, snap: 'up' }),
    ).toEqual(['9 or more, 8 * n + 1, an off-grid value rounds up'])
  })

  it('states min_length for a string or a list by the default type', () => {
    expect(hints({ min_length: 1 }, '')).toEqual(['a non-empty string'])
    expect(hints({ min_length: 3 }, 'abc')).toEqual(['at least 3 characters'])
    expect(hints({ min_length: 1 }, [])).toEqual(['a non-empty list'])
    expect(hints({ min_length: 2 }, ['a'])).toEqual(['at least 2 entries'])
    expect(hints({ min_length: 1 }, null)).toEqual(['a non-empty string'])
  })

  it('states max_words_per_second with seconds_from as a warning', () => {
    expect(
      hints({ max_words_per_second: 2.5, seconds_from: 'seconds' }, ''),
    ).toEqual(['about 2.5 words a second of seconds (a warning)'])
  })

  it('leaves required to the field mark, and a bare reason gives no line', () => {
    expect(hints({ required: true, reason: 'a voice' }, null)).toBeUndefined()
  })

  it('keeps the reason as text', () => {
    expect(hints({ min_frames: 1, reason: '<img src=x onerror=1>' })).toEqual([
      '1 or more - <img src=x onerror=1>',
    ])
  })

  it('falls back to key: value for a key it has no label for', () => {
    expect(hints({ min_frames: 1, new_rule: 7 })).toEqual([
      '1 or more, new_rule: 7',
    ])
  })

  it('shows a list-entry rule under the list it reaches', () => {
    const def = {
      variables: { shots: [{ name: 'a', num_frames: 124 }] },
      variable_constraints: {
        num_frames: { modulus: 17, remainder: 5, min_frames: 124 },
      },
      steps: [
        {
          name: 'shot',
          for_each: 'variable:shots',
          pipeline: { arguments: { num_frames: 'item:num_frames' } },
        },
      ],
    }
    expect(constraintHints(def)).toEqual({
      shots: ["each entry's num_frames: 124 or more, 17 * n + 5"],
    })
  })

  it('gives nothing for a workflow without constraints', () => {
    expect(constraintHints({ variables: { prompt: 'x' } })).toEqual({})
    expect(constraintHints(null)).toEqual({})
  })

  it('has a label for every key the schema declares', () => {
    const declared = Object.keys(schema.$defs.variable_constraint.properties)
    expect(declared.length).toBeGreaterThan(0)
    expect(
      declared.filter((k) => !CONSTRAINT_LABELLED_KEYS.includes(k)),
    ).toEqual([])
  })
})

describe('presetView and presetPlaceholder', () => {
  const definition = {
    variables: {
      resolution: '480p',
      width: 832,
      height: 480,
      steps: 30,
      fps: 24,
    },
    presets: {
      resolution: {
        '480p': { width: 832, height: 480 },
        '768p': { width: 1344, height: 768, steps: 40 },
      },
    },
  }

  it("lists each selector's bundles and the union of what they set", () => {
    expect(presetView(definition)).toEqual({
      resolution: {
        bundles: ['480p', '768p'],
        sets: ['width', 'height', 'steps'],
      },
    })
    expect(presetView({ variables: {} })).toEqual({})
  })

  it("the default bundle is the selector's declared value", () => {
    const shifted = {
      ...definition,
      variables: { ...definition.variables, resolution: '768p' },
    }
    expect(presetPlaceholder(shifted, {}, 'width')).toBe('1344')
    expect(presetPlaceholder(definition, {}, 'width')).toBe('832')
  })

  it('a picked bundle moves the placeholders', () => {
    const picked = { resolution: '768p' }
    expect(presetPlaceholder(definition, picked, 'width')).toBe('1344')
    expect(presetPlaceholder(definition, picked, 'height')).toBe('768')
    expect(presetPlaceholder(definition, picked, 'steps')).toBe('40')
  })

  it('an explicit value suppresses the placeholder', () => {
    expect(
      presetPlaceholder(
        definition,
        { resolution: '768p', width: '1000' },
        'width',
      ),
    ).toBeUndefined()
    // an empty field is no value: the bundle still shows
    expect(
      presetPlaceholder(definition, { resolution: '768p', width: '' }, 'width'),
    ).toBe('1344')
  })

  it('a variable the chosen bundle does not set keeps its declared default', () => {
    // fps: no bundle sets it; steps: only the unchosen bundle does
    expect(presetPlaceholder(definition, {}, 'fps')).toBeUndefined()
    expect(presetPlaceholder(definition, {}, 'steps')).toBeUndefined()
    expect(
      presetPlaceholder(definition, { resolution: 'nope' }, 'width'),
    ).toBeUndefined()
  })

  it('a reference in the selector names no bundle', () => {
    const referenced = {
      ...definition,
      variables: { ...definition.variables, resolution: 'variable:other' },
    }
    expect(presetPlaceholder(referenced, {}, 'width')).toBeUndefined()
  })

  it('reads the real pika-h3 template: picking the other bundle moves width and height', () => {
    const view = presetView(pikaTextToVideo)
    const [selector] = Object.keys(view)
    const bundles = (
      pikaTextToVideo as {
        presets: Record<string, Record<string, Record<string, unknown>>>
      }
    ).presets[selector]
    expect(view[selector].bundles).toEqual(Object.keys(bundles))
    const declared = (pikaTextToVideo.variables as Record<string, unknown>)[
      selector
    ] as string
    const other = view[selector].bundles.find((b) => b !== declared)!
    for (const name of ['width', 'height']) {
      expect(presetPlaceholder(pikaTextToVideo, {}, name)).toBe(
        String(bundles[declared][name]),
      )
      expect(
        presetPlaceholder(pikaTextToVideo, { [selector]: other }, name),
      ).toBe(String(bundles[other][name]))
    }
  })
})
