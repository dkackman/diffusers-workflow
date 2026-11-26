import js from '@eslint/js'
import ts from 'typescript-eslint'
import svelte from 'eslint-plugin-svelte'
import prettier from 'eslint-config-prettier'
import globals from 'globals'

// Empty.svelte is imported only by src/lib/ui/LoadState.svelte, so an empty
// list can never stand in for one still loading or one that failed.
const emptyPattern = {
  group: ['**/Empty.svelte'],
  message:
    'Render an empty state through src/lib/ui/LoadState.svelte, never Empty directly.',
}
// The change feed has one subscriber: resource(), whose quiet reloads
// keep edits, focus and selection, and whose latestOnly drops a reply
// overtaken by a later read. A view watching a kind declares `watch`.
// Nothing else imports the feed's values, so no spelling of a subscription
// - a chain, a feed held in a const, a `new ChangeFeed` - is left to write
// (the ChangeKind type stays importable). Pinned by scripts/lint-rules.test.ts
const feedMessage = 'Subscribe through resource() (src/lib/resource.svelte.ts)'
const feedPattern = {
  regex: '(^|/)changes\\.svelte(\\.[jt]s)?$',
  importNames: ['changes', 'ChangeFeed'],
  message: feedMessage,
}
const bitsPath = {
  name: 'bits-ui',
  message: 'Import a wrapper from src/lib/ui/ - Bits UI stays behind it.',
}
// The whole package carries every language Monaco ships; the JSON view
// needs one. A part of it (`monaco-editor/basic-languages/...`) is imported
// only by the two modules that choose which parts load. Pinned by
// scripts/lint-rules.test.ts
const monacoMessage =
  'Import Monaco through src/lib/monaco.ts, which loads the JSON language only.'
const monacoPath = {
  name: 'monaco-editor',
  allowTypeImports: true,
  message: monacoMessage,
}
const monacoParts = {
  group: ['monaco-editor/*'],
  allowTypeImports: true,
  message: monacoMessage,
}
// The modules that pick Monaco's parts: the editor and its contributions
const MONACO_OWNERS = ['src/lib/monaco.ts', 'src/lib/monacoContribs.ts']
/** The no-restricted-imports options for a block: flat config replaces a
 * rule's options rather than merging them, so each block lists them all */
const restricted = ({
  bits = false,
  empty = false,
  feed = false,
  parts = true,
}) => [
  'error',
  {
    paths: [monacoPath, ...(bits ? [bitsPath] : [])],
    patterns: [
      ...(parts ? [monacoParts] : []),
      ...(empty ? [emptyPattern] : []),
      ...(feed ? [feedPattern] : []),
    ],
  },
]
// A caught error's raw message reaches the screen nowhere: describeError
// (src/lib/apiErrors.ts, which alone reads it) gives the words, ErrorLine,
// notify.failure and Details show them with the engine text folded. A
// caught error is read by `.message` behind an `instanceof Error` (a
// ternary, an `if` or an `&&`) or a cast, or stringified whole (`String(e)`,
// `${e}`) - the last two matched by the names a caught error goes by.
// Pinned by scripts/lint-rules.test.ts
const rawMessage =
  'Show a caught error through describeError (ErrorLine, notify.failure, Details), never its raw .message.'
const CAUGHT = '/^(e|err|error)$/'
const messageSyntax = [
  "ConditionalExpression[test.operator='instanceof'][test.right.name='Error'] > MemberExpression.consequent[property.name='message']",
  "IfStatement[test.operator='instanceof'][test.right.name='Error'] > .consequent MemberExpression[property.name='message']",
  `LogicalExpression[operator='&&'][left.operator='instanceof'][left.right.name='Error'] > MemberExpression.right[property.name='message']`,
  "MemberExpression[property.name='message'][object.type='TSAsExpression'][object.typeAnnotation.typeName.name='Error']",
  `CallExpression[callee.name='String'][arguments.length=1][arguments.0.name=${CAUGHT}]`,
  `TemplateLiteral > Identifier.expressions[name=${CAUGHT}]`,
].map((selector) => ({ selector, message: rawMessage }))
const feedSyntax = [
  {
    selector:
      "CallExpression[callee.property.name='watch'][callee.object.callee.name='changes']",
    message: feedMessage,
  },
  {
    // import('./changes.svelte'), which no import rule sees
    selector: 'ImportExpression[source.value=/(^|\\/)changes\\.svelte/]',
    message: feedMessage,
  },
]
// A repeating timer is a `poll` (src/lib/poll.ts), which alone keeps the
// hidden-tab, pending-skip and rejection promises: `setInterval` is named
// nowhere else, in any spelling. Pinned by scripts/lint-rules.test.ts
const timerMessage = 'A repeating timer is poll() (src/lib/poll.ts)'
const timerSyntax = [
  "Identifier[name='setInterval']",
  "MemberExpression > Literal.property[value='setInterval']",
].map((selector) => ({ selector, message: timerMessage }))
// Destroying something the person made asks one way: destroyConfirm (the
// P5 policy) names it, counts it, says how many are starred and puts the
// button in danger, so no other module writes a Delete or Remove button of
// its own - as a `confirmLabel` (a string or a template), a chooseDialog
// choice's `label` (seen only when the choices array is written inside the
// call), or a component's `confirmLabel=` attribute. Pinned by
// scripts/lint-rules.test.ts
const destroyMessage =
  'A delete asks through destroyConfirm() (src/lib/destructive.ts)'
const DESTROYS = '/^(Delete|Remove)/'
const destroyLabel = (property) => [
  `Property[key.name='${property}'][value.value=${DESTROYS}]`,
  `Property[key.name='${property}'] > TemplateLiteral.value > TemplateElement:first-child[value.raw=${DESTROYS}]`,
]
const destroySyntax = [
  ...destroyLabel('confirmLabel'),
  ...destroyLabel('label').map(
    (s) => `CallExpression[callee.name='chooseDialog'] ${s}`,
  ),
  `SvelteAttribute[key.name='confirmLabel'] SvelteLiteral[value=${DESTROYS}]`,
  `SvelteAttribute[key.name='confirmLabel'] Literal[value=${DESTROYS}]`,
  `SvelteAttribute[key.name='confirmLabel'] TemplateElement:first-child[value.raw=${DESTROYS}]`,
].map((selector) => ({ selector, message: destroyMessage }))
/** The no-restricted-syntax options for a block, listed whole for the same
 * reason as `restricted` */
const syntax = ({
  feed = false,
  message = false,
  destroy = true,
  timers = true,
}) => [
  'error',
  ...(timers ? timerSyntax : []),
  ...(feed ? feedSyntax : []),
  ...(message ? messageSyntax : []),
  ...(destroy ? destroySyntax : []),
]
// The one module that reads a caught error's message, and the tests
const MESSAGE_OWNERS = [
  'src/lib/apiErrors.ts',
  'src/lib/test/**',
  'src/**/*.test.ts',
]
// The module that words a delete
const DESTROY_OWNERS = ['src/lib/destructive.ts']
// The feed's own module, the one subscriber and the tests that drive it
const FEED_OWNERS = [
  'src/lib/changes.svelte.ts',
  'src/lib/resource.svelte.ts',
  'src/lib/test/**',
  'src/**/*.test.ts',
]

export default ts.config(
  // Generated from the server's OpenAPI document (npm run gen:api)
  { ignores: ['src/lib/generated/**'] },
  js.configs.recommended,
  ...ts.configs.recommended,
  ...svelte.configs['flat/recommended'],
  prettier,
  ...svelte.configs['flat/prettier'],
  {
    languageOptions: { globals: { ...globals.browser } },
  },
  {
    files: ['scripts/**'],
    languageOptions: { globals: { ...globals.node } },
  },
  {
    files: ['**/*.svelte', '**/*.svelte.ts', '**/*.svelte.js'],
    languageOptions: { parserOptions: { parser: ts.parser } },
  },
  {
    rules: {
      // Workflow definitions are open JSON by design - the engine accepts
      // anything schema-valid, so the editor models them as any
      '@typescript-eslint/no-explicit-any': 'off',
    },
  },
  {
    files: ['src/**/*.{ts,svelte}'],
    ignores: ['src/lib/ui/**'],
    rules: {
      'no-restricted-imports': restricted({
        bits: true,
        empty: true,
        feed: true,
      }),
    },
  },
  {
    files: ['src/lib/ui/**/*.{ts,svelte}'],
    ignores: ['src/lib/ui/LoadState.svelte'],
    rules: { 'no-restricted-imports': restricted({ empty: true, feed: true }) },
  },
  {
    files: ['src/lib/ui/LoadState.svelte'],
    rules: { 'no-restricted-imports': restricted({ feed: true }) },
  },
  {
    files: FEED_OWNERS,
    ignores: ['src/lib/ui/**'],
    rules: {
      'no-restricted-imports': restricted({ bits: true, empty: true }),
    },
  },
  {
    files: MONACO_OWNERS,
    rules: {
      'no-restricted-imports': restricted({
        bits: true,
        empty: true,
        feed: true,
        parts: false,
      }),
    },
  },
  {
    files: ['src/**/*.{ts,svelte}'],
    ignores: [...FEED_OWNERS, ...MESSAGE_OWNERS],
    rules: { 'no-restricted-syntax': syntax({ feed: true, message: true }) },
  },
  {
    files: FEED_OWNERS,
    ignores: MESSAGE_OWNERS,
    rules: { 'no-restricted-syntax': syntax({ message: true }) },
  },
  {
    files: MESSAGE_OWNERS,
    ignores: FEED_OWNERS,
    rules: { 'no-restricted-syntax': syntax({ feed: true }) },
  },
  {
    files: DESTROY_OWNERS,
    rules: {
      'no-restricted-syntax': syntax({
        feed: true,
        message: true,
        destroy: false,
      }),
    },
  },
  {
    files: ['src/lib/poll.ts'],
    rules: {
      'no-restricted-syntax': syntax({
        feed: true,
        message: true,
        timers: false,
      }),
    },
  },
  {
    ignores: ['dist/', 'node_modules/', 'playwright-report/', 'test-results/'],
  },
)
