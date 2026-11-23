import js from '@eslint/js'
import ts from 'typescript-eslint'
import svelte from 'eslint-plugin-svelte'
import prettier from 'eslint-config-prettier'
import globals from 'globals'

// Empty.svelte is imported only by src/lib/ui/LoadState.svelte, so an empty
// list can never stand in for one still loading or one that failed. Off
// until Task 12 moves the last page onto LoadState; Task 12 sets it true.
// Flat config replaces a rule's options rather than merging them, so the
// pattern rides in the bits-ui block and again, alone, for src/lib/ui/**
const EMPTY_OUTSIDE_LOADSTATE = false
const emptyPattern = {
  group: ['**/Empty.svelte'],
  message:
    'Render an empty state through src/lib/ui/LoadState.svelte, never Empty directly.',
}
const emptyPatterns = EMPTY_OUTSIDE_LOADSTATE
  ? { patterns: [emptyPattern] }
  : {}

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
      'no-restricted-imports': [
        'error',
        {
          paths: [
            {
              name: 'bits-ui',
              message:
                'Import a wrapper from src/lib/ui/ - Bits UI stays behind it.',
            },
          ],
          ...emptyPatterns,
        },
      ],
    },
  },
  {
    files: ['src/lib/ui/**/*.{ts,svelte}'],
    ignores: ['src/lib/ui/LoadState.svelte'],
    rules: {
      'no-restricted-imports': [
        EMPTY_OUTSIDE_LOADSTATE ? 'error' : 'off',
        { patterns: [emptyPattern] },
      ],
    },
  },
  {
    ignores: ['dist/', 'node_modules/', 'playwright-report/', 'test-results/'],
  },
)
