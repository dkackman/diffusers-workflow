import js from '@eslint/js'
import ts from 'typescript-eslint'
import svelte from 'eslint-plugin-svelte'
import prettier from 'eslint-config-prettier'
import globals from 'globals'

// Empty.svelte is imported only by src/lib/ui/LoadState.svelte, so an empty
// list can never stand in for one still loading or one that failed.
// Flat config replaces a rule's options rather than merging them, so the
// pattern rides in the bits-ui block and again, alone, for src/lib/ui/**
const emptyPattern = {
  group: ['**/Empty.svelte'],
  message:
    'Render an empty state through src/lib/ui/LoadState.svelte, never Empty directly.',
}
const emptyPatterns = { patterns: [emptyPattern] }

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
      'no-restricted-imports': ['error', { patterns: [emptyPattern] }],
    },
  },
  // The change feed has one subscriber: resource(), whose quiet reloads
  // keep edits, focus and selection, and whose latestOnly drops a reply
  // overtaken by a later read. A view watching a kind declares `watch`
  {
    files: ['src/**/*.{ts,svelte}'],
    ignores: ['src/lib/resource.svelte.ts', 'src/lib/test/**', '**/*.test.ts'],
    rules: {
      'no-restricted-syntax': [
        'error',
        {
          selector:
            "CallExpression[callee.property.name='watch'][callee.object.callee.name='changes']",
          message: 'Subscribe through resource() (src/lib/resource.svelte.ts)',
        },
      ],
    },
  },
  {
    ignores: ['dist/', 'node_modules/', 'playwright-report/', 'test-results/'],
  },
)
