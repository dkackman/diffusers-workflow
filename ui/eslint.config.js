import js from '@eslint/js'
import ts from 'typescript-eslint'
import svelte from 'eslint-plugin-svelte'
import prettier from 'eslint-config-prettier'
import globals from 'globals'

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
        },
      ],
    },
  },
  {
    ignores: ['dist/', 'node_modules/', 'playwright-report/', 'test-results/'],
  },
)
