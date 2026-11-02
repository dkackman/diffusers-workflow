import { beforeEach, expect, it } from 'vitest'
import { adoptTokenFromUrl, getApiToken, setApiToken } from './token'

beforeEach(() => {
  setApiToken('')
  window.history.replaceState(null, '', '/')
})

it('saves a login link token and drops it from the address bar', () => {
  window.history.replaceState(null, '', '/?token=abc123&x=1#/jobs/42')
  adoptTokenFromUrl()
  expect(getApiToken()).toBe('abc123')
  expect(localStorage.getItem('dw-api-token')).toBe('"abc123"')
  expect(window.location.search).toBe('?x=1')
  expect(window.location.hash).toBe('#/jobs/42')
})

it('leaves a URL without one alone', () => {
  setApiToken('kept')
  window.history.replaceState(null, '', '/#/workflows')
  adoptTokenFromUrl()
  expect(getApiToken()).toBe('kept')
  expect(window.location.hash).toBe('#/workflows')
})

it('an empty token keeps the saved one', () => {
  setApiToken('kept')
  window.history.replaceState(null, '', '/?token=')
  adoptTokenFromUrl()
  expect(getApiToken()).toBe('kept')
  expect(window.location.search).toBe('')
})
