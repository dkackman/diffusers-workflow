import { expect, it } from 'vitest'
import { cardReadings, meterText, meterTitle } from './cardMemory'
import type { MemoryInfo } from './types'

const card = (device: string, allocated: number, total = 24576) => ({
  device,
  live: true,
  stale: false,
  reason: null,
  age_seconds: 0,
  info: {
    gpu_available: true,
    gpu_device_name: 'NVIDIA GeForce RTX 3090',
    gpu_memory_allocated_mb: allocated,
    gpu_memory_total_mb: total,
  },
})

it('reads every card from workers, not just the top-level first card', () => {
  const a = card('cuda:0', 18636.8)
  const memory: MemoryInfo = { ...a, workers: [a, card('cuda:1', 3174.4)] }
  const cards = cardReadings(memory)
  expect(cards.map((c) => c.device)).toEqual(['cuda:0', 'cuda:1'])
  expect(cards[1].pct).toBeCloseTo(12.9, 1)
  expect(meterText(cards)).toBe('18.2·3.1/24.0 GB')
  expect(meterTitle(cards).split('\n')).toHaveLength(2)
})

it('falls back to the top level when the server sends no workers', () => {
  const single: MemoryInfo = { ...card('cuda:0', 1024), device: undefined }
  const cards = cardReadings(single)
  expect(cards).toHaveLength(1)
  expect(meterText(cards)).toBe('1.0/24.0 GB')
})

it('pairs each card when their sizes differ', () => {
  const a = card('cuda:0', 1024, 24576)
  const cards = cardReadings({
    ...a,
    workers: [a, card('cuda:1', 2048, 12288)],
  })
  expect(meterText(cards)).toBe('1.0/24.0 · 2.0/12.0 GB')
})

it('a card without a reading is listed but not metered', () => {
  const a = card('cuda:0', 1024)
  const b = { ...card('cuda:1', 0), info: null, live: false, age_seconds: null }
  const cards = cardReadings({ ...a, workers: [a, b] })
  expect(cards[1].available).toBe(false)
  expect(meterText(cards)).toBe('1.0/24.0 GB')
})
