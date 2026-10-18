import { existsSync } from 'node:fs';
import { describe, expect, it } from 'vitest';
import { MODEL_IDS } from '../support/matrix';

// The per-model sweeps run from one file per model (so they run in parallel): a new model must get its own.
describe('per-model test files', () => {
  it.each(MODEL_IDS)('%s has its graph and popup suites', (id) => {
    expect(existsSync(`tests/model/graph/${id}.test.ts`)).toBe(true);
    expect(existsSync(`tests/ui/popups/${id}.test.ts`)).toBe(true);
  });
});
