// The production build in a real browser: every model loads and renders without errors, and the controls, popups and
// generation loop work end to end.
import { expect, test, type Page } from '@playwright/test';
import { MODELS, type ModelId } from '../src/model/config';
import { formatCount } from '../src/format';

const IDS = Object.keys(MODELS) as ModelId[];

/** Fails the test on any uncaught exception or console error in the page. */
function watchErrors(page: Page): string[] {
  const errors: string[] = [];
  page.on('pageerror', (e) => errors.push(`pageerror: ${e.message}`));
  page.on('console', (m) => {
    if (m.type() === 'error') errors.push(`console: ${m.text()}`);
  });
  return errors;
}

async function load(page: Page, query: string) {
  await page.goto(`./${query}`);
  await expect(page.locator('#panel')).toBeVisible({ timeout: 30_000 });
  await expect(page.locator('#stats .stat').first()).toBeVisible();
}

const stat = (page: Page, name: string) =>
  page.locator('#stats .stat', { has: page.locator(`span:text-is("${name}")`) }).locator('b');

const select = (page: Page, label: string) =>
  page.locator('#controls label.row', { has: page.locator(`.row-head span:text-is("${label}")`) }).locator('select');

/** The canvas has drawn a scene: a screenshot of a blank canvas compresses to almost nothing. */
async function expectDrawn(page: Page) {
  await page.waitForTimeout(1500);
  const png = await page.locator('#app canvas').screenshot();
  expect(png.byteLength).toBeGreaterThan(20_000);
}

for (const id of IDS) {
  for (const backend of ['', '&webgl']) {
    test(`${id} loads and renders${backend ? ' (WebGL2)' : ''}`, async ({ page }) => {
      const errors = watchErrors(page);
      await load(page, `?model=${id}${backend}`);
      await expect(page.locator('#title')).toContainText(MODELS[id].name);
      if (backend) await expect(page.locator('#title')).toContainText('WebGL2');
      await expect(stat(page, 'Parameters')).toContainText(formatCount(paramsOf(id)));
      await expect(page.locator('.op-label').first()).toBeAttached();
      await expectDrawn(page);
      expect(errors).toEqual([]);
    });
  }
}

/** Published totals (see tests/model/reference.test.ts, which checks the builder against them). */
function paramsOf(id: ModelId): number {
  return {
    'gpt2': 124_439_808, 'gpt2-xl': 1_557_611_200, 'llama-3-8b': 8_030_261_248, 'llama-3-70b': 70_553_706_496,
    'mixtral-8x7b': 46_702_792_704, 'deepseek-v2-lite': 15_706_484_224, 'gemma-3-1b': 999_885_952,
  }[id];
}

test('falls back to Llama-3-8B for an unknown model', async ({ page }) => {
  await load(page, '?model=nonexistent&webgl');
  await expect(page.locator('#title')).toContainText('Llama-3-8B');
});

test('controls change the pass', async ({ page }) => {
  const errors = watchErrors(page);
  await load(page, '?model=gemma-3-1b&webgl');

  await select(page, 'Attention').selectOption('eager');
  await expect(stat(page, 'Eager score tensors, one layer')).toBeVisible();

  await select(page, 'Mode').selectOption('decode');
  await expect(page.locator('#title')).toContainText('decode');
  await expect(stat(page, 'FLOPs per generated token')).toBeVisible();

  await select(page, 'KV cache').selectOption('static');
  await expect(stat(page, 'KV cache filled')).toContainText('of 32,768 slots');

  await select(page, 'Model').selectOption('deepseek-v2-lite');
  await expect(page.locator('#title')).toContainText('DeepSeek-V2-Lite');
  await expect(select(page, 'MLA cache')).toBeVisible();
  await select(page, 'MLA cache').selectOption('latent');
  await select(page, 'Model').selectOption('gpt2');
  await expect(select(page, 'MLA cache')).toHaveCount(0);

  await page.keyboard.press('o');
  await page.keyboard.press('f');
  await expectDrawn(page);
  expect(errors).toEqual([]);
});

test('an op label opens its popup, and Esc closes it', async ({ page }) => {
  const errors = watchErrors(page);
  await load(page, '?model=llama-3-8b&webgl');
  // Labels can overlap in the 3D view: dispatch the click to the label itself.
  await page.locator('.op-label', { hasText: /^RoPE$/ }).first().dispatchEvent('click');
  const popup = page.locator('.op-popup');
  await expect(popup).toBeVisible();
  await expect(popup.locator('h2')).toHaveText('RoPE: rotary position embedding');
  await expect(popup.locator('section.op-stage.focus')).not.toHaveCount(0);
  await popup.locator('input[type=range]').first().fill('500');
  await page.keyboard.press('Escape');
  await expect(popup).toHaveCount(0);
  expect(errors).toEqual([]);
});

test('Generate steps through decode, one token at a time', async ({ page }) => {
  const errors = watchErrors(page);
  await load(page, '?model=gpt2&webgl');
  // Start from a short prompt: GPT-2 opens at its full 1,024-token context, where generation has nowhere to go.
  const row = page.locator('#controls label.row', { has: page.locator('.row-head span:text-is("Tokens")') });
  await row.locator('input[type=range]').fill('500');
  const tokens = row.locator('.row-value');
  await expect(tokens).toHaveText('32 in the prompt');
  const before = await tokens.textContent();
  await page.getByRole('button', { name: '▶ Generate' }).click();
  await expect(page.locator('#title')).toContainText('decode');
  await expect(tokens).not.toHaveText(before!);
  await expect(tokens).toContainText('in cache, +1 new');
  await page.getByRole('button', { name: '❚❚ Pause' }).click();
  await expect(page.getByRole('button', { name: '▶ Generate' })).toBeVisible();
  expect(errors).toEqual([]);
});

test('hovering the scene shows a tooltip', async ({ page }) => {
  await load(page, '?model=gpt2&webgl');
  await page.keyboard.press('f');
  await page.waitForTimeout(1500);
  const canvas = page.locator('#app canvas');
  const box = (await canvas.boundingBox())!;
  // Sweep the middle of the view until something is under the pointer.
  let shown = false;
  for (let y = 0.3; y <= 0.8 && !shown; y += 0.05) {
    for (let x = 0.15; x <= 0.7 && !shown; x += 0.05) {
      await page.mouse.move(box.x + box.width * x, box.y + box.height * y);
      shown = await page.locator('#tooltip').isVisible();
    }
  }
  expect(shown).toBe(true);
  await expect(page.locator('#tooltip .tt-title')).not.toBeEmpty();
});
