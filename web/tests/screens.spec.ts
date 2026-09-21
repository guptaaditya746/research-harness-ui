import { expect, test } from '@playwright/test';
import type { Page } from '@playwright/test';

async function openScreen(page: Page, scene: string) {
  await page.goto(`/?screen=${scene}`);
  await page.evaluate(async () => { await document.fonts.ready; });
}

const screens = [
  ['idle', 'Recent'], ['running', 'Agentic research'], ['readings', 'Four readings'],
  ['agentic_result', 'Subschema'], ['baseline_result', 'Unstated — inferred by the extractor'],
  ['failed', 'Try deeper research'], ['runs', 'Every run keeps its artifacts'],
  ['corpus', 'Add PDFs'], ['settings', 'Fast baseline budget'],
] as const;

for (const [scene, label] of screens) {
  test(`${scene} screen renders from API fixture`, async ({ page }) => {
    await openScreen(page, scene);
    await expect(page.getByText(label).first()).toBeVisible();
    if (scene === 'idle') await expect(page.getByRole('button', { name: /Across this corpus, which author/ })).toBeVisible();
    await expect(page).toHaveScreenshot(`${scene}.png`, { fullPage: true, animations: 'disabled' });
  });
}

test('evaluation sidebar is present on result and supports plans and rating', async ({ page }) => {
  await openScreen(page, 'agentic_result');
  const sidebar = page.getByRole('complementary', { name: 'Evaluation' });
  await expect(sidebar.getByText('Triple precision')).toBeVisible();
  await sidebar.getByRole('button', { name: 'Compare plans' }).click();
  await expect(sidebar.getByText('A · author · citations · rank')).toBeVisible();
  await sidebar.getByText('Useful', { exact: true }).click();
  await expect(sidebar.getByRole('radio', { name: 'Useful', exact: true })).toBeChecked();
  await expect(sidebar).toHaveScreenshot('evaluation-sidebar.png');
});

test('reading choice switches answer, schema and triples together', async ({ page }) => {
  await openScreen(page, 'readings');
  await page.getByRole('button', { name: /First-author publication count/ }).click();
  await expect(page.getByText(/Tomer Wolfson ranks first/)).toBeVisible();
  await expect(page.getByText('Interpreted as: First-author publication count.')).toBeVisible();
  await page.getByRole('button', { name: 'Graph' }).click();
  await expect(page.locator('.graph-row')).toHaveCount(2);
  await page.getByRole('button', { name: 'Turtle', exact: true }).click();
  await expect(page.locator('section[aria-label="Triples"] pre')).toContainText('xsd:integer');
});

test('runs open an artifact inline from the API', async ({ page }) => {
  await openScreen(page, 'runs');
  const row = page.locator('.run-row[data-run-id="r-0412"]');
  await row.locator('summary').click();
  await row.getByRole('button', { name: 'query.rq' }).click();
  await expect(row.locator('pre')).toContainText('SELECT');
});
