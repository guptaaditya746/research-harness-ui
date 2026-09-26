import { expect, test } from '@playwright/test';
import type { Page } from '@playwright/test';

async function openScreen(page: Page, scene: string) {
  await page.goto(`/?screen=${scene}`);
  await page.evaluate(async () => { await document.fonts.ready; });
}

const screens = [
  ['idle', 'Evaluation questions'], ['running', 'Agentic research'], ['readings', 'interpretations of this question'],
  ['agentic_result', 'Subschema'], ['baseline_result', 'Unstated — inferred by the extractor'],
  ['failed', 'Try agentic research'], ['runs', 'Every run keeps its artifacts'],
  ['corpus', 'papers indexed'], ['settings', 'Fast baseline budget'],
] as const;

for (const [scene, label] of screens) {
  test(`${scene} screen renders from API fixture`, async ({ page }) => {
    await openScreen(page, scene);
    await expect(page.getByText(label).first()).toBeVisible();
    if (scene === 'idle') await expect(page.getByRole('button', { name: /List the authors of FACT-AUDIT/ })).toBeVisible();
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

test('choosing an interpretation starts research without rewriting the question', async ({ page }) => {
  await openScreen(page, 'readings');
  // The fixture fills the question after loading; read it only once it is there.
  await expect(page.getByRole('textbox', { name: 'Question' })).not.toHaveValue('');
  const question = await page.getByRole('textbox', { name: 'Question' }).inputValue();
  await page.getByRole('button', { name: /First-author publication count/ }).click();
  await expect(page.getByText('Stop run').or(page.getByText('Researched as:')).or(page.locator('.notice')).first()).toBeVisible();
  await expect(page.getByRole('textbox', { name: 'Question' })).toHaveValue(question);
});

test('gold editor keeps a partially typed JSON draft and reports invalid JSON on save', async ({ page }) => {
  await page.route('**/api/datasets/candidates', route => route.fulfill({ json: [
    { id: 'gold-1', run_id: 'r-1', question: 'Draft question?', gold_triples: [], gold_claims: [], notes: '', status: 'draft' },
  ] }));
  await page.goto('/');
  await page.getByRole('link', { name: 'Gold' }).click();
  await page.getByText('Draft question?').click();
  const claims = page.getByRole('textbox', { name: 'Claims for Draft question?' });
  await claims.fill('[{');
  await expect(claims).toHaveValue('[{');
  await page.getByRole('button', { name: 'Save and approve' }).click();
  await expect(page.getByRole('status')).toContainText('Invalid JSON');
});

test('runs open an artifact inline from the API', async ({ page }) => {
  await openScreen(page, 'runs');
  const row = page.locator('.run-row[data-run-id="r-0412"]');
  await row.locator('summary').click();
  await row.getByRole('button', { name: 'query.rq' }).click();
  await expect(row.locator('pre')).toContainText('SELECT');
});

test('question library runs a question without typing', async ({ page }) => {
  await page.goto('/');
  // Only the 60 evaluation questions (nlp01-nlp30, nlp32-nlp61) are listed by default.
  await expect(page.getByRole('button', { name: /Question library \(60\)/ })).toBeVisible();
  await page.getByRole('button', { name: /Question library/ }).click();
  const library = page.getByRole('region', { name: 'Question library' });
  await expect(library.getByText('List the authors of FACT-AUDIT.')).toBeVisible();
  await library.getByRole('textbox', { name: 'Filter questions' }).fill('SafeRAG');
  const row = library.locator('.library-row').filter({ hasText: 'Which ACL 2025 paper introduced SafeRAG?' });
  await row.hover();
  await page.screenshot({ path: 'test-results/library-hover.png' });
  await row.getByRole('button', { name: 'Fast', exact: true }).click();
  await expect(page.getByRole('textbox', { name: 'Question' })).toHaveValue(/Which ACL 2025 paper introduced SafeRAG/);
  await expect(page.getByText('Fast baseline').first()).toBeVisible();
});
