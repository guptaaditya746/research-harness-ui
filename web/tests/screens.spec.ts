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

test('results show numbered references, removed sentences and the run timeline', async ({ page }) => {
  await page.goto('/?screen=baseline_result');
  const answer = page.getByRole('region', { name: 'Answer' });
  await expect(answer.locator('sup.ref a')).toHaveText('1');
  await answer.getByRole('button', { name: /sentence the verifier removed/ }).click();
  await expect(answer.getByText('His work defines the field.')).toBeVisible();
  await expect(page.getByRole('region', { name: 'Run timeline' })).toContainText('4 model calls');
  await expect(page.getByRole('button', { name: /Model status|Models|Harness API|Checking models/ })).toBeVisible();
});

test('model status table shows vLLM waiting counts and a row for harness runs waiting for a slot', async ({ page }) => {
  await page.route('**/models/status', route => route.fulfill({ json: {
    reachable: true, detail: null, overall: 'busy', updatedAt: '2026-09-27T10:00:00Z',
    models: [
      { alias: 'heavy-model', roles: ['schema and answers'], state: 'busy', running: 4, waiting: 3, kvCache: 0.83, queueP95Seconds: 2.4 },
      { alias: 'lite-model', roles: ['agent'], state: 'healthy', running: 1, waiting: 0, kvCache: 0.12 },
    ],
    runs: { active: 2, queued: 5, capacity: 2 },
  } }));
  await page.goto('/');
  await page.getByRole('button', { name: /Models busy · 5 runs queued/ }).click();
  const table = page.getByRole('dialog', { name: 'Model status' }).locator('table');
  await expect(table.getByRole('row', { name: /heavy/ }).getByRole('cell').nth(3)).toHaveText('3');
  await expect(table.getByRole('row', { name: /lite/ }).getByRole('cell').nth(3)).toHaveText('0');
  const runs = table.getByTestId('harness-runs-row');
  await expect(runs.getByRole('cell').nth(2)).toHaveText('2');
  await expect(runs.getByRole('cell').nth(3)).toHaveText('5');
});
