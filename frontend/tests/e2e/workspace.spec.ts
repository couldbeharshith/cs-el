import { expect, test, type Page } from "@playwright/test";

async function openHistory(page: Page, mobile: boolean) {
  if (mobile) await page.getByRole("button", { name: "Toggle chat history" }).click();
}

test("opens a new chat, streams and reloads it, then renames and deletes it", async ({ page, isMobile }, testInfo) => {
  await page.goto("/");
  await expect(page.getByRole("heading", { name: "What are you researching?" })).toBeVisible();
  await expect(page.getByRole("button", { name: /Notepad Summary/ })).toBeVisible();
  await page.screenshot({ path: testInfo.outputPath("chat-workspace.png") });
  const conversation = page.url();

  const profile = page.getByLabel("Chat profile", { exact: true }).filter({ visible: true });
  await profile.click();
  await page.getByRole("option", { name: "Local LLM2" }).click();
  await expect(profile).toHaveText("Local LLM2");

  const prompt = `Policy verification ${Date.now()}`;
  await page.getByRole("textbox", { name: "Message" }).fill(prompt);
  await page.getByRole("button", { name: "Send message" }).click();
  await expect(page.getByRole("article", { name: "Assistant response" }).getByText("documented exclusions", { exact: true }).filter({ visible: true })).toBeVisible();
  await expect.poll(async () => {
    const result = await page.request.get(`http://127.0.0.1:4316/rest/v1/messages?conversation_id=eq.${conversation.split("/").at(-1)}`);
    return (await result.json()).length;
  }).toBe(2);
  await page.reload({ waitUntil: "networkidle" });
  await expect(page.getByRole("article", { name: "Assistant response" }).getByText("documented exclusions", { exact: true }).filter({ visible: true })).toBeVisible();
  await page.screenshot({ path: testInfo.outputPath("chat-transcript.png") });

  await openHistory(page, isMobile);
  await page.getByRole("button", { name: `Options for ${prompt}` }).click();
  await page.getByRole("menuitem", { name: "Rename" }).click();
  await page.getByRole("textbox", { name: "Chat title" }).fill(`Reviewed ${prompt}`);
  await page.getByRole("button", { name: "Save title" }).click();
  await expect(page.getByRole("link", { name: `Reviewed ${prompt}` })).toBeVisible();
  await page.getByRole("button", { name: `Options for Reviewed ${prompt}` }).click();
  await page.getByRole("menuitem", { name: "Delete" }).click();
  await page.getByRole("button", { name: "Delete chat" }).click();
  await expect(page).not.toHaveURL(conversation);

  await page.goto(conversation);
  await expect(page.getByText("This page could not be found.")).toBeVisible();
});

test("navigates dashboard data and carries a research draft into chat", async ({ page }, testInfo) => {
  await page.goto("/dashboard");
  await expect(page.getByRole("heading", { name: "Market overview" })).toBeVisible();
  await expect(page.getByText("Sample snapshot", { exact: false })).toBeVisible();
  await page.screenshot({ path: testInfo.outputPath("dashboard.png"), fullPage: true });
  await page.getByRole("tab", { name: "Losers" }).click();
  await expect(page.getByRole("link", { name: /Paytm/ })).toBeVisible();
  expect(await page.evaluate(() => document.documentElement.scrollWidth - window.innerWidth)).toBeLessThanOrEqual(1);
  await page.getByRole("link", { name: /Tata Motors Analysis/ }).click();
  await expect(page.getByRole("textbox", { name: "Message" })).toHaveValue("Tata Motors Analysis");
  await page.getByRole("button", { name: "Toggle theme" }).click();
  await page.getByRole("menuitem", { name: "Dark" }).click();
  await expect(page.locator("html")).toHaveClass(/dark/);
  await page.screenshot({ path: testInfo.outputPath("chat-dark.png") });
});

test("shows provider failures and transcribes a recording", async ({ page }) => {
  await page.goto("/chat");
  const composer = page.getByRole("textbox", { name: "Message" });
  await composer.fill("Fixture provider error");
  await page.getByRole("button", { name: "Send message" }).click();
  await expect(page.getByRole("alert").filter({ hasText: "could not finish responding" })).toBeVisible();
  await expect(page.getByRole("button", { name: "Retry", exact: true })).toBeVisible();
  await page.getByRole("button", { name: "Voice input" }).click();
  await expect(page.getByRole("button", { name: "Stop recording" })).toBeVisible();
  await page.getByRole("button", { name: "Stop recording" }).click();
  await expect(composer).toHaveValue("Fixture voice transcription");
});

test("rejects invalid chat and transcription input", async ({ request }) => {
  expect((await request.post("/api/chat", { data: { messages: [] } })).status()).toBe(400);
  expect((await request.post("/api/transcribe", { multipart: { other: "empty" } })).status()).toBe(400);
});

test("renders and persists a chart tool result", async ({ page }) => {
  await page.goto("/chat");
  const composer = page.getByRole("textbox", { name: "Message" });
  await composer.fill("Chart fixture company");
  await page.getByRole("button", { name: "Send message" }).click();
  const result = page.getByRole("button", { name: "Chart generation results" });
  await expect(result).toBeVisible();
  await result.click();
  await expect(page.getByText("Fixture company performance", { exact: true })).toBeVisible();
  await expect(page.locator(".recharts-line-curve")).toBeVisible();
  await expect(page.getByRole("button", { name: "Stop response" })).toHaveCount(0);
  await page.reload();
  await expect(result).toBeVisible();
});
