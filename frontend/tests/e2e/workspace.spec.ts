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
  const favicon = await page.locator('link[rel="icon"]').first().getAttribute("href");
  expect(favicon).toContain("/favicon.ico");
  expect((await page.request.get(favicon!)).ok()).toBe(true);

  const model = page.getByLabel("Chat model", { exact: true });
  await expect(model).toHaveText("OpenAI");
  await model.press("Enter");
  await expect(page.getByRole("option")).toHaveText(["OpenAI", "Gemini", "Groq"]);
  const search = page.getByPlaceholder("Search model...");
  await search.fill("Gem");
  await expect(page.getByRole("option", { name: "Gemini" })).toHaveAttribute("aria-selected", "true");
  await search.press("Enter");
  await expect(model).toHaveText("Gemini");

  const id = Date.now();
  const prompt = "Policy verification " + id + " for the fixture";
  const title = "Coverage Review " + id + " for Policy";
  await page.getByRole("textbox", { name: "Message" }).fill(prompt);
  await page.getByRole("button", { name: "Send message" }).click();
  await expect(page.getByRole("button", { name: "Stop response" })).toBeVisible();
  await expect(page.getByRole("article", { name: "Assistant response" }).getByText("documented exclusions", { exact: true }).filter({ visible: true })).toBeVisible();
  await expect.poll(async () => {
    const result = await page.request.get(`http://127.0.0.1:4316/rest/v1/messages?conversation_id=eq.${conversation.split("/").at(-1)}`);
    return (await result.json()).length;
  }).toBe(2);
  await openHistory(page, isMobile);
  await expect(page.getByRole("link", { name: title, exact: true })).toBeVisible();
  await page.reload({ waitUntil: "networkidle" });
  await expect(page.getByRole("article", { name: "Assistant response" }).getByText("documented exclusions", { exact: true }).filter({ visible: true })).toBeVisible();
  await page.screenshot({ path: testInfo.outputPath("chat-transcript.png") });

  await openHistory(page, isMobile);
  await page.getByRole("button", { name: `Options for ${title}` }).click();
  await page.getByRole("menuitem", { name: "Rename" }).click();
  await page.getByRole("textbox", { name: "Chat title" }).fill(`Reviewed ${title}`);
  await page.getByRole("button", { name: "Save title" }).click();
  await expect(page.getByRole("link", { name: `Reviewed ${title}` })).toBeVisible();
  await page.getByRole("button", { name: `Options for Reviewed ${title}` }).click();
  await page.getByRole("menuitem", { name: "Delete" }).click();
  await page.getByRole("button", { name: "Delete chat" }).click();
  await expect(page).not.toHaveURL(conversation);

  await page.goto(conversation);
  await expect(page.getByText("This page could not be found.")).toBeVisible();
});

test("navigates dashboard data and carries a research draft into chat", async ({ page }, testInfo) => {
  await page.goto("/dashboard");
  await expect(page).toHaveTitle("Sampada AI");
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

test("streams with Groq and retries with OpenAI after an API key failure", async ({ page }) => {
  await page.goto("/chat", { waitUntil: "networkidle" });
  const model = page.getByLabel("Chat model", { exact: true });
  await model.click();
  await page.getByRole("option", { name: "Groq" }).click();
  const composer = page.getByRole("textbox", { name: "Message" });
  await composer.fill("Fixture Groq response");
  await composer.press("Enter");
  const responses = page.getByRole("article", { name: "Assistant response" });
  await expect(responses.first().getByText("documented exclusions", { exact: true })).toBeVisible();

  await composer.fill("Fixture Groq error");
  await page.getByRole("button", { name: "Send message" }).click();
  const alert = page.getByRole("alert").filter({ hasText: "Groq rejected its API key" });
  await expect(alert).toBeVisible();
  await model.click();
  await page.getByRole("option", { name: "OpenAI" }).click();
  await page.getByRole("button", { name: "Retry", exact: true }).click();
  await expect(responses.nth(1).getByText("documented exclusions", { exact: true })).toBeVisible();
  await expect(alert).toHaveCount(0);
});

test("rejects invalid chat and transcription input", async ({ request }) => {
  expect((await request.post("/api/chat", { data: { messages: [] } })).status()).toBe(400);
  expect((await request.post("/api/transcribe", { multipart: { other: "empty" } })).status()).toBe(400);
});

for (const provider of ["OpenAI", "Gemini", "Groq"]) {
  test(`renders and persists a chart tool result with ${provider}`, async ({ page }) => {
    await page.goto("/chat", { waitUntil: "networkidle" });
    if (provider !== "OpenAI") {
      await page.getByLabel("Chat model", { exact: true }).click();
      await page.getByRole("option", { name: provider }).click();
    }
    const composer = page.getByRole("textbox", { name: "Message" });
    await composer.fill("Chart fixture company");
    await page.getByRole("button", { name: "Send message" }).click();
    const result = page.getByRole("button", { name: "Chart generation results" });
    await expect(result).toBeVisible();
    await result.click();
    await expect(page.getByText("Fixture company performance", { exact: true })).toBeVisible();
    await expect(page.locator(".recharts-line-curve")).toBeVisible();
    await expect(page.getByRole("article", { name: "Assistant response" }).getByText("documented exclusions", { exact: true })).toBeVisible();
    await expect(page.getByRole("button", { name: "Stop response" })).toHaveCount(0);
    await page.reload();
    await expect(result).toBeVisible();
  });
}
