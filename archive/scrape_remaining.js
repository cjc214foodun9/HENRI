const { chromium } = require("C:\\Users\\chan\\AppData\\Local\\npm-cache\\_npx\\0d29dd9f4e472da9\\node_modules\\patchright");
const fs = require("fs");
const path = require("path");

const CATALOG_PATH = "c:\\Users\\chan\\Desktop\\HENRI 7B SWARM\\HENRI\\archive\\source-catalog.md";
const OUTPUT_DIR = "c:\\Users\\chan\\Desktop\\HENRI 7B SWARM\\HENRI\\archive\\raw_sources";

function cleanFilename(title) {
  let cleaned = title.replace(/[\\/*?:"<>|]/g, "");
  cleaned = cleaned.replace(/ /g, "_");
  return cleaned.substring(0, 100);
}

async function run() {
  if (!fs.existsSync(OUTPUT_DIR)) {
    fs.mkdirSync(OUTPUT_DIR, { recursive: true });
  }

  // Read catalog
  const data = fs.readFileSync(CATALOG_PATH, "utf-8");
  const lines = data.split("\n");
  const sources = [];
  let currentTitle = null;

  for (let line of lines) {
    line = line.trim();
    if (!line) continue;
    const mUrl = line.match(/^Verified URL:\s*(https?:\/\/\S+)/);
    if (mUrl) {
      const url = mUrl[1];
      const title = currentTitle ? currentTitle : url;
      sources.push({ title, url });
      currentTitle = null;
    } else {
      if (line.match(/^\d+$/)) continue;
      if (line.includes("[URL]")) {
        currentTitle = line.replace("[URL]", "").trim();
      } else {
        currentTitle = line;
      }
    }
  }

  console.log(`Parsed ${sources.length} sources from catalog.`);

  // Filter for sources that are missing or small (< 1000 bytes)
  const targets = [];
  for (const src of sources) {
    const safeTitle = cleanFilename(src.title);
    const mdPath = path.join(OUTPUT_DIR, `${safeTitle}.md`);
    const pdfPath = path.join(OUTPUT_DIR, `${safeTitle}.pdf`);
    
    let needsScrape = true;
    if (fs.existsSync(mdPath)) {
      const stats = fs.statSync(mdPath);
      if (stats.size >= 1000) needsScrape = false;
    }
    if (fs.existsSync(pdfPath)) {
      const stats = fs.statSync(pdfPath);
      if (stats.size >= 1000) needsScrape = false;
    }

    if (needsScrape) {
      targets.push(src);
    }
  }

  console.log(`Found ${targets.length} sources to scrape.`);
  if (targets.length === 0) {
    console.log("No sources need scraping.");
    return;
  }

  // Launch browser via patchright (custom Chromium designed to bypass antibot fingerprints)
  console.log("Launching Chromium browser via patchright...");
  const browser = await chromium.launch({
    headless: true,
    channel: "chrome",
    args: [
      "--disable-blink-features=AutomationControlled",
      "--disable-dev-shm-usage",
      "--no-first-run",
      "--no-default-browser-check",
    ]
  });

  const context = await browser.newContext({
    userAgent: "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
    viewport: { width: 1280, height: 800 }
  });

  for (let i = 0; i < targets.length; i++) {
    const target = targets[i];
    console.log(`[${i+1}/${targets.length}] Scraping: ${target.title} -> ${target.url}`);
    
    let targetUrl = target.url;
    // Map arXiv abstract URLs to direct PDF URLs if not already downloaded
    if (targetUrl.includes("arxiv.org/abs/")) {
      const arxivId = targetUrl.split("arxiv.org/abs/").pop().trim();
      targetUrl = `https://arxiv.org/pdf/${arxivId}.pdf`;
    }

    const page = await context.newPage();
    const safeTitle = cleanFilename(target.title);

    try {
      // Set up download listener in case the page navigation initiates a direct file download
      let downloadOccurred = false;
      const downloadPromise = page.waitForEvent('download', { timeout: 6000 })
        .then(async (download) => {
          downloadOccurred = true;
          const downloadPath = path.join(OUTPUT_DIR, `${safeTitle}.pdf`);
          await download.saveAs(downloadPath);
          console.log(`  [SAVED PDF via download event] ${downloadPath}`);
          return true;
        })
        .catch(() => false);

      await page.goto(targetUrl, { waitUntil: "domcontentloaded", timeout: 45000 });
      await page.waitForTimeout(4000); // Wait for Cloudflare/Akamai and Javascript execution

      if (downloadOccurred) {
        await page.close();
        continue;
      }

      const currentUrl = page.url();
      const isPdfUrl = currentUrl.toLowerCase().endsWith(".pdf") || targetUrl.toLowerCase().endsWith(".pdf") || targetUrl.includes("/pdf/");

      if (isPdfUrl) {
        console.log(`  Detected PDF. Fetching binary data inside browser context...`);
        const dataUrl = await page.evaluate(async (fetchUrl) => {
          const res = await fetch(fetchUrl);
          const blob = await res.blob();
          return new Promise((resolve) => {
            const reader = new FileReader();
            reader.onloadend = () => resolve(reader.result);
            reader.readAsDataURL(blob);
          });
        }, targetUrl);

        const buffer = Buffer.from(dataUrl.split(',')[1], 'base64');
        const pdfPath = path.join(OUTPUT_DIR, `${safeTitle}.pdf`);
        fs.writeFileSync(pdfPath, buffer);
        console.log(`  [SAVED PDF] ${pdfPath}`);
      } else {
        // HTML Page: Strip layout noise and extract readable body text
        const result = await page.evaluate(() => {
          const title = document.title;
          const scripts = document.querySelectorAll('script, style, nav, footer, header, aside, form, iframe, noscript');
          scripts.forEach(s => s.remove());
          return { title, text: document.body.innerText };
        });

        const mdPath = path.join(OUTPUT_DIR, `${safeTitle}.md`);
        const header = `# ${target.title}\nSource URL: ${target.url}\n\n`;
        fs.writeFileSync(mdPath, header + result.text, "utf-8");
        console.log(`  [SAVED MD] ${mdPath} (${result.text.length} chars)`);
      }
    } catch (err) {
      console.log(`  [ERROR] Failed to scrape ${target.title}: ${err.message}`);
    } finally {
      await page.close();
    }
  }

  await browser.close();
  console.log("Finished scraping remaining sources.");
}

run().catch(console.error);
