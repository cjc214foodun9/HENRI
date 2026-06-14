const { chromium } = require("C:\\Users\\chan\\AppData\\Local\\npm-cache\\_npx\\0d29dd9f4e472da9\\node_modules\\patchright");
const fs = require("fs");
const path = require("path");

function copyRecursiveSync(src, dest) {
  const exists = fs.existsSync(src);
  const stats = exists && fs.statSync(src);
  const isDirectory = exists && stats.isDirectory();
  if (isDirectory) {
    if (!fs.existsSync(dest)) {
      fs.mkdirSync(dest, { recursive: true });
    }
    fs.readdirSync(src).forEach((childItemName) => {
      copyRecursiveSync(path.join(src, childItemName), path.join(dest, childItemName));
    });
  } else {
    if (src.includes('SingletonLock') || src.includes('Lock') || src.endsWith('.tmp')) {
      return; // skip lock and temp files
    }
    try {
      fs.copyFileSync(src, dest);
    } catch (err) {
      console.warn(`[WARN] Skipping file: ${src} -> ${err.message}`);
    }
  }
}

async function run() {
  const baseProfileDir = "C:\\Users\\chan\\AppData\\Local\\notebooklm-mcp\\Data\\chrome_profile";
  const uniqueId = `temp_profile_${Date.now()}`;
  const tempProfileDir = path.join("c:\\Users\\chan\\Desktop\\HENRI 7B SWARM\\HENRI\\archive", uniqueId);
  
  copyRecursiveSync(baseProfileDir, tempProfileDir);
  
  const launchOptions = {
    headless: true,
    channel: "chrome",
    viewport: { width: 1280, height: 800 },
    args: [
      "--disable-blink-features=AutomationControlled",
      "--disable-dev-shm-usage",
      "--no-first-run",
      "--no-default-browser-check",
    ]
  };

  const context = await chromium.launchPersistentContext(tempProfileDir, launchOptions);
  const page = context.pages().length > 0 ? context.pages()[0] : await context.newPage();
  
  const notebookUrl = "https://notebooklm.google.com/notebook/904717d3-7bf6-4cc4-b779-ff30158fad4e";
  await page.goto(notebookUrl, { waitUntil: "domcontentloaded", timeout: 60000 });
  await page.waitForTimeout(15000);
  
  console.log("Clicking source-catalog.md...");
  const clicked = await page.evaluate(async () => {
    // Find all elements in ARTIFACT-LIBRARY or containing source-catalog.md
    const elements = Array.from(document.querySelectorAll('*'));
    const target = elements.find(el => el.innerText && el.innerText.trim() === "source-catalog.md");
    if (target) {
      target.click();
      return { success: true, text: target.innerText, tagName: target.tagName };
    }
    
    // Fallback: look for contains
    const fallback = elements.find(el => el.innerText && el.innerText.includes("source-catalog.md"));
    if (fallback) {
      fallback.click();
      return { success: true, text: fallback.innerText, tagName: fallback.tagName };
    }
    
    return { success: false, error: "source-catalog.md element not found" };
  });
  console.log("Click result:", clicked);
  
  console.log("Waiting 10 seconds for artifact content to load...");
  await page.waitForTimeout(10000);
  
  const screenshotPath = "c:\\Users\\chan\\Desktop\\HENRI 7B SWARM\\HENRI\\archive\\artifact_clicked.png";
  await page.screenshot({ path: screenshotPath });
  console.log("Screenshot saved to:", screenshotPath);
  
  // Dump text elements outside of the sidebar/libraries
  const viewerTexts = await page.evaluate(() => {
    const results = [];
    const elements = Array.from(document.querySelectorAll('*'));
    
    // We look for elements containing markdown text of the catalog (which has list of sources)
    const keywords = ["source-catalog", "catalog", "verified url"];
    
    for (const el of elements) {
      const className = typeof el.className === 'string' ? el.className : '';
      const text = el.innerText || '';
      
      if (className.includes('scroll-area-desktop') || className.includes('single-source-container') || el.tagName === 'BODY' || el.tagName === 'HTML') {
        continue;
      }
      
      if (text.length > 500) {
        const matches = keywords.filter(kw => text.toLowerCase().includes(kw));
        if (matches.length >= 1) {
          const children = Array.from(el.children);
          const hasChild = children.some(c => (c.innerText || '').length > 500 && keywords.filter(kw => (c.innerText || '').toLowerCase().includes(kw)).length >= 1);
          if (!hasChild) {
            results.push({ tag: el.tagName, class: className, textLength: text.length, sample: text.slice(0, 1000) });
          }
        }
      }
    }
    return results;
  });
  
  console.log("Candidate artifact contents:");
  console.log(JSON.stringify(viewerTexts, null, 2));
  
  // Write the longest text block found to a file
  if (viewerTexts.length > 0) {
    // Sort by length descending
    viewerTexts.sort((a, b) => b.textLength - a.textLength);
    const longestText = viewerTexts[0].sample; // wait, we only returned sample (first 1000 chars) in the evaluate block.
    // Let's grab the full text of the longest element
    const fullText = await page.evaluate((selectorObj) => {
      // Find element by tag, class and text length matching
      const elList = Array.from(document.querySelectorAll(selectorObj.tag));
      const el = elList.find(e => {
        const c = typeof e.className === 'string' ? e.className : '';
        return c === selectorObj.class && (e.innerText || '').length === selectorObj.textLength;
      });
      return el ? el.innerText : '';
    }, viewerTexts[0]);
    
    const catalogPath = "c:\\Users\\chan\\Desktop\\HENRI 7B SWARM\\HENRI\\archive\\source-catalog.md";
    fs.writeFileSync(catalogPath, fullText, "utf-8");
    console.log("Full catalog text saved to:", catalogPath);
  }
  
  await context.close();
  try {
    fs.rmSync(tempProfileDir, { recursive: true, force: true });
  } catch (err) {}
}

run().catch(console.error);
