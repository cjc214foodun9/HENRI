import { AuthManager } from "file:///C:/Users/chan/AppData/Local/npm-cache/_npx/0d29dd9f4e472da9/node_modules/notebooklm-mcp/dist/auth/auth-manager.js";

async function run() {
  const auth = new AuthManager();
  console.log("Opening Google Authentication window...");
  console.log("Please sign in or complete the verification in the Chrome window that opens.");
  
  const sendProgress = async (msg) => {
    console.log(`[PROGRESS] ${msg}`);
  };
  
  const success = await auth.performSetup(sendProgress, true); // true = show browser
  if (success) {
    console.log("\n[SUCCESS] Authentication state saved successfully!");
  } else {
    console.log("\n[ERROR] Authentication failed or was cancelled.");
  }
}

run().catch(console.error);
