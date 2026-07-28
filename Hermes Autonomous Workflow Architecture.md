# **Hermes Autonomous Research, Testing, and Governance Pipeline**

**Google Drive Ingestion, Local RAG, Telegram Interactive Governance, and Automated Benchmark Execution**

## **Executive Summary & System Overview**

This specification details the complete end-to-end automation of your research-to-code cycle across **Google Drive**, **Obsidian**, local **Vast.ai** compute, **Telegram API**, and your **Hermes Multi-Model Ensemble** (Gemini 3.6 Flash driver, Kimi K3, Sakana Fugu Ultra).

### **The Automated Life-Cycle**

  \+-----------------------------------------------------------------------------------+  
  | 1\. SENSORY INGRESS: Google Drive Watcher & Local Embedding Engine                 |  
  |    \- Drop research PDF into GDrive folder "\~/Research\_Inbox/"                      |  
  |    \- Vast.ai daemon downloads PDF, extracts text/equations via PyPDF2/pdfplumber    |  
  |    \- Notes auto-formatted into Obsidian Vault with YAML metadata                   |  
  |    \- Fast, zero-API local embedding model (bge-m3 / nomic-embed) indexes vector DB  |  
  \+-----------------------------------------------------------------------------------+  
                                           │  
                                           ▼  
  \+-----------------------------------------------------------------------------------+  
  | 2\. AGGREGATOR PLANNING: RAG Context Retrieval & Speculative Plan Drafting         |  
  |    \- Aggregator (Gemini 3.6 Flash) pulls top 3-5 relevant vector chunks from DB   |  
  |    \- Drafts an actionable \`implementation\_plan.md\` \+ unified \`patch.diff\` proposal  |  
  \+-----------------------------------------------------------------------------------+  
                                           │  
                                           ▼  
  \+-----------------------------------------------------------------------------------+  
  | 3\. TELEGRAM INTERACTIVE GOVERNANCE (Google Pixel Interface)                       |  
  |    \- Hermes Telegram Bot pings your phone with:                                  |  
  |      • Executive Research Summary                                                 |  
  |      • Proposed Implementation Plan                                               |  
  |      • Key Git Diffs / Parameter Calibrations                                     |  
  |    \- Interactive Buttons sent to your Pixel: \[ ✅ APPROVE & EXECUTE \]  \[ ❌ REJECT \]|  
  \+-----------------------------------------------------------------------------------+  
                                           │  (User taps "APPROVE" on Pixel)  
                                           ▼  
  \+-----------------------------------------------------------------------------------+  
  | 4\. UNATTENDED CI/CD EXECUTION & BENCHMARK EVALUATION                             |  
  |    \- Vast.ai applies \`git apply patch.diff\`                                       |  
  |    \- Runs headless benchmark (\`production\_arc\_run.py\`) on GPU                      |  
  |    \- Local \`telemetry\_preprocessor.py\` compresses logs to 10-line JSON summary     |  
  \+-----------------------------------------------------------------------------------+  
                                           │  
                                           ▼  
  \+-----------------------------------------------------------------------------------+  
  | 5\. TELEGRAM RESULTS REPORTING                                                     |  
  |    \- Bot sends instant scorecard report back to your Pixel:                       |  
  |      • Target Env Scores (e.g. WINS / Deltas)                                     |  
  |      • EFE Range, Residual RMS, Admissible Counts                                 |  
  |      • Option to commit branch or rollback                                        |  
  \+-----------------------------------------------------------------------------------+

## **I. Component Specifications & Code**

### **1\. Google Drive & Obsidian Ingestion Sync (gdrive\_obsidian\_watcher.py)**

This service runs in the background on your Vast.ai instance (or home server). It watches a dedicated Google Drive folder via Google Drive API or rclone, saves new PDFs directly to your **Obsidian Vault**, extracts key text and equations, and updates a local vector index (**ChromaDB** using nomic-embed-text or bge-m3).

\# gdrive\_obsidian\_watcher.py  
import os  
import time  
import json  
from pathlib import Path  
import pypdf  
import chromadb  
from chromadb.utils import embedding\_functions

\# Configuration Paths  
OBSIDIAN\_VAULT\_PATH \= Path("/workspace/obsidian\_vault/Research/")  
GDRIVE\_SYNC\_DIR \= Path("/workspace/gdrive\_inbox/")  
CHROMA\_DB\_DIR \= Path("/workspace/chroma\_db/")

\# Ensure directories exist  
OBSIDIAN\_VAULT\_PATH.mkdir(parents=True, exist\_ok=True)  
GDRIVE\_SYNC\_DIR.mkdir(parents=True, exist\_ok=True)

\# Initialize local embedding function (zero external API tokens consumed)  
default\_ef \= embedding\_functions.SentenceTransformerEmbeddingFunction(  
    model\_name="BAAI/bge-m3"  
)  
chroma\_client \= chromadb.PersistentClient(path=str(CHROMA\_DB\_DIR))  
vector\_collection \= chroma\_client.get\_or\_create\_collection(  
    name="henri\_research\_papers", embedding\_function=default\_ef  
)

def process\_pdf(pdf\_path: Path):  
    """Extracts text from PDF, formats Obsidian markdown, and embeds chunks locally."""  
    reader \= pypdf.PdfReader(str(pdf\_path))  
    full\_text \= \[\]  
    for idx, page in enumerate(reader.pages):  
        text \= page.extract\_text()  
        if text:  
            full\_text.append(f"--- Page {idx+1} \---\\n{text}")

    content \= "\\n\\n".join(full\_text)  
    paper\_title \= pdf\_path.stem.replace(" ", "\_")  
      
    \# 1\. Write structured note into Obsidian Vault  
    obsidian\_note \= OBSIDIAN\_VAULT\_PATH / f"{paper\_title}.md"  
    frontmatter \= f"""---  
title: "{paper\_title}"  
date\_added: "{time.strftime('%Y-%m-%d %H:%M:%S')}"  
source\_file: "{pdf\_path.name}"  
tags: \[type/paper, status/unprocessed\]  
\---

\# Paper Abstract & Ingested Text

{content\[:3000\]}  \<\!-- Truncated preview in Obsidian front note \--\>

\#\# Full Text Reference  
{content}  
"""  
    with open(obsidian\_note, "w", encoding="utf-8") as f:  
        f.write(frontmatter)

    \# 2\. Chunk and insert into local Vector Database  
    chunk\_size \= 1000  
    overlap \= 200  
    words \= content.split()  
    chunks \= \[\]  
    for i in range(0, len(words), chunk\_size \- overlap):  
        chunk \= " ".join(words\[i:i \+ chunk\_size\])  
        chunks.append(chunk)

    ids \= \[f"{paper\_title}\_chunk\_{i}" for i in range(len(chunks))\]  
    metadatas \= \[{"paper": paper\_title, "chunk\_id": i} for i in range(len(chunks))\]  
      
    vector\_collection.add(  
        documents=chunks,  
        metadatas=metadatas,  
        ids=ids  
    )  
    print(f"✅ Ingested '{paper\_title}' into Obsidian and ChromaDB ({len(chunks)} chunks).")  
    return obsidian\_note, paper\_title

def check\_inbox\_loop():  
    print("👀 Watching Google Drive Sync Folder for new research papers...")  
    processed\_files \= set()  
    while True:  
        for pdf\_file in GDRIVE\_SYNC\_DIR.glob("\*.pdf"):  
            if pdf\_file.name not in processed\_files:  
                note\_path, title \= process\_pdf(pdf\_file)  
                processed\_files.add(pdf\_file.name)  
                \# Trigger Hermes plan generation for newly ingested paper  
                os.system(f"python hermes\_telegram\_orchestrator.py \--paper '{title}'")  
        time.sleep(10)

if \_\_name\_\_ \== "\_\_main\_\_":  
    check\_inbox\_loop()

### **2\. Telegram Interactive Governance Bot (hermes\_telegram\_orchestrator.py)**

This script manages communication with your **Google Pixel via Telegram**. It queries local vector storage for the new paper, invokes Gemini 3.6 Flash / Kimi K3 / Sakana Fugu via Hermes MoA to draft a patch.diff and implementation plan, sends an inline approval button to your Pixel, and awaits your command.

\# hermes\_telegram\_orchestrator.py  
import argparse  
import json  
import subprocess  
import os  
import requests  
from telegram import InlineKeyboardButton, InlineKeyboardMarkup, Update  
from telegram.ext import Application, CommandHandler, CallbackQueryHandler, ContextTypes

\# Telegram Bot Credentials (Set via environment variables)  
TELEGRAM\_BOT\_TOKEN \= os.getenv("TELEGRAM\_BOT\_TOKEN", "YOUR\_BOT\_TOKEN")  
TELEGRAM\_CHAT\_ID \= os.getenv("TELEGRAM\_CHAT\_ID", "YOUR\_CHAT\_ID")

PENDING\_PLAN\_FILE \= "pending\_implementation\_plan.json"

def get\_rag\_context(paper\_title: str) \-\> str:  
    """Retrieves key chunks from local vector store for the aggregator model."""  
    import chromadb  
    from chromadb.utils import embedding\_functions

    ef \= embedding\_functions.SentenceTransformerEmbeddingFunction(model\_name="BAAI/bge-m3")  
    client \= chromadb.PersistentClient(path="/workspace/chroma\_db/")  
    coll \= client.get\_collection(name="henri\_research\_papers", embedding\_function=ef)  
      
    results \= coll.query(  
        query\_texts=\[f"key algorithms methods implementation equations {paper\_title}"\],  
        n\_results=5  
    )  
    documents \= results.get("documents", \[\[\]\])\[0\]  
    return "\\n\\n--- CHUNK \---\\n\\n".join(documents)

def generate\_hermes\_plan(paper\_title: str) \-\> dict:  
    """Calls Hermes Agent Aggregator (Gemini Flash \+ MoA) to draft plan and diff."""  
    context \= get\_rag\_context(paper\_title)  
      
    prompt \= f"""  
    You are the Lead Engineer for Project HENRI.  
    A new paper '{paper\_title}' was uploaded to Google Drive.  
      
    RELEVANT PAPER CONTEXT:  
    {context\[:4000\]}  
      
    Task:  
    1\. Summarize the actionable insight for Project HENRI in 3 bullet points.  
    2\. Propose an exact python implementation plan or parameter calibration.  
    3\. Output a unified git diff patch to be applied to \`efe\_planner.py\` or \`production\_arc\_run.py\`.

    Return ONLY a JSON payload with keys: 'summary', 'plan', 'git\_diff'.  
    """  
      
    \# Mocking Hermes MoA CLI call or direct API call  
    \# In production, this executes your Hermes MoA wrapper  
    mock\_payload \= {  
        "summary": f"Insights from {paper\_title}: Apply calibrated Sagnac gating and PEARL repair.",  
        "plan": "1. Update constraint\_reject\_thresh to 0.36 in efe\_planner.py\\n2. Enable PROGRESS\_VALENCE=1.0 in production\_arc\_run.py",  
        "git\_diff": "--- a/efe\_planner.py\\n+++ b/efe\_planner.py\\n@@ \-80,1 \+80,1 @@\\n- constraint\_reject\_thresh \= 0.25\\n+ constraint\_reject\_thresh \= 0.36"  
    }  
      
    with open(PENDING\_PLAN\_FILE, "w") as f:  
        json.dump(mock\_payload, f, indent=2)  
          
    return mock\_payload

async def send\_plan\_to\_telegram(paper\_title: str):  
    """Sends implementation plan with interactive buttons to Pixel via Telegram."""  
    plan\_data \= generate\_hermes\_plan(paper\_title)  
      
    message\_text \= (  
        f"📄 \*NEW RESEARCH INGESTED\*: \`{paper\_title}\`\\n\\n"  
        f"💡 \*Summary\*:\\n{plan\_data\['summary'\]}\\n\\n"  
        f"🛠️ \*Implementation Plan\*:\\n{plan\_data\['plan'\]}\\n\\n"  
        f"📝 \*Proposed Git Diff\*:\\n\`\`\`diff\\n{plan\_data\['git\_diff'\]}\\n\`\`\`\\n\\n"  
        f"Approve execution on Vast.ai GPU?"  
    )  
      
    keyboard \= \[  
        \[  
            InlineKeyboardButton("✅ Approve & Execute", callback\_data="approve\_exec"),  
            InlineKeyboardButton("❌ Reject Plan", callback\_data="reject\_plan")  
        \]  
    \]  
    reply\_markup \= InlineKeyboardMarkup(keyboard)  
      
    url \= f"\[https://api.telegram.org/bot\](https://api.telegram.org/bot){TELEGRAM\_BOT\_TOKEN}/sendMessage"  
    payload \= {  
        "chat\_id": TELEGRAM\_CHAT\_ID,  
        "text": message\_text,  
        "parse\_mode": "Markdown",  
        "reply\_markup": reply\_markup.to\_dict()  
    }  
    requests.post(url, json=payload)

async def button\_callback(update: Update, context: ContextTypes.DEFAULT\_TYPE):  
    """Handles Pixel user tapping Approve/Reject buttons."""  
    query \= update.callback\_query  
    await query.answer()  
      
    if query.data \== "approve\_exec":  
        await query.edit\_message\_text(text="🚀 \*Plan Approved\!\* Applying patch and executing benchmark on Vast.ai GPU...")  
          
        \# Apply patch and execute benchmark in background  
        subprocess.run("python hermes\_ci\_loop.py", shell=True)  
          
        \# Read compressed telemetry summary  
        with open("latest\_telemetry\_summary.json", "r") as f:  
            summary \= json.load(f)  
              
        result\_msg \= (  
            f"🏁 \*BENCHMARK COMPLETE\*\\n\\n"  
            f"• \*Status\*: \`{summary.get('status')}\`\\n"  
            f"• \*Total Steps\*: \`{summary.get('total\_steps')}\`\\n"  
            f"• \*Admissible Count Mean\*: \`{summary.get('admissible\_count\_mean')}\`\\n"  
            f"• \*PEARL Repair Rate\*: \`{summary.get('pearl\_repair\_rate')}\`\\n"  
            f"• \*Fallback Rate\*: \`{summary.get('fallback\_rate')}\`\\n"  
            f"• \*EFE Mean\*: \`{summary.get('efe\_mean')}\`\\n\\n"  
            f"Scores per Env: \`{json.dumps(summary.get('env\_scores'))}\`"  
        )  
        await context.bot.send\_message(chat\_id=query.message.chat\_id, text=result\_msg, parse\_mode="Markdown")  
          
    elif query.data \== "reject\_plan":  
        await query.edit\_message\_text(text="❌ \*Plan Rejected.\* Changes discarded. Awaiting next command or upload.")

if \_\_name\_\_ \== "\_\_main\_\_":  
    parser \= argparse.ArgumentParser()  
    parser.add\_argument("--paper", type=str, help="Ingested paper title")  
    args \= parser.parse\_args()  
      
    if args.paper:  
        import asyncio  
        asyncio.run(send\_plan\_to\_telegram(args.paper))  
    else:  
        \# Start Telegram Bot listener for interactive button clicks  
        app \= Application.builder().token(TELEGRAM\_BOT\_TOKEN).build()  
        app.add\_handler(CallbackQueryHandler(button\_callback))  
        print("🤖 Telegram Governance Bot listening for Pixel approvals...")  
        app.run\_polling()

## **II. End-to-End Operational Lifecycle**

| Step | Action | Executed By | Token Cost |
| :---- | :---- | :---- | :---- |
| **1\. Paper Drop** | User drops paper PDF into Google Drive folder from phone or desktop | User | **0 Tokens** |
| **2\. Local Ingest** | rclone / GDrive daemon downloads PDF to Vast.ai; gdrive\_obsidian\_watcher.py converts to Obsidian Markdown & embeds with local bge-m3 | Local Vast.ai CPU | **0 Tokens** |
| **3\. Plan Draft** | Hermes Aggregator performs local vector search, retrieves 4k characters of context, and generates plan \+ patch.diff | Gemini 3.6 Flash | ![][image1] **Tokens** |
| **4\. Pixel Notification** | Telegram bot messages user on Pixel with summary, plan, git diff, and inline \[APPROVE\] button | hermes\_telegram\_orchestrator.py | **0 Tokens** |
| **5\. Execution** | User taps \[APPROVE\] on Pixel. Vast.ai applies git apply patch.diff and runs 10-env benchmark | Vast.ai GPU | **0 Tokens** |
| **6\. Reporting** | telemetry\_preprocessor.py parses .jsonl output into 10-line JSON summary; Telegram bot sends results to Pixel | Telegram Bot API | **0 Tokens** |

## **III. Setup Checklist for Your Vast.ai Server**

1. **Install Dependencies**:  
   pip install pypdf chromadb sentence-transformers python-telegram-bot requests

2. **Setup Google Drive Sync (rclone)**:  
   rclone sync gdrive:Research\_Inbox /workspace/gdrive\_inbox/ \--daemon

3. **Set Environment Variables**:  
   export TELEGRAM\_BOT\_TOKEN="your\_bot\_token\_from\_botfather"  
   export TELEGRAM\_CHAT\_ID="your\_personal\_telegram\_user\_id"

4. **Launch Background Daemons**:  
   nohup python gdrive\_obsidian\_watcher.py \> watcher.log 2\>&1 &  
   nohup python hermes\_telegram\_orchestrator.py \> telegram.log 2\>&1 &

### **Key Advantage of This Integration**

This setup gives you **complete mobile governance over your AI research and compute cluster** without needing to open a terminal, copy-paste telemetry logs, or manually trigger benchmark runs.

1. **Drop a PDF into Google Drive from anywhere.**  
2. **Review the AI's proposed code diff on your Pixel.**  
3. **Tap Approve to run GPU tests on Vast.ai and get instant scorecard alerts.**

[image1]: <data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAEAAAAAWCAYAAABwvpo0AAAAqElEQVR4XmNgGAWjYBSMglEwCkYBMUBeXt4biN8pKirqo8sNWyAuLs4N9PRHIP6moqLChy4/rIG0tLQw0OO/5OTk8kEBgS4/bAHQw/OBHv+voKDgjy43bAHQs+ZATx8H0ruAXEZ0+WENgB7XBMU4EK9DlxsxAJQCQLE/IlMAOhiRZQA2AK0FvgFxkoyMDCe6/IgASO2AjyOuHYAO5EdiS3AUjIJRMKwBAHu/IBrhI8ixAAAAAElFTkSuQmCC>