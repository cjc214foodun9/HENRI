# **Hermes Closed-Loop Autonomous Research & CI/CD Workflow**

**Token-Optimized Automation Blueprint for Vast.ai, Obsidian, arXiv, and Multi-Model Agent Ensembles**

## **Executive Summary: Eliminating Token Leaks Across the Workflow**

Your current manual loop leaks tokens in four main areas:

1. **Paper Ingestion**: Reading full arXiv PDFs into LLM context windows (![][image1] input tokens per paper).  
2. **Full-File Code Rewrites**: Requesting LLMs to write full 500+ line Python scripts (![][image2] output tokens per iteration).  
3. **Raw Telemetry Dumps**: Copy-pasting raw .jsonl benchmark output logs (![][image3] input tokens per run).  
4. **Manual Orchestration**: Interacting with chat interfaces step-by-step.

### **The Automated Solution**

By deploying **deterministic local pre-processors on Vast.ai**, we shift the heavy lifting to local Python scripts. The LLM receives **only concise structured deltas, git diffs, and compressed diagnostic summaries**, transforming the loop into an automated, event-driven engine.

## **I. Workflow Transformation Matrix**

  \+-----------------------------------------------------------------------------------+  
  |                       MANUAL LOOP (High Token Waste)                              |  
  \+-----------------------------------------------------------------------------------+  
  |  Upload PDF  ──\>  Chat Prompts  ──\> Manual Code  ──\> Run Tests ──\> Copy Log Files  |  
  | (50k tokens)     (10k tokens)      (5k tokens)     (Manual)       (20k tokens)     |  
  \+-----------------------------------------------------------------------------------+  
                                           │  
                                           ▼  AUTOMATION & LOCAL PRE-PROCESSING  
  \+-----------------------------------------------------------------------------------+  
  |                  HYPER-OPTIMIZED AUTOMATED LOOP (85% Token Reduction)             |  
  \+-----------------------------------------------------------------------------------+  
  | Local PDF Extractor ──\> Gemini Flash Triage ──\> Git Diff Patch ──\> Automated Cron  |  
  |  (3k RAG context)       (1k plan prompt)       (500 tok diff)     (Local Exec)    |  
  |                                                                        │          |  
  |                                                                        ▼          |  
  |                                                             Telemetry Pre-processor|  
  |                                                            (10-line JSON summary) |  
  \+-----------------------------------------------------------------------------------+

| Phase | Old Manual Process | Token-Optimized Automated Process | Token Reduction |
| :---- | :---- | :---- | :---- |
| **1\. Research Analysis** | Upload full arXiv PDF to agent chat | Local RAG / PyPDF extracts key sections (Abstract, Methods, Equations) into vector store | ![][image4] **Input Tokens** |
| **2\. Design & Plan** | Freeform prompt discussion | Gemini Flash generates a structured implementation\_plan.json spec | ![][image5] **Context** |
| **3\. Implementation** | LLM regenerates whole files | Hermes generates standard git diff patches applied automatically via git apply | ![][image6] **Output Tokens** |
| **4\. Testing & Review** | Copy-pasting raw .jsonl logs | Vast.ai script parses .jsonl into a 10-line metric summary (telemetry\_summary.json) | ![][image7] **Input Tokens** |

## **II. End-to-End Automation Infrastructure on Vast.ai**

### **Component 1: Local Telemetry Pre-Processor (telemetry\_preprocessor.py)**

This script executes automatically on Vast.ai when a benchmark run completes. It converts hundreds of megabytes of raw JSONL telemetry into a compact, diagnostic metric payload.

\# telemetry\_preprocessor.py  
import json  
import sys  
from pathlib import Path

def summarize\_telemetry(jsonl\_path: str) \-\> dict:  
    path \= Path(jsonl\_path)  
    if not path.exists():  
        return {"error": f"File {jsonl\_path} not found"}

    total\_steps \= 0  
    admissible\_counts \= \[\]  
    rms\_residuals \= \[\]  
    efe\_scores \= \[\]  
    pearl\_repairs \= 0  
    fallback\_count \= 0  
    scores \= {}

    with open(path, "r", encoding="utf-8") as f:  
        for line in f:  
            data \= json.loads(line.strip())  
            total\_steps \+= 1  
              
            \# Extract key operational metrics  
            admissible\_counts.append(data.get("admissible\_count", 0))  
            rms\_residuals.append(data.get("constraint\_penalty", 0.0))  
            efe\_scores.append(data.get("best\_efe", 0.0))  
              
            if data.get("pearl\_repaired", False):  
                pearl\_repairs \+= 1  
            if data.get("fallback\_executed", False):  
                fallback\_count \+= 1  
                  
            env\_id \= data.get("env\_id")  
            if env\_id:  
                scores\[env\_id\] \= data.get("score", 0.0)

    \# Compute concise aggregates  
    n \= max(1, total\_steps)  
    summary \= {  
        "total\_steps": total\_steps,  
        "admissible\_count\_mean": round(sum(admissible\_counts) / n, 2),  
        "rms\_residual\_mean": round(sum(rms\_residuals) / n, 4),  
        "efe\_mean": round(sum(efe\_scores) / n, 4),  
        "pearl\_repair\_rate": round(pearl\_repairs / n, 3),  
        "fallback\_rate": round(fallback\_count / n, 3),  
        "env\_scores": scores,  
        "status": "HEALTHY" if fallback\_count / n \< 0.1 else "OVER\_CONSTRAINED\_PARALYSIS"  
    }  
      
    return summary

if \_\_name\_\_ \== "\_\_main\_\_":  
    if len(sys.argv) \> 1:  
        log\_file \= sys.argv\[1\]  
        summary \= summarize\_telemetry(log\_file)  
        print(json.dumps(summary, indent=2))  
        with open("latest\_telemetry\_summary.json", "w") as f:  
            json.dump(summary, f, indent=2)

### **Component 2: Autonomous CI/CD Loop Orchestrator (hermes\_ci\_loop.py)**

This controller links the research plan, git patching, test execution, and agent evaluation into a fully automated loop.

\# hermes\_ci\_loop.py  
import os  
import subprocess  
import json  
import time

def run\_cmd(cmd: str) \-\> str:  
    result \= subprocess.run(cmd, shell=True, capture\_output=True, text=True)  
    return result.stdout.strip()

def execute\_automated\_iteration(target\_script: str, patch\_diff: str \= None):  
    print("=== \[1/4\] Applying Patch \===")  
    if patch\_diff:  
        with open("temp\_patch.patch", "w") as f:  
            f.write(patch\_diff)  
          
        \# Apply git diff natively  
        patch\_res \= run\_cmd("git apply temp\_patch.patch")  
        print(f"Git Apply Status: {patch\_res}")

    print("=== \[2/4\] Executing Vast.ai Benchmark Run \===")  
    \# Execute benchmark run non-interactively  
    run\_output \= run\_cmd(f"python {target\_script} \--num\_envs 10 \--max\_steps 50")  
    print("Benchmark execution complete.")

    print("=== \[3/4\] Pre-processing Telemetry \===")  
    \# Find latest telemetry file  
    latest\_log \= run\_cmd("ls \-t telemetry\_logs/\*.jsonl | head \-n 1")  
    summary \= run\_cmd(f"python telemetry\_preprocessor.py {latest\_log}")  
      
    print("=== \[4/4\] Compact Telemetry Summary \===")  
    print(summary)  
      
    \# Return compact string for LLM evaluation (under 500 tokens\!)  
    return summary

if \_\_name\_\_ \== "\_\_main\_\_":  
    \# Example execution trigger  
    execute\_automated\_iteration("production\_arc\_run.py")

### **Component 3: Speculative Diff System Prompt for Hermes Agent**

To force the agent ensemble (Kimi K3 / Sakana Fugu / Gemini Flash) to output **diffs only** instead of entire code files, update your Hermes prompt configuration:

\# hermes\_agent\_config.yaml  
system\_prompt: |  
  You are an expert AI scientist and software engineer working on Project HENRI.  
    
  STRICT CODE GENERATION RULES:  
  1\. DO NOT return entire source code files.  
  2\. Output updates strictly as UNABLE TO BREAK UNIFIED GIT DIFFS wrapped in \`\`\`diff \`\`\` blocks.  
  3\. Ensure all mathematical equations adhere to complex wave mechanics on S^(d-1).  
  4\. Optimize logic to minimize fallback\_executed rate and bound EFE within \[-1.0, \+2.0\].

  FORMAT:  
  \--- a/efe\_planner.py  
  \+++ b/efe\_planner.py  
  @@ \-100,5 \+100,5 @@  
  \- constraint\_reject\_thresh \= 0.25  
  \+ constraint\_reject\_thresh \= 0.36

## **III. Step-by-Step Execution Lifecycle**

Here is how your loop runs automatically once set up on Vast.ai:

1. **Ingest Research Paper (Local Extraction)**:  
   * You drop an arXiv PDF into a watched folder (\~/research\_inbox/).  
   * A background cron script runs pdf\_chunker.py, extracting key equations and method sections, and indexing them into a local vector DB (e.g., Qdrant on Vast.ai).  
   * Token cost to agent: ![][image8] **tokens** (retrieved chunks only vs. ![][image9] for raw PDF).  
2. **Generate Patch (Speculative Diff)**:  
   * Gemini 3.6 Flash drafts a proposed implementation plan (plan.md).  
   * Kimi K3 / Sakana Fugu Ultra review the plan and output concise structural critiques as git diff suggestions.  
   * Token cost: ![][image10] **tokens** (diffs only vs. ![][image11] for full python scripts).  
3. **Apply & Test (Vast.ai Unattended Execution)**:  
   * hermes\_ci\_loop.py executes git apply patch.diff.  
   * It launches production\_arc\_run.py on your Vast.ai GPU (e.g., RTX 5090 / H100).  
   * Token cost to agent: **0 tokens** (runs locally on bare metal).  
4. **Review & Iterate (Compressed Diagnostic Payload)**:  
   * When finished, telemetry\_preprocessor.py compresses the output log into a 10-line JSON summary.  
   * If status \== "HEALTHY" and scores \> 0, commit the branch.  
   * If status \== "OVER\_CONSTRAINED\_PARALYSIS", feed *only* the 10-line summary back to Gemini Flash to adjust constraint\_reject\_thresh and repeat.  
   * Token cost to agent: ![][image12] **tokens** (vs. ![][image13] for raw log copy-paste).

## **IV. Expected Token & Efficiency Savings**

| Metric | Manual Method | Automated Hermes Engine | Total Savings |
| :---- | :---- | :---- | :---- |
| **Tokens per Cycle** | **![][image14]** | **![][image15]** | **![][image16] Token Reduction** |
| **Cycle Latency** | 30–60 minutes (manual copy-pasting) | 3–5 minutes (unattended script execution) | ![][image17] **Speedup** |
| **API Costs / Throttling** | High rate-limit risks | Operates comfortably within free/basic tier limits | **Zero Rate-Limiting** |

**An analysis of your setup reveals that while you have built a powerful, modern stack—Vast.ai for raw compute, automated CI/CD telemetry loops, deep knowledge tools (Obsidian/arXiv), and a multi-model ensemble—your current MoA implementation is likely increasing token consumption rather than saving costs.**  
**Below is an honest breakdown of your token economics, why your current workflow is leaking efficiency, and a blueprint to transform it into a hyper-optimized Hermes Agent setup.**

## **Part I: Token Economics & Cost Reality Check**

### **Are you saving costs on tokens with standard MoA?**

**Short answer: No. Standard Mixture-of-Agents (MoA) is an accuracy multiplier, not a cost saver.**  
**In a standard MoA pipeline:**

1. **Prompt Ingestion: The same prompt (plus context) is sent to $N$ models simultaneously (e.g., Kimi K3 \+ Sakana Fugu Ultra).**  
2. **Generation: Both models generate full, independent responses ($T\_{\\text{Kimi}} \+ T\_{\\text{Sakana}}$).**  
3. **Aggregation: The aggregator model (Gemini 3.6 Flash) ingests the original prompt PLUS all $N$ full candidate responses to output a final synthesis.**

**$$\\text{Total Tokens} \= \\underbrace{(N \\times \\text{Prompt})} \+ \\underbrace{\\sum\_{i=1}^{N} \\text{Output}\_i} \+ \\underbrace{(\\text{Prompt} \+ \\sum\_{i=1}^{N} \\text{Output}\_i)} \+ \\text{Final Output}$$**  
**For a 2-model ensemble \+ 1 aggregator, you consume $3\\times$ to $5\\times$ more tokens per query compared to calling a single model.**

* **If paying per API token: MoA is significantly more expensive.**  
* **If using flat-rate monthly subscriptions/fixed API caps: Token *cost* in dollars may be capped, but you are burning latency, API rate limits, and compute throughput.**

## **Part II: The Bottlenecks in Your Current Setup**

1. **Unrouted MoA (Brute-Force Aggregation): Running MoA on *every* task (e.g., writing simple bash hooks, parsing telemetry logs, formatting JSON) wastes model bandwidth. Small tasks do not need Kimi K3 or Sakana Fugu Ultra.**  
2. **Context Bloat from Obsidian & arXiv: Dumping full markdown research notes or PDF pages into agent contexts quickly saturates context windows and forces high input-token overhead on every call.**  
3. **Passive Telemetry Consumption: Having cron jobs retrieve telemetry is great, but if the LLM isn't dynamically triggered *only* when telemetry deviates from expected physical bounds (e.g.,** fallback\_executed \== True **or** admissible\_count \== 0**), you are polling models unnecessarily.**

## **Part III: The Hyper-Optimized Hermes Agent Architecture**

**To transform your setup into an industrial-grade, cost-efficient research and engineering engine, restructure your workflow into a Tiered Dynamic Cascade with Speculative Critique.**  
                          \+------------------------+  
                           |  USER / CRON / AGENT   |  
                           \+------------------------+  
                                       |  
                                       v  
                     \+------------------------------------+  
                     |  TIER 1: TRIAGE ROUTER             |  
                     |  (Gemini 3.6 Flash \- Fast/Cheap)   |  
                     \+------------------------------------+  
                                  /    |    \\  
                                 /     |     \\  
               \+----------------+      |      \+-------------------+  
               |                       |                          |  
               v                       v                          v  
    \[Task: Deterministic\]    \[Task: Context/Research\]    \[Task: Deep ML / Math\]  
    \- Run Bash / Tests       \- Query Obsidian / arXiv    \- Refactor Core Math  
    \- Parse Telemetry Logs   \- Hybrid BM25/Vector RAG    \- EFE / Sagnac Calibration  
               |                       |                          |  
               v                       v                          v  
      Executed Directly       Synthesized by Flash       \+-----------------+  
     by Gemini Flash 3.6         w/ RAG Context          |  SPECULATIVE    |  
                                                         |  CRITIQUE MOA   |  
                                                         \+-----------------+  
                                                                  |  
                                                  \+---------------+---------------+  
                                                  |                               |  
                                                  v                               v  
                                          \+---------------+               \+---------------+  
                                          |    KIMI K3    |               |  SAKANA FUGU  |  
                                          |  (Diff Only)  |               |  (Diff Only)  |  
                                          \+---------------+               \+---------------+  
                                                  \\                               /  
                                                   \+--------------+--------------+  
                                                                  |  
                                                                  v  
                                                        \+------------------+  
                                                        |  FINAL AGGREGATOR|  
                                                        | (Gemini Flash 3.6|  
                                                        \+------------------+

### **Step 1: Implement Dynamic Tiered Routing (Cascading)**

**Never send raw prompts directly to the MoA ensemble. Use Gemini 3.6 Flash as a strict, tool-calling Triage Router:**

* **Tier 1 (Execution & Telemetry Triage): Handled 100% by Gemini 3.6 Flash.**  
  * ***Tasks:*** **Git commits, running test harnesses, parsing** .jsonl **telemetry logs, generating basic boilerplates.**  
  * ***Cost/Latency:*** **Near zero, sub-second execution.**  
* **Tier 2 (Domain Knowledge & Synthesis): Gemini 3.6 Flash \+ Local RAG.**  
  * ***Tasks:*** **Summarizing arXiv papers, searching Obsidian vault notes, cross-referencing past test runs.**  
* **Tier 3 (Deep Architectural Reasoning & Complex Bug Fixes): Triggers the MoA Ensemble.**  
  * ***Tasks:*** **Solving numerical instability in high-dimensional tensor spaces, refactoring $65,536$-dim wave dynamics, resolving zero-score bottlenecks in ARC tasks.**

### **Step 2: Replace "Full Generation MoA" with "Speculative Critique"**

**Instead of having Kimi K3, Sakana Fugu, and Gemini all generate full code implementations from scratch:**

1. **Drafting: Have Gemini 3.6 Flash draft the initial code patch or architectural proposal (10x faster, low token cost).**  
2. **Speculative Critique: Send *only the draft \+ problem spec* to Kimi K3 and Sakana Fugu Ultra.**  
   * **Instruct them: *"Do not rewrite the code. Provide ONLY a JSON structural critique or code diff highlighting errors in physical/mathematical logic."***  
3. **Aggregation: Gemini Flash merges the concise diffs into the final patch.**

**Token Impact: Reduces MoA output tokens by 70–80%, while retaining the reasoning density of large frontier models.**

### **Step 3: Optimize Knowledge Base Integration (Obsidian \+ arXiv RAG)**

**Instead of feeding raw files into the prompt:**

1. **Local Vector Database on Vast.ai: Deploy a lightweight local vector store (e.g.,** Qdrant **or** pgvector **inside a Docker container on your Vast instance).**  
2. **Hybrid Search Index: Index your Obsidian Vault and downloaded arXiv PDFs using a local embedding model (e.g.,** bge-m3 **or** nomic-embed-text **running on CPU/GPU).**  
3. **Exact Context Injection: Embed the top 3–5 relevant chunks into the agent's prompt rather than entire files.**

### **Step 4: Closed-Loop Telemetry Automation (Vast.ai \+ Cron Integration)**

**Connect your cron-driven CI/CD directly to the agent's decision loop:**  
**Python**  
\# System Workflow Script on Vast.ai Server

def auto\_benchmark\_eval\_loop():  
    \# 1\. Run headless benchmark  
    telemetry \= run\_arc\_benchmark(num\_envs=10)  
      
    \# 2\. Check automated threshold triggers  
    if telemetry\['admissible\_count\_mean'\] \< 2.0 or telemetry\['score'\] \== 0.0:  
        \# Construct focused diagnostic payload  
        payload \= {  
            "error\_type": "OVER\_CONSTRAINED\_PARALYSIS" if telemetry\['reject\_rate'\] \> 0.8 else "SOLIPSISM",  
            "telemetry\_summary": telemetry.get\_summary(),  
            "target\_files": \["efe\_planner.py"\]  
        }  
          
        \# Trigger Hermes Agent with Tier 3 Speculative MoA  
        patch \= trigger\_hermes\_moa\_debug(payload)  
          
        \# Apply patch and stage for re-testing  
        apply\_git\_patch(patch)  
        stage\_test\_run()

## **Part IV: Recommended Action Items**

**To align your setup for maximum efficiency and performance:**

1. **Enable Prompt Caching: Ensure your API requests to Gemini use context caching for system instructions, static mathematical definitions, and base code files.**  
2. **Implement Router Rules: Define strict threshold criteria for when Hermes escalates a task to Kimi K3 / Sakana Fugu Ultra.**  
3. **Shift to Diff-Based MoA: Update system prompts in your Hermes MoA setup so worker models return structured Git diffs rather than full-file completions.**  
4. **Local RAG Indexing: Run a background cron job on Vast.ai that auto-indexes your Obsidian Vault whenever** .md **files are updated.**

[image1]: <data:image/png;base64,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>

[image2]: <data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAJMAAAAZCAYAAAA1zhyrAAAFYUlEQVR4Xu1ZTYgcRRTuYREU40/Udcn+TM3urC7xD2EhIkQEETEHETZ4EfFgDsnBW8DgLSJePEkMOSzmoKAHPckSFQkSPIV4MIdEQQioSC5BFoUcjGzG7+t+b/bNS3VPj8tsJ1AfFNP11atX76t+XV1dk2UJCQkJCQkJCQmNI4Swa2Zm5n5ctnzbCGjRx+zs7EO+waHVbrf3Tk5O7vANiuXl5dsWFhba8DXj27Yb1MN4cTnh20aB6qE236ZgW6fTebbKhqjja9sxPz+/B8F/icsJJNQ+lHVM3C/ebhjQ7zDKj+j7SFb4uobr351ZC3yPdtLnDdph/CetEbivUU4sLi7ezSJ9vrA22wHMDYYNP+sNw/WHjA038h5vWwXR01M90Pux10Ofqpt1sen5ser4agQi4Ozc3NwzyqH+JoNF1t9hbYcBff5Bn0WtQ+QhPxkcB9xlb4fyvdYzSTj7xKG+Ro5txm6skBXic9ywSUPrw/Cu4YYh70NfSnBuvR76tLrFZg393labrKavRhCKlYiTw2ByYDVZFm6fta0CkuQx64PgKwzcGZRTyuF6HRNxMGLHCbpdbJjMf1obiemq7ztOYLwVr0l4avrN82VQPSi7HT+gh2OV6O7HUNdXk5iwqwAEHKGAUVYm2H9kRStkCf5X67SB/5esjeH34rUyhetLwd0s1HeRQ/nB8uNEkAdNXtV2BdkA94ExrYTqoQbH53rga6fo7pXoZgw7pT7Ul+UbBTfPSIBzCGzdt1UB9qco2vOSTDnPlUcm7IYVT3iuBLtD8eRdcu2aTH9YfpzodrsPYrwLjI06uD/h65n7Se6lvH0ZVA8TxvG5Hsz5rOjmHMR092gj9aG+LD8U8o58FeU8B/PtBi1MwiFPxoB9zcPwdVkEfeLbhyEUS/9Vz9tk0tcZl25vJ/wRfZ3Rn2vXZLphjHEDY16ReWGM/+fDJNfjv1xVDzXr66xEd3/O6viyfCX4tHDZRccV2SB+y8GQYPd5W342st3zVZBE5StrpJtGgbE+t3oyIf4Ox8TPy/j9i3FKOexty6B6qhKgkWSCqKNuieV5zetwdMFwOcC/Bn7N88PAMxUKwDh7fFsZKJBiPL+dyYR4n0D7CZTVOsXfEA/Oc7t4peXzYI4oWErj8KBtGJIAjSQTOnzlOUKeIK5Qj2fFGc8Brkp+0Bhgd6/nRFjtd7BNmipeJia6AQ/FhleTJrYRJT+wpxgnQrE/+dXzTDDGu7S0dJdvi0H1UEOEz/c/oq9XopsPdr5HquPL8pVAh/2es8DNewc2Z2M3LAbY/i0izhi6JVztZMJ4z7OP5czRQP8LDNfXOyVHA+3NL5ZjIf6JzKfyPcuPE5F5yaGxjJBMuZ4Q+Zy3emS8mG77MNby1QgQ7DcMtmNOoFF/mpw7yLxIjvbKebBd/o7JAZ8vgruOPo8ajgeU5yJ2V7QeO4RD/Ti5UY4rtoqwedwx8BdKKA5Q+1+7YfNL7HhmYlaonjB40Jk/sFaP+LW6aUPd/a1KXV+NgPsBBLLO/Qbr3Mij/p0T1U+m4J4cC7SdRlnVMyt9HWTGj5y4088q67ovQXlLbQjGFMwqjOsNctZm3NDYgvy9QchnPGOxsWkyXZyenn5AeQvRs2Hq+70evRfqW2w2yFu7Or4aBVaHpxDQpyjvozzn2xVoO+85C/FDHye5YfXtggnadYr91IEyO9zIV0KxoT6WbfEP1q1Ajk1OMl7c2Bd8u4JzEyqOaqampu5UPdwW+HaF6P5MbKK66/q6qRHMXyMJA+CrhkmfUAfdbncOE7bi+YRio8zi+YQImEQoP3k+oQDm5rTnEhISEhISEhISbm38B/+sPAFXXX6IAAAAAElFTkSuQmCC>

[image3]: <data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAKgAAAAZCAYAAACl3WVkAAAFbklEQVR4Xu1aPYgkRRSe4RRO/P9Zl9tdpmZ3R5YLxGBAE8HExMDEUxTvMDlEAyMD8TLlMBEDWeQCs0sEfzBZ/EEEFy4RDDTYQzjY4A4jQRaETVb2zu/reW9987q6u2bZ6cWb+qCYrq9fv3rvq+rq6urpdDIyMjIyMjIyMjJuT3SXlpbu8qSi3++f7PV6T6+urj7qz02C4XB4J3w9w19/zmJlZaWHeBbr7FJ9TRuI4YEQwin++nMWqiEOT/hzk4DaNOWd2l/Q+LGjiGka6C4uLj4MYc+j7KPcQqBDb7S8vIxTYQsJv8s6rlmibZ04MUDU+3kdyiXW4e8y6+StHbjvyA8Gg/tYxO5La5Pqqw2g3T2038dhF/q9LnH9aG1Ew33VkOdpd1gNqQvrqo3POyT0l8akk5LE9JG3OzbMzc3dg4A2GTyEvSG/pQGKRL/gORx2lRMBXjZmjcA1F+lLBaAw4DbAXTBmXbZFOyXEzref4mvqQJvs5V8NVcTv4xUNLyrHWe0QGtJ3kbcSqo3Nm5r49n1b8uTRmApITDcnjKkd9GUGigxQFfyqJWH3PJPB77OWr4P4OW05tidiqs1bKH95O9R3EeMbpt7oqw2gvbcllucMx/otzFBPCFVouLCw8IjaEKqh5eoQRtqU8kZ91+TNtj4JDf0lNqWYxH9yTK2haoBSZAl60/I6GFC+snwV4Gde7E9ZnnVp90Gpb6Ncj9iR+4XHqb7aguvk0gyqGvKJZezsDXWH5asg2sTypjZF3tLW36Ghv8SmFFP4b7AnxdQaqgYo7zhJbGxNZRK+YvkqwO407Tm4HF8MKq6TpM7Zcztix074Q46TfB0H8MgdMAboc0051XB+fv5ua6sarq2t3Wv5Kog2sbyLAcq8pa2boaG/xKYUU5ABmhpTCbLmeBXlt+DuJIcuBt2bnqxCzQDlo4FrnMuO14SvW74Kah+5Y3XWK9oVgTYjduyEXR6n+moTYTS7/Ymyry8wCtXQcsKrhnX9eADRJpa3zqDD1P6S41JM2kZqTGPgIrY3epl5QRa5P9AZBu1D3pbbMzzv+SrkAXo0YF9IDDeUm5kBikbfw/QeDMVtjdfgbMtwBcCfA7/h+SrMyADl+vAsyqcpBTm/4h2kANf+wzi0r2ZmgOKibz1HIJg+neLufRzVEzg+z9nTJ1KHmgGqiW1W8GNrnSow4Vjiyuu6KozEZvF25LbtNRGbMV8tgUupk5ZADJsSX7Heq7qhlPfrwCqIBrG8C555i8/iJrc2vr/EphST8qkxjQEXnvGcBYR6HzY/8y7y55pQNUApviQW27Ygf9bydRD72q0hHK+Him0mlA9MvdFXG5A4xtpE/YrwxQBVDf2WjmpouTqEkTalvIMMKh5LW9+Ehv4Sm1JMYpMcU2uoGqAE+C0fNOy+R/m4I1spFC3IW6ZyHji3EcxmNX/DaD/uYCmiG89ip6AdnxAHn2FTfLUBtPc7yk+OY/yMd2C4Leh1TuvMxWioNsXuBPPoRDTUjxGSt6LQxuYtG+61/aUfCnxMvM7GdOzAwHwqjD517kmiDPpF++IF7gzKPh4hTwpViLJs1sPBDFB/Vyp4Pf3QH+sRvwXA7ZA3ddrtWJtUX9OGtPu11nWbCWUvYrej8UHjd2gX0ZDXXm3SUOver+Fr+0tsxnRlTCjXvN3/BmE0kD9D+bAjd6IHZ+BQs8Dm2gY2L9FPb/RVI/oHBdrB5hLKutiVkOpr2pBPi+u42T/nMqtTEYf53wPj5ZMqqmFo2EI02mjeUaT0l8TEl0KN6fZGGD2eMg4Pznbrnsw4IvTM/l/G5OAsNhMz2XEAj7cL/m9fGZMhJG7bZWRkZGRkZGRkZMw6/gUjRqVjwwhbhAAAAABJRU5ErkJggg==>

[image4]: <data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAC8AAAAWCAYAAABQUsXJAAACiElEQVR4Xu2WP2hTURTGX6gOoqigMTT/Xv4t0kE0xU5uDooIgoqDjoJLN8Eugi6ii0vRCpEiDiLo6ODi0Mmls3+wOCjFJVRRUKhg6+9r3iknl6RqksWSDz5yz3fPu+fc8+49L1E0xBD/P+I4noU/i8XizXDOUK/Xt5ZKpVuh3jNYbDdBD6XT6R3hnEFBK5VKMZ/P58I5gTWewGMak/wY633NZDLbvQ9aGa4yTHm9V6S0WKFQGGc8QtBr2PO5XC7vndCew5larbZT1DPwaeCz6jeP/arUwiXGDdY+ye9F7An/XM9gsdPwu5PWNkOgB6GmypuA/Uya5pwmex3Yr71dLpdj1n0XDaLqLDSlgFTiodfRFn1ijCfhkvfh2bo2raqa1mEzbcljNwdZ9bmkylOB/kE6lToAM4zfSwt8RhO/eactZ7PZvYmZ8m+PuQb3ZZfZfeNPyaMf5Xc/XNIGAh9LftG05MKeS8YT3KMjzr9p44Fgg2PzTTo8b8cDzgU+lry/L9KvwrdwUrbuid5AUPURN+4ZerVXCPTJBAXBXlHy6g7/mnwI5hcUQ2M62B7sl/A4fLTmoLaG0fhL3oajQYDLcavS4nTszrxLfv1sJ89Y8m0X2UNHiDUOmx23GkFX/55RrVb32aYseV0+l2S3C9t2FzyYW4iCVhqu0xeozqlS8KlOgiw7W2+jY6uEN7xuYP6Cr7ow0ORdG1wxLW59tJoEHzONvwPbFDhqr+IdaZozzaCuw9yLUFecgSUPtrDYY4JdT2x9SVXNE95JQPuijTn7lzTvY0D/DA920N/AH6HeF6jymbjV8+9GG7Qx/M7iMwOnoy5+qnqntyEkbfMez9/n92M4P8QQmx2/AVPJ25IuD8JvAAAAAElFTkSuQmCC>

[image5]: <data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAC8AAAAWCAYAAABQUsXJAAACiklEQVR4Xu2WPWhUQRSF3yIWomihy5r9e7vLNmIlCwZBsRK0sBKxUDvBJpWCNlqKNoJESWBFQgobTSdiYyEINjY2/mCwUNKFRCwiRInrdzYz4e4lEbP7GmUPHN7cO3funLnzw0uSIYb495Gm6QP4o1qt3vR9Ea1Wa2utVrvl/QMhJD2qr+8T5G80GtVyuVzyfQJjH8HjaiN+P4v4VigUttsYfHXYoZmz/r6BmKYSwruymfgO7SUbg/0MTjSbzZ1iiH/sYjr5fH6Hsd/WVnGRdpu8J/lewB614wYCST+S8EwSqhGEqToROdl2R7CfhJi1Croxst9Zu16vp5oryarqJDvHJD+tL4hfMfYYXLAxjGvhW1JVTZxfTI947PnMqk6yQ3AZzvi+CKpVoP8T/Gz92CPywdfGt1wsFvcEM8cCp0xfm/uyK9oDg4SXVC2qMc1E52m/EbkD20zMPrigBbixUfxc9IULq+On9milUjli4udjOxOQ8KnEw8V4nvUNvuey4/GAL9zYKN5f7GvwAxyTrXzaAVf1LabdHyQoCH3p/BLV0ZO4WfEe9M+S44rapVJpN/YreAI+7AbgLGO0/5K34YjG6bikq+JvmPnsos4a8WtnO8RE8T0X2UJHiDtzMNrEzv0pflOQaIlE4FXn74qX34jc6ML23AUL+mYT95T6PH0jDZdRO+D83WOD+MPBHlecjTE70rNrEXqCbdWFTMULTHJMIqwP+1dq3nm9Ppo46a3iPfnsyxShVycNF94i5M1OfHxdmHCv7PirEC9ZBL6v8JSxV+SzMRH4F+GBdfzv4XfvHxgkvQ5n4GXfF8GCTtM/AceTDZ47VX293RDCsznJ+Pt8v/j+IYb43/EbwjvdZVwFZfwAAAAASUVORK5CYII=>

[image6]: <data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAC8AAAAWCAYAAABQUsXJAAACmElEQVR4Xu2WO2hUQRSG7xItxFeh65J93X3BGrSRBdNoKWhhExEbCwtBi3RCKlvRxiZIAisSUthoaWGTQhBsbH1hsFCCTYjBIoLCZv3+3ZnN2WEjZnMbcX/4yTn/nJk559yZ2UTRCCP8+4jj+BH8VSwW74ZjHo1GY2+pVLoX6kMjl8sdYcMz6XT6QDjmoU0rlUoxn8/nwjGBhJ7A87JZ6wRFfM9kMvttDFoZtjFTVh8WKRZrlcvljBw2PafOwSkbhP8cztVqtUOiEoBPg5i2LR7/bamLG9hN1r7I3+v4k3be0CgUCmdZ7JrV2OQjm3w1kgpsq/NewH8WdtD5PeC/sz4NirV2lETX6/X6QTZ4yZHJW911qJcI9jRcC2IaaBvqqokLi+lLHn81sa7rPLLgkjbVUfC66+qqbB0n7E/w89bMTsy4NPjaaD+z2exR56YocMGMNbkvh72fCOjETSWvZHXekcawN7FnNI49AddUgJ1nkl/xmruwV5w9qSNp4jvNSBzqkCugQ6QxM9Y5HvDF1oy+5DcC/Tb8AKfl655o/aDrvfV3C3W6BSfUOVPEGw3uNPkQjC/7r6jnGP8VvAAfdwJ04XCaf8n7cFzzfGLVavWY38w8g20++0mTfO9sC1oj7ibfd5EtdIS4M6e9T+zKn+J3BBaaVZKhrk65AqZMkttd2L67YMHYchQ8peE6QyN2L02oC3RtUcnLjrtFDnwq4R2rezB+1XZdSDR581YfD3T9yrYw98jn34F9rkjbxQfSNOY1D3d3lkIdbTOx5KPuWzwDv8h2mi6wOtq3Of46vGR8XfJ1G+OB/g2eGqC/hz9CfVdwP1Zz7qjcCsc9KPKy4uBstM1zp64P+hqCezbnmf/QNWyEEf4r/AaoGtbyM61wcQAAAABJRU5ErkJggg==>

[image7]: <data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAC8AAAAWCAYAAABQUsXJAAACgklEQVR4Xu2WvWtUQRTF3xIFRUFB18X9mre7ZQrRFVOJICKKCKKChZaCjZ2gjaAWYhRsAiKsiFiI4F+gaJHKJq1fGCwiYhNUFCLEIvo72Zl4d3wxYfMacQ9cdu6ZM/femTfvvk2SAQb49+Gcu4v9qNfr1+K5gHa7vTpN09GY7xsE20jSHcVicX08J5RKpXUxx5o1kf8IO6AxxQ8T72u8Dq6B/WRYsHy/KChYrVbbyXiIpJfwJyqVStWKpIkN7ZNYYzeP/zLt4gzjDvrD/J7GH7Hr+gbBjmEzhprfDInuGe6P4ingerVaXRtrIv+V9RuNhiPu2ySPUyfQBV/IfcvDffCFLCTBnzKSTGSs6SkefzrPUx9XQm0i4qfEc1LbLGc1WUAzWy6XN3u3YJ8ec51ms7kh+CvGUsXD77OcuoTeDTa13+oD/At7wo9H0O4Oczr138oc8Jdr8008dtJw8vcyLOiue/+FWRZ0F7E32Fn52rCeQHTqQ2bcN/Roz5PoYyCUBH9Oxak7BJ7usymMBeYfSBNa42JAM6kcGisG/nPsoNbPC9TWcDrLtJvY1ijBOee7CDbmMu58jPDUsGfxXICuEDF2Bd91G8Enq8kFrVZrS9hUKD68fBSxB/+p1Zvixy1vwdxk0tt9pF/yxV82KOxIGn2qfZJZjfXRUYHikt5Crnrd2MJCAzZ3yp66kGvxBC8R7B02FzjX/WhNk3zYcEfj/gz3RcXohbS8oK7jMq6T8uRWPFhFsIcku+x9fV1nsENW5DvGe83LR5/6U5ywugD4z9j2DP419j3mVwQKO+66V+NWsngb08ZuoH1M7VfiyQCdevy3IcAfwm3i3PGHMcAA/xV+AQqUzt9cP+eoAAAAAElFTkSuQmCC>

[image8]: <data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAEgAAAAZCAYAAACSP2gVAAADN0lEQVR4Xu1XPWhTURR+oQoV/xCNwSbNTdJIKAgOgbo46iTi4KCgm4iL4CCIu7gUhRKKQhfp4qDiJv7QIZDBwcWhXYQOBaeKikKKKG39vvRcejh5ec1LSlvlfXDIu+d897xzv3vfvTdBkCBBggT/L1LFYnFseHj4BJ9tMA5KpVI+l8tlq9XqbhvzKBQKg/l8/vTIyMhRG9NAnuPCGbCxrULKOXcbxT7OZrM5FH4Z7VXYcwz0oCVHAX1es2+5XD5AQ65p5tEcTAJcbhaxu2zznexjxRTeMgTaI5wZ2APN2RJAmLN48a9ArRq0GyyaoinqRqDQqxj4M+/g4OhjzPsYtz4R7JJvUyzh3fM+riK0V3y7byDZOdg3zMRJG9MAZ4EFwx56H4Q5JD4OpCuAexP2FTZq/E0M9oY0WyLC5jQH7zsP3wp+z7CN50nyhoaGjmgefZ7TEzKZzF4kuIVES1ziNh4GcKfkxdeNnwOJI9C8WxP7mPHT94Gic7Ikb11zEKuK/4X0+cl2Op3ep3nwNYWzS/u7Ar7Tw+j8G7ZIoWw8AgN2o6xUKvul4DgCcfXMQ4SM8VOgz9xHOPuSd0ZzlEAN6bPCth2HWxOowfq0PxJI/oTJsIwv2FivQM4J5oyzB0nx9ZBZp0BNiiCfEmud1hwl0IL04XPb5Mg72lZpG/CCUyC9x++7oM8j2YInF4vjjNtYFHaMQAiOSoKXNtYveHog7xQKfmtjG0GK336BCK4grp5NXkE8Yf64Hu8aUnhb8eJr7U1KiLrmKH9rbxIhOm3SM3ZvisRm7UHI8SVQYstG3do0uwG4NdfhmIfd5zNqHBQhwo55+q9In1dshx3znhMbcootwRZ5QbPxTuCNFQV+kpdba3qeW/+0J4OQVesvhU5d7gK59+h60J6lT3Eo0BvYRCB55VI4C99Vzwm7dMaGugf96PYexM9UBhZmrT2BcOsCzdmZ9UDsO2xZtS/SZzj0LWNixsTVEpETFcLzfVMY1x1OpOb0BdflTToukPOji9gkOUmIP4LV8hG3XsSvwZ7CxoMOq0K+Cl5kx7lP2fhOBGe7Zp0JBHJU/xMzuS1w5i9CggQJEiRYx1+YIzOSIRCGngAAAABJRU5ErkJggg==>

[image9]: <data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAD0AAAAZCAYAAACCXybJAAAC+UlEQVR4Xu1WPWhTURROCIKiKCIhhvy8NIlDrVvQQQQXBx1crN2cdFCh4NBBnIuLi1AkUx1cHHQMRQeHzHZxUUHooIhb6dRBpcbvC+ckJ+fd1xcHkdb3weHd873vnp/37r3v5XIZMmTYd+h0OgdwyXu+0Wic9BxRqVROVKvVU8Vi8Yi/9yfQOJ63YG2o46LUmIhms1lHrEqaboQoisqwgbd6vb7gdEvCn4ZbwPgmxl+sZhoE4vzwcdDEMamjSx+NP6NP3urAvSLfbreP0kT30mqCiMJNfwzovsO+OS5WSBp8HBR6x8eBvwz+hb45vMVD4HrgHqgGyHMedUqIbsB7RhcHRGU86fuet6jVamckwW3Lg+vD1sAftHwSQnG4TTSOctTAZtUnUGNHGlLNImzT6+Bv+zpjmKZpaFaZELqrlpfl9BP8BcsnIS0OxzMzMyVpumw19GXucfE3YJ8DOnLrqguCk7RpFgOb8wcCNGtSyBXL616DXbN8EtLiiGaWYzZvNayTPA7Aqvh8yxsBHZv+qrogNFgkhwau52FbsCWj6cO2ucTGM8fFpq0URVocjnUZ+6+D1qlzGYfxAjo2HcvhUfB7EnvvLBPwYKCfVuxebDoGTRDJIZFW7J5rGjcfI+ndnDnmTYLhgWOaCx1Ag8jt0SSkxeFYc/NqNcrrHpbmaF5HLrbXJyAJJp4Mxk3hh28a/iX6jfAna/eT0iAUx3yy1pWzuRW6AoxmJUr4ZMEeWi4GJnSndR7cE0423LAQFPvWcb9QzJzx31MH7rXVWfg4GF8OxOnBlnPj1ac19VSjPyKiU1A3OosSgWVwDsIuf+PoI/k9+DuwRauD/4YB9QFhXkQ/N7kthk3DNkcTHXwc5Pvk40hNrGFe5szTJ68a4fmV2TE+dVtWk4hSqXQYyW9gQh/X67s8Kf4rP4I9hd3yNxW4985zDqM4+rA9pKYFaJ5zW4AqeA1BHTRd2Iro/g0i80v5X6DVatWiKf/Q9gXYLOyD5zNkyJDhb+E3Wq9KQobKmosAAAAASUVORK5CYII=>

[image10]: <data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAEgAAAAZCAYAAACSP2gVAAAC8ElEQVR4Xu1XPYgTQRTecCec+NNoDJe/SUgkKIJFRBtBERFFRFDBQjs7GzuxFTuxOOQqEcTGQsRO8CdFIIWFjUWusbsrT1AUvEK40+/bfWNehs26mz1jlP3gsfvefPMy883M24nnZciQIcP/jxmxxCgUCtvcWK1Wm3NjBOPVavVoo9HY47ZplMvlvcIZa0ybgRljzDzsOuyH2LxLigPV/5dBhFeaU6/XETZ9CHSLfqlUKpPXbre3hPDWIdBW4XRg9zRnIjCBOMsyoY/y3CyB/AlqDoR5yja85mzMBIJdtj7FEt4dG+Mugr9h/dRAsrOwT1iJg27bKIDflYmNKxCFjuqbk/xLOohddo6Tx/MkfbwvklcsFndrHmOWMxZYA5DgBhKtNZvNnW777/CnBeJiSf6ujmPMbYk/o4/nV/r5fH675iH2TTizOh4LOKe70Pk7bDWsWMYBBy4DHTnJKBgRqFKpHIIYpzgm3c7Vl/wdJ24F6tHHc4O+Ow8TCNRrtVo7dDwSSP6IyXBmz7ttSWHSC8QjsILXHGsP/Iewvm2Xo8SxPlbdtEDL9OWddWoIJhAocpf6wA8cAektnq89VezSwqQUSHaMHo9fczDO03QmIhAa90mC525bWpiUAoVB8t3n+0QEIriDuHumaQdhLMfQ7w3vLDou+bosuEqIruaouF+bTCDEqCLdcWtTJKalBtm+HI8TZz5/B2GMc+KHfeYZvyJ9XtAP+8xbTmLIV2wNtupezuLCRAhkBkd70QvZtYhf4M7WMXzNDrCPviXD7zOmeRDoJWzBk7xyKewjdtVypOgPXTATQ92DviS4B81iYsfN8F+NHvJcwvOMJZmBQEvuyhIUAX1W0H6CPnLW8P6OpnnwL8LWcQ04LCG/kPOvRQjvs+Ug903YB81JBTPGTToOkPO9CdlhAk5kP3cE9LmNVW+6BAvkuAZ7ArvrjdgVcioekMM65bZPI7jafj3JEAKu4r+ykn8FxvmLkCFDhgwZBvgJDs0UcI3G894AAAAASUVORK5CYII=>

[image11]: <data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAADIAAAAZCAYAAABzVH1EAAACw0lEQVR4Xu1WPYtTQRRNEGEFQQujmK95SVQQRIvgdjaLK4LYaCFop/9gBVfEZitBrRarZbew0MrGQrDYImCnrSIoFlqsRRBBEFTIruck94b7buZlN1YR3oHLm3vmzMy98/WmUMiRI8dEaDQas0mS7Pf8JED7mWq1erTVah30dRYcp1arnfC8BTUTxRRC+AV7imIRDU+i4Wv4N71uO6DNW7S9jUT2VCqVKvwt2EOnuVSv17/gOwe3iG8X9tNqRPdHNdDPUwO74HVDYNAjEN5DcZdyzWZzHxr9NrJt0W63dzNwFIvKMTHYppGR28BKnFFfguxhAhLlGJMfnxrYJ8ulgMon6Ph0hB+ZpTHgzD6CvbMkgrzI5PA9Sx+rfUiSTQE5PLZtUX4PW45otsrl8gHLD4HKVQogfG5oBtYz/lggwFPQ/4B1LI8E2pwQ2DP6+C7EEoFu0fKS/GJMA7tm+SFkaSmgXQdVRFJXUF7x2ixwxqHfhK07XhN5RR/fZRuw0Y0kAluIaWB3LJ8CKu+KSG3Va8ZBtxCX3/GayGf6uj2sRnQjibDPmMaPYcFtxMHewB6YZLpemIWpSASVl2HrvKnoJ4Ort59MqVTa6/UxTEsiqSvT8LzusvejgQm4k8H3z44P2OhiiWQd9tTZGSLWMSENd5QIZmkG2hch4/oNctPgezw2nqzUV/VR/hYyrl/02bT8EKj8GAZ/UM937daSgLi0561OwSeJDxKDvhTO/iR7/OGpz2cKuO/QzivHMrgN9Qlq2NZyKYTBGeHyXxWKh38OnX1wun4iwc2UBQfCP2VWXPbDGfT9rARzJlmmji8D1egrQTXy0uDY438JbIgAznH5YEtoeMxrFJmHTYDBbkiw9wuRs0ewf44j2+Wwrxfwf7aE+rXwD+++seAToy7Pjf8avAB2eiVPLbC8HSRyy/M5cuSYfvwFpcAHbhK8rYsAAAAASUVORK5CYII=>

[image12]: <data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAADUAAAAZCAYAAACRiGY9AAACh0lEQVR4Xu1WTWtTURB9oQiV+rGQNJR83LwkGgtChYCuuhDUjYjQbgSX/gDBlb+huHBRKBREuiwUd4JIkIILl24qhUIXikSouGjBLpS0npPMlMltL0lsIrS8A8O7c2bu3Jn78e6NogQJEpwoOOcmKD5vUSwWR3O53OVyuTzu2yziOL6Rz+ev+fx/AwqZLRQKC2imarXaGehNSL1UKl30/NZQ1DMUdTabzeag70Oeez4ziPUVzRQF+g/IPeszdGDAGPINyV413DwTRnIvlJNi96N2suq3BtlTHcVWoDewQtPKIcYdcJuIX1Tun8HZgXzCNpjybRbwecpkJeEWkMh94XaE4oyz0M/qY/3wvU0d7XUbRyGxOvr2jEwmM4YBniDAbqVSueDbQ+Aq4DOiOmK8kkTmqXNiWCBkVX3ErwbuF2SFuvQJFXWI7wrs8Uvo+BuyxeJ8ez9AjAYS3kAxjjpXAtwepG79TFEfpN+RyYf4IHRWsWUf+LZ+wHPACZEEOg6/bjOMseTxWtQX6qHkQ3wHEPwmnD7i+y4yB3dQQOz3/IMh/nXqQy8Kxklxeu3bBgXEvsUxuAWpD70ogivFVRrQSqV4Hi2RTqfPaSLVavW8SX7V+hm+ddZCyYf4II57pnRAuwosRHm2YRtF+40L/NIhj6jj+/Oo5MXnu893hfz9diFbvPF9ewhO7ha+EJRDEQ8lkYZyfBb5CaOot8K1dkuhfdE2eQmrD59K5Gg76NgPzD213es9Bd9ZyB9IrBzPkmtfDzOeb5NvOlF5IR+cO0JeHYvO/PrZJid34fHgenxRKOB3Fyu0zG3Is+LbFYj5WBKfiwLnGW/GK7C/ZCzX5YGcIEGCBAlODf4CebDk3F62tksAAAAASUVORK5CYII=>

[image13]: <data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAE0AAAAZCAYAAAB0FqNRAAADYUlEQVR4Xu1XPWsUURSdEARF8QNZF9mPN5tdhaiFsIiFiqIGtLCJ/yBbaJFKQWsRm1SSwiLEwlbrYJNCLGNhowiBIIoYMMhikUIliefsvKvXu28yG9ZIlDlwmXfPu2/vfWffx0wU5ciRI8d/imazuS2O47OFQmGX7dMolUr7y+Xyoay4LCDX3kqlcszyBgPVavV0Vq6hoaEqaipxDrZvU4CEe5xzX2AtioHnN9gabFzHwb9JHpM4AncQ7TG03+uYXoBxoxyH53m4A3guwZZNGHnW8NKPGWNdEPq4DgL3lHGNRmM3Df2P4D/RMZsCTOA+xVL+KU4CtqLj4H+FfTTcGkXXXBb4G1hhZ8RHvhHmwoRj4djPONTVEA7d12HPxY+8sOAeC4H4HeTYp+KCQN4LlusZTOKLPicctwR5PJv0uY18gdd+Dow6Y5/BZsBv13waarVa0U/qN/gV8lp8tNs2F7eor6GTC+1x2GfYsI6Dv2zHhsCclusZEGYeiRZ1crQPetGueH9a+wI/2e8UWfNpQOyNkGgYf1vzoVyK7+RCewH2jrWaGHIvELdP8xZ9icZ/zq4UrKwTpsAZ+rDLOs6LRn5U82lA3CTjLa9FYy2hXITO5ZJVtsDVa2Io2gdcWGXNW/QlmkW9Xj+ApK+cOtNcsg2XZbsKRDROWvNpkHjLa9FkG9pchM7FeliXvVm9aF21WvxR0ZCwzeL09Z2Ltg6w1J0v7Jbm/2XR0J5KsbcBbqq6kVu1WCzuZDLLE0qc0EWw5gLnTwhanPX4UC7hJZdLxKGFLoKus86i75XGlYVEnzTHy0D+LTwvsuA4/MqReVMJEDscEs2Lvyg+2qs2l1qBnVwuuVSCrxywe5oLoS/RkOAqiwmZXvr0kWjOjF3FJI4qv3M7wto6TgN9K/ql1b8DtvE7I8LFyYvsHD/ZFHcJcUviy4ss7K5wkX/hZZ/iguhXtM7BHzITN0tOLgg5/yL19u1+ida1mgQuOTtm5UvC/i7hP+34O1P0mQuiznNHSAzhktr1Lc8FkPqHafQl2gbBb84J2ENYy3YKsgqCKIcRc4dxzpxJCoPoP+ljWvy2tAGEP4sfwCZ5jNj+NGTV+Ffht9sby281bOiW3Gy45JNr2vI5UoCviQrPFcvnyJEjR46tiR/3cVxZTeS17QAAAABJRU5ErkJggg==>

[image14]: <data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAJ0AAAAWCAYAAADaUZ7NAAAD1ElEQVR4Xu2aPWgUURDHLxgh4hd+nIe54/ZyOQ2pLA6sRCGIYCFCgihql8ZWBDG1XbAQDRaihRYWYhf8QFIcpBFsLJRAQDBpJQQFyyT+/3vzkrnxNtl4yRl1fvDYnXmz70125t7XJpNxHMdxHMdxnDUoFApHisXiCdxus3WBUqnURZve3t5Dtm49VKvV7WjrFPrM2zoNfZK+Nt0np4309PREYIqJQBn391Fel8vlvdoOuo8I8C3e5/P5AuSl8Exa2CbbRnlAuVKp7GE7zfpCWUDS7ZC+JlDuaBvxu2WfnDYjI85zBD+r1B0MHsptbUcd64JOAn4xyGlgmzYxII+jnZEgh750/xzFIC8GmdDvjfDJaTMI0qAEzuprKDMiMgnHUD5pG0xp55gIuJ7W+tWQZJrTOjxf1T6Evrq7uw9qO9qovsIPo2WfnDaDAJ2VYM5mGkeMBeju8h7T2DHI31FqoZ6EZEF5ofVJoJ2c2IdkjoF8WHzYJ3LcVzab3WXsfkhfneIT26ppm/X65PwBZNri+mkJ09ITrrGwjqogeNNcM9GGowbqF1Em9LMqwJNanwTs+sX+s9HHScc1mchxX7lcbqexY9JN9vX17Raf2FZLPjkp4eIaL/UyygcGzNYrOpBI16yyGWjnqwSLI860rpMpK05Kow8Bbhi5klD2Na3n3yD9VkX+pS/RM+lmaL9RPjkp4MiEFzuLlzoom4C3fMlIxP3WlscRrLd6C2xKDCgu53H9JkFjucH6jQqwJ91fig1YQBJxQOsgTyEol7TOAptHDJLRDUjg4p1hCHBk1kkqwA2L+SSU/XutjyTpUPpFDn11GjsmXbzB+F2f5Id6HfUP0xQsMU7aNv478CJeWR2R0Yov+ynEbbgOc5Szi3EL7OZQvlg9gjfN9mT9xEAy4DVjEwLcsK5KIlpJrqYbCW40RI77sr6LPl7rJY2a6/XJSQFe5pDVabgJgM07jgS2rhnNAkdCojHpkLxduH8ZNT+e4PNXtH41xH6tI5O4r2ZHJqEv8Ylyyz45bSZamV4bPjNBN44yH2TZ5TZMwwjwm2L9WCU+aonqu1OOnA0Hthpp1x7o8lxuPMihL7R9Nehk82Sfi3fdQSbWJ2cLwmMRmUrjz1JEPictRGZUpQ72x0WMD2fDsYrULyedHaUCfB7187pt0+6yjnYicl15k34am6G1fHK2MOVy+SgC9pi7QQTtjK0PwGYY5RnKaCZhNJHpcrWjHNpckHbuZRI+5iP5D0T1Bf0o27T1gTQ+Of84CP6Y1TnOplKsf1JznPaA6XnE/puS4zhOan4CX8l9uma5Q04AAAAASUVORK5CYII=>

[image15]: <data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAIIAAAAWCAYAAAAM9ESoAAAC+UlEQVR4Xu2Yv2sUQRTHLwRBURSV8wj3Y+5XE1GbFduAWNjYJApia6GFvaSU4D8QQgoRxEaIlgF/kSKlcE0KxeoKxTZaXXnE7/d2Jj4fe7cjbAx67wMPZr7z3bfJvtmZ2SuVDMMwDMMw/iLVavWs1prN5lGtxdJutxu1Wq2aJMkRPRbgGO6xMMlDYnIZBdFoNB465/ZkQHunfXnguje8ttvtnmSg0M/RfyU9KOwpf4919r1nj7r0xeQyCiZjIgzxFh7TvhxmeC0K9jIIzEGNY0FDf4We8IZ7zya05eApReYyCsZPhDmt/wm4/gFiFzGv9AEKek/0OdF+8+D+iS9y8ETlMgqmoInQR3zRebzWY7vValX8RNCeOer4O077fm4u4wAIE6Fer19G+2bW4TEPl77BfRZb6SzeN9+eZ8EzPKOJgPvWfD83l1Ha3y/vIHb4EPW4YAZL6X0tajgREF9Lfv9FzqeIj8o2EfgHiO1yuXxC6SzegO2wBWR4woqQ+H5urqmn0+mcY9HwQBb9J9h7PkRMjjPay88ujmtd4x+4PISFw9p1oU0kpng2EQqED0hrxE+Oq1JD/zOKeVtqsbAwiK1KpXJcj2Xhi9cL+7zQWbxdtsNEyPCMJoLzh8OYXFlgK7mE8XXEk5jQE+2fAv/Aa60RFLzp0pXhIrqzaN/lapD3z8KzAG8v7M8B5nIZb+U4XFqkcQe8vm+HgmvPSA9ngphcUw8exJLWJCjsI3g+4G26oceygHebRYD/mdJZsFWpTYJeN+aTD/FY9Jk37/MxKpdRIHiwi4jvUsPXwwUWRv6s6wvDIq5JbyD84INYEfLorCF/nEJ/03vCmYSeNerBE5vLKBAW26V75v75Au0eQ/rcr4nwSeoSjP1ADEV/iZr0YPm/Qg/HhGdIXfpichkHAJbn84i32Fo28NZ19XgAxdjRmoSHS5ce2FaR75oeJ/Rg7BY8L7xnVntITC7jcODyHH1uMP5TeKgL3/rGlIItYxmrwZbWDcM4JH4CFIMh66EOLY0AAAAASUVORK5CYII=>

[image16]: <data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAACEAAAAWCAYAAABOm/V6AAACkElEQVR4Xu1VPWtUURB9QYRo/ALZbNyv997uyhIUVBZNo50WaWORgHaKImxlYUgXCGIvopAYJIWNhSC2KSRFfkFEEC0UETTokkIhgrue8zJznQy7+QGageHdOXPm4867970o2pV/XZIkGYnj+HelUrlXKpX2eb8KOPMeyyRN01NIcsTjVsDJU5vN5l7vQ+wzJO9wjSZOYL0OvWY5sFPwXkBnLE7HL2ibiRlEG6TTjrMMfQd8UBKtcMeO04W+MvZErVYbBvcm1vPQ+9Dr0HUTlu3sHMAOiJOKwW6xiOWRA72tdrlcPsmi3LHhdBG3pDZ8TWhVbcHe9ppC1n0ulzugWKPROEhc3ynHL7xuPp8fIka+YI9drqdq+ybgu4wGxtQO4runmALjivGs2EZZgBxORDHYm9DXYg5g/YBPGnzVfrpBdmqC79LiRlhgjhyuFUzMweSOsf6svr5ToMiOt70OM/67loskg5jAR/GtRaYBlXq9fgi+99CW3iDEPPFTKBQK+4PBDqGbGOsF2nJDFlkIwdOBuF0GkHSSnF5X1Qp4CfLciaRhxKxC22g2h+cbS2TCr7LD5ziQx2UdzkQvEc4naOp9KvB9MesU+g06Sht1p/4yt4Tv+RgX5mBmZEqxWDxKTmBHoQnqhMVVzBQyAW8c+kHrhPw8LDAW8BxRMgLPM7na8daHisV+KCZ43yaQ4yrw7w6bdk1kz1AgkZsgH682Ai5pMOwWMXBuKVatVg9LA3ORm5Ccq2VyLI6cF20TOIdnMwcSz0KX9CMkp5+f45BYCm5Arwi0BzEzbMIXosgUznhcbt1PLc7NBad0yO/64k5/PvwHyuRAH7KQ91NkCi89roK4R/B38LwR9/uT7sp/LX8A3hbLKpTg45AAAAAASUVORK5CYII=>

[image17]: <data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAACAAAAAWCAYAAAChWZ5EAAABUUlEQVR4Xu2UvUoDQRSFYyHYGWyC7M/sjwii3dY2AUtBxCexsU6ZNmW6+BzBF7DVxtJCCwsrm0A05+gMDHfnRjbZcj8YNnPm7NwzN8n0eh0dAcqyTKTmk2VZP0mSM6lvDDbcw7jCGBtjfvCcSQ/B2nWapq94DjHdwfMD40v6GoNNL1kYY4HxvSbAG05+7uZ47wLaEv7Ms20HTxQKkOf5gCGlTi/0Z6lvjBYA+m0oALpwJ3WEhWSeiqLY93VHFEUx1udS/2VNgIksREIBiA1RK2KLP2rh1AC21bVCWgBiQzyI+VwtTtoMQFxR+E79MCptByDwvNATx/GBXKuhBdAKabqjtQ5APwkVsp15lzpp9TdAsLZEG4/cnNcxtE9eSL6PNPoX8LvBJjcoPOIpzd+NeE/Ne5cBpv4J+Jn+qqp2fd9W98B/YNNjBrWtP5TrHR1NWQGxUn9jZ9CJjwAAAABJRU5ErkJggg==>