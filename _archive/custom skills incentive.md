### **Part 2: How Custom Skills Maximize Token Efficiency**

Custom skills are the single most effective lever for reducing token burn. Instead of letting the LLM do multi-step tool calls over chat (which inflates context), **offload multi-step logic into Python code inside the skill**.

#### **4 Core Skill Architecture Patterns:**

1. **Local Work Encapsulation (Zero-Token Sub-Loops):**  
   * *Problem:* Letting Hermes run 10 separate shell commands via chat burns tokens on every turn.  
   * *Solution:* Combine the 10 shell commands into a single Python skill. The skill executes locally and returns **a 2-line clean summary** to the LLM context.  
2. **Deterministic Context Compression:**  
   * *Problem:* Dumping raw test logs, database queries, or `git diff` outputs into chat fills up the context window.  
   * *Solution:* The skill intercepts raw output, strips stack traces, extracts only failing assertion lines, and returns a compressed summary.  
3. **Conditional Model Escalation:**  
   * *Problem:* Using Kimi k3 for simple coding tasks wastes expensive output tokens.  
   * *Solution:* The skill attempts execution with DeepSeek first. If unit tests fail, the skill sends only the failing code \+ diff to Kimi k3 for targeted repair.  
4. **Local Obsidian/RAG Retrieval:**  
   * *Problem:* Sending full 50-page research documents over MCP.  
   * *Solution:* The skill queries local vector indexes and injects only the 3 most relevant paragraphs into prompt context.

I have generated a ready-to-use Hermes Skill suite and model router implementation (`hermes_token_optimizer.py`) to demonstrate these patterns.

