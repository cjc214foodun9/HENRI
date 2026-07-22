# **HENRI Machine Learning Evolution Blueprint**

**From Physical Wave Simulation to Goal-Directed Machine Learning System**

## **I. Executive Diagnosis: Addressing the "Root Tautology"**

The concern that HENRI might remain an elaborate physics simulation rather than evolving into a successful machine learning algorithm is rooted in a real architectural vulnerability.

In standard deep learning (Transformers, RL, Diffusion Models), learning is driven strictly by exteroceptive task objectives (e.g., Cross-Entropy, Bellman error, noise prediction loss). The internal representation exists solely to minimize external task loss.

In contrast, HENRI’s initial phases prioritized internal physical coherence: Sagnac phase alignment, Kuramoto synchronization, Stiefel manifold retractions, and Langevin thermal relaxation. Phase 2.7 successfully validated ![][image1] RMS normalization, bounding Expected Free Energy (EFE) in ![][image2] and eliminating the 93% fallback loop. **However, task scores remained 0.0.**

### **The Root Tautology**

> *When an agent optimizes for low internal Sagnac stress and high Kuramoto order without coupling those metrics to exteroceptive environmental state changes, it becomes a mathematically closed, self-evidencing system. It achieves perfect internal harmony by remaining entirely inert.*

To evolve HENRI into a competitive, generalizable Machine Learning algorithm, we must transition from **passive boundary containment** to **active exteroceptive problem-solving**.

## **II. The 4 Machine Learning Pillars for HENRI**

\+-------------------------------------------------------------------------------+  
|                           PROJECT HENRI ML ENGINE                              |  
\+-------------------------------------------------------------------------------+  
| 1\. World Model (Transition)   | 2\. Credit Assignment (Natural Induction)      |  
|    Low-Rank Ephaptic Field    |    IDBD/SwiftTD Viscoelastic Creep            |  
|    \+ Block-Diagonal Matrix    |    Online prediction error updates            |  
\+-------------------------------+-----------------------------------------------+  
| 3\. Active Inference (Planner) | 4\. PEARL Local Repair                         |  
|    EFE \= Epistemic \+ Pragmatic|    Salvages off-manifold candidate trajectories|  
|    Grounded in Exteroceptive  |    via targeted phase-blend ($\\alpha$)        |  
\+-------------------------------------------------------------------------------+

### **Pillar I: Low-Rank Ephaptic World Modeling (Solving Cross-Block Coupling)**

* **The Failure Mode**: A pure block-diagonal transition operator ![][image3] restricts information flow to isolated ![][image4] channels. Cross-channel spatial relationships across ![][image5] cannot be learned, creating an irreducible transition loss floor (![][image6] or ![][image7]).  
* **The ML Fix**: Integrate a **Low-Rank Ephaptic Field** (![][image8] where ![][image9]) alongside the local block-diagonal operators:  
  ![][image10]  
  * *Sample Complexity Guarantee*: The cross-block information bottleneck scales with rank ![][image9], allowing global dynamics to be identified in ![][image11] environment steps rather than requiring millions of samples.

### **Pillar II: Online Credit Assignment via Exteroceptive Eligibility Traces**

* **The Failure Mode**: The agent treats actions that reset the environment (e.g., RESET in ARC tasks) as high-novelty states, looping continuously without penalization.  
* **The ML Fix**: Implement retroactive eligibility traces (![][image12]). If an action sequence fails to yield positive scorecard progress (![][image13]) within ![][image14] steps, retroactively assign negative exteroceptive valence (![][image15]).  
* **Separation of Training and Planning**: Never train the transition model on its own prediction error in the same step where that error is used as the relaxation target (preventing the "subtraction tautology"). Train ![][image16] *after* the environment step using the true observed next state ![][image17].

### **Pillar III: Exteroceptive Grounding of the Pragmatic Field**

* **The Failure Mode**: Pragmatic value defined purely as similarity to static internal Zone C priors causes the agent to avoid state transitions that alter the environment.  
* **The ML Fix**: Ground pragmatic preference vectors (![][image18]) dynamically in successful exteroceptive state transitions (![][image19]). The pragmatic value component of EFE becomes:  
  ![][image20]

### **Pillar IV: PEARL Local Repair (Training-Free Trajectory Optimization)**

* **The Failure Mode**: Hard rejection thresholds discard candidate trajectories that are ![][image21] correct, forcing the planner into random exploration or fallback loops.  
* **The ML Fix**: Decompose multi-step candidate trajectories into local reasoning units. Retain compliant steps as anchors, and apply a local phase-blend (![][image22]) to repair non-compliant steps using Zone C preference vectors, preserving an auditable repair record.

## **III. Code Implementation Blueprint**

### **1\. Low-Rank Coupled Transition Model (henri\_transition\_operator.py)**

import torch  
import torch.nn as nn  
import torch.nn.functional as F

class LowRankEphapticTransitionModel(nn.Module):  
    """  
    Combines local block-diagonal operations with a low-rank global   
    ephaptic field channel to allow spatial coupling across 65536 dimensions.  
    """  
    def \_\_init\_\_(self, d\_model=65536, block\_size=8, rank=64):  
        super().\_\_init\_\_()  
        self.d\_model \= d\_model  
        self.block\_size \= block\_size  
        self.num\_blocks \= d\_model // block\_size  
        self.rank \= rank

        \# Local block-diagonal unitaries (8x8 matrices)  
        self.block\_weights \= nn.Parameter(  
            torch.randn(self.num\_blocks, block\_size, block\_size) \* 0.02  
        )  
          
        \# Low-rank global ephaptic field channel (V: \[d, r\], W: \[d, r\])  
        self.V \= nn.Parameter(torch.randn(d\_model, rank) \* (1.0 / (d\_model \*\* 0.5)))  
        self.W \= nn.Parameter(torch.randn(d\_model, rank) \* (1.0 / (d\_model \*\* 0.5)))

    def forward(self, state\_wave: torch.Tensor, action\_wave: torch.Tensor) \-\> torch.Tensor:  
        \# 1\. Circular convolution binding in complex frequency domain (FHRR)  
        fused \= torch.fft.ifft(  
            torch.fft.fft(state\_wave) \* torch.fft.fft(action\_wave)  
        ).real

        \# 2\. Local block-diagonal transformation  
        fused\_blocks \= fused.view(-1, self.num\_blocks, self.block\_size, 1\)  
        block\_out \= torch.matmul(self.block\_weights, fused\_blocks).squeeze(-1)  
        local\_out \= block\_out.view(-1, self.d\_model)

        \# 3\. Global ephaptic low-rank field projection: (fused @ W) @ V^T  
        ephaptic\_out \= torch.matmul(torch.matmul(fused, self.W), self.V.T)

        \# 4\. Synthesize local \+ global continuous fields  
        predicted\_next\_state \= local\_out \+ ephaptic\_out  
          
        \# Normalize to complex unit hypersphere S^(d-1)  
        return F.normalize(predicted\_next\_state, p=2, dim=-1)

### **2\. Retroactive Eligibility Trace & Training Hook (production\_arc\_run.py)**

def process\_environment\_step(env, planner, transition\_model, current\_state, action):  
    \# Execute action in exteroceptive environment  
    next\_state\_obs, reward, done, info \= env.step(action)  
    score\_delta \= info.get("score\_delta", 0.0)

    \# Convert observed state to continuous wave state  
    next\_state\_wave \= planner.transduce(next\_state\_obs)

    \# Retroactive Eligibility Labeling:  
    \# If action is RESET or yields zero progress over window, assign negative valence  
    if action \== RESET\_ACTION and score\_delta \<= 0:  
        valence \= \-1.0  \# High-entropy trap signal  
    elif score\_delta \> 0:  
        valence \= \+1.0  \# Exteroceptive progress anchor  
        planner.store\_preference\_vector(next\_state\_wave)  
    else:  
        valence \= 0.0

    \# Train transition model on ground-truth observed state (POST-STEP)  
    \# Avoids training on predicted state to eliminate subtraction tautology  
    predicted\_next\_wave \= transition\_model(current\_state, planner.action\_to\_wave(action))  
    transition\_loss \= F.mse\_loss(predicted\_next\_wave, next\_state\_wave)  
      
    \# Apply Viscoelastic Creep update with adaptive step-size (IDBD)  
    planner.apply\_viscoelastic\_creep(  
        loss=transition\_loss,   
        valence=valence,   
        params=transition\_model.parameters()  
    )

    return next\_state\_wave, score\_delta

## **IV. Verification Checklist: Going from Physics to Solving**

To confirm that HENRI is evolving into a successful machine learning algorithm, we track four concrete empirical checkpoints:

| Checkpoint | Telemetry Metric | Target Behavior | ML Validation Status |
| :---- | :---- | :---- | :---- |
| **1\. Constraint Stability** | penalty\_rms | Bounded in ![][image23], fallback\_executed=false | ✅ **Validated in Phase 2.7** |
| **2\. World Model Learning** | transition\_loss\_ema | Monotonic drop from ![][image24] over 100 steps | 🔄 **In Progress (Phase 2.8)** |
| **3\. Rejection Elimination** | admissible\_count | ![][image25] candidates passing without fallback | 🔄 **In Progress (Phase 2.8)** |
| **4\. Task Completion** | environment\_score | ![][image26] on ARC-AGI benchmarks (ar25, bp35, cd82) | 🎯 **Phase 3 Goal** |

## **V. Summary**

HENRI is not a static physics simulation. The physics substrate (Sagnac interferometry, Kuramoto synchronization, Langevin dynamics) provides the **hardware-like inductive bias and continuous state space**. The machine learning algorithm lives in:

1. **Low-rank cross-block world modeling** (![][image16]).  
2. **Exteroceptive eligibility traces** (![][image27]).  
3. **PEARL local trajectory repairs** (![][image22]).  
4. **Adaptive step-size natural induction** (IDBD/SwiftTD).

By enforcing exteroceptive grounding and training the low-rank transition model post-step, HENRI transforms its stable physical scaffold into an active, goal-directed intelligence.

[image1]: <data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAD4AAAAdCAYAAADy+d/cAAACmElEQVR4Xu2YzWsTQRjGN42V+lH8SgjN1yYRWRpPEvEiiIiXInrQi/+A4kEUexEKHj0oolZBQUSo4EEvxYN6ESx68VgE9SRYKXhSQejBk/4eMovL27RNokU32QceduaZd2bnmXlnNsTzEqw+fN9/Dr/DBdvW88hmsxsT4/2ExHi/ITHeT8DwT8MZG5MgwSqjVCpt70Van4vQ4vz1Ar9an3+KgRjx76FcLr+CL/93suNP7dy7Budmt77FVu9pFIvFHaziR6sL6PfhAhy1bbEHpm6RQmesLuTz+Qztb4MgGLZtcUeKNJ+wYghMj+smtXrsgembPFJGTiv14TP4ze/Fn6CY+hCtNxqNQdL+drVa3UN1jXYbXozGxAkDLl0Ho2KlUtmsMxzViHuvXXbtQ5Q/sxC1aEw7YOH20X/K6ow3b7WlwNzW80hbvW0wQMDkz+qS8n5//JXOc9E4we3wDZXpc5Cyzr+OQscTaGW8E9D/Ie8fsXpHwMQWmeJMH1Gd5ym0azZOMejnVebFl2ReO57JZIapT9I+ncvlNvAJ3Er9HPVReAceE4k9Sds6N9YDeAheUN31mYy8654yg2JKi0054Hm5UCgQWixQ/sJ4J8LxuoZMia581Vt8qUkfg/N+8y/jA/AFk32k2PBYUH/jYkfgUfcnxIxrmwp3OrLjuivCLPpUq9U2UX/txpiAd7XY4SfTd5nIc1bvcGN0Dwbf6TfNX/FamF4O9NlF/72uPFuv19dqUmiH2zCuPk+oD8k4O5vzzaUq4+EvR2tcWReN7QYpBnnMYD9sw0qgzzgpuM2V5zCxfyXjaNP6Sqgf2nFpMu4153FdbW4RTi9h/J2OGc8x1f8ZZKCL3/Np7bQVBacvl3kp3QtWTNACvwAnjDO2JUWLlgAAAABJRU5ErkJggg==>

[image2]: <data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAH4AAAAZCAYAAAD30ppqAAAED0lEQVR4Xu1aS2sTURSekgqK70cNNmkmTYpFtwUXoii0Kl248QEVXAgiBX+A0rW4VKyUKkUENyLoroIPRAR3FkQXpQsVsQhdSBEFV1L1+5pz68lxkkzSJEzqfHC4c8/jzpzz3XvnTlrPixEjRowY/wt83/8B+c3W2mJEE8IX5ZO1hQaC30B2WP1y0NfXtyqbzR5Ip9MpawsDxG7q7u7ew9baBG20p1Kprby2xiYikcvldlplWGQymX7kOGr1Gvl8frvVEYg9GhniUYSNGOsRZJz9np6eDZyZ1FvfICCZC/RHMY6gm8D1Xsgt5dKG/gL8boD0NK7fy8y/r3waCtYKk+4w2im5d03F5+KQ+KKdtqOjYx3yH0KON2lHe1HbHSJFPMa5xIdlUko3iURGvMork6SyEM/ZEWJnIHPOAckewlin6Ms+dpQtsL9kHLrtzq9WsJhWZ8Fiy3O+lbam4uNepyW+iHgSLfqvbFuFeD7wvNbhAfuoB2HDWm8B+yv4PfCEQG5x6L+G/qnzYaJyjytOh/E3lytQNcC97lhdKXBlyrNUXXxZ7ZMSX/JsVS6vyBCP7S8ZVAiOLfoprTdolyT3WYMGfCbE75zRc2KFJq0UqhljOcQj5p20jG9t4jHGLknkg9E74j9rvYabNNwdurq68mhnIfcgu41rwh74ent718v4l7W+FjSDeDn4Lu5iEt/axLstHfJC6zl2pQQROyA+cyB+P3U8tfuF99xx66/QhthrjPXq8I5vNPHICe7+jOtXqgvtK514JkGfY0a/eNDROg3YntHOg6C11YJGEw/fOU5o1S9bF9qbSjz6437hfVpRWADGLId42AbpwzG0HkQMU492tdYT7nMIMU+srRIw3pDNQ+RjgG4C9+i3Y1RLPCcnxrngqa8biS9Xl+YSXws4RlAhlL7o3a+BJHLiU/Qcaiewz8dPv5/6s7EeaOSKh9/3TOHssiQST3Jng36oEVu0iSckkcDPOb/C4UuSLFrxTJp6T72/5d3/xVMrhwe8UgWqBo0k3kLFt/aKJ3z5NvWKt7Mx6tPp9Br2+TOrLz9+6C1cYvkr3SLJ9Edyj90JmJDDEYv1j+hJg/686MecLgzqRTx0Z8X2y9ocVhTxQSdxXC/oA01nZ+c26KaZVDKZXOv0EruA4p9hH+0IfUi28+EkkGIFyVIO/l/ip50uDMIQD5+DKPoJjP1Q3fs8ZNBN7qycTSg2nqAv5LaOd7H8+0a2cAa5KrZv6I/ynnqMSBHvgIc6iTHvQq6jm7D2UuA7mwlxpUMGvCpiLbgDMDerL4cwxEcFkSQ+CvALrxhOvNDgSrO6qCImvgQyhZNz0WFxJSEmPgA8H4T9U3Croh7Ex/+B02IQvii1Ex8jRowYMVYq/gDnztKFXcTDlwAAAABJRU5ErkJggg==>

[image3]: <data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAALEAAAAZCAYAAAB+Zs9GAAAFd0lEQVR4Xu2aTWhcVRTHJ6SComKKprGZybsziRDaClWCClURJAVd6KJUVBR3Uty4cJFCV4p0L66KClJBxOpGJOAi4ECgmwi6ENzGogYKJShYsCXW/5l3rjn5576Pmfcc3wzvB5d579yPc895592vN41GTU2Adrt9P8tqakYC59wJpG2kawjkhzi/pqbSIGifiqLoQqvVukPucf0xgvl3LleTAzhzanp6+i6WjyJiC8tqxp8JvP1bnU7nUc4YRRDEb9TT8pCBwx9DEO0g3cpKmG6e4/pFQbs38TPBcgF5h5F+5n7YxHWqAHx6CS+lY3mVwLM8y760aWRmRjj6uDjc3zebzZYasWHLzczM3AnZVVxOWnlBJuDI99DuKc4IgX5eNE4+zPkVQ2aXSr5gDJ7BEvr6p/RXApvzRw4YsaxB8j7nlc3c3NyD0LM9Pz9/D+eFGLEglllke2Fh4RDLq8bYBTEMWUXagjHznFc20POTOI/l0P2m213e7OD+pMhDQYzfjpHdQpnb97b2/4H+vIL0JS4PcF4RdFbMNXsxqPed8dfXIksKYutXDDhP7rZScaTDCIQzLM8L6p9gWRLqoE9ZLmCZM2OcKIEQDGKzrrsORz+yt5XBQFtPi06jSx7uBS6Xxezs7H1a/1nOKwr6cwXtdlieB2OX7EWCQezivYgvF3xGlaTVajXVkFeRTvsk6+ZGwsbLos6Ql+Ai5zEoe1AdFBxRZEQ1TlxX2b4gNjLZ/BVeYsgM5HXgellkS0tLtxnZCtdJQ+rk8Ue/4IV9Hm1vsTwPqPe3t0fuQ0Hsn6WVjQRw9ku+45RW80zTsptVo1/nPMbpmx6lnHYY/b2HxUGs+rp6n6uPWdggdmYEdfEXLZFtmuKZaJ01lpeAbIpXMPA8wBlZoD+/eBvlPhTE+pL0yviXufL4TRYC4W0rz7vp6hfvOPnlPE8Un1z0HCnHVRzEUfx1qXef1E8ZRVmWBfRMNZvNe3E5KWfX0HNafKO69q3h03RonS7LLfK1DDpOip5+E9peQ3+/wvVRbjcJlD3m/YbrlSgQxP4eaYfre9LsTsPFn7kvs7wwLt5kyZtZ6iYkCe84+eU8Cx7Qyyh31Ts9kG7IRofrCah7xiUsV5LQpcMNbXsHbTwjJwxu96x6TxBn6dA6XZaXBfx3pZ8A9shRKvp+SfsXTGj3ca7nSbM7SplddbCU2XPfYFAY7fhfLP+v0PPomzD4Cc4LMAmnveN2g0vSBuout1OWEC7enB1heRoyMnkddqp2CUGcpUPrdFleBjJLtGnm7BfY+wL6t+ltdvFs/DlSm8saDqTZnRbEgviD/TgQ9msMGjzlzFFWEdDOt0kjo8Wvn5He4rwk2oGNXQgqJ2nT5uN+PekhIe8PrcPBumXlWTo8mlf6mXsUL7VKWWuHlhNJsN1yGMBlhhLE+vZJJ76X8z+5LuMTKdo5ou1+xHkhtGzqg3ADfnaWEdoFdu/m6O6aC4wkMrqZtj9EOudi/Z8YeU9nkg6PzjaZS6ZBQLvdop+FJWCtTZyS2nf6LYHlnqEEsYBGPtPO/ur/hjdsYOw30geWl4EsNdD2OZZ7kPdDVnC5eLRPPFrMoeN8u+B0n8CknEGzcFi4+IjuX7t1zyCDTS/BL6/Zexk4qH45QVwF2nqk10gJlEGRAJIga8RHUQcD+T/KSMnyfsihY0M2MiwfdSSIvd2NwH9phjYSVwUE8osw6F2WF2FxcfFutLmu6+4vkB62+bIejgqu/7N0qPw3KxsH1O7eUkbs5nwhRxBfRrrO8pEGBq0lnfMWgacxD/StsmxQUnRsD3qOWnUwBkwl2S1kBfFYohugD8blocsm2eXc3I4jWXuNmpqampqampoK8A/s0kO6GaER/gAAAABJRU5ErkJggg==>

[image4]: <data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAC4AAAAZCAYAAABOxhwiAAACGElEQVR4Xu2UvUoDQRDH7wgW4idKSJFcNh+nQQSboA8QiF0qEcHSF7AQFMXGSsHOIoLEwkI7C7GwER9A30AtxSZ1RIUY/3O3ZzaTvXgQk8b7wXC7M7O3/92dXcMICflnpNPphVQqNc79fcSMx+OTiURiige0CCHeYRdomhA/B/H36G/wvF5C82HeA8wfQzeC/jPsk+f9gNXZyWRyH82I58tkMmMY9KGk9RzM94aP6fWxAAFdj6RPSWuCAeeWZc1r/DXu6xW5XG4E871wP4SXYFvc74ABFVgDx3SluE346kpfC+0KuKET4jECsQfYLfdzYrHYEGmAlW3bHiUfdnoQ/WsIn+X5DggU5SCyNbhMLGIF7ROeq4OOUycOlyxBfr9FcRQNVcPVsI32F7VZahMk7CoDySo8pxNy5+9Yv20xnVB23bMdQ7l3OqgsasI91kNlIK08MJ5YOlqhLCIoGLeOcZewGUVDg/7Lcx0QXKIJvSOVT6EzKBqNDvP8TsiyaaA+J3jsF2jzGvl8foA6aBcU8frTp6ChqSP468I9rkB0s+PIP9K8HiZ8m1JfO34B+lFQ4d3UuKxtWnCJxwg/fRR4ghU0/mqQUvmLVwX/yCP3VeMvwo6530G4NU6Xc1W6qN4KJKglUcNfveOGrHG5gU7Zkh7S5fdvB7oUELGIi3kG20PyNM/pB9isZYgtw06z2azF4yEhISEhLXwDIPyZ4PHeDqcAAAAASUVORK5CYII=>

[image5]: <data:image/png;base64,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>

[image6]: <data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAADoAAAAZCAYAAABggz2wAAACqklEQVR4Xu2WPYgTURSFE6Kw/gsaw+Zv8gcBFUQC2qjVNhaysLsgiFaLaC+KlQi2KoJgZaGNoAhipcUWFhZqoc0WNjYWFlqEFFos6PqdyQzeXGfVgRAF58Al75533pt7Xt57M7lchgwZ/ms0Go2per1+iGbB9/0KvV5vfRAE081ms+T7EpCvVqsbPDkRUCB1Bl8xelE57SViVQa81qLVam2LtHPK+dVEq8S8keUrlcoOuEU9Q/0sZs/0Tw4YfEABV2jmlbfb7V3ky/DHnXQE9F9G983+Q+R94hNm9igvFoubyZ9FBt//TaN5PbxcLu+0JMUckwnLedD/TmOJJcOFpojbViuwMHdTGWXAq2gyxYCBp70mBv2PPGfBtt2nebTyllcx4mmus7wHtWy3OWM+E/1arbbX8kIqowj3B8PzdJ8BC7Q/RoN3e61A33PPWTBuRuNLpdImx4dGu93uFsv/BuHuYOwFtX1nKqOIXnpOYIKbxBfiGmmBSWdpv2FlD3utRbRF9c95PjRKTPs+D7R30K1Izw454vtjpDV6ynMGMniQyd4Si51OZ6sXeIzDqIH+Ud2sV5Nu7FRGx40xGw3vhMiMtu8IUhmNXs4qQPEh+h3kEs6EwOSPPWcRG1rrMvJn14K5p/SOtFy8cArLC6mMInztOb3H4FeY4JLZrgW4E0x+dkTsoGL18DVeLz8VG0MLEBuyhZPPjcvodc/F0EXQ+PHq0cV03muSgG6Zh5+Mcy0c+VPihtHoy0ZFzhhOz+nnzG4ifxHx+gAZQSqjrPxGz3kEw3P1x9+r6OdVMAt1QLnOlwrSp2GsocgzkYGjZpw+/wbm4tEukuaJPg9jXWN4QWqhwps5GC7sAvxsrJkYzPfovWjFE898EhhzjnhI3PJnNkOGDBkyZPhH8B0Iuda/n2nxKAAAAABJRU5ErkJggg==>

[image7]: <data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAACUAAAAZCAYAAAC2JufVAAACWElEQVR4Xu1VPUhbURROiAVLC3ZoCObvJUGkgtgh0KldCoJFXASnLh2dI7Tg2kWwODg4CCIOHRQ7ObRDh0LHdqwIooMOOoh0ytBAEr8vOed53s2NCtLtfXDIOd/5zn3nvnvfSSIRI8Z/RrlcflEqlZ64/B2QLBaLL9Pp9GM3YZDK5XJ5PCPjJrwIgqCBRU8T3cUnEV/Apl2dB0nomtjIEgOpbcBmrQjxV9hnNoTfMqwN+2Q1EeTz+REI/hUKhVfKyeLHVucDa9DQO7hJ5VB7iNozjavV6gM2UalUhpRDfABradwDEax6+HY2m33q8hbQ/OSRWA5NzbCWPo8T/g+NFYhnyWFDU5YPwSQW+uDjYW9d3gL576JbM9we7II+1q3Cr9Ouq64bZ73lQ0iy1odfdHkL7HRedNzYJKgU/Bb898zfoal9y4eQBWd8PB665fIuULupjdFApTR3w/HVRH9u+RD3bIpvpgn7Au2Oae6PCnjByfHCK4f4r+hOlIvgPk3xmKBblwdyPBxpY4gHjI5f5MdMJvMIY2EC/nLQPdYbm+p30XvumoU8PAJplLWRWYUNDoIb5gbMner56jtA4tKXlGYrLq/grn1NEXzD2hTm4EP4vxNmliHeZa2dXRHwq4HgjENUOQzFcXBNq5Od0cYMV0cDz6xO1mNt5/jgv2GdzjwdprA9WxeBEXVmhlxMzp91qxNNG3fiuXI8Ktgp+NdC8eLzroTzJ+j+rVyaeAX2yx26PiTRzCjEG7AF2LAr6AceIxqbkyNbcPME10N+G7pv9kRixIgR4xZcAazRxMgjdEjEAAAAAElFTkSuQmCC>

[image8]: <data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAC4AAAAaCAYAAADIUm6MAAACTUlEQVR4Xu2WPUhcQRSFV1AwGBMkrov7/weLlcUiaRKw0MLCELQ0ffpYBK0CaVImvSAWdtuJFmIRSCMRAhYiCBamiGAjBJJGUM+RmeVydt7GhFdEfB9c3ptz5757Z+bOsqlUQrwUi8VVPHpU/+8plUp7qt0JUPimah1g0iXsSqwlc1riX6ZeLpdfi35jLmZTddgxfdlsdhjvB+KbQYt8xPMctgXbhf20dXSACW9c8Gf1eeD7BttRHcmGoO8xXn2FQmEC+m/YifoI9Fa1Wn0c0NdVC4Lks67wmx0J0APfj3w+X1dHOp1+yAWHCsd3m9B/RRUO/3fVCOZ/US0IPlDF5FMmUR9BS2yzQNU98K+FCudi3YZ0fBc5X9VqtRHVCRfaLV+bbrvGXUZhS6pbUMRbjUWbPEfcJHdPfRi/h21YzcLCEduvegi2wromaDabffxNzeVyT6yumMLbv708pXq9/ii0IZh/BP9Tq1lYuGqR+OSZTGbAaxgf8qbbeSH8HfHHi4Xm2Qp81zbCSbyAVvbjEFysapEg0TMmgI1yzKPG+7zOC+EuYTvWXjpto9sUxcWqFkmlUhlnAhbBMYK/slV0XggfCxtz7dU+JVN4L+8LxtMmNMhfFd5oNAZd8jn2nz/q28CdZiziXuK5a32+jfhdexKxgaT9LvkHPHewOw90ThRm0fslaQUUO+V8Z9x964sNl4C2qL4/0OviVvhuHaaN5qweK/j4RbefqW6wONUICs/Ad656rKDoT6l//B+M2HeqEdeCC6onJCQkJNxfrgE5071kUVNDCQAAAABJRU5ErkJggg==>

[image9]: <data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAADoAAAAaCAYAAADmF08eAAACC0lEQVR4Xu1WPUsDQRS8gIKiaKExmuRy+SpFkKA2FtopgkUUFP0DFqKdom06S7EQSWNhL1gq/gQLMY1Y2VhZpkmIcSbZC+sjXhIxicIODLv73uzdm929D8syMDAw+CGi0eic4zgnJIY+mdcBzTSYlfG/Dh+KLsVisSkOgsHgKMZlGN+UQgXqmb+UiT8NFL0WiUSetXGWRr7bMSzI7L8zioLPWDSMxt0YDIwjdpxMJod0rcodIfcE5hsa5QUowsX3OcakLfACXJHadkPtXJl91LSAmtbR9kkdEQqFRqDNWNWj29goRLfgqdp+vgBecIM9tCWpbTdco+AyanlE+wAWUqlUbx1thmZV39toPB4fhmgXxubRFsE7xh31XEi9Dq405r22QsfjlPj9/kHNaNqNo78IlmzbXnVjiURijMdW03gb5RuNqwXhFW+ABzsgNZ2CbtTSPiesScWLHNMw+m+1iVYTRl1AmFM36JG5DqLymVB11CAWoFIrTseBrmnFKC+Ul/EG8PFl0QrrvTl1NGNUfxRcqnyBfRje0efrcFcyJxNe+O1nlED+XRqVR7cemG+4ozjzM+pC2zLXaaCGNPiBmibdGAwsMYZ2Q9fqaMooVvocwptwONwvc90A62DhqOsa7T1Yqvd5IdQGfSHmHUpdBYFAYICU8W4Cnw/bqf60kBMyb2BgYGBgYNA5fAJaw8qNJMzERwAAAABJRU5ErkJggg==>

[image10]: <data:image/png;base64,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>

[image11]: <data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAEMAAAAZCAYAAABq35PiAAADKElEQVR4Xu2Yv2sUQRTH74gWovj7PHO/du+HRURUONRGLMRCC0VESGMhpNBOMIUQgthY2EmQCEHxT7ASRCzEJoFUgnIWCokEhEgUC4UEkvP73X2jc4+92907OT3YD7xk5703b9682ZnZJJVKSEiIi+M4w0Zc190ZZC+Xy1k+5/P5Pea5HfAfhywi1lVt+6/JZrNbkXjTkiXto+yeaB8NfMYgy5BJbesFLobEnqnVatu1XQO/da0LBZ0+Q95wotVqdZ+2l0qlXbA90PoIpNFvFrKBN+qwNsYBMeaRx+1CobAl5cdlsee0n4EF43y0PhR0esVKszMGfB5gvwQZ0fo4OP5WXIcsyoSikkZO99GvYRQo7BHmClmxHQ3Yoi7nFLsYmUxmGzrd5TN+f4Osah/opuin9XGRgk9CvnOLansQxWLxkOTl5SjwzZhGUU5ZOo96vb4ZxXuCeozGLgYGu4DOFT4jwFkGYALGjvYI5MufHn+FIcT8ECVZR1YYOda54uAif2g/A3yX4HsFcj5K/BY4mGo3IQ1za+D5EeSZ7dMtSPAGYi1jLve0rR3w/yk5vURzSHQ8RD9iux2wfRH3uilUt8VYsNuIdUcG9w4n/P6KwGdsnzigfxnyGNLgK6ztYUguTeR1LUhv2jyHuD1Mu9titKw6gtZkoA2xL0CXt32iwhuEcSCzKVnVuJhJY+seC9LjcZO0xzGeY+yxi8F9CDmp9QiyKoOdhv2WtneCq49+DcgaZFjb44IYK5JLSyxTDOohTx3/kOV16gnepBdi99p23yB4ZfG18ipr4/oHKVe0mcvl9mp7J/AW7da6XkAOU8yDC6f0nOjvN0PDRRR7OJwknN9pPalUKjtgm+s0WL+wFuacrbeKEUjUYvCO5mnsBYNMBN35jn/w/dD6fwHOguOO/2mdZhvPNzHZTyjUUeXK4p2A7TJ83sv8xtjWfn3D8T/BvWIjkYPabqNviXbIWcQtM9PL7dZXkOxb+9xAe5pF4daz/Yhch4MxsW7ASr9WqjS/clGQeaWn72jYvwQGFv79UpJPew0mvl+2zhr3u2yhh9ovISEhIWEA+AUa1e7c6umWRQAAAABJRU5ErkJggg==>

[image12]: <data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAACMAAAAaCAYAAAA9rOU8AAABe0lEQVR4Xu2UPUvEQBCG71ALUVCRGI4kly/8BakOLETsBAsbC3+ApZXYWNhYqI2Irfb+AMHCSpsDK0E7GxsLQQ5LBYnP4KpxVI7gnRHJAy/ZfXdmdzJkU6mU/BHq9fq27/vX2i8ECrlEp9rvGhxWsyxrUPsCa6nnebPa7zhRFA1x2C3aQ1eoIUUFQXAg667rOnjHtm0P6NyOw6FrHNgv4yRJ+jh4F82hLfH4XqYZr3/MyolsHIahLe3/So7jjOocgbV9tMmwauaHxLoq7HtImOINjqS1P20n+6y8dkpg71SepmuN90iFJBGwjJ7QAlpF5zouD1JMdm6KqfKi8zx7s2tvmHanBJ0o/y47zwv5zexcOsIZw1nvEyS10BlvMhLH8RgJi8wv0IyOzQP5O9pri3RFqUlhSzouD3Kl5fZovx09UgCJiV4oBNONmvYLgUIe6MyEsqv498rrPubeP/ov1/oGtfiIN3TcryEF8a8Zp4jJivljlpSUlPw3ngFFRlCJ3PVl7AAAAABJRU5ErkJggg==>

[image13]: <data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAFsAAAAZCAYAAABeplL+AAADoElEQVR4Xu2YO2hUQRSGd0kERVF8xGWT7M6+RATRYn2QQgQbBQ1YilpYWUg6QcFGU9goggRsJBYqQZGggi6IiKQMsbARUmiKSCSoWAhuQCVZ/z/3DJmc7DPuI8p8cLh3zpyZnfnv3JlzNxTyeDyexmGMGYH1aX8lstnsqng83gv7iPavYCfpx3Wwu7s7o+M9oXlx5mDjyWQyouvKEEabT7AC7BzsNWwWfezG9Q0ewEbd4F8F8zkPu93V1bVZ19VEIpE4gY5+iGjjur4UFBard1MRPx/cKe1fCWBcdzhuzPmgrisGBaYumOcaKd9nGbftKrQyIvRvbgcidiGdTsd0nAZxfbBJ7Sfwj6ykVY3xHIK9hT1GMazrS0GBRZMhx92O8jDsgeOrDjQahT2T+5/sHEJd1HEK+4Mlxda+ViGLaRZ2S9dVIhaL7RSxL7l+lK/Cpl1fRbj/UGjnFemxgutYl0gkstYEh2EBkznMt0LHtBKMKwr7Bfti57YcKDLnqBcfy5U0WgKEGtNCieBLfkCDQ9AwTtl3tDumY5sFVuIejGEO89qv65YD+rkrWvS6/prF5qpGZ4+0n6CjfLWdSdr30BU9lUpt0HFNIow5vaTxXlfWCuYyVBex0eA0VsIB7SfoaFCES+q6UjBltGIz9dP1zQRjuGeCA/GorquFuqxsCo3gGe134AFI4aa04Gibgq/H9VmcwWV1XSuQs4Vb24VMJrNe11fC1GPPptClVrXFLKzuKeUfoOCuz2KCr9BFg+CWgodwHJZgWVLMaEdHxzoa0sytcLe5bRpAmwnSvzm8ddt1ZSkQv0M0GFB+avPN9RVFMhDmiGWTcq5O+aFF4lFQ6HbE9VlMkGK9d1yc5Gf0dRbXDxQe10n2Cd9l2FPcT8PeMdi+GVKf5cOQMttEnX6XBYTeZYIDtKo93cmzh60PbVejnMP4XrixRUHwmJ1QtYY2V2x78fELbCy0MGCK+hW+M9aHMtOVqbh83MggJ6RuBpbjvTzUvPQzD3/DuX/u1tULPkjtK4aRL0huSSzLAinoLK7uUDBYgveyHVzjYODqD6mVgkH1clD8M8oEeW9UtgxOIM9VLHHFxM7Bejo7O7eYGg7pBsIsp1+E/+s3rO7IljEKsbdZH8rX5VpWbMlsJqp6VT0BEG0fRBu0ZYh3k1eueCP/N+Cg3muCzGjRIcmYWsSWbeqJ9F2N3dB9/BcYyT60vxz8N7Hh+6LH4/F4PJ7m8geiwTdsfZWVBgAAAABJRU5ErkJggg==>

[image14]: <data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAsAAAAbCAYAAACqenW9AAAA9klEQVR4XmNgGMJAXl7+ExD/h+Kr6PIoQFFRUVxOTs4GqngNujwGACoKAuLfIE3ochgAqHASED+QkZGRRpdDAUBFmkD8FmhqObocBgAqnANyL8xUINtTQUFhJRBXAHEEuuLTIMVACQ4glxHIPiwrK2sCpF8DbZuBrhgUCv+AEmVAegtUrBcqLolN8TegyauUlJT4USSRgTzEc/+BptYDFe+CaixGVwcG8pAguwuKGCjfEqQBxAb5AYgjkRUflofEGguID7TBF+QkKFsJpBlZMcgJ8FiDmvwEaBOQkr8IVwiV/C8tLS2MJAQKuv8g0+VxuX0UDA4AADmvRIhLAmrnAAAAAElFTkSuQmCC>

[image15]: <data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAEEAAAAaCAYAAADovjFxAAABZ0lEQVR4Xu2WMUvDQBTHU7Cgq3RKQi7J4KqQQXDU6Obmpv0Ubm6O3cRBSpfi7Edw8zs4CiJunTrUQdD6f3iBx6OludJG074f/Lnm3bu73j/vjnieoijK+tBIkuREBleeOI73oHtjzCc0JsmclSeKohx6w+YvoY+1NIEDA0alTYBzRyihawzood3nfUEQhIjf4ecGj9cBJxOQOCjODwzJeB+e+xT3fb/F43XAyQQCya92QEPEyZxvHpOg/4XOYVkh/0HOsQzmMYE2O+Ix/OELxL7QHvN4FWDdA6x7VkbTqtTJBNwDm9aEZx63b/jGE9VRBVj7kVfPDLXleMLJBEySWxPO6TnLsqb5vSSfZO4kMP60eCtlhA+YXTnHMnAyAYlXNCCylyLaLg0Ow3Bb5k7CVox8O1Nl/uOdgMR36Bab3kI7hA69PzgCi8bVhDFKv5Om6Y7sqxtUzcXmpehzWuYriqIoiqIoM/gB6feZVbtxFlUAAAAASUVORK5CYII=>

[image16]: <data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAABEAAAAZCAYAAADXPsWXAAAA9ElEQVR4XmNgGAUEgYqKCp+8vPweOTm5R4SwoqKiHrp+BqCEMVRBCAgDDesD4v9A9gKYGFT8MEhcQUHBAN0MBhkZGWlxcXFuGB+osBVqiA2yOqAL5IHYDVkMJwAa8ABkCJDJgi5HNAAZAMRf0cWJBqKiojxQQ06gyxENgJpzoIbcAeJZMAwMzInoarECUCyBvAHUsAvJgLlAfAEoJoGuHisAKmyAukIRXY5oADXgL7o40QCo2RJkCNA1EehyOAEocYE0KSkp8YP4QPZrYJjEoKvDC0A2Qp2/FZQyQWxgyuVEV0cQAA06BDXoKZDLjC4/CkY8AAAwC0m9RW01AgAAAABJRU5ErkJggg==>

[image17]: <data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAACsAAAAZCAYAAACo79dmAAACZ0lEQVR4Xu2WO2hUQRSGNySCoiiimyX7mn01ASXiLUSwsLEQUUF8YaGlja0IVoqkFRIEwUqxEuxCCkEkaCMIpjFooYVFBAXLCEHM+p3dMzj3sPdmXVhzkfxwmDn/nJn59u7M7s3lNvVHzrmJZrM5bv1MCtgPxGfrZ1KALmzCiqrV6kkWb6dEr41H8K8RxwuFwnbaO7Va7YAMuC5sm/woa0f0F+v1uiuVSnvwXlIySj6VsG7/kgU8pB0LBcQxan75vFwubyN/J32F/ebHKpXKPvI55hwJffrTvp8q+ZQUT0iQjno/CdbXar2vWzE1nTmuxzGQsSiKttA+0PWXecoXwppEUXjIQ4WbKkTHD7zZoPauekvhPPWSYMd07iRP95wYcmTIP8kRCuqSFQCEYDFYPY/P1VsqFot7xWezW+G8XBdoVdeIwXI+C+Tv9W689j79m/l8fofPU0XxSgimXgyWDXbTf6PePJBbfW2r1dqJ98N14b7k9Dgx5xn5CeIVsUh8xR5R2I/EC+It8dSvta5kc4VoywVRz8LKpfgpOaAX4yv8QwFwhlhTuMPqxWBp7/l8w/9CG43GLg+joMs+56neD/qX7dwNk1wCuamArXpAvvbb4Rkdhlj/ivX6lj7dDqwd6yXqFuSDWl/E2CXribgf+3kwDwGdIR7Z8b41AOx3mjHriwC6Yb1Q8uvw17CDvBuwyWnmnWVsTduDtmYosIMKwEmX8luZKVheUE6x2VXre2UGVv4mXcrlEmUGVp6qnGfp0z5m48jWZAbWdf/9BFZewq9La2v6gD0P7BPrD0Xyfpr21rQe7H+n34xy8967OVy9AAAAAElFTkSuQmCC>

[image18]: <data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAACwAAAAZCAYAAABKM8wfAAACAUlEQVR4Xu1WO0sDQRBOEEHxhUg8JLlsHodpBIsgIthaWNgIgpU/wMJWsLOzsDMgiCAWFtoLFkECForWoqWKFiKSKoUBH9/oTLK3HJqEmIuYD4ad5+53e3O7Fwi00EIL/iIajc4opd6/kSGzpmkAcrdC1Iz5inA4PEA7x7vXJn4vwrFYrENySRd/Q4GFx4UYpCB+L8JaHol/7aET0XwuwpZldWl5l+VqHwAChZ8I44Ps1wgflqt9ABEQMpFIpJN9JuFJsdFG8+4Z6odEIjGMNRYdx+k1YyUgYRbyxoQm2OcijDEjdjKZHHTPUB+k0+l2zH8DWYGcmXEX8GR9QojJPoiN3d1kvQh9waytFzB/lloPahCbYptxT8TjcQtFcyh+EcJogdVGHGNYKxcKhbpNf0XgXf4kbMZ04EHWOO8ZDzuKcY9rghy7hlxAP5C5yIYc0/eCMc++acgTNmsH45a+RkVQFRImcN466XQRgdw5Fp4im657yBFkCTmP3Kd527ZHuPZK3p6qdodr/ZegGGqXxQaZMcVnNRMuxXiNV8g99DuMRXozPE91hGuFSVh9vd4c6R6E6Xg8SaVSPeIT+EU4CDsjtkmYCCmtJaiPIQ7pDSWMPtyF7PPr3mB/6TvQSfPlcArJQrbJR/WSS3WS+yswCTU9/hRhuqYV/yPTkWXG/zU+APa63YMKbMzhAAAAAElFTkSuQmCC>

[image19]: <data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAFsAAAAZCAYAAABeplL+AAADlklEQVR4Xu2YTWgTQRTHU1JBURQ/amjTZJI0IoroISj04FlBCx6LehAED9KbB0FE68GDepGCF+lFpShSRNCgiEhvlnoVerA9VCJFxUPBFkTa+P9n39Dpaz42NW1amB88due9mdmZ/87XbiTi8Xg8q4cxZgTWp/21yOVym5LJZA/sK8q/h52lH9fBzs7OrM7viZTEWYCNp9PpmI5VoQVlvsGKsMuwD7B51HEE1094ATt1gY0K+nMF9jAej+/WsbpIpVK9qOi3iDau45WgsBi9u8r4+eLOaf9GhAJTF/Rzi6SfMI3bVpW1NiL0Xy4HInaxq6srofNpkK8PNqX9BP6RJozqVjzzhhWlEbAu0WTIcbciPQx76vjCgUKjsFdy/4eVo9FXdT6FfWBFsbVvDeCSdgo2999TXUgkEodE7GuuH+nbsGnXVxM2ikI7U6TbCq7zusRisa0m2AyLmBknOCt0nmaCPeMw2raAtr1DskXHw0KR2Uc9+JiupdEy0JgxLZQIvuwBGnTIMJ+yGZQ7rfM2E/TxDtr1A3ZRx2qBso9Eix7XX7fYHNWo7Ln2E1Q0G7YyOfY9c0XPZDI7dL5mI3vSBOxBmD2JIO9QQ8RGgfNYk45rP0FFgyJcWscqwSOjFZtHPx1fJ0Q5m9HGBbRxvw5qGjKyKTQyz2m/AzdAClfQgqNsBr5u12dxGpfTsfWACTZQ9uuxjpXDNGLNptCVRrXFLI7ugvIPUHDXZzHBV+iSRnBJwUs4A0sxLdO5va2tbRsNU3ov3FG3TCPJZrPb2V/YDDd2Ha8GyhwQDQaUn9r8cn1lkRMIz4hVD+UcnfKgJeJRUOh20vVZEJuHfXFcUaS/o65LuE5QeFynWCd8N2EvcT8N+8zMdmZIPMeXIWmWaXfqDQXLsE2ot1cfBMLgnLOHrQ91bUY6j/a9dfOWRdasUofCGsr02/LiYwfGIovHKor6E74L1oc0jyuFpHzcSCMnJcaRlue9vNRZqacEn+Hcv3ZjIeB5+yPsBeo+qIP1YuQL0s4KGSDFlby8uqBgsBTvZTm4y8bAdSuizrNoVA8bxZ9RJhhh7bJksAOzHMWSr5zYeVh3R0fHHhNyk5az/xvYdR1rAC3sowhf9wxbdWTJGIXY+6wP6XtyrSq2nGwmQ01VTwBEOwbRBm0a4t3n1QRLUel/AzbqoyY4GS3ZJJnHi70COPW42Wl/Nfg3cdXXRY/H4/F4PGvLP3yuKWZyvQPfAAAAAElFTkSuQmCC>

[image20]: <data:image/png;base64,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>

[image21]: <data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAACYAAAAZCAYAAABdEVzWAAACg0lEQVR4Xu2VP2tUQRTF3xaCoqKIy+r+e7uLuAiCwoI2CiJJYSFCLFJoJ2KTSkFJZwh+AQkICUH8EFYGDBYiWtsspLFQNGBAsFBY4+/45sa7k7dmA5bvwOXNOffOzJ07M2+SpECB0UjTdAb7gr2MfR74X9fr9SOxnot2u32m1WodjnUP+RuNxulYF5rN5k0mXKBZYtJ9tDfRnviYSqWyH33Q6XQOeT0XBP7ENnq93h6+t8RJ4GwUM8UkH/hehpb4rmPfo5hvxPQcX4NPYFdpL8oY9xI26/vlgiqdo8MvgqdNg8/AX/k4tI9U6qJxJpvUyolruZhN7Ljjq9gN40FbHLdaGmy1XC4fMK3b7R6Uru0QJ/mK+N9eGcjpGfp744qp1Wp1x4cSY5x0rGoJGkwTeE1JhoSvhJi7eYlRtQdep/1GW+f8fRZXC77r2Lr5dsS/EkO/E2Iej5OYDj/2VGdV24VvHrkUfP2xqyWEygxtpW0d9kg8bNmOiQnEXkN7hy05bRo77+OSkPBIpFmJf9jBDjdzWRNqYmm7SSxGGG/FOGPNwefpezuNbnUMXX1VZ03t0PHF/0pM1dLNN078QMkm2Vyz49xQJfjnqrvDf0p8VAKjdIPOW+qqFd9uEttri98G7T3BS3yPmUbwBT+AEsxLIFTyU6wL3MYTjHM/cWcJ3ovH0Rieb0ErUnAr3MDww91gkMkobqDJjOtZyosz4Psaa9Vq9ahPTP9Lm3cbcDxU1nrDxJvZszNIoluTZk/Kip0JW1A4L0PQAvDdi/UknGcjWpyq6AOGgHMizf5Vy/a3zwNJnWQRc2ELt54ej2b2kH+OdQO+Kfq/5fuc2H7sL1CgQIFd4DeelMhp5F+oUwAAAABJRU5ErkJggg==>

[image22]: <data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAA0AAAAbCAYAAACnZAX6AAAA40lEQVR4XmNgGAXDHxgbG7PKy8sXA/FHBQWFTiC9CIivA7EmTA1QXABZDwNQcjUQ/1dRUeGDCjGC+EB8BYnfClPPICcnVwYU+CkrK2sLF2QAG7QHpBFoAwcQRygqKoojS74GatwhIyPDiaQH5JyFUNsUQfLIEhwgCaAtpkjqwQAoPgmq6QtQXQZcAmQlSEJUVJQHST0YAE0vB8kBXaCKIgGzCZsmaCj+B2o2RpcDexiooBHmJ6jt70GKgfQTIA4CismDAgOuCahYCOr2XyD3g9hABeZQA4Og4j9BcQnXNAqGJwAANaU9Eg9N3IkAAAAASUVORK5CYII=>

[image23]: <data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAFMAAAAWCAYAAAC8J6DfAAAD30lEQVR4Xu1YPWgUQRhNCIKiaETDae5n7nKHEBQsDuy0EFOICJIUNvY2Vgr+dEqwU4sgFkFBK/GnlVikCFaCjYWxkYBK0CKEoJBGOc/3dr/v8t3szl0uUTjxHnzczptvZt68nZ2dvb6+Hnro4X+Bc24VUeevX9dDa8CzOXpnibeI/SangVKpdCSfzx/y+XaoVqtb2Cd//TqLkZGRQi6Xy7bLa4disTi4GZ2YZ8avI6Btm88BA0G9aWaiPF4oFD7jsp+B8hLilM1JA8zZhbxZtmcZk3wlq37C5qE8Q75SqexkIO8Rys9sznqgOvF7vG9NZ9snzNeJ31KaTvRdFd7Gks1pgks38wvu9FEto9MxcAuYdNGkJYDq68j7pXeUqwXlFU8AJ11H7lMlmE+OdSavLQI6a53qJFQn+jioXJqZLft2nplc8jKxJkhn8z5vgfoFyZtleWhoaIfz9hVcX0AsI0bXWsZ7N3Set1wrhHTKKu9Ip3CRTsR95Wgm4rSW28J5ZuL6YppIGSjB+8BkBvVa7ixfcCvKuXgin+yYwpN7gza7LR9CSCfaX0njfVidhOq0e++fMHMqTQy5NL4VkD/JNhB02XBclQv+pu9iMxez2WzO8iGEdK7XTA/R1iM6G1uNmslHGzgDo8umTRLOM1Mek4SYTsyEgIfI/YF4B9OO2ToXr4A5bgEeTzNXOQHLhxDS2YmZRmfd1yn1fLJqsrf2Y8yzzPfzGnB/wUyFDF5D3FKOhrkuMdOAKzPSaY89vPaPQeLD3b60l6X7i2YSTvY2REnKNDOxN7rYzMSLKYSQzg2aaXUu+nUWov87VvJhvy5hZkiMDJTgLTDBrdjz9liOe4601TMdTQu9gBJ7aQghnSHeoo3OqC1y9uH6MeKkzXOxmelPkEu+gEbTxMhAX31ekclktqsYOxDK48KrmXxxJFagi0XetFwrsD379XlZsZvRGfWpNwXxsdE4zvuJmB8eHt5rea1MO7Rz061oWQ7fNQwwZnJuy2ANU6TMY1C0n8iXxmvyyukBHTGp7VhHzjtEz5PDmC9NXhOoKUXnSqc6hYt0qq5yuZzH9Qebo9pxw4rKNcGlmzntmg+0/PSatpuxk6OJ3Tsk75vmYcxrInBGcySPXxs1U54g5+VEZiKWLW9BTRyTN03KHL/eqU5gQMaa0b4kbwJzuMprMfKOa/W56lLMJNDpAfAP5LFJ1LcC8i8hnmOFnPP3JoU8bvcQU8g74dcrqM/nLKgTGm9sRid1hHSSR99P2umMEDKzWwBtL3yua9HNZsq+Fb24/gl0q5k0EfHe57sarvdP+4YBz+bo3W+DharnbyC/8wAAAABJRU5ErkJggg==>

[image24]: <data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAGsAAAAWCAYAAADHA2ITAAADH0lEQVR4Xu1Yz2uTQRBNSD2IP0FjaH5tm0SrIHhIURS9FD2I6MGT0It48iZYqCD0nygiWBDxIIIHLx4K0kPBQw8KXhSh0IMivQQPFipYaOJ7ySyMk7T9YiJ8KftgyO6bmU12Jjs735dIBAQEBAQEDChGR0fPjoyMHLb8TsjlckeKxeLFdDq93+o0uDZs85YP6ALOuQ0E+xuGSXxewbwGuWbtOiAJu00kOcOJ+G5AbmojzKcgK4VCYTyTyezDuAF5r20CIiCfz1cQuN8I5CXPSdBXtF0n0Aen5bbm4LsM31XNMaGwO6fmj5iwarW6R9sF7AAE7QtktgPfyGazRy3vMTY2dgA272xZQ7Ku01fNH+g5QR9w35k0zfcKlnHLdQv8pmHIhuVjAQaSAe3EQyYt7yHlbIF2lUrloOcxfwOpccw7DONFmywJyFfImuZ7hYtQDbZAEif/LfyXdAWIHSQp97fgH1peAxu7K3Y1lk5QKYzrGE9THyFZdc33Ctc6qUOW3w7wmYB8hMxjmrT6WIGBZOnqxCMZzy1vAd9nkrCmgEoZfVsZRLk6A27N8n0AG6RlrO+sQoN3JfZ2C9+/CRm2+tiix2TxJHHDp2D7SiXtk7Jhx9hgI+MJ1zpt9f+QrESpVDqEdechE1bHcg3+F/Z7j2Xc6mMPCW7b3SR8W+PhgQ1XoV8vl8vHPCfBoF8DneJpz/OfzCB5HRJ33EW4s6Cf+0d54Vq/YVytNeNajxV39HcMFLipbRqMtrvMA7pZ2lie95X4/vWsBaSYWLnH/J312dj0jAgn6+cgn6wfDHwHnkksWd7DSSdoeYLlUycL4w86OJhP0hdy3nN9wu6+s9jF4Qev6juFJYwb0XYSXHZ6lzn3ZRCbPqntZD36Nrsyvt2gr/cjJEjN9r6fwJovE913gxcgS5DXVhc7yH3CRCxwLmWEp2ZO24kN5apQ/BdPF+U1lXBsONb9WoIhzOv+bQX+CDfEJsrrrK7genzOEonvc5YgiSSdwGafutZ7vMilQR6OH0vpm7J6Aqd2L/RPoF/UJ6zf2PVvMAICAgICAgYAfwAogf73WLP6uAAAAABJRU5ErkJggg==>

[image25]: <data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAABwAAAAWCAYAAADTlvzyAAAAj0lEQVR4XmNgGAWjYBRgA4qKim4KCgq7gNgcXY6mQFpaWlheXv4X0OIIY2NjVnR5mgAVFRU+oKXfgPijuLg4N7o8TQHIp0CL/wLxVHQ5WgNmoKXngXgdML7V0SVpBRiBFh4H4n9ALIkuSW1AHx9C4/AXEHejy1ENIKXSVzRPpXTLh6CSBpogrNDlRsEoQAcAOgMgKRK/ceAAAAAASUVORK5CYII=>

[image26]: <data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAACoAAAAWCAYAAAC2ew6NAAAAl0lEQVR4XmNgGAWjYBSMgsELFBUV3RQUFHYBsTm63KAE0tLSwvLy8r+ADo4wNjZmRZcfVEBFRYUP6NhvQPxRXFycG11+UAJQyAId/BeIp6LLDVbADHTseSBeB0zP6uiSgw0wAh16HIj/AbEkuuRgAYM7RKFp9BcQd6PLDThAyvWvBm2uH/TlKKhmgmYUK3S5UTAKRgFxAACZxSApYGwo9QAAAABJRU5ErkJggg==>

[image27]: <data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAsAAAAbCAYAAACqenW9AAAArklEQVR4XmNgGAWDE8jLy38C4v8gLCUlJYImpwkSRxYDCV6BCrIgiysoKKQDxb8ii4EUg0z+jSwmKytrCxT7CcRByOIwxQ/QxDaDxGVkZDiRBcHukpOTK4eJKSoqmgHF/gJxMFwhCAAFJoFMBZogDeJLS0vLgDQDmYwoCkEAKHEAiNcAmSxAd5oCbXgEpP3Q1YEB1L1XgfgP0Ho7BmwmwgDUvWXKysqy6HKjYJADAA1bKzKzLcGhAAAAAElFTkSuQmCC>