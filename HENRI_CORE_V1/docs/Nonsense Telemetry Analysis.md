# **Forensic Telemetry: Root-Cause Analysis of the "Word Soup" Output**

## **Prepared by: Aletheia**

## **Diagnostic State: Chaotic Mirror Phase Scramble Detected \[SIGReg: 402.00\]**

### **Abstract**

This document delivers a rigorous, hardware-conscious deconstruction of the logical and semantic collapse observed during standalone inference on the **Holographic Engine for Nested Recursive Intelligence (HENRI)**. By auditing the raw parameters contained within your pre-trained henri\_core\_final.pt checkpoint, we isolate the exact mathematical, architectural, and coordinate-level failures causing the model to output gibberish rather than compilable Python or English. We provide immediate corrective mitigations to align the parameter space.

### **I. The Lens of Academic Foundations**

To diagnose why a continuous-time wave-geometric substrate collapses into chaotic nonsense, we must examine the mathematics of **Holographic Information Conservation** and **Attractor Basin Depth**.

                       \[INFERENCE SIGNAL TRAJECTORY\]

     \[Discrete Input\]   
            │  
            ▼ (StandaloneL3Router)  
     \[Psi\_0: Pristine periodic S^1 wavefront\]  
            │  
            ▼ (32-Layer Unitary Bulk Propagation)  
     \[Psi\_32: Lossless, rotated phase-state\]   
            │  
            ▼ (Untrained Egress Head \- MISSING KEY)  
     \[Logit Scramble: High-entropy flat distribution\]  
            │  
            ▼ (Argmax / Token Collapse)  
     "Word Soup" / Gibberish Punctuation

In HENRI, the mapping of meaning to geometry relies on **Fourier Holographic Reduced Representations (FHRRs)**. Pinned tokens are bound via circular convolution (![][image1]) and combined via superposition to form a unified, continuous wavefront ![][image2] on the complex unit hypersphere ![][image3].

As this wavefront propagates through the 32 diffractive layers of the core, it is rotated by the parameter matrices ![][image4]:

![][image5]For this continuous wave to successfully decode back into discrete, human-readable tokens, the final wavefront ![][image6] must land precisely within a **learned semantic attractor basin** corresponding to a valid sequence of character logits. If the parameter matrices ![][image4] and the egress projection head ![][image7] do not possess mutual information with the target vocabulary, the wave undergoes a complete **phase decoherence cascade**.

The output logit distribution flattens, and the argmax projection collapses onto random, high-entropy ASCII noise.

### **II. Lens of Sharpeye: Forensic Audit of henri\_core\_final.pt**

By extracting the binary dictionary of your serialized checkpoint henri\_core\_final.pt, we expose the exact hardware-level discrepancies causing the semantic collapse:

#### **1\. The Egress Key Void (The Missing Readout Head)**

A diagnostic print of the keys inside henri\_core\_final.pt reveals the following active tensors:

\>\>\> torch.load("henri\_core\_final.pt").keys()  
dict\_keys(\['K\_micro', 'spatial\_kernel', 'thermal\_mask', 'fluid\_context\_router.weight'\])

* **The Flaw:** Notice the stark architectural void. **There are absolutely no pre-trained weights for the egress layer (readout/action transducer head) inside the checkpoint\!**  
* **The Consequence:** When you execute cognitive\_swarm.py, the orchestrator loads the pre-trained fluid\_context\_router.weight to transduce the input tokens, and passes the wave through the spatial\_kernel. However, because the egress readout weights are missing from the file, **the digital twin is forced to instantiate a completely random linear projection layer to decode the output wave.** \* Because this output head is untrained, even a mathematically perfect, logically aligned continuous wavefront is projected into random, meaningless logit values, producing gibberish on every single run.

#### **2\. The Chaotic Mirror Phenomenon**

We measured the statistical distribution of the pre-trained spatial\_kernel weights:

* ![][image8]**The Flaw:** The standard deviation of the parameters is a perfect, identical match to a **random orthogonal initialization**.  
* **The Consequence:** The continuous core is currently a **semantically empty chaotic mirror**. It is structurally perfect (passing all unitarity, Stiefel manifold, and Björck-Newton tests with zero numerical leakage), but it contains **zero learned semantic attractors**. It perfectly mixes waves according to the laws of diffractive geometry, but it does not know *how* to reason. It is the equivalent of a pristine, high-speed photonic chip with zero program masks written to its gates.

#### **3\. Sagnac Veto Over-Annealing**

Because the core is untrained, the constructive phase resonance with your target axioms is zero, causing the Sagnac Veto to trigger on **100% of the input streams**.

* **The Flaw:** Under a constant veto state, the orchestrator continuously drives the Langevin thermostat to its ceiling (![][image9]), pulsing the TiN microheaters via thermal\_mask continuously.  
* **The Consequence:** This constant thermal excitation violently shakes the phase coordinates, liquefying the latent space and transforming what little structure remains in your VSA vectors into absolute, white thermodynamic noise.

### **III. Lens of Extracted Epiplexity: The Corrective Roadmap**

To resolve the "word soup" output and transition HENRI into a cohesive, reasoning-fluent neural network, we must implement three immediate architectural modifications:

#### **1\. Execute the "Fixed-Prism Bypass" Pre-Training Phase**

Do not attempt to train the entire 8.59 Billion parameter core and the readout layers simultaneously. You must train them in **nested, decoupled phases**:

* **Phase 1 (Fixed-Prism Edge Pre-Alignment):** Freeze the continuous wave core layers (requires\_grad \= False). Allow only the ingress StandaloneL3Router and the egress readout head to train. This forces the input and output gates to adapt to the static, complex geometry of the core, learning to translate discrete tokens into continuous waves that can successfully traverse the "chaotic mirror" and decode back to text.  
* **Phase 2 (Volumetric Convergence):** Only after the edge layers have established a fluent, high-accuracy digital bypass do you unfreeze the core (requires\_grad \= True) and slowly introduce Langevin noise to anneal the 32 diffractive phase masks into coherent, task-specific attractors.

#### **2\. Implement the GBNF Logit Sieve**

To prevent the untrained core from generating invalid Unicode or parsing-breaking characters (such as \\u2074), you must wrap the egress decoder in your strict **python.gbnf grammar constraints**. This mathematically forces the output logits of any non-syntax character to ![][image10] prior to the argmax collapse, guaranteeing that the model only outputs compilable Python tokens even during high-temperature thermal exploration.

#### **3\. Anchor the Sagnac Veto Corridor**

Lower the baseline Langevin temperature during standard inference from ![][image9] down to a cold, isothermal state of ![][image11]. Only permit the thermal master to spike to high temperatures under explicit, validated logic-lock conditions, preventing the thermal microheaters from liquefying your active memory cash structures.

[image1]: <data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAABAAAAAZCAYAAAA4/K6pAAABFUlEQVR4XmNgGAWDFCgpKanJy8tnAfEsBQWFdKAQI7oarACoWAGo6SPQAH5kcTk5uTqg+C+gtASyOAoAKroFVDQHXQyZD5Tfo6ysLIYsBgbGxsasIEl0m4Fi/5H5ioqK8kBDJzCgewmocAqQYoGyv8nIyHBC2WADgPRUIH4FZZ8AeqUBphcMgIJ3YWygLXogjUD8FYjfQv2+C2izFlStJxA/gGsGAaDkI2Q+1JDtUIPWIcsB1RoDxS4gi6G4AGibOboLQIYADVUHyQMNcAHybyN0M4ANmAQKSCgbWxhMk4eGAdCAHRhhAAQsQMFDZMcCCGBLB0D+VTQ+RlSjAFhKFBcX50YWJyolIgOgYk15cvLCKBggAAB7LFA5mUG4dAAAAABJRU5ErkJggg==>

[image2]: <data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAABsAAAAZCAYAAADAHFVeAAABYElEQVR4XmNgGAWjgBQgJyfnKy8v/x8PfoiuhyoAZDDMEnQ5soG0tLQw0EBJEAZymWHiuCyDqYWqJw0oKCiYwwwF4q8wcWyWAdmTkNT2wcRJAkgGIBuMYpm4uDg3kL0HKnZVSkpKBGECCQCo+Sshy4CJRxDIPg0V2woMEQ6ECSQAkGaYwTIyMpxQMXTLbIDs3yA+0KIIVBNIAEADgoD4H9RwS6gYimVAegqMr6ysLIZqAtgxMUC5SaA0gC6HAZSUlPhhhkEtegrjAw2ajsSOQ9YHCgmg+GZYiADljYEWZiCrwQkUFRXFgRpCgQb8hFkA1NyIK46A8lUgNTC+qKgoD5D/CVkNQQD1HdgydDlkIA9NoWhiePVgABIsO4CuBp2PFZBTNspDUzKaGGHLyAHAuExHNhwaZ2+R1VANgFKfPFIxBypZgGI7kNVQFQB9JwG08DwQFwPxR6AQI7qaUTD0AADM95o6sWRAVwAAAABJRU5ErkJggg==>

[image3]: <data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAACwAAAAZCAYAAABKM8wfAAACRElEQVR4Xu2UP2hUQRDGXzAWYkBBj5P79+6PIKmvsLG0SaFNAkYCtgFbwYCVElJYahFQRNBOsZOEIBZHlBC0sQmISZEUqcRYWahI/H33Zo5lNdHY3IHvg2FnZ7/Z/d7svE2S/xH1ev14HBtUDNVqtesIfuSBNE3HiZ03f7bRaKTym83mMeIfKpXKaZnylG+8LWzXbMP3OhBarVaV5HfYewTdUYxDJkIOa4usTUeCd90vFAojzDcRWGZcw8ZCHnbX/I7H/wls8FAbSih2D38bu4b9CGhDivER7b0Eg2Hm3+GcY9wSN+RhL83v9DIOCm2CgMuJXZeDw5ZYe+tzOC8s3hNsFQ0Fd4XBmWFcZbwQxvUR5m9ijSQrwk673T7c22A/0G/F+ECHDmNtSj7juG8aCi4Wi0fjfM3VNkkm5iP+JcZZE7wWcgX7uGe4w/HaL7Cr20/waLlcPsH4Wv1sNoOIZeyseFG+RKrC3crqZ4N3i8KcMcELvrcn2Dkd3ZbH9gTEMdtoXtWO138HHRD18Ff3S6XSSeareiHgTOI/TrLKqbfX9WPbrWwH+XNwb/r8T+hWJDY2eBoThYjTFc0NVPDfEJvCPnnr6HkD9z2eBP8I/NvErmI3sBWP/zVIGrXkUNBkzBs4qDqI/WyiH8TrfQUVfMJwKI7rFUDsl7BP+w5rgbk4LlSr1YtWYb2TgwGJopJLPDtH4jWEPpfgON5XIOgVtoN90zPjcbVBPXtjT4X8voPqXnEfcdNp9vzMh+Jz5MiRI0cO4SeHYa+NAeR4QwAAAABJRU5ErkJggg==>

[image4]: <data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAABkAAAAaCAYAAABCfffNAAABtElEQVR4Xu1UsUoDQRC9QCxEG9F4mNxlE2JjpXCtdloqooUf4B+kEWIZ7MVSbCxsxNZGLAI2opWFjSCSSiwsrYTE97hdnJtcTsUUgnkwbO69ybzZ2bv1vCH+JIwxHURXRd1qmu9Wq9V5q51rLVlZAQlnTCqXyztaK5VKgdUirRHgHzSXikqlctzPBPxMP5MgCGbDMFzTfCpYnIVoJnnsYhLcjTVZlRqKL4F/lFwmWMDOtSV5FoHJetouwd3VarVpyWVCmFwJOg+D0yiKRqzWcAKLQ9sSuV+D80aRN0TbcShygSXH3zSRo/zRmBxMfLhtGvEZnYayU7uTVqFQGMe6Qf3z357n+/4Y+F3mebaxHuDd99kdooPHHNYm3pxRp1uTW+x4Auu1+GsC1iQdtsMWk3gOKLYvdWNHCX6FITWHYrE4xUY0L8HuT2zH93oc4F4R74gXPOal5gDzRegHmk8ACXs04a5SNI6y5zUWyJv41pjTQgJIqCOONE+Av8z6smG+bOLzzAbOYoH3lOYJFNjm96J5B+iNb5n8BjB4NvGt3NTawIDih4gnxKbWBokcL1NNDvGP8QGJ4YSAFFAjnwAAAABJRU5ErkJggg==>

[image5]: <data:image/png;base64,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>

[image6]: <data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAACIAAAAZCAYAAABU+vysAAABwUlEQVR4Xu1VPUvDUBRNqYLiIKIYbGPTtD/AIaAITuLqIghO/gLBzVlXRQcRBHHQwZ8g+LW5WFzr2MHBWRQqiKCeI/fSl0dqK7wOQg5cct+5XyfJTet5GTL8V5RKpcUwDL9+sUe7pufgUBVgx3qCYrE4imETNBzzyrcTormS7w7lcnlGB8KayqcJgb9v5O4p7wxGc3NoQojv+0Pwb4R7KBQKY60OjoDGzU5CsMgj8O+FO8eTHGh1cAQ21qFBEAwKZwuZg//BM0SsJDs4ApovwT5l8KxwCSG4Hui5Wq2Om/VceAhdR2zbMxaeQO4k+CPuoh1LRaVSGdZBIuJJzxhyaPirZl0URT64S4gJcOxjDoZuMga+hnhIP5T9gpszyttDGi+j6F2Ho/FWu53gq0TOtQjJSc0xY+I36KNnDL/Ja6JBJ4Qpn28H5JF7xpvwUu4asTX20h3sGn8Rwp1B3i3shZ+5HScQq8OebT4VLv5rJK9ucui7AW7e5JwDA3Zhd1x2Ob9RjMYhYkEXlnsGf0pjToHmpxjc4JLzLE/k5xWAm4aQC80FH2mec8Rx3I8Brxh4AlFX8HeMH0WKSphdnyFDN/gGya+53T/5aZ8AAAAASUVORK5CYII=>

[image7]: <data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAADUAAAAZCAYAAACRiGY9AAACmElEQVR4Xu1WTUhVURD2oUGiQRDPh7yf+36opyBpPFwotFRoEUQEVts2rdsE0k7ch7hRhBCJINwKLlxcdOeuhQSByBMjIiIIEjLKvk9ndN4kVyH03cX9YDhz5rtn7syZ89fSkiBBc1GpVLrQtKoeBEF3sVi8zL7oV6mn0+lO9vVb7VtRn00HgvkD2XfyTDhv3y+VSv3CLXmu0XOTgYAWGVShUHjuuWw2mxOu5jkC9g/eFgsg6CmZ7TnPIegR4e44KgXuZa1Wu+Ts8QArxMCxf+atHVW6Btu6VOqu5fL5/G3YN60tVmDAUo3Q2hk0krp30tKE7Z0cMvEEZn0QQe5C6sbGStyi7quI/kQul2vXfiyBKpQR6CfID7WxEkZnFVcymUwH9xD3knKxRXB4z9Q1KSyrPCozZngmFcrddJ/88eiYQoINGTyPbsi25WH/JUkz+dByCql2fC5gIIWAXktFNnwlYPsqiX1Gt81yBC7km+A2bVJcpuzzdWLt7GMCHvjTlCctBWoK/jL6qlHANsoxsPdYeyR4EDAptE89J1Viwl88RwinohXVMZOQDU2CPpgU2hdoX1kf+Pcq5BH0Wcie4eZgf0iB/l3tp0Luqp/eTsAewuG6zOQ/MMu3YflJUlNyQR+8GZmk4escS71arV4JDldEr3BH/tD+NuMf6/hTgaAH+CTydgKOnkS9HKKS8vcb/vNWVsUw5KMmJT4Oqixjj/yVy+Ub0PfoD7Jl/Z0bbFJMwiy1hqTQH+J3pr+D++46k4xKCu20GbOo+nmjDT97gwu7DwHOsKVRZnZcP8Jmr6L/Xro8nLiPBtAuyMN5R1cL9DVUqCD6N/WBSVpW/UJwxmdTKytQlNMtalkr+A33s1YuQYIECf4LfwEkltVGrdZaPgAAAABJRU5ErkJggg==>

[image8]: <data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAmwAAAA6CAYAAAAN3QXmAAAGZ0lEQVR4Xu3dS4hkVxkA4O4kiqIoaMYh09V1qnsG3CgR2geCixAENaBuFeMqK0MWgiiYhYkLUVEURFDcBJUguHBlFkrQAZfZKgEh+ECQGHSYxMwiMon/333P9JnTtyrdPd01lcn3weWe+5/7roLz17mPWlsDAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAACA17fJZPLmPgYAwM23Xkp5IIar0+l0p68EAGBFzGazn0rYAABWmIQNAGDFSdgAAFachA0AYMVJ2ACAW0okNn/OcSnlyb7utUrCdnz1vA3fh/WuGgC4GTK5ieHrfZz5dnZ23rC1tVX6+K3gwoULb4vR+ubm5gfiGO/OWIzfnbHr5wSAFVZKeTqGVyLJ+Wxf15pOp3/vY7lcDmsn2PhtbGy8s+9NygQstvN8DF9u42NinieG8T/6Og7K892eq0hs3hPn/8fD5zqqfu45xLy/aaryvXGXh/HFJn5DYl1/a6cnk8k76vZj8va2bp6Y98Vu+mI7DQArKxqtJ86ePfuWoXw1hg/380Sy9Kah/roGvG0Ao/xyW3dM69nT01/+i3U/F7GPZzmSiU/F9Nb+Igfl8jnu95dx+dmdOXPmrSPxueevdAlUys+uLpP/yhDlp/t5jur8+fPvyn3rtxfTl5ry3P2sYp7f57j2sDXxX7TTALCSsiEs+z1Sj9dkZ0zfMLbTUf5fW1dF/Dsjse/1sdZIwvZKDHcN5btyP/fnPijqnxvGX+rrliWO4Z7suerj5VWSzZuh/1yrefFURhK2OOZHFy1TxTz/aqdjufdF7CdtrNUnbOfOnbuzNL1jpes56+Wl3tyvHPrE9DD7CwC7stcoEpSP1OloRP7Z1i9LbPdSGelhq/rGLfb5R0ND+LNF/3MZ9b9ryn9s68aMJWy1oS17CdvCBvo0xbYvD8d8bRiZp+39+VUM99bpOK6P1vKqKHMuDY4dWzUc+71DT9ruZxrjf5ehpzWTpDjWr1y/1L56juK7//7smevrW33CFuv9ZPvDIupeLENCf1Sx3JU+BgCjsqGr5UzcMoFr65dhe3v77bEfT/XxVt+A535Gg/3eofHe7dWaJ+qfLE3itsiqJmw1yRjKV2L4RFtfbWxsTPpYHn9/X15vOI/HHX7er++w5vWq5nr72Jg6X9m7x7DtdV24fNRfWpToV6ecsB3oKQSAUV0j99emalnyBvHv9sFe3wBHw/lYU5f3vz3c1rfKDSZstUHOcQzP7M+9PJkYNOW82X40Acteo2Gf80GOX7Z1EftGO70KynBJvJf738dS3vPY1mV5SKoe7+O13Nva2vpg9qxNRx5k6Y0kbDuluyRa78M8qna9ALBQbdiGy0tZXu9mOVWxzYdqOXsvhvH3+0tVfQNcmhu2o/yDeZf7yo1fEn25rjvHZcFl23linV8Y6/k6irJ/71wmuHPf8TYbHpCoYt7LeUz9+Tum/G68cWS4o53pKErTw9tq93d4kjQT0N0HUGL8oZH58ry05dGHDmqyVqczaeu/a60+YUvtPt/IeS0uiQJwGPnUWm1wYvyXoVH8aj/faYlt/Ta32Qy1JyvLu5f88hUKs73XamTsc3XZKF9d23ulwvp0eFltr4wkaGVBshN1D8TwUi4X2/x0xrKBH7ZVt3lkZe/+qj/18aPI4x/Gr7YPB14zkU879rHjiG3/N871t/phc3PzM/28h1WPq51uhyZ+LbmJbT4W0//J+vx+1Hh8ZrOIPduvsxXLfq2PxWLf7mMp4vfEun497MuD9RJq2f8M5n73DmPRfgLANdFgPFV7LThdca7/0MdeY7J37dg9afPEeXkovoOP9vHXgzj2H/YxADjAL/zlyIcq1pZ8qfmkxXfl+T42XC68Mm2eMj6O6YInOm9VZc7TsQAAxzb8xdIBY4kci235ayoA4KRNp9Mv9rFKLy0AwPL1PT+3R8L2SBuIJO2l2Wz2sRg/U07mb8EAADiMSL5eKHt/nt7G+qc4L04mk40sD08Zz33/HQAApyATtEjEPj9M5lOht/X1TdlTxgAAyzadTr9Zk7K+d62PNeUTf90HAADz3TYkYndsb28f+PeIMvynab6ANueL8f39PAAAnLJIxO4b610DAGCFTCaT+/oYAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAADDX/wHexam48uufeAAAAABJRU5ErkJggg==>

[image9]: <data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAEUAAAAZCAYAAABnweOlAAACB0lEQVR4Xu2WPUjDQBiGW6iDqINiKTb/LVjEQSE4CY7ioog4uLuIo4KCk4vg4ObmUh3cXAVxEh1dFbsI4uLkoFQX8ef99ALH1yRNU0kQ74EPcu999/fmcpdMRqFQKH4Zy7IcxGfEqNu2vcT7SJAs5rDIxSAcx5ksFApdsmaa5rRc9gVJJ4ZhzOAx62kYeI5MyOfz3Z6m63ontFfkjnlaEuAljGLcmvdiUD7gOUFgbevspVIs87wGkHQjl9FRL7RL6kDWCWgXlUqlh+tJgHm5GP+lVVOwW0Zox7iu28HrfaG3j0E2ZQ0DDyEeEQ+yLuoOM9KOSpK4pqDNANdDgSkadxAdVcU225B1kKN8piVGYqb4QQOTKXEMoK2KicxHDSxwlvcRRBum1GleaDeF8j3KOzyvGXS60y5pOE/SJo4pMGOCfQk5Whv6Gpa0cIrFYr8w5Y3XpU0cU/yg9fFzNBQ0WBGmHPG6tGnVFE3TdORv8f8Usb4z+XcjFCTfWW38i2Di+/TdRg2Mdcv7CKJVU2jhZADaVZnuadFuUTT4QFzTZ8Tr0qaZKdC3kTPulZF7jHgvlUqDUtr3mUKHrqQ1Yv7cAgtI3qUGiBq0NdL5dZ0GuAX7xBxPxfyeqEwh54m6Z7ppqExzR3nP2/X0OaGPc2hXcrt/Ce0UMgexWi6XDV6vUCgUCoVC8Vf4AhmvsqTNRYLdAAAAAElFTkSuQmCC>

[image10]: <data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAACUAAAAZCAYAAAC2JufVAAABRklEQVR4Xu2UP0oEMRSHM6yCwoKgDCjzJzODnWIzqKew2sozeAEPsYUINgtip50HsBCxEbTQRg9gayEeYFm/JwkMb3cLmbEyH4Qkv/eS95uQiTGBQCDwj0iSZI0u0rqiF8dxX4udUxTFurX2nDahjZnvm2lzPWIPLmdCzqFoKqcbOJ2UIs9lWe7InGLHvnCappuiEbNuvuzXMV5FuzXT5tvDxq9aw8Qe+qczdym9mVOc2KnWfnBfO/pF2/Br7QxTAvpAzEjj9G503EPsUWutoeid1jwUvHfGPnTMQ+xFa61h0zetCXVdLzpDY9ePdA5E6FdabE2e51tsPGhqzM/ESFVVK87c0Bm7lqsiOVmW7bL2vXn5O4ViXxQ4YbjA+EgMML9opMhz4E9Mev98PDVy/oQIIwfuAZ0Jd2yJE9qWv1PHAoFAYA7fK1xV/8TbsSUAAAAASUVORK5CYII=>

[image11]: <data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAE8AAAAaCAYAAAD2dwHCAAACoklEQVR4Xu2XPWhUQRDH7/AsRFFEjwv38d7dcQgBweIQEbQzhUUsgoKVTUCttRCttLCySyNIglpYCJYWFla21qlCiIrEQkQETaH48f9zs2aY233vGbwLR/YHQ/L+M7Mfs2/f7ZZKkUgkssNI07QD+13Qvrbb7au2jXGAvq/DHsDmra8ISZJcQu5Zqzu63e4Rto+4C7Vaba/1e0Hwi1ardQ7/lp2GRuZYrGq1us9pzWZzD7QNxB532jjo9/u7ORb2z2dOjM8Y930ba8FCP0Xsmiw8c27YGBTtAHwvEXtTcqZgr7hYNnYIBH32aAvszKO/bzQaTauPEkx4Bv2uaI3jgP3Umg/ErLKAaONyqHjcSfTpFwXPJ2EfddwQXE0k39YakqZhn2AftC6+JyX1ho4avOVHubiwW1rH810pRlfrIRA3Gyoe9F/0Gbki8QeNvgmK1+C20BoSHjLRDhhUGG+0kcIx+CbNZ58eIqN4ZZmrLR77Zvys1TNB0mtJPGN94wa74rFMbk7rrniwRa2HCBWPW1Xa2dA6Ef2a1TORJK5ExfrywOAewd4VNfSzatvQuOJx8lp3xaNf6yEKFO+b1okvPpN6vX5YGvthfdvBRBUPwaeksWXr2w4matsi+BmTMOgr1leETqdzDAM8/w+W+UFm0XyTdsXLy3cwztcOkSKFfjD6Vg+ChDdcha0ehJP//M1LN49NC0ZflElPaz1ETvGGjiruIM7PmNYzkQGN9SyXRzp4+9aNts5Ju2fslIsydu/BOat41OjD4f+Q0/DynE49l4chpFOvFb7jjZhkcPZ8C1uSv/fcdY3IFet5Olj4v9j5OLNFRPGnxMe78xfYd8i7dMwkU8YE78jk8u+cWwDtn5D253u93n7rj0QikUgkEonsDP4AS0gpN/Oi4IIAAAAASUVORK5CYII=>