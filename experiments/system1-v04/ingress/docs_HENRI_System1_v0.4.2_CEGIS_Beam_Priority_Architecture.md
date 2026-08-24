       +-----------------------------------------------------------------------+
       |             THE 50% CAPABILITY CEILING & THE PROMOTION GAP            |
       +-----------------------------------------------------------------------+
       | 1. Passive Diagnostic vs. Active Operational Efficacy                  |
       |    - E_φ correctly predicts pass probability (AUROC = 0.7531).        |
       |    - However, E_φ was evaluated ONLY as a passive post-hoc filter     |
       |      over a frozen, greedy token decoder.                             |
       | 2. Un-Differentiable Egress Bottleneck                                 |
       |    - The egress decoder's 50% pass rate ceiling squashes all 4 arms. |
       |    - Swarm particles generate identical greedy token completions,     |
       |      preventing E_φ from steering alternative trajectory generation. |
       +-----------------------------------------------------------------------+
