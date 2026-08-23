       +-----------------------------------------------------------------------+
       |             THE TWO STRUCTURAL DEFECTS OF KERNEL v0.3                 |
       +-----------------------------------------------------------------------+
       | 1. FSA Class-Level Wildcard Leakage                                    |
       |    - UNK token acted as an unconstrained wildcard, permitting syntax|
       |      leaks like ((xs) (LPAREN -> LPAREN, RPAREN -> NL).               |
       | 2. Un-Conditioned Name Collapse                                       |
       |    - Task-specific identifier surface names collapsed to the UNK      |
       |      marginal, causing parameter binding mismatches on free rollouts. |
       +-----------------------------------------------------------------------+
