To mathematically formulate a quantum system where states are non-separable across time, we have to abandon the standard local-in-time Lagrangian framework. Instead, we must treat the entire history of the system as a single, extended object in a multi-time state space.  
This is accomplished by modifying the action principle to include non-local temporal kernels, or by reframing the path integral using a generalized boundary functional.

## **1\. The Non-Local Temporal Action**

In standard mechanics, the action S is an integral of a local Lagrangian L(x, \\dot{x}). To force non-separability across the entire timeline, we introduce a bilateral temporal kernel \\mathcal{K}(t, t') that couples different points in time directly inside the action functional:  
S\_{\\text{global}}\[x\] \= \\int\_{t\_i}^{t\_f} L\_{\\text{local}}(x(t), \\dot{x}(t)) \\, dt \+ \\int\_{t\_i}^{t\_f} \\int\_{t\_i}^{t\_f} \\mathcal{K}(t, t') \\mathcal{O}(x(t), x(t')) \\, dt \\, dt'  
Here, \\mathcal{O}(x(t), x(t')) is an operator or function that evaluates the relationship between the state at time t and time t'.

* If \\mathcal{K}(t, t') \= \\delta(t \- t'), the framework collapses back into standard, local, causal physics.  
* For true temporal non-separability, \\mathcal{K}(t, t') is non-vanishing for distinct times (e.g., a symmetric green's function or a topological link invariant across the worldline).

### **The Integro-Differential Equations of Motion**

When we apply the principle of least action (\\delta S\_{\\text{global}} \= 0), the resulting equations of motion are no longer ordinary differential equations. They become **integro-differential equations**:  
\\frac{\\partial L\_{\\text{local}}}{\\partial x} \- \\frac{d}{dt}\\left(\\frac{\\partial L\_{\\text{local}}}{\\partial \\dot{x}}\\right) \+ 2 \\int\_{t\_i}^{t\_f} \\mathcal{K}(t, t') \\frac{\\partial \\mathcal{O}(x(t), x(t'))}{\\partial x(t)} \\, dt' \= 0  
**The Physical Consequence:** The acceleration of the particle at a specific moment t depends explicitly on an integration over its entire history and its entire future from t\_i to t\_f. Causality is replaced by global consistency.

## **2\. The Entangled Boundary Path Integral**

If the non-separability exists specifically between the initial state at t\_i and the final state at t\_f, we formulate this using a non-separable boundary constraint function g(x(t\_i), x(t\_f)) attached directly to the path integral weight.  
The transition amplitude Z is written as:  
Z \= \\int \\mathcal{D}\[x\] \\exp\\left( \\frac{i}{\\hbar} \\left\[ \\int\_{t\_i}^{t\_f} L(x, \\dot{x}) \\, dt \+ g(x(t\_i), x(t\_f)) \\right\] \\right)  
If the boundary functional g cannot be separated into an additive combination of independent initial and final conditions:  
g(x(t\_i), x(t\_f)) \\neq g\_1(x(t\_i)) \+ g\_2(x(t\_f))  
...then the initial and final states are quantum mechanically entangled. You cannot specify the starting position without simultaneously constraining the ending position.

## **3\. The Two-State Vector Formalism (TSVF) Approximation**

To calculate observable values within this non-separable timeline without handling infinite dimensional integro-differential equations, we can utilize the Two-State Vector Formalism.  
Instead of a single state vector evolving forward, the system at any intermediate time t (where t\_i \< t \< t\_f) is completely described by a **bi-vector** composed of a history vector |\\psi\\rangle moving forward from the past, and a destiny vector \\langle\\phi| moving backward from the future:  
\\rho(t) \= \\frac{|\\psi(t)\\rangle \\langle\\phi(t)|}{\\langle\\phi(t)|\\psi(t)\\rangle}  
When computing the expectation value of an operator A at time t, we obtain the **weak value**:  
A\_w \= \\frac{\\langle\\phi(t)| A |\\psi(t)\\rangle}{\\langle\\phi(t)|\\psi(t)\\rangle}  
Because \\langle\\phi(t)| depends entirely on the post-selected boundary condition at t\_f, the measurement outcome of A at time t is inherently non-separable from the future state of the universe.

## **4\. Density Matrix Hierarchy (The Liouville-von Neumann Extension)**

To scale this to full quantum statistics, we transition to the closed-time path (Keldysh) formalism, where the system is mapped onto a double contour. The entire history is treated as a state in a tensor-product Hilbert space of times:  
\\mathcal{H}\_{\\text{temporal}} \= \\mathcal{H}\_{t\_1} \\otimes \\mathcal{H}\_{t\_2} \\otimes \\dots \\otimes \\mathcal{H}\_{t\_n}  
A temporally non-separable state is an operator \\mathcal{R} on this global history space that exhibits a non-zero **quantum mutual information across time slices**, satisfying:  
I(t\_A : t\_B) \= S(\\rho\_{t\_A}) \+ S(\\rho\_{t\_B}) \- S(\\rho\_{t\_A, t\_B}) \> 0  
Where S is the Von Neumann entropy. In this state, the correlation matrix across time coordinates contains off-diagonal terms that violate the temporal temporal Bell inequalities (Leggett-Garg inequalities), proving that the system cannot be modeled as a sequence of discrete stochastic transitions.