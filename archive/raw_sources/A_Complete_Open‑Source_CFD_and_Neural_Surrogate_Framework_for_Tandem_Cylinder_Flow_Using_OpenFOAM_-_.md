# A Complete Open‑Source CFD and Neural Surrogate Framework for Tandem Cylinder Flow Using OpenFOAM - Preprints.org
Source URL: https://www.preprints.org/manuscript/202603.1921

Loading [MathJax]/jax/element/mml/optable/GeometricShapes.js
Celebrate 10 Years of Open Sharing Explore All Events
Cite
Add to My List
Share Comments
 Download PDF
Version 2

Submitted:

27 March 2026

Posted:

27 March 2026

You are already at the latest version

Subscription

Notify me about updates to this article or when a peer-reviewed version is published.

Preprint
Article

This version is not peer-reviewed.

A Complete Open‑Source CFD and Neural Surrogate Framework for Tandem Cylinder Flow Using OpenFOAM
Muhammad Bilal  *
,Muhammad Sabeel Khan  *
Muhammad Bilal  *
,Muhammad Sabeel Khan  *
Version 2

Submitted:

27 March 2026

Posted:

27 March 2026

You are already at the latest version

Abstract
The flow past two cylinders in tandem arrangement is of fundamental importance in engineering applications such as heat exchangers, offshore structures, and power transmission lines. This study presents a complete open‑source simulation pipeline using Gmsh for mesh generation and OpenFOAM for the finite‑volume solver, combined with a long short‑term memory (LSTM) neural network surrogate for fast predictions. A distance‑based refinement strategy resolves the flow accurately, with characteristic mesh sizes as low as around the cylinders. The methodology is validated against the classical Schäfer–Turek single‑cylinder benchmark at (Re=100), showing satisfactory agreement for force coefficients and Strouhal number. The main analysis focuses on a tandem configuration at 
𝑅
𝑒
=
1.0
×
10
5
 with unequal diameters (
𝐷
1
=
0.1
m
, 
𝐷
2
=
0.15
m
) spaced 
1.0
m
 centre‑to‑centre. The results reveal strong wake interaction: the downstream cylinder experiences higher mean drag 
(
𝐶
―
𝐷
=
0.997
)
 and significantly larger lift fluctuations 
(
𝐶
𝐿
′
=
0.340
)
 than the upstream cylinder 
(
𝐶
―
𝐷
=
0.947
, 
𝐶
𝐿
′
=
0.129
). Both cylinders shed vortices at the same frequency 
𝑓
=
2.041
Hz
, yielding Strouhal numbers 
𝑆
𝑡
𝐴
=
0.204
 and 
𝑆
𝑡
𝐵
=
0.306
. An LSTM neural network trained on the force coefficient time series achieves near‑perfect predictions of the downstream lift and correctly reproduces the shedding frequency, providing a fast and accurate surrogate model. The fully reproducible open‑source workflow, including all CFD setup files and the neural network training code, is made available, enabling future studies on bluff‑body interactions and facilitating the adoption of data‑driven methods in fluid mechanics.
Keywords: 
tandem cylinders
;  
OpenFOAM
;  
Gmsh
;  
CFD
;  
vortex shedding
;  
turbulent flow
;  
wake interaction
;  
force coefficients
;  
strouhal number
;  
LSTM
;  
neural network surrogate
;  
machine learning
Subject: 
Computer Science and Mathematics  -   Computational Mathematics
1. Introduction
Flow past bluff bodies has been a central theme in fluid mechanics for decades, owing to its relevance in numerous engineering systems. Among the many configurations, the tandem arrangement of two circular cylinders one placed behind the other exhibits intricate wake interactions, vortex shedding, and unsteady loading that are critical for structures such as heat exchanger tube bundles, offshore risers, power transmission lines, and marine cables (Figure 1). Understanding these flows is essential for predicting fatigue, noise, vibration, and for optimising structural layouts.
Figure 1. Engineering applications of tandem cylinders: (a) heat exchanger tube bundles, (b) offshore risers, (c) power transmission lines, and (d) marine cables.
The canonical case of a single circular cylinder has been extensively investigated. Foundational work by King [1] and Williamson [2] established the fundamental characteristics of vortex shedding and the dependence on Reynolds number, while Norberg [3] provided detailed insights into fluctuating lift. For two cylinders arranged in tandem, the flow becomes considerably more complex due to the interference between the bodies. The upstream cylinder sheds vortices that interact with the downstream cylinder, altering wake structure, force coefficients, and shedding frequency. The centre-to-centre spacing ratio is a key parameter, and the flow regimes have been categorised in comprehensive reviews by Zdravkovich [4] and Sumner [5]. These works classify the regimes into (i) a single-bluff-body regime at very small spacings, (ii) a reattachment regime where shear layers from the upstream cylinder reattach onto the downstream one, and (iii) a co-shedding regime where both cylinders shed vortices independently.
Experimental studies have provided valuable data. Zhang et al. [6] combined experiments and numerical simulations to examine three-dimensional flow in a duct, highlighting the influence of confinement. Wang et al. [7] investigated the effect of a nearby wall on the tandem arrangement, showing how wall proximity alters wake development. Khan et al. [8] studied flow-induced vibrations for cylinders of different diameters and spacing ratios, emphasising the importance of diameter disparity.
Numerical simulations have greatly expanded the parameter space. Early finite-volume and finite-element computations, such as the benchmark of Schäfer and Turek [9], established reference solutions for laminar flow around a single cylinder at 
𝑅
𝑒
=
100
. For tandem cylinders, Vu et al. [10] performed simulations at low Reynolds numbers using the lattice Boltzmann method, while Jiang et al. [11] investigated two- and three-dimensional instabilities in the wake of a cylinder near a moving wall. Large eddy simulations (LES) have been applied to capture turbulent structures at higher Reynolds numbers; Liang and Papadakis [12] used LES for pulsating flow over a circular cylinder. Elgendy [13] conducted a detailed numerical study of oscillating tandem cylinders with two degrees of freedom.
More recent contributions have explored a variety of configurations. Xu et al. [14] examined the effect of spacing on flow-induced vibrations of two tandem circular cylinders at subcritical Reynolds numbers. Zhao et al. [15] studied flow past wall mounted finite length square cylinders in tandem. Chang et al. [16] investigated wake structures and hydrodynamic characteristics of flows around near-wall cylinders in tandem and parallel arrangements. Nguyen et al. [17] provided a state-of-the-art review of flows past confined circular cylinders. Vu et al. [18] used the lattice Boltzmann method for a comprehensive study of low-Reynolds flow through two tandem cylinders with various configurations. Kumar et al. [19] examined wake interference in tandem square cylinders. Wang et al. [20] studied the first instability of the flow past two tandem cylinders with different diameters. Zhang et al. [21] investigated flow induced vibrations of ten tandem cylinders at low Reynolds number. Sarkar et al. [22] analysed flow around two wall-mounted trapezoidal bluff bodies in tandem. Zhao et al. [23] considered free-surface effects on the flow around two circular cylinders in tandem.
Despite the wealth of existing work, several gaps remain. Many numerical studies are confined to low Reynolds numbers or rely on commercial software that is not universally accessible. High-fidelity simulations at fully turbulent Reynolds numbers and are still relatively scarce for tandem cylinders, especially with unequal diameters. Moreover, a complete, open-source based methodology that integrates mesh generation, solution, and post-processing in a fully reproducible manner is lacking. Such a framework would democratise access to advanced CFD and enable transparent validation.
At the same time, machine learning is increasingly being used to accelerate fluid dynamics research. Recurrent neural networks, particularly long short-term memory (LSTM) networks, have shown great promise in learning temporal dynamics from time-series data [24,25]. However, applications to tandem cylinder flows, especially when combined with open-source CFD, remain rare.
In this study, we address these gaps by developing a fully open-source simulation pipeline for flow past two cylinders in tandem arrangement using Gmsh for mesh generation and OpenFOAM for the flow solver. The methodology is validated against the well-known single-cylinder benchmark of Schäfer and Turek [9]. We then apply the approach to a tandem configuration at 
𝑅
𝑒
=
1.0
×
10
5
 with an upstream cylinder diameter 
𝐷
1
=
0.1
m
 and a downstream cylinder of larger diameter 
𝐷
2
=
0.15
m
, spaced 
1.0
m
 centre-to-centre. The main contributions are:
A detailed, open-source workflow for tandem-cylinder simulations, including a distance-based mesh refinement strategy that ensures accurate resolution of near-wall gradients and wake structures.
Validation of the solver against the Schäfer–Turek benchmark, providing confidence in the force predictions and vortex-shedding characteristics.
Comprehensive analysis of the flow field at a high Reynolds number, including instantaneous and time-averaged contours of velocity, pressure, turbulent kinetic energy, dissipation rate, and eddy viscosity.
Quantification of force coefficients and Strouhal numbers for both cylinders, highlighting the effect of wake interference.
Detailed probe data at two downstream locations, offering a quantitative reference for future comparisons.
A long short-term memory (LSTM) neural network trained on the force coefficient time series, which accurately predicts the downstream lift coefficient (
𝑅
2
=
0.9895
) and recovers the correct shedding frequency. This surrogate model provides a fast and reliable tool for further parametric studies.
By providing a fully transparent and reproducible methodology including all CFD setup files and the neural network training code, this work aims to facilitate future studies on similar bluff-body problems and encourage the adoption of open-source tools in industrial and academic settings.
2. Methodology
The computational domain is a rectangular channel of length 
𝐿
=
2.2
m
 and height 
𝐻
=
0.41
m
. Two circular cylinders are arranged in tandem, the upstream cylinder (A) has diameter 
𝐷
1
=
0.1
m
 with its centre at 
(
0.2
,
0.205
)
m
 and the downstream cylinder (B) has diameter 
𝐷
2
=
0.15
m
 with its centre at 
(
1.2
,
0.205
)
m
. The centre-to-centre spacing is therefore 
1.0
m
. The domain is extruded in the z-direction by a thickness 
𝑡
=
0.01
m
 to create a quasi-two-dimensional geometry suitable for OpenFOAM’s 2.5D treatment.
Figure 2. Schematic of the computational domain with the two cylinders.
Mesh generation was performed using Gmsh [26]. Characteristic lengths were set to 
0.01
m
 on the channel boundaries and 
0.001
m
 on the cylinder surfaces. A distance-based refinement field was defined around the cylinders as
Field
[
1
]
=
Distance
,
Field
[
1
]
.
CurvesList
=
{
5
,
6
,
7
,
8
,
9
,
10
,
11
,
12
}
,
and a threshold field smoothly transitions from the fine mesh (
𝐿
𝑐
=
0.001
) near the cylinders to the coarser background mesh (
𝐿
𝑐
=
0.01
) over a distance of 
0.2
m
. The surface mesh was generated using the frontal-Delaunay algorithm with recombination to obtain quadrilateral elements. The two-dimensional mesh was then extruded in the z-direction with a single layer, resulting in a thin three-dimensional mesh of wedge elements. The final mesh contains approximately 
34
716
 cells; a quality check with OpenFOAM’s checkMesh utility confirmed acceptable non-orthogonality and skewness.
Figure 3. Computational mesh around the cylinders, showing the refined region.
The flow is assumed incompressible, Newtonian, and isothermal. The instantaneous continuity and Navier–Stokes equations are
∂
𝑢
𝑖
∂
𝑥
𝑖
	
=
0
,
(1)
∂
𝑢
𝑖
∂
𝑡
+
∂
(
𝑢
𝑖
𝑢
𝑗
)
∂
𝑥
𝑗
	
=
−
1
𝜌
∂
𝑝
∂
𝑥
𝑖
+
𝜈
∂
2
𝑢
𝑖
∂
𝑥
𝑗
∂
𝑥
𝑗
,
(2)
where 
𝑢
𝑖
 is the velocity component, p the pressure, 
𝜌
 the constant density, and 
𝜈
 the kinematic viscosity.
Turbulence is modelled using the Reynolds-averaged approach. Decomposing the variables into mean and fluctuating parts, 
𝑢
𝑖
=
𝑢
𝑖
¯
+
𝑢
𝑖
′
 and 
𝑝
=
𝑝
¯
+
𝑝
′
, and averaging (1)–() yields the RANS equations:
∂
𝑢
𝑖
¯
∂
𝑥
𝑖
	
=
0
,
(3)
∂
𝑢
𝑖
¯
∂
𝑡
+
∂
(
𝑢
𝑖
¯
𝑢
𝑗
¯
)
∂
𝑥
𝑗
	
=
−
1
𝜌
∂
𝑝
¯
∂
𝑥
𝑖
+
∂
∂
𝑥
𝑗
𝜈
∂
𝑢
𝑖
¯
∂
𝑥
𝑗
−
𝑢
𝑖
′
𝑢
𝑗
′
¯
.
(4)
The Reynolds stresses 
−
𝑢
𝑖
′
𝑢
𝑗
′
¯
 are closed with the standard k-
𝜀
 model [27]. The Boussinesq hypothesis gives
−
𝑢
𝑖
′
𝑢
𝑗
′
¯
=
𝜈
𝑡
∂
𝑢
𝑖
¯
∂
𝑥
𝑗
+
∂
𝑢
𝑗
¯
∂
𝑥
𝑖
−
2
3
𝑘
𝛿
𝑖
𝑗
,
where 
𝜈
𝑡
 is the turbulent viscosity, 
𝑘
=
1
2
𝑢
𝑖
′
𝑢
𝑖
′
¯
 the turbulent kinetic energy, and 
𝛿
𝑖
𝑗
 the Kronecker delta. The turbulent viscosity is computed from
𝜈
𝑡
=
𝐶
𝜇
𝑘
2
𝜀
,
with 
𝐶
𝜇
=
0.09
. The transport equations for k and its dissipation rate 
𝜀
 are
∂
𝑘
∂
𝑡
+
∂
(
𝑢
𝑗
¯
𝑘
)
∂
𝑥
𝑗
	
=
∂
∂
𝑥
𝑗
𝜈
+
𝜈
𝑡
𝜎
𝑘
∂
𝑘
∂
𝑥
𝑗
+
𝑃
𝑘
−
𝜀
,
(5)
∂
𝜀
∂
𝑡
+
∂
(
𝑢
𝑗
¯
𝜀
)
∂
𝑥
𝑗
	
=
∂
∂
𝑥
𝑗
𝜈
+
𝜈
𝑡
𝜎
𝜀
∂
𝜀
∂
𝑥
𝑗
+
𝐶
𝜀
1
𝜀
𝑘
𝑃
𝑘
−
𝐶
𝜀
2
𝜀
2
𝑘
,
(6)
where the production term 
𝑃
𝑘
=
𝜈
𝑡
(
∂
𝑢
𝑖
¯
/
∂
𝑥
𝑗
+
∂
𝑢
𝑗
¯
/
∂
𝑥
𝑖
)
∂
𝑢
𝑖
¯
/
∂
𝑥
𝑗
. The model constants take their standard values:
𝐶
𝜀
1
=
1.44
,
𝐶
𝜀
2
=
1.92
,
𝜎
𝑘
=
1.0
,
𝜎
𝜀
=
1.3
.
Near solid walls, standard wall functions are used to avoid resolving the viscous sublayer. The dimensionless wall distance is 
𝑦
+
=
𝑦
𝑢
𝜏
/
𝜈
, with friction velocity 
𝑢
𝜏
=
𝜏
𝑤
/
𝜌
. For 
𝑦
+
>
11.63
, the logarithmic law
𝑈
+
=
1
𝜅
ln
(
𝐸
𝑦
+
)
applies, where 
𝜅
=
0.41
 (von Kármán constant) and 
𝐸
=
9.8
. The turbulent kinetic energy satisfies a zero-gradient condition at the wall, while 
𝜀
 at the first cell centre is prescribed as 
𝜀
=
𝐶
𝜇
3
/
4
𝑘
3
/
2
/
(
𝜅
𝑦
𝑝
)
.
The simulations were performed with OpenFOAM® version 11 using the finite volume method. The transient term was discretized with a first-order implicit Euler scheme. Convective terms in (), (5) and () were discretized with second-order upwind differencing; diffusion terms used second-order central differencing with explicit non-orthogonal correction. Pressure–velocity coupling was handled by the PIMPLE algorithm with two corrector steps. The pressure equation was solved with a geometric agglomerated algebraic multigrid (GAMG) solver, while the velocity and turbulence equations were solved using a smoothSolver with Gauss–Seidel smoother. Residual tolerances were set to 
10
−
7
 for all equations.
Boundary conditions were specified as follows:
Inlet: Uniform velocity 
𝑢
¯
𝑥
=
𝑈
∞
=
1
m
/
s
, 
𝑢
¯
𝑦
=
𝑢
¯
𝑧
=
0
. Turbulence intensity 
𝐼
=
5
%
 gave inlet values
𝑘
∞
=
3
2
(
𝑈
∞
𝐼
)
2
=
0.00375
m
2
/
s
2
,
𝜀
∞
=
𝐶
𝜇
3
/
4
𝑘
∞
3
/
2
𝑙
,
with the turbulent length scale 
𝑙
=
0.07
𝐷
1
=
0.007
m
, yielding 
𝜀
∞
=
0.0066
m
2
/
s
3
.
Outlet: Zero gradient for all variables except pressure, which was fixed at zero (gauge pressure).
Cylinder surfaces and channel walls (top and bottom): No-slip condition for velocity (
𝑢
𝑖
¯
=
0
), zero normal gradient for pressure, and wall functions for turbulence quantities as described above.
Front and back planes: Because the domain was extruded in the z-direction, these boundaries were treated as empty, enforcing two-dimensionality.
The initial velocity field was obtained by solving a potential flow solution using OpenFOAM’s potentialFoam utility, ensuring a divergence-free field that respects the inlet and outlet conditions. Turbulence quantities were initialised with the constant inlet values.
Table 1. Mesh quality statistics.
Table 2. Boundary conditions for all patches.
2.1. Surrogate Model Development
We constructed a data-driven surrogate model to capture the unsteady dynamics of the downstream cylinder. The aim was to learn a mapping from past values of the lift coefficient to its next value, enabling fast predictions without re-running the expensive CFD solver. For this purpose, we employed a long short-term memory (LSTM) network, which is well-suited for time-series forecasting due to its ability to retain information over long sequences. As baselines, we also trained a simple multi-layer perceptron (MLP) that uses only the instantaneous upstream lift to predict the downstream lift, and a classical autoregressive integrated moving average (ARIMA) model.
The time series of lift coefficients for both cylinders were extracted from the OpenFOAM simulation. The downstream lift 
𝐶
𝐿
,
𝐵
(
𝑡
)
 was used as the target. The data were sampled with a constant time step 
𝛥
𝑡
=
0.012893
s
, corresponding to a sampling frequency of 
77.56
Hz
, which is more than sufficient to resolve the shedding frequency. We created sequences of length 
𝐿
=
10
 to predict the next value. The dataset was split into 
80
%
 for training and 
20
%
 for testing, with the training set used for hyperparameter tuning via early stopping.
The LSTM network consisted of a single LSTM layer with 50 hidden units, followed by a dense output layer. The input shape was 
(
𝐿
,
1
)
. The model was compiled with the Adam optimizer and mean squared error loss. Training was performed for up to 200 epochs with a batch size of 32, using early stopping (patience 
=
10
) based on the validation loss. The best weights were restored at the end of training.
For comparison, we implemented:
MLP: A feed-forward network with two hidden layers of 64 neurons each and ReLU activation, mapping the upstream lift 
𝐶
𝐿
,
𝐴
(
𝑡
)
 to the downstream lift 
𝐶
𝐿
,
𝐵
(
𝑡
)
. This tests whether a simple instantaneous mapping is sufficient.
ARIMA: An autoregressive integrated moving average model fitted to the training portion of the downstream lift series. The order 
(
𝑝
,
𝑑
,
𝑞
)
=
(
5
,
1
,
0
)
 was selected by minimising the Akaike information criterion.
The models were evaluated on the test set using the mean absolute error (MAE), root mean squared error (RMSE), and the coefficient of determination 
𝑅
2
. The dominant frequency of the predictions was extracted via fast Fourier transform and compared to the actual shedding frequency to verify that the LSTM captured the correct periodicity.
All neural network models were implemented in Python using TensorFlow/Keras, and the ARIMA model was built with the statsmodels library. The complete code is provided as Supplementary Material.
3. Validation
To assess the reliability of the numerical framework, a single-cylinder benchmark at 
𝑅
𝑒
=
100
 was simulated and compared with the reference data of Schäfer and Turek (1996), case 2D-2 (mesh 8a) [9]. The comparison is summarised in Table 3.
Table 3. Comparison of force coefficients and Strouhal number for the single-cylinder flow at 
𝑅
𝑒
=
100
.
The maximum drag coefficient from the present simulation agrees well with the benchmark, indicating that the solver captures the peak unsteady force accurately. However, the mean drag coefficient is significantly higher, while the maximum lift and the Strouhal number are lower than the reference values. These differences can be attributed to several factors:
Inlet profile: The benchmark uses a parabolic velocity profile at the inlet, whereas the present simulation employs a uniform inflow (as used in the tandem-cylinder study). The different inlet conditions alter the boundary layer development and the strength of vortex shedding, directly affecting the time-averaged forces.
Mesh resolution: Although the mesh was refined around the cylinder (
𝑐
𝑙
cylinder
=
0.001
), the number of cells may still be insufficient to fully resolve the near-wall gradients and the small-scale vortical structures, leading to an overprediction of drag.
Time-averaging interval: The simulation was run for 30s, but a longer integration period might be needed to obtain fully converged mean values, especially for the low-frequency components of the flow.
Despite these discrepancies, the present model reproduces the essential unsteady behaviour periodic vortex shedding and the characteristic frequency with reasonable accuracy.
4. Results and Discussion
The flow characteristics around two cylinders arranged in tandem are analysed at a Reynolds number of 
𝑅
𝑒
=
1.0
×
10
5
 using the Reynolds-averaged Navier–Stokes (RANS) approach with the standard k–
𝜀
 turbulence model. The key flow parameters employed in the simulations are summarised in Table 4. The chosen Reynolds number corresponds to a turbulent flow regime in which strong vortex shedding, wake interaction, and complex unsteady flow structures are expected.
Table 4. Flow parameters.
As shown in Table 4, the inlet velocity is fixed at 
𝑈
∞
=
1
 m/s, and the fluid properties correspond to water, yielding a high Reynolds number flow. Under these conditions, the flow is fully turbulent and characterised by significant inertial effects relative to viscous forces. The imposed turbulence intensity of 
5
%
 ensures realistic inflow conditions, promoting the development of turbulent structures downstream of the cylinders.
The selected time step, 
𝛥
𝑡
=
0.001
 s, is sufficiently small to resolve the transient evolution of the flow and capture vortex shedding phenomena with reasonable temporal accuracy. The total simulation time of 10 s allows the flow to reach a statistically steady state, enabling meaningful computation of time-averaged quantities such as drag, lift, and Strouhal number.
The transient evolution of the velocity field around the tandem cylinders is shown in Figure 8. At the initial stage (
𝑡
=
0
, Figure 4), the flow remains nearly uniform due to the potential flow initialisation. As time progresses to 
𝑡
=
1
s (Figure 5), flow separation begins to develop around both cylinders, leading to the formation of small recirculation zones. At later times, 
𝑡
=
5
s and 
𝑡
=
10
s (Figures 6 and 7), a fully developed wake is observed, characterised by low-velocity regions behind the cylinders and accelerated flow around their surfaces. The wake interaction between the upstream and downstream cylinders becomes clearly visible, indicating strong hydrodynamic coupling.
Figure 8. Temporal evolution of the velocity magnitude U around the tandem cylinders.
The distribution of turbulent kinetic energy is presented in Figure 13. At 
𝑡
=
0
 (Figure 10), turbulence levels are negligible. As the flow develops (
𝑡
=
1
s, Figure 10), shear layers form around the cylinders, increasing turbulence production. At 
𝑡
=
5
s and 
𝑡
=
10
s (Figures 11 and 12), high values of k are concentrated in the shear layers and wake regions, especially in the interaction zone between the two cylinders, indicating strong energy transfer from the mean flow to turbulent fluctuations.
Figure 13. Temporal evolution of the turbulent kinetic energy k around the tandem cylinders.
The evolution of the turbulence dissipation rate is shown in Figure 18. Initially (
𝑡
=
0
, Figure 14), 
𝜖
 remains very low throughout the domain. As vortical structures develop (
𝑡
=
1
s, Figure 15), localised dissipation regions appear near the cylinder surfaces. At later times (
𝑡
=
5
s and 
𝑡
=
10
s, Figures 16 and 17), the dissipation rate is highly concentrated in the shear layers and wake regions, indicating the breakdown of turbulent eddies into smaller scales.
Figure 18. Temporal evolution of the turbulence dissipation rate 
𝜖
 around the tandem cylinders.
The pressure field evolution is illustrated in Figure 23. At 
𝑡
=
0
 (Figure 19), the pressure distribution is nearly symmetric. As the flow develops (
𝑡
=
1
s, Figure 20), a high-pressure region forms at the upstream stagnation points of the cylinders, while low-pressure regions develop in their wakes. At later times (
𝑡
=
5
s and 
𝑡
=
10
s, Figures 21 and 22), the pressure field becomes asymmetric due to vortex shedding and wake interaction, particularly affecting the downstream cylinder.
Figure 23. Temporal evolution of the pressure field p around the tandem cylinders.
The variation of the velocity magnitude at a selected downstream location is shown in Figure 24. The profile clearly reflects the influence of the wake generated by the upstream and downstream cylinders. A significant velocity deficit is observed in the central region of the channel, corresponding to the wake zone, while higher velocities appear near the upper and lower walls due to flow acceleration around the cylinders. This non-uniform distribution indicates strong momentum loss in the wake and recovery toward the free-stream velocity away from the centreline.
Figure 24. Velocity magnitude profile at a selected downstream location.
The turbulent kinetic energy distribution at the same location is presented in Figure 25. The profile shows peak values in regions corresponding to strong shear layers, particularly near the wake boundaries. The elevated turbulence levels indicate intense mixing and energy transfer from the mean flow to turbulent fluctuations. Lower values of k are observed away from the wake, where the flow becomes more uniform.
Figure 25. Turbulent kinetic energy profile at a selected downstream location.
The dissipation rate profile is shown in Figure 26. Similar to the turbulent kinetic energy, 
𝜖
 attains higher values in the shear layer regions where turbulent eddies break down into smaller scales. The peaks in 
𝜖
 indicate zones of strong viscous dissipation, while lower values away from the wake region suggest reduced turbulent activity.
Figure 26. Dissipation rate 
𝜖
 profile at a selected downstream location.
The turbulent viscosity distribution is illustrated in Figure 27. The profile shows increased turbulent viscosity in the wake and shear layer regions, where turbulence intensity is high. This enhanced viscosity reflects the increased momentum transport due to turbulent mixing. In contrast, near-wall and free-stream regions exhibit lower turbulent viscosity, indicating weaker turbulence effects.
Figure 27. Turbulent viscosity distribution at a selected downstream location.
The pressure distribution at the selected location is shown in Figure 28. The profile indicates a pressure deficit in the wake region due to flow separation and vortex shedding. Away from the wake, the pressure gradually recovers toward the free-stream value. The asymmetry in the pressure distribution reflects the unsteady nature of the wake and its interaction with the downstream cylinder.
Figure 28. Pressure profile at a selected downstream location.
4.1. Force Coefficients and Vortex Shedding
The time-averaged drag coefficient and the root-mean-square (RMS) lift coefficient for both cylinders are reported in Table 5. The upstream cylinder (A) shows a mean drag coefficient of 
𝐶
𝐷
¯
=
0.947
, whereas the downstream cylinder (B) has a slightly larger value of 
𝐶
𝐷
¯
=
0.997
. This increase indicates that the downstream cylinder experiences a stronger net resistance due to its exposure to the unsteady wake of the upstream cylinder.
Table 5. Time-averaged drag and RMS lift coefficients.
The RMS lift coefficient is much smaller for the upstream cylinder, 
𝐶
𝐿
′
=
0.129
, than for the downstream cylinder, 
𝐶
𝐿
′
=
0.340
. This clearly shows that the downstream cylinder is subjected to stronger lateral unsteadiness and more intense vortex-induced loading. In tandem-cylinder configurations, the wake of the first cylinder typically impinges on the second cylinder, amplifying the lift fluctuations and making the downstream body more sensitive to flow oscillations.
The temporal variation of the drag and lift coefficients is shown in Figure 29. The plot indicates an initial transient stage, during which both coefficients adjust from the starting condition toward a periodic or statistically periodic regime. After this initial adjustment, the drag coefficient exhibits comparatively weaker oscillations than the lift coefficient, which is consistent with bluff-body wake dynamics. The lift history contains stronger fluctuations because the alternating shedding of vortices produces an unsteady transverse force on the cylinders.
For the upstream cylinder, the oscillations are relatively moderate because it interacts primarily with the incoming flow. For the downstream cylinder, however, the amplitude of the lift oscillation is larger, reflecting the effect of wake interference and the stronger unsteady forcing imposed by the upstream cylinder’s shed vortices. Overall, the force histories confirm that the downstream cylinder experiences a more vigorous unsteady aerodynamic loading than the upstream cylinder.
Figure 29. Temporal variation of the drag and lift coefficients for the tandem cylinders.
The vortex shedding characteristics of the tandem cylinders are summarised in Table 6. Both cylinders shed vortices at the same dominant frequency, 
𝑓
=
2.041
 Hz, indicating that the wake interaction couples the shedding modes. The corresponding Strouhal numbers, based on the individual diameters, are 
𝑆
𝑡
𝐴
=
0.204
 and 
𝑆
𝑡
𝐵
=
0.306
, which lie in the typical range for a circular cylinder at 
𝑅
𝑒
=
10
5
.
Table 6. Strouhal numbers and shedding frequencies.
The frequency spectrum of the lift coefficient is shown in Figure 30. A clear dominant peak appears at the shedding frequency, confirming that the lift fluctuations are governed by a periodic vortex-shedding mechanism. The presence of a strong spectral peak indicates that the flow has reached a quasi-periodic state after the initial transient stage. Since the downstream cylinder is directly exposed to the wake of the upstream cylinder, its lift response is expected to contain stronger unsteady components, which is consistent with the larger RMS lift reported in Table 5. Overall, the FFT results confirm the existence of coherent oscillatory motion in the wake and provide a reliable estimate of the shedding frequency.
Figure 30. Frequency spectrum of the lift coefficient showing the dominant vortex-shedding frequency.
4.2. Probe Statistics and Time Histories
The statistical quantities extracted at the two probe locations are summarised in Table 7. Probe 0, located at 
(
1.5
,
0.2
,
0.005
)
, lies closer to the near-wake region and therefore records a lower mean velocity magnitude, 
𝑈
mag
=
0.236
 m/s, with comparatively strong fluctuations. Probe 1, at 
(
1.8
,
0.2
,
0.005
)
, is farther downstream and shows a much larger mean velocity magnitude of 
0.791
 m/s, indicating partial recovery of the flow toward the free-stream condition. The RMS values suggest that unsteady velocity fluctuations remain significant at both points, with probe 1 exhibiting a slightly larger velocity fluctuation level than probe 0. For pressure, both probes record negative mean values, which is consistent with the low-pressure wake behind the cylinders. The more negative mean pressure at probe 1 indicates that it is more strongly influenced by the downstream wake structure.
Table 7. Mean and RMS values at the two probe locations.
The time history of the streamwise velocity component at the probe locations is shown in Figure 31. The signal exhibits clear unsteady oscillations, reflecting the periodic passage of vortices in the wake. The fluctuation amplitude is larger at the downstream probe, which indicates that the wake-induced unsteadiness persists farther downstream and remains dynamically important.
Figure 31. Temporal variation of the streamwise velocity component at the probe locations.
The cross-stream velocity component is presented in Figure 32. Compared with the streamwise component, the cross-stream signal is typically more sensitive to vortex shedding because it directly reflects the alternating lateral motion of the wake. The oscillatory pattern confirms the presence of a strong periodic transverse flow caused by the unsteady separation behind the tandem cylinders.
Figure 32. Temporal variation of the cross-stream velocity component at the probe locations.
The pressure histories at the same probe points are shown in Figure 33. The pressure fluctuates periodically due to the alternating vortex shedding and the associated wake development. The amplitude of the pressure variation is substantial, especially near the wake region, which confirms that pressure unsteadiness plays a major role in the aerodynamic loading on the cylinders. Together, Figure 31, Figure 32 and Figure 33 demonstrate that both velocity and pressure remain strongly time-dependent in the downstream wake, consistent with the force fluctuations reported earlier.
Figure 33. Temporal variation of pressure at the probe locations.
4.3. Neural Network Surrogate Model
To further exploit the generated dataset and provide a fast predictive tool, we trained a long short-term memory (LSTM) neural network on the time series of force coefficients. The LSTM is designed to capture temporal dependencies in periodic signals. As baselines, we also trained a simple multi-layer perceptron (MLP) that maps instantaneous upstream lift to downstream lift, and a classical ARIMA model. The data preparation, architecture, and training details are described in sub Section 2.1. Here we present the key results.
The performance of the models on the test set is given in Table 8. The MLP fails to capture the relationship between upstream and downstream lift (
𝑅
2
=
0.421
), confirming that a simple instantaneous mapping is insufficient. In contrast, the LSTM achieves near-perfect predictions for the downstream lift (
𝑅
2
=
0.9895
, MAE=0.0308), successfully learning the temporal dynamics. When applied to the downstream drag, the LSTM performs poorly (
𝑅
2
=
0.3226
), likely because drag variations are smaller and more dominated by mean flow. The ARIMA model yields an 
𝑅
2
 of only 
0.2195
, highlighting the advantage of recurrent neural networks for nonlinear, periodic signals.
Table 8. Model performance on downstream lift and drag prediction.
The dominant shedding frequency was extracted from the actual lift signal and from the LSTM predictions via fast Fourier transform. Both give a peak at 
2.041
 Hz, confirming that the LSTM correctly captures the periodic dynamics. Figure 34 shows the frequency spectra; the close agreement further validates the surrogate model.
Figure 34. Frequency spectrum of downstream lift: comparison between CFD data and LSTM predictions.
The probability distribution of the downstream lift coefficient is shown in Figure 35. The histogram confirms that the LSTM reproduces the full range of fluctuations. The phase-averaged lift coefficient (Figure 36) demonstrates that the LSTM captures the correct waveform shape and phase relationship.
Figure 35. Probability distribution of downstream lift.
Figure 36. Phase-averaged downstream lift coefficient: comparison between LSTM predictions and actual CFD data.
Residual analysis (Figure 37) shows that the errors are uncorrelated (within the 95% confidence bands), indicating that the LSTM has extracted all predictable information from the signal. The cross-correlation between upstream and downstream lift (Figure 40, left) reveals a strong positive correlation near zero lag, confirming that the wake of the upstream cylinder directly influences the downstream cylinder’s lift. The cross-correlation between LSTM predictions and actual lift (Figure 40, right) shows a sharp peak at zero lag, indicating that the model faithfully reproduces the temporal structure.
Figure 37. Autocorrelation of LSTM prediction residuals (95% confidence bands shown).
Figure 40. Cross-correlation analyses.
Figure 41 presents an extended forecast of the downstream lift coefficient for two seconds beyond the CFD simulation. The model continues the periodic pattern for several cycles before error accumulation; this demonstrates a practical prediction horizon of approximately 1.5s, which is valuable for control and reduced-order modeling applications.
Figure 41. Extended LSTM forecast of downstream lift (2s beyond CFD data).
Overall, the LSTM successfully learns the periodic dynamics of the downstream lift coefficient, as evidenced by high 
𝑅
2
 values, correct reproduction of the dominant frequency, and close agreement in phase space and probability distributions. The poor performance on drag suggests that drag is less deterministic in the turbulent regime, possibly due to higher-frequency fluctuations not captured by the RANS model. The cross-correlation analysis quantifies the wake interaction, and the error analysis provides insight into the model’s predictive limits. The open-source code and the trained models are available in the supplementary material, enabling future studies to use this surrogate for rapid parameter sweeps or reduced-order modeling.
5. Conclusions
This work presented a comprehensive numerical investigation of the flow past two cylinders in tandem arrangement using a fully open-source workflow. The complete pipeline from mesh generation with Gmsh to finite-volume simulation with OpenFOAM was described in detail, ensuring reproducibility and accessibility. The methodology was validated against the classical Schäfer–Turek single-cylinder benchmark at 
𝑅
𝑒
=
100
, showing reasonable agreement for the dominant shedding frequency and force coefficients, thus providing confidence in the solver’s capabilities.
The main analysis focused on a turbulent tandem configuration at 
𝑅
𝑒
=
1.0
×
10
5
 with an upstream cylinder of diameter 
𝐷
1
=
0.1
m and a downstream cylinder of larger diameter 
𝐷
2
=
0.15
m spaced 
1.0
m centre-to-centre. The results revealed strong wake interaction: the downstream cylinder experienced a higher mean drag (
𝐶
¯
𝐷
=
0.997
) and significantly larger lift fluctuations (
𝐶
𝐿
′
=
0.340
) than the upstream cylinder (
𝐶
¯
𝐷
=
0.947
, 
𝐶
𝐿
′
=
0.129
). Both cylinders shed vortices at the same frequency 
𝑓
=
2.041
Hz, yielding Strouhal numbers 
𝑆
𝑡
𝐴
=
0.204
 and 
𝑆
𝑡
𝐵
=
0.306
, which are physically realistic for a circular cylinder at this Reynolds number.
Beyond the conventional CFD analysis, a long short-term memory (LSTM) neural network was trained on the force coefficient time series to serve as a fast surrogate model. The LSTM accurately predicted the downstream lift coefficient (
𝑅
2
=
0.9895
, MAE=0.0308) and correctly reproduced the dominant shedding frequency, confirming the periodic dynamics. In contrast, a simple multi-layer perceptron and an ARIMA model performed poorly, highlighting the advantage of recurrent architectures for such nonlinear periodic signals. The surrogate model also enabled additional insights, including phase-averaged waveforms, cross-correlation analysis, and an estimate of the practical prediction horizon (approximately 1.5s).
All simulation setups, post-processing scripts, and the neural network training code have been made publicly available, allowing researchers to reproduce the results and adapt the workflow to other bluff-body configurations. This work thus contributes not only a detailed reference dataset for tandem cylinder flow at high Reynolds number but also a data-driven surrogate that can be used for rapid parametric studies, control applications, or reduced-order modeling.
Future work will extend the analysis to a wider range of spacing ratios and diameter disparities, incorporate more advanced turbulence models (e.g., LES or DES), and explore the use of physics-informed neural networks to further enhance the surrogate’s fidelity and generalizability.
Nomenclature
Table 9. List of symbols.
Table 10. List of abbreviations.
References
King, R. A review of vortex shedding research and its application. Ocean Engineering 1977, 4, 141–171. [Google Scholar] [CrossRef]
Williamson, C. K. Vortex dynamics in the cylinder wake. Annual Review of Fluid Mechanics 1996, 28, 477–539. [Google Scholar] [CrossRef]
Norberg, C. Flow around a circular cylinder: aspects of fluctuating lift. Journal of fluids and structures 2001, 15, 459–469. [Google Scholar] [CrossRef]
Zdravkovich, M. Review of flow interference between two circular cylinders in various arrangements. 1977. [Google Scholar] [CrossRef]
Sumner, D. Two circular cylinders in cross-flow: A review. Journal of fluids and structures 2010, 26, 849–899. [Google Scholar] [CrossRef]
Zhang, X.F.; Yang, J.C.; Ni, M.J.; Zhang, N.M.; Yu, X.G. Experimental and numerical studies on the three-dimensional flow around single and two tandem circular cylinders in a duct. Physics of Fluids 2022, 34. [Google Scholar] [CrossRef]
Wang, X.; Zhang, J.X.; Hao, Z.; Zhou, B.; Tan, S. Influence of wall proximity on flow around two tandem circular cylinders. Ocean Engineering 2015, 94, 36–50. [Google Scholar] [CrossRef]
Khan, H.H.; Islam, M.D.; Fatt, Y.Y.; Janajreh, I.; Alam, M.M. Flow-induced vibration on two tandem cylinders of different diameters and spacing ratios. Ocean Engineering 2022, 258, 111747. [Google Scholar] [CrossRef]
Schäfer, M.; Turek, S.; Durst, F.; Krause, E.; Rannacher, R. Benchmark computations of laminar flow around a cylinder. In Flow simulation with high-performance computers II: DFG priority research programme results 1993–1995; Springer, 1996; pp. 547–566. [Google Scholar]
Vu, H.C.; Ahn, J.; Hwang, J.H. Numerical simulation of flow past two circular cylinders in tandem and side-by-side arrangement at low Reynolds numbers. KSCE Journal of Civil Engineering 2016, 20, 1594–1604. [Google Scholar] [CrossRef]
Jiang, H.; Cheng, L.; Draper, S.; An, H. Two-and three-dimensional instabilities in the wake of a circular cylinder near a moving wall. Journal of Fluid Mechanics 2017, 812, 435–462. [Google Scholar] [CrossRef]
Liang, C.; Papadakis, G. Large eddy simulation of pulsating flow over a circular cylinder at subcritical Reynolds number. Computers & fluids 2007, 36, 299–312. [Google Scholar]
Elgendy, A.M. NUMERICAL INVESTIGATION OF CROSS-FLOW OVER OSCILLATING TANDEM CYLINDERS WITH TWO DEGREES OF FREEDOM. PhD thesis, Khalifa University of Science, 2024. [Google Scholar]
Xu, W.; Wu, H.; Jia, K.; Wang, E. Numerical investigation into the effect of spacing on the flow-induced vibrations of two tandem circular cylinders at subcritical Reynolds numbers. Ocean Engineering 2021, 236, 109521. [Google Scholar] [CrossRef]
Zhao, M.; Mamoon, A.A.; Wu, H. Numerical study of the flow past two wall-mounted finite-length square cylinders in tandem arrangement. Physics of fluids 2021, 33. [Google Scholar] [CrossRef]
Chang, X.; Yin, P.; Xin, J.; Shi, F.; Wan, L. Wake structures and hydrodynamic characteristics of flows around two near-wall cylinders in tandem and parallel arrangements. Journal of Marine Science and Engineering 2024, 12, 832. [Google Scholar] [CrossRef]
Nguyen, Q.D.; Lu, W.; Chan, L.; Ooi, A.; Lei, C. A state-of-the-art review of flows past confined circular cylinders. Physics of Fluids 2023, 35. [Google Scholar] [CrossRef]
Vu, V.T.; Duong, V.D.; Ngo, I.L. A comprehensive study of low-Reynold flow through two tandem cylinders with various configurations using the Lattice Boltzmann method. Marine Geophysical Research 2025, 46, 7. [Google Scholar] [CrossRef]
Kumar, R.A.; Kumar, K.S.; et al. Numerical Investigation of Wake Interference in Tandem Square Cylinders at Low Reynolds Numbers. Symmetry 2025, 17, 2038. [Google Scholar] [CrossRef]
Wang, J.; Shan, X.; Liu, J. First instability of the flow past two tandem cylinders with different diameters. Physics of Fluids 2022, 34. [Google Scholar] [CrossRef]
Zhang, L.; Zhang, Z.; Chen, W.; Srinil, N.; Zhu, H.; Bao, Y.; Ji, C. Flow-induced vibrations of ten tandem cylinders at low Reynolds number. Physics of Fluids 2023, 35. [Google Scholar] [CrossRef]
Sarkar, S.; Gupta, N.; Debnath, K.; Lawrence Raj, P.R. Computational Analysis of Flow Around Two Wall-Mounted Trapezoidal Bluff Bodies Arranged in Tandem Position. Journal of Fluids Engineering 2025, 147, 021301. [Google Scholar] [CrossRef]
Zhao, F.; Wang, R.; Zhu, H.; Cao, Y.; Bao, Y.; Zhou, D.; Cheng, B.; Han, Z. Free-surface effects on the flow around two circular cylinders in tandem. Journal of Fluid Mechanics 2024, 1001, A7. [Google Scholar] [CrossRef]
Brunton, S.L.; Noack, B.R.; Koumoutsakos, P. Machine learning for fluid mechanics. Annual review of fluid mechanics 2020, 52, 477–508. [Google Scholar] [CrossRef]
Raissi, M.; Perdikaris, P.; Karniadakis, G.E. Physics-informed neural networks: A deep learning framework for solving forward and inverse problems involving nonlinear partial differential equations. Journal of Computational physics 2019, 378, 686–707. [Google Scholar] [CrossRef]
Geuzaine, C.; Remacle, J.F. Gmsh: A Three-Dimensional Finite Element Mesh Generator, 2025. Version 4.12.
Launder, B.E.; Spalding, D.B. The numerical computation of turbulent flows. Computer Methods in Applied Mechanics and Engineering 1974, 3, 269–289. [Google Scholar] [CrossRef]
	
Disclaimer/Publisher’s Note: The statements, opinions and data contained in all publications are solely those of the individual author(s) and contributor(s) and not of MDPI and/or the editor(s). MDPI and/or the editor(s) disclaim responsibility for any injury to people or property resulting from any ideas, methods, instructions or products referred to in the content.
Copyright: This open access article is published under a Creative Commons CC BY 4.0 license, which permit the free download, distribution, and reuse, provided that the author and preprint are cited in any reuse.

Preprints.org is a free preprint server supported by MDPI in Basel, Switzerland.

Contact UsRSS
MDPI Initiatives

SciProfiles

Sciforum

Encyclopedia

MDPI Books

Scilit

Proceedings

JAMS

Important Links

Advisory Board

Collections

How It Works

Preprints Friendly Journals

Reading List

News

Statistics

2023 Popular Award

2025 Book Day Events

2016–2026 Popular Award

Subscribe

Choose an area of interest and we will send you notifications of new preprints at your preferred frequency.

Subscribe

© 2026 MDPI (Basel, Switzerland) unless otherwise stated

Disclaimer

Disclaimer

Terms of Use

Privacy Policy

Privacy Settings

© 2026 MDPI (Basel, Switzerland) unless otherwise stated