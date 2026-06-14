# Simulating gravitational dynamics via scalar field propagation - Frontiers
Source URL: https://www.frontiersin.org/journals/physics/articles/10.3389/fphy.2025.1672745/full

## ORIGINAL RESEARCH article

Front. Phys., 11 November 2025

Sec. Statistical and Computational Physics

Volume 13 - 2025 | https://doi.org/10.3389/fphy.2025.1672745

# Simulating gravitational dynamics via scalar field propagation

* BTBrendan Toupin  *

Brendan Toupin  *

* DIRECTV LLC, El Segundo, CA, United States

DIRECTV LLC, El Segundo, CA, United States

Article metrics

## Abstract

Introduction:

We study whether gravity-like kinematics (bending, time-delay, redshift-like shifts, capture/orbits) can arise as media analogs from a deterministic scalar-field propagation model without invoking mass or spacetime curvature.

Methods:

We evolve a real scalar field under a spatially varying symmetric positive-definite transport tensor R(x) and non-negative damping field Λ(x); with source off (S≡0). Thirteen simulations quantify deflection, transit delay with escape thresholds, collapse/trapping and orbital containment, anisotropy-induced drift, repulsion under curvature inversion, and interference. We monitor energy budgets (Rayleigh loss + boundary flux) and check spectral safety and robustness.

Results:

Observables are reproducible on 256 × 256 grids with 512 × 512 confirmations for key cases. Bending scales with ∥∇R∥ and flips sign under gradient reversal; transit delay increases monotonically with ∫Λdx and can prevent exit; bounded orbits satisfy a/p≤1.15 over a finite capture band; radial drift in 1/r2 profiles follows |r ̇|∝r^(-α) with a≈2; transverse drift sign matches sign(Rxy); interference visibility follows a cosine in relative phase.

Discussion:

Results constitute operational gravitational analogs—transport and loss in structured media—rather than statements about spacetime curvature. We release code/configs/outputs for full reproducibility and outline laboratory test paths.

Graphical Abstract

## 1 Introduction

Gravitational phenomena—trajectory bending, path-dependent time-delay, redshift-like frequency shifts, capture, and rebound—are traditionally explained via spacetime curvature and mass [1–3]. Here we ask a narrower, operational question: to what extent can the kinematics of such effects be reproduced as gravitational analogs by a deterministic scalar-field propagation model moving through a structured medium?

### 1.1 Model ingredients at a glance

We evolve a real scalar field  in two dimensions (and  for a single 3D scalability demonstration in the Supplement). A spatially varying, symmetric positive-definite transport (resistance) tensor  sets local propagation speed and directionality; its gradients and anisotropy bend paths and steer energy flux. A non-negative damping field  regulates loss and enables controlled sinks or absorbing layers. In homogeneous regions the directional effective speed along unit vector  isso spatial variation in  alone can generate curved characteristics, while  controls attenuation. The full evolution law, energy analysis, stability (CFL/Courant) bounds, and boundary conditions appear in Methods. This damped, anisotropic wave form is a generic effective model for transport in structured media (e.g., acoustics in inhomogeneous or lossy materials, metamaterial waveguides, or diffusion-wave hybrids). Here  encodes direction-dependent conductance (or stiffness), while  encodes local dissipation, providing a compact way to design and test kinematic analogs without invoking mass, force, or curvature.

### 1.2 Operational use of “analog”

We call an outcome a gravitational analog when the model reproduces the dimensionless kinematic observables of a target phenomenon (e.g., deflection angle, path-delay ratio, frequency-ratio shift) within stated tolerances—without asserting equivalence to Einstein’s equations or invoking spacetime curvature. For context, our benchmark observables refer to classic tests such as solar-limb deflection, radar-echo delay, and gravitational redshift [13–15].

### 1.3 Scope (what this paper is—and is not)

This study investigates kinematic analogs in a linear scalar-transport model. It does not solve Einstein’s field equations, include back-reaction of energy on geometry, or model gravitomagnetic effects arising from spacetime curvature. Conservation statements apply in uniform-,  subdomains; with , energy decays according to a derived law. We use idealized boundary conditions (reflective, absorbing, periodic) and disclose them in every figure. Unless stated otherwise, results are 2D.

### 1.4 Inverse design

While this work solves the forward problem (given ,  observed kinematics), the framework also invites the inverse question: given a target behavior—e.g., a bound trajectory with specified aphelion/perihelion ratio or a prescribed transit delay—what ,  fields realize it subject to smoothness and physicality constraints? Because observables are differentiable functionals of , gradient-based or bilevel schemes (forward solver + regularized optimizer) are natural next steps, enabling ‘transformation-acoustics-style’ design of analog gravitational media.

### 1.5 Relation to prior work

Methodologically, our approach is adjacent to analogue gravity in acoustics and optics—where structured media reproduce aspects of gravitational kinematics [4–9]—yet remains distinct from numerical relativity, which directly solves Einstein’s equations under gauge and constraint handling [10–12]. We use this literature to situate scope, not to claim equivalence.

### 1.6 Contributions

* Unified formulation and mathematical spine. We make explicit the governing evolution law, the associated energy functional and decay law, stability/Courant bounds under symmetric positive-definite , and boundary-condition treatments (Methods).

Unified formulation and mathematical spine. We make explicit the governing evolution law, the associated energy functional and decay law, stability/Courant bounds under symmetric positive-definite , and boundary-condition treatments (Methods).

* Thirteen simulations under one rule. We demonstrate bending (geodesic analog), Shapiro-like delay, redshift-like shifts, inverse-square-like radial drift, collapse-like trapping, rebound, interference, and related variants—each tied to a specific structure in  (Section 5).

Thirteen simulations under one rule. We demonstrate bending (geodesic analog), Shapiro-like delay, redshift-like shifts, inverse-square-like radial drift, collapse-like trapping, rebound, interference, and related variants—each tied to a specific structure in  (Section 5).

* Predictions and falsification. We define testable, dimensionless observables (deflection , delay ratio , frequency ratio , provide simple scaling relations, and state clear falsifiers (e.g., chromatic bending in a static ; rotation-induced effects in symmetric ) (Section 6).

Predictions and falsification. We define testable, dimensionless observables (deflection , delay ratio , frequency ratio , provide simple scaling relations, and state clear falsifiers (e.g., chromatic bending in a static ; rotation-induced effects in symmetric ) (Section 6).

* Reproducibility and robustness. We release code, configurations, and figure-regeneration scripts via public DOIs (Data and Code Availability). Robustness studies—larger grids, alternate sources and boundary conditions.

Reproducibility and robustness. We release code, configurations, and figure-regeneration scripts via public DOIs (Data and Code Availability). Robustness studies—larger grids, alternate sources and boundary conditions.

### 1.7 Paper organization

Section 2 surveys related work. Section 3 overviews the modeling ingredients and maps phenomena to transport/damping structures. Section 4 presents the governing equation, energy law, stability bounds, and boundary conditions. Section 5 reports thirteen simulations with standardized, dimensionless metrics. Section 6 gives benchmarks and falsification tests. Section 7 discusses scope and limitations. All data and code are archived on Zenodo; DOIs are listed in the Data Availability statement. Robustness checks and additional figures are provided in the Supplement.

## 2 Related work

Before presenting our model, we situate it among classical general relativity, analogue-gravity programs, and numerical relativity. Classical GR attributes gravitational phenomena to spacetime curvature sourced by stress–energy [1–3]; analogue gravity shows that structured media can reproduce many kinematic signatures (e.g., bending, delay) [4–9, 16–18]; numerical relativity solves Einstein’s equations directly in strong-field regimes [10–12]. Our contribution is a single-law, scalar-transport formulation that yields acceleration-like kinematics as analogs—curved trajectories and path-dependent delays—through spatially varying transport and damping fields, without solving Einstein’s equations. We quantify outcomes using dimensionless observables (deflection angle, delay ratio, frequency ratio) and state falsifiers, developed in Sections 5, 6.

### 2.1 General relativity and classic tests

General relativity (GR) explains gravitational phenomena as spacetime curvature sourced by stress–energy, with predictions verified from weak-to strong-field regimes [1–3, 20, 40]. The benchmark observables we reference—solar-limb light deflection, radar-echo time-delay, and gravitational redshift—are canonical GR tests [12–15, 49]. Our aim here is operational: reproduce these dimensionless kinematic observables as gravitational analogs using a linear scalar-transport model, without solving Einstein’s equations.

### 2.2 Analogue gravity: acoustics and optics

Analogue-gravity programs show how structured media can mimic geodesic-like transport. In acoustics, effective-metric ideas (Unruh; Visser) emulate horizons and geodesic behavior in flowing or inhomogeneous media [4, 5], with broad reviews by Barceló, Liberati and Visser [6, 7]. In optics, transformation-optics frameworks (Leonhardt; Pendry–Schurig–Smith) use spatially varying constitutive parameters to bend rays and shape phase fronts in ways formally analogous to geodesic transport [7, 15, 39]. Laboratory demonstrations include fiber-optic analogue horizons and related effects [8, 26]. Closely related graded-index (GRIN) constructs (e.g., the Luneburg lens; standard treatments in Born and Wolf) realize achromatic bending via smooth index profiles [16, 17, 27, 48].

Terminology crosswalk (reader note). Transformation-optics “effective metrics” and GRIN “index profiles” play roles analogous to our transport (resistance) tensor : all shape local propagation speed and directionality. We remain agnostic about emergent metrics and work directly with a symmetric positive-definite  and a non-negative damping field  (Methods).

### 2.3 Numerical relativity (contrast in scope)

Numerical relativity (NR) integrates Einstein’s equations with gauge/constraint handling to model strong-field spacetimes (e.g., binary black holes) [9–12]. By contrast, we evolve a single real scalar under spatially varying  and  to produce kinematic analogs of bending, delay, and frequency shifts. There is no methodological overlap: our results are analogs assessed by dimensionless observables, not GR parameter inference.

### 2.4 Boundary treatments and stability in wave simulations

Open-domain wave simulations rely on artificial boundary treatments. Classical absorbing layers and non-reflecting boundary conditions appear across acoustics, seismics, and electromagnetics (e.g., Cerjan et al.; Berenger’s PML) [18, 19, 34–36]. We use a tapered-damping (“sponge”) layer—implemented by smoothly increasing  near boundaries—which suppresses reflections and preserves simple energy accounting (Methods §4.5). Stability follows a CFL (Courant) bound tied to the largest eigenvalue of  and the grid spacings; explicit bounds and the time-stepping scheme are given in Methods.

### 2.5 Relation to general relativity (scope boundary)

We study operational analogs of gravity-like kinematics in structured media, not solutions of Einstein’s equations. Our evolution law is a damped, anisotropic wave transport model on a fixed Euclidean lattice; it does not include curvature, geodesics, or mass-energy back-reaction. GR is used as a yardstick for kinematic motifs (bending, time delay, precession), not as the theory being solved [21–25, 30–33, 43–46, 49].

### 2.6 Relation to analogue gravity

Our scope aligns with analogue gravity: reproducing curved-spacetime-like kinematics in non-gravitational media to probe mechanisms and guide experiments. Classic results (e.g., acoustic horizons) motivate the approach of designing media parameters to elicit GR-reminiscent observables without asserting spacetime curvature [4, 6, 7, 48].

### 2.7 Relation to numerical relativity

This is not numerical relativity: we do not evolve the Einstein field equations, solve constraints, or manage gauge/radiative boundaries. We evolve a single scalar field with prescribed  and  and compare the resulting observables to GR-like kinematics for intuition only [11, 45].

### 2.8 Predictive value and benchmarks

The framework yields design-forward mappings from media to observables:

* Bending (Section. 5.1): deflection vs. local gradients of .

Bending (Section. 5.1): deflection vs. local gradients of .

* Containment (Section 5.5): azimuthal drift  and radial period  vs. basin shape/smoothness. These predictions are suited to metamaterial/GRIN-style testbeds where ,  can be engineered. For verification, we provide 5122 repeats with matching metrics (Supplementary Appendix D-E) and exact configs in Section 9 [15].

Containment (Section 5.5): azimuthal drift  and radial period  vs. basin shape/smoothness. These predictions are suited to metamaterial/GRIN-style testbeds where ,  can be engineered. For verification, we provide 5122 repeats with matching metrics (Supplementary Appendix D-E) and exact configs in Section 9 [15].

* Discriminants (analogs vs. generic wave effects). We tag a behavior as a gravity-like analog only when it (i) depends monotonically on a controlled feature of ,  (e.g., bend ∝ local  magnitude/direction), (ii) survives nuisance changes (e.g., modest window/crop, sampling), and (iii) fails under ablation (e.g., no bend when  is spatially constant; no collapse-like decay when ). These controls separate designed kinematics from generic diffraction/refraction. Cross-referenced ablations, definitions, and scripts are in Supplementary Appendix C and Section 9.

Discriminants (analogs vs. generic wave effects). We tag a behavior as a gravity-like analog only when it (i) depends monotonically on a controlled feature of ,  (e.g., bend ∝ local  magnitude/direction), (ii) survives nuisance changes (e.g., modest window/crop, sampling), and (iii) fails under ablation (e.g., no bend when  is spatially constant; no collapse-like decay when ). These controls separate designed kinematics from generic diffraction/refraction. Cross-referenced ablations, definitions, and scripts are in Supplementary Appendix C and Section 9.

### 2.9 Limitations and differences

* No curvature/no EEP or PPN tests: we do not test GR, PPN parameters, or the equivalence principle [46, 47].

No curvature/no EEP or PPN tests: we do not test GR, PPN parameters, or the equivalence principle [46, 47].

* Media analogs only: any frame-drag-like effects in §5.7 are media phenomena, not GR gravitomagnetism [48].

Media analogs only: any frame-drag-like effects in §5.7 are media phenomena, not GR gravitomagnetism [48].

* Dissipation by design:  models physical loss (absorbers/sponge); energy accounting follows Supplementary Appendix C [41].

Dissipation by design:  models physical loss (absorbers/sponge); energy accounting follows Supplementary Appendix C [41].

* Forward design, not inverse GR: we design ,  to achieve target kinematics; we do not infer spacetime metrics from data [38].

Forward design, not inverse GR: we design ,  to achieve target kinematics; we do not infer spacetime metrics from data [38].

### 2.10 Summary positioning and predicted observables

* Theory anchors: GR sets the gold standard for gravitational dynamics [1–3, 10–12].

Theory anchors: GR sets the gold standard for gravitational dynamics [1–3, 10–12].

* Method lineage: Analogue-gravity shows that structured media can reproduce many kinematic signatures without GR dynamics [4–9, 16–18].

Method lineage: Analogue-gravity shows that structured media can reproduce many kinematic signatures without GR dynamics [4–9, 16–18].

* Our contribution: a single-law scalar-transport formulation that (i) makes those analogs explicit in terms of  and ; (ii) quantifies outcomes via deflection angle , delay ratio , and frequency ratio ; and (iii) states falsifiers (e.g., chromatic bending with static ; rotation-induced signatures in symmetric ), expanded in Section 6.

Our contribution: a single-law scalar-transport formulation that (i) makes those analogs explicit in terms of  and ; (ii) quantifies outcomes via deflection angle , delay ratio , and frequency ratio ; and (iii) states falsifiers (e.g., chromatic bending with static ; rotation-induced signatures in symmetric ), expanded in Section 6.

## 3 Gravity-like behavior as emergent propagation in structured fields

Traditional theories attribute gravitational acceleration to mass—either via long-range forces (Newtonian mechanics) or spacetime curvature (general relativity) [6, 7, 48]. In both, mass–energy is the source term. Operationally, however, what is measured are kinematic outcomes—deflected paths, path-dependent time-delays, frequency shifts. This suggests a complementary question: can gravity-like kinematics arise as analogs from structured propagation alone, without solving Einstein’s equations [28, 29, 37]?

We explore this possibility with a constructive, deterministic model in which gravitational analogs emerge from scalar-field transport modulated by two spatial structures [

,

]:

* A resistance field  (symmetric positive-definite), which sets local propagation speed and directionality; its gradients and anisotropy steer energy flux and bend characteristics, and

A resistance field  (symmetric positive-definite), which sets local propagation speed and directionality; its gradients and anisotropy steer energy flux and bend characteristics, and

* A damping field , which introduces controlled loss, enabling localized sinks and absorbing boundary layers.

A damping field , which introduces controlled loss, enabling localized sinks and absorbing boundary layers.

The real scalar field

volves under these structures according to a second-order update law (given explicitly in Methods, §4). Together,

and

form a structured substrate that steers, delays, or attenuates propagation. In this substrate we observe the following operational analogs of familiar gravitational effects:

* Acceleration-like drift toward high-delay regions. Packets exhibit net drift toward zones that increase cumulative travel-time (via  sinks or graded ), producing sustained, direction-biased motion without external forcing [35].

Acceleration-like drift toward high-delay regions. Packets exhibit net drift toward zones that increase cumulative travel-time (via  sinks or graded ), producing sustained, direction-biased motion without external forcing [35].

* Curved trajectories (geodesic analogs) from . Smooth spatial gradients of  bend characteristics and focus/defocus packets, including lensing-like patterns [17, 18].

Curved trajectories (geodesic analogs) from . Smooth spatial gradients of  bend characteristics and focus/defocus packets, including lensing-like patterns [17, 18].

* Redshift-like frequency changes. Weak gradients in  across a cavity or standing-wave region yield measurable frequency-ratio shifts [6, 7].

Redshift-like frequency changes. Weak gradients in  across a cavity or standing-wave region yield measurable frequency-ratio shifts [6, 7].

* Escape thresholds from integrated delay. Sufficient cumulative delay (from  wells or sinks) produces capture-vs-escape thresholds analogous to potential-well intuition.

Escape thresholds from integrated delay. Sufficient cumulative delay (from  wells or sinks) produces capture-vs-escape thresholds analogous to potential-well intuition.

We quantify these outcomes by dimensionless observables—deflection angle

, delay ratio

, and frequency ratio

—and report them for every simulation (

). In the linear regime we use frequency-independent

; predicted bending and delay are therefore achromatic. Observation of chromatic bending with static

would falsify this description (

).

* Scope note. We seek kinematic analogs, not equivalence to curvature dynamics. The model does not include spacetime curvature, back-reaction of energy on geometry, or gravitomagnetic effects. Conservation statements apply in uniform-,  regions; with , energy decays according to a derived law (Methods, §4).

Scope note. We seek kinematic analogs, not equivalence to curvature dynamics. The model does not include spacetime curvature, back-reaction of energy on geometry, or gravitomagnetic effects. Conservation statements apply in uniform-,  regions; with , energy decays according to a derived law (Methods, §4).

This approach is simulatable, constructive, and testable. It models gravity-like behavior from first principles using only scalar transport with locally specified  and , providing a concrete foundation for the evolution law, energy analysis, stability bounds, and boundary treatments presented next (Methods, §4). Unless stated otherwise, results are 2D.

## 4 Methods

This section makes the modeling contract explicit. We specify the field, domain, and notation; state the governing equation; derive the energy and decay law; interpret the tensor-divergence (anisotropy/steering); and give the discrete scheme, stability bound, and boundary/initial conditions. The goal is a paper-faithful, constructive recipe: every simulation in Section 5 can be regenerated from these ingredients without hidden parameters.

### 4.1 Transparency and materials

The full simulation engine, discretization details, update rule, and example YAMLs/outputs are archived (Section 9). Implementation specifics—including stencil choices, stepper policy, and figure scripts—are documented in Supplementary Appendix C and mirrored in the software record.

### 4.2 Fields and assumptions

We model a scalar  obeying , where  is symmetric positive-definite (SPD) and  is a scalar (or diagonal) loss field. Unless noted, ,  are time-independent and piecewise-smooth; typical forms include (i) radially symmetric wells , (ii) anisotropic basins with off-diagonal coupling, and (iii) thin absorbing aprons  at the boundary. Nondimensionalization and units are specified in Supplementary Appendix C.

### 4.3 Notation and domain

We evolve a real scalar field  on a rectangular domain . For a single scalability demonstration in the Supplement we use  on . The transport (resistance) tensor is , assumed symmetric positive-definite (SPD) everywhere; the damping field is . Bold symbols denote vectors; indices  with  in the main text and  in the Supplement [42].

### 4.4 Boundary conditions

We use two BC families: (i) reflective (Neumann-type) for the core region in containment tests, and (ii) absorbing aprons (thin  sponge) to remove far-field clutter. The energy identity above shows how BCs enter via the surface flux term; reflective cores null the flux, while absorbing aprons intentionally dissipate outgoing energy. Each figure caption and YAML specifies the BC choice [18, 19].

### 4.5 Governing equation of motion

The field obeys a linear, second-order evolution law

with time-independent  (SPD) and . In a homogeneous, undamped region , plane waves satisfy ; the directional effective speed (Equation 4.2) along unit vector  is

[38]. Thus gradients and anisotropy of  bend characteristics and steer energy flux;  regulates loss.

### 4.6 Energy functional and decay law

Define the energy density  and flux  (Equation 4.3); the global identity appears in Equation 4.4:

Multiplying (4.1) by , integrating by parts, and using time-independent coefficients gives

Consequences. In undamped, closed subdomains  the energy is conserved. For  the energy decays monotonically aside from boundary flux. We report energy budgets per run (and reflection fractions for absorbing boundaries).

#### 4.6.1 Damping is not potential/curvature

introduces loss, not forces or curvature; collapse-like behavior arises from dissipation and resistance shaping, not from a gravitational potential. For numerical robustness,  is kept non-negative and typically smoothed across a few grid cells to avoid stair-step reflections; sharp discontinuities may cause artificial echoes and are avoided in the released configs.

#### 4.6.2 Energy identity (summary

For time-independent  and , define (Equation 4.5)

Multiplying the evolution law by , integrating, and using the divergence theorem gives

i.e., monotone decay from damping () plus any boundary flux. With reflective core BCs the surface term vanishes; with a thin absorbing apron it captures intended outflow. The discrete energy  we monitor follows the same structure (Supplementary Appendix C.1), matching the trends reported in §5 and Supplementary Appendix D, E. (Full derivation and the time-dependent  extension are in Supplementary Appendix C.5) [41]

### 4.7 Tensor divergence and anisotropy (interpretation)

The operator expands component-wise as

so diagonal terms  set directional speeds and off-diagonals  rotate flux, steering characteristics. SPD  guarantees real, bounded  [35].

### 4.8 Discretization and time stepping

#### 4.8.1 Spatial discretization (conservative divergence form)

On a uniform Cartesian grid with spacings  and cell-centered , we discretize  in flux-conservative form (Equations 4.6–4.8). In 2D:

with face-averaged coefficients  (arithmetic or harmonic) to preserve symmetry and discrete conservation [35, 36]. The 3D stencil is analogous.

#### 4.8.2 Time integration (damping-stable, second order)

Let . We use a leapfrog-type update with semi-implicit damping (Crank–Nicolson split), which is unconditionally stable in  while keeping transport explicit [36]:

#### 4.8.3 Stability (CFL) bound and coefficient conditions

Let  be the maximum eigenvalue of  on . A sufficient CFL bound is

[34]. We enforce SPD bounds  and . Discontinuities in  are tapered to limit numerical reflections (Section 4.6.2).

### 4.9 Boundary and initial conditions

#### 4.9.1 Reflective (Neumann, no-flux)

#### 4.9.2 Absorbing sponge (tapered damping)

To emulate open boundaries we use a smoothly increasing  within a shell of thickness  adjacent to :

where  is the distance to the boundary. Outside the sponge . We report the reflected-energy fraction (target: ) for absorbing runs [35, 36].

#### 4.9.3 Periodic

Variables and fluxes wrap across opposing faces identically.

#### 4.9.4 Initial data

We use localized pulses (Gaussian, Ricker), narrowband wave packets, and cavity modes as specified per figure. Each caption reports the source definition and parameters.

### 4.10 Dimensionless observables (measurement procedures)

We evaluate outcomes via dimensionless kinematic observables reported in captions and summarized in Section 5.

(a) Deflection angle  (bending/geodesic analog). Let  be the packet centroid,

and  the unit tangent to its path in the far field (measured over a window where  is homogeneous and ). If  is the initial propagation direction, define [17, 18]

(b) Delay ratio  (Shapiro-like delay). With and without a delay structure in , record arrival times  and ​ at a downstream plane; let  be the homogeneous-medium path length. Then

(c) Frequency ratio  (redshift-like shift). For a cavity or standing-wave region straddling a weak gradient in , compute spectral peaks  and  from local time-series on either side and report the ratio.

Achromaticity. In the linear regime we use frequency-independent ; thus predicted bending and delay are achromatic. Observation of chromatic bending with static  would falsify this description (see Section 6) [6, 7].

### 4.11 Stability and δt policy (CFL)

With explicit second-order time stepping, stability follows a CFL-type bound determined by the discrete spatial operator for . Let ​ denote that operator on the chosen grid; then a sufficient bound iswhere  is the spectral radius and  is a safety factor (recorded per run). Equivalently, on a uniform grid with spacing  and SPD  with largest eigenvalue , a practical estimate is

with  a scheme-dependent constant absorbed into  [34, 36]. We constrain the fields to

and use an auto-CFL policy (safety factor ) whose chosen  is written to each run’s metadata. See Supplementary Appendix C for operator definitions and how  is estimated in practice.

### 4.12 Reporting standards (reproducibility hygiene)

Every figure/caption states: grid , time step , CFL margin (ratio to the bound in (4.5.3)), boundary condition (reflective/absorbing/periodic), explicit  forms, run duration, and the measured observable(s) . Energy budgets and (for sponges) reflection fractions appear in the Supplement.

### 4.13 Provenance and versioning

All main-text figures were recomputed with an updated implementation (EOM-v1) of the governing Equation 4.1. On the original configurations from the reviewed submission, EOM-v1 reproduces the reported dimensionless observables—deflection angle , delay ratio , and frequency ratio —within . We archive the original submission’s figure files and their exact configuration files for provenance; the executable EOM-v1 code and “regen-all” scripts are provided via DOI in Data and Code Availability.

## 5 Simulation results: Gravitational behavior from structured fields

This section reports operational analogs of gravitational phenomena produced by a scalar field  evolving under anisotropic transport  and Rayleigh-type damping . We focus on observables, validation, and reproducibility; the full PDE, discretization, stability bounds, and energy identities are in §4. For every run we report the discrete energy proxy  and attribute changes per the identity in §4.3 (derivation App. C.1).

Notation and dimensionality. We write  with . Unless stated, runs are 2-D with ; selected confirmations are 3-D with  and are labeled “(3-D)” in captions and the table. We follow §4:  with ​; . Measures use .

Code and data (reproducibility). All §5 configs (YAML), engine source, and outputs (.npz recorders with fields + metrics) are archived with commit hashes at < DOI/URL>. Each figure caption lists the config slug, grid(s), , and the bundle ID.

Acceptance gates (applied to every §5. x).

* Energy budget closure after transients () with tallied Rayleigh loss and boundary flux (definitions in §4).

Energy budget closure after transients () with tallied Rayleigh loss and boundary flux (definitions in §4).

* The section’s primary metric meets its pre-registered threshold.

The section’s primary metric meets its pre-registered threshold.

* Robustness across grid size (; selected 3-D where noted), seed shape, and relevant boundary swaps.

Robustness across grid size (; selected 3-D where noted), seed shape, and relevant boundary swaps.

* Spectral safety: content remains sub-Nyquist (anti-aliasing guard).Predictions and falsifiers. Each §5. x states a concrete prediction for its primary metric and a matching falsifier; global statements are summarized in §4.

Spectral safety: content remains sub-Nyquist (anti-aliasing guard).Predictions and falsifiers. Each §5. x states a concrete prediction for its primary metric and a matching falsifier; global statements are summarized in §4.

Boundary conditions (policy). Absorbing sponges for open domains, reflective for containment basins, periodic for controls; flux tallies verify low reflection (see §4). Profiles with interfaces are -smoothed over  unless intentionally sharp; measured reflection is reported when interfaces are sharp by design.

Grid sizes and robustness. Figures in §5 use 2562 grids unless labeled; 5122 repeats for free-fall (§5.1) and containment (§5.5) are reported in Supplementary Appendix D,E. Boundary variants (reflective core + absorbing apron) are included; additional pulse-shape sweeps are earmarked for follow-on work.

Scope and limits. Results are media analogs arising from structured propagation in —not claims of mass, forces, or spacetime curvature. Comparisons to geometric-optics/eikonal predictions for  are treated as observable mappings, not equivalences (see §4). Damping is a loss channel, not a potential: collapse-like outcomes here arise from focusing in  plus dissipation in , not from forces or curvature.

Falsification routes. Each case in §5 defines a primary observable and an acceptance gate. A reproduction fails if (i) the observable falls outside the gate under the published YAML and seed, (ii) prescribed ablations (e.g., flatten , set ) do not suppress the effect, or (iii) grid refinement (Supplementary Appendix D,E) reverses qualitative behavior.

How to read §5. Each subsection states the objective and minimal setup (domain, , , BCs, seed), declares the primary metric and threshold, and reports results with robustness checks. Every run includes energy budgets (kinetic, structural, Rayleigh loss, boundary flux). Genesis is off unless explicitly stated.

A summary of all cases appears in Table 1. Figure conventions. Panels typically include: (A) timeline montage; (B) geometry/path; (C) energy budget; (D) the primary metric with acceptance band; (E) a sweep (ICs or profile). Captions include grid(s), , config slug, and hashes.

TABLE 1

Section | Phenomenon | Primary Metric | Key profiles , | Config slug
5.1 | Free-fall from structural asymmetry | Lateral deflection vs.  (linear, sign-correct) | Diagonal  with weak monotone gradient | grav_5_1_free_fall.yml
5.2 | Collapse/sink (dissipative focusing) | Time-to-collapse vs.  depth and focusing strength (decreasing) | Central  well with focusing | grav_5_2_collapse_sink.yml
5.3 | Geodesic-like convergence | Centerline curvature vs.  gradient; residuals to eikonal fit | Smoothly graded | grav_5_3_geodesic_convergence.yml
5.4 | Escape threshold and “redshift-like” delay | Transit delay vs.  (monotone); escape map | Slab with  ramp; near-uniform | grav_5_4_escape_redshift.yml
5.5 | Orbital containment (limit cycle) | Bounded radius ; flat ; budget closure | Annular  support +  ring (radial loss) | grav_5_5_orbital_containment
5.6 | Equivalence-/inertial-like response | Path overlap of packets under amplitude/width scaling ( threshold)e | Uniform/weakly graded ;  as noted | grav_5_6_equivalence_inertial.yml
5.7 | Directional drift (anisotropy-induced) | Lateral drift rate vs. anisotropy/off-diag in | with controlled anisotropy | grav_5_7_directional_drift.yml
5.8 | Curvature without coordinates | Extrinsic curvature  vs. designed  pattern | Spatially varying | grav_5_8_curvature_without_coords.yml
5.9 | Local collapse trap | Capture probability vs. well depth/width | Local  well embedded in smooth | grav_5_9_local_collapse_trap.yml
5.10 | Reversible rebound (conservative basin) | Restitution coefficient; repeatability ( threshold) | Conservative  basin | grav_5_10_reversible_rebound.yml
5.11 | Inverse-square-like drift | Capture radius or drift trend vs. radial | Radial  gradient; small  for noise control | grav_5_11_inverse_square_drift.yml
5.12 | Repulsion (curvature-inversion analog) | Divergence of trajectories vs. sign of | gradient sign-reversed | grav_5_12_repulsion_inversion.yml
5.13 | Interference and stacking | Contrast vs. initial phase ; budget integrity | Two coherent seeds; uniform ; low | grav_5_13_interference_stacking.yml

Simulation suite overview: phenomenon, primary metric, key profiles R,Λ, and configuration slug for Figures 1–13.

Each §5. x details its primary metric, prediction and falsifier, and gates. Acceptance also requires budget closure, spectral safety, and robustness across grid, seed, and boundary swaps.

Seeds are compact packets (Gaussian/top-hat variants) unless stated. Profiles are  to minimize artificial reflections; when sharp interfaces are intentional, measured reflection coefficients are reported.

Baseline runs are 2562; selected 5122 and (3-D) confirmations are labeled where applicable.

### 5.1 Free-fall acceleration from structural asymmetry

#### 5.1.1 Objective

Demonstrate that a compact packet acquires a systematic lateral deflection when traversing a weak spatial gradient in  with . In the weak-gradient regime the primary metric—net bend angle —is expected to be linear in the gradient magnitude and to flip sign when the gradient is reversed (see §4 for derivation/limits).

#### 5.1.2 Minimal setup

* Domain and BCs: 2-D grid (2562), absorbing boundaries with graded sponge; boundary flux tallied (§4).

Domain and BCs: 2-D grid (2562), absorbing boundaries with graded sponge; boundary flux tallied (§4).

* Profiles:  with a weak,  monotone ramp in ​ constant. .

Profiles:  with a weak,  monotone ramp in ​ constant. .

* Seed: Compact packet launched straight across the gradient (zero initial lateral velocity).

Seed: Compact packet launched straight across the gradient (zero initial lateral velocity).

* Genesis: Off.

Genesis: Off.

* Config: grav_5_1_free_fall.yml (commit/hash in caption).

Config: grav_5_1_free_fall.yml (commit/hash in caption).

#### 5.1.3 Primary metric and gates

* Metric: Bend angle  from the centroid path, estimated by a straight-line fit of  over the middle segment of the transit (method in §4).

Metric: Bend angle  from the centroid path, estimated by a straight-line fit of  over the middle segment of the transit (method in §4).

* Acceptance gates (this subsection):1. Non-zero, sign-correct  under the applied gradient;2. Energy budget closure within 1%–3% post-transient, with Rayleigh loss = 0 and decline explained by boundary flux;3. Spectral safety (sub-Nyquist content);4. Robustness: reproducible under seed-shape variant; grid-refinement confirmation at 5122 provided in Supplementary Appendix D.

Acceptance gates (this subsection):

* 1. Non-zero, sign-correct  under the applied gradient;

1. Non-zero, sign-correct  under the applied gradient;

* 2. Energy budget closure within 1%–3% post-transient, with Rayleigh loss = 0 and decline explained by boundary flux;

2. Energy budget closure within 1%–3% post-transient, with Rayleigh loss = 0 and decline explained by boundary flux;

* 3. Spectral safety (sub-Nyquist content);

3. Spectral safety (sub-Nyquist content);

* 4. Robustness: reproducible under seed-shape variant; grid-refinement confirmation at 5122 provided in Supplementary Appendix D.

4. Robustness: reproducible under seed-shape variant; grid-refinement confirmation at 5122 provided in Supplementary Appendix D.

#### 5.1.4 Results (2562 main run)

A small, sign-consistent bend accumulates across the graded region; from the centroid path we obtain  for this run. Energy decreases smoothly due to absorbing boundaries; with , Rayleigh loss is zero and the discrete budget closes within tolerance. Optional checks (not shown) confirm sign flip under gradient reversal and  for the uniform control .

Interpretation. The “free-fall” is an analog arising from spatial inhomogeneity of : rays refract toward slower directions (lower effective transport), consistent with the eikonal picture for  (see §4). No forces or curvature are invoked.

#### 5.1.5 Falsification route

* Reverse the gradient:  must change sign.

Reverse the gradient:  must change sign.

* Null profile  must be within the null noise band. Failure of either falsifier invalidates the claim for this setup.

Null profile  must be within the null noise band. Failure of either falsifier invalidates the claim for this setup.

Repro bundle. Figure assets and recorder outputs (.npz/.csv) for Figure 1 are archived with engine commit < hash> and bundle ID < ID>; see Data and Code Availability.

FIGURE 1

Free-fall from structural asymmetry. A compact packet traverses a weak  ramp in  with  and absorbing boundaries. Timeline of . Final frame of . The packet acquires a small, sign-consistent bend; from the centroid path we measure  (method §4, mid-segment linear fit of ). Energy declines monotonically due to boundary absorption; Rayleigh loss = 0, and the discrete budget closes within 1%–3% after transients (§4 identity). Falsifier: reversing the  gradient must flip ; the null profile  must yield . Config grav_5_1_free_fall.yml; 2-D 2562;  auto (CFL); sponge parameters as in §4. Grid refinement: a 5122 repeat reproduces  within error (Supplementary Appendix D)

### 5.2 Collapse/sink (dissipative focusing)

#### 5.2.1 Objective

Show that a compact packet undergoes irreversible collapse/trapping when traversing a region that combines focusing transport  with positive damping . The observable is a rapid increase of core energy fraction within an inner mask and a monotone energy decay explained by Rayleigh loss + boundary flux (no potential energy is invoked).

#### 5.2.2 Minimal setup

* •Domain and BCs: 2-D grid (2562), absorbing boundaries with graded sponge; boundary flux tallied (§4).

•Domain and BCs: 2-D grid (2562), absorbing boundaries with graded sponge; boundary flux tallied (§4).

* Profiles: Radially focusing  (SPD, smooth), plus a central  well whose depth increases toward ; both profiles are -smoothed over .

Profiles: Radially focusing  (SPD, smooth), plus a central  well whose depth increases toward ; both profiles are -smoothed over .

* Seed: Compact packet launched toward the well center.

Seed: Compact packet launched toward the well center.

* •Genesis: Off.

•Genesis: Off.

* Config: grav_5_2_collapse_sink.yml (commit/hash in caption).

Config: grav_5_2_collapse_sink.yml (commit/hash in caption).

#### 5.2.3 Primary metric and gates

* Metric (collapse time TcT_cTc). Let  be the fraction of total field energy inside an inner disk of radius  (specified in caption). Define ​ as the earliest time such that  for at least  consecutive recorder frames (e.g., ).

Metric (collapse time TcT_cTc). Let  be the fraction of total field energy inside an inner disk of radius  (specified in caption). Define ​ as the earliest time such that  for at least  consecutive recorder frames (e.g., ).

* Acceptance gates (this subsection):1. Monotone rise of  crossing the threshold and staying above it (collapse achieved);2. Energy budget closure within 1%–3% post-transient, with Rayleigh loss  and boundary flux accounting for all decay (definitions/identity in §4);3. Spectral safety (sub-Nyquist content);4. Robustness: reproducible under small  changes and seed-shape variants; 5122 confirmation in Supplementary Appendix D reproduces  within error.

Acceptance gates (this subsection):

* 1. Monotone rise of  crossing the threshold and staying above it (collapse achieved);

1. Monotone rise of  crossing the threshold and staying above it (collapse achieved);

* 2. Energy budget closure within 1%–3% post-transient, with Rayleigh loss  and boundary flux accounting for all decay (definitions/identity in §4);

2. Energy budget closure within 1%–3% post-transient, with Rayleigh loss  and boundary flux accounting for all decay (definitions/identity in §4);

* 3. Spectral safety (sub-Nyquist content);

3. Spectral safety (sub-Nyquist content);

* 4. Robustness: reproducible under small  changes and seed-shape variants; 5122 confirmation in Supplementary Appendix D reproduces  within error.

4. Robustness: reproducible under small  changes and seed-shape variants; 5122 confirmation in Supplementary Appendix D reproduces  within error.

#### 5.2.4 Results (2562 main run)

The packet is drawn inward by the focusing ; radial components are preferentially eliminated by , and the field locks into a compact core.  crosses the acceptance threshold and remains high thereafter, while total energy decreases smoothly. The Rayleigh tally is strictly positive and, together with boundary flux, explains the full budget drop; the discrete energy identity from §4 holds within the stated tolerance.

Interpretation. Collapse here is a deterministic analog of trapping from focusing + dissipation:  steers energy inward;  irreversibly removes radial motion. This is not a gravitational potential well: damping is a loss channel, not a stored energy term (limits discussed in §4).

#### 5.2.5 Falsification route

* Remove  (negative control): with  focusing but , the packet must fail to achieve irreversible collapse (rebound/breathing expected).

Remove  (negative control): with  focusing but , the packet must fail to achieve irreversible collapse (rebound/breathing expected).

* Flatten  (transport control): with  present but  uniform, the packet must not focus sharply nor meet the  gate. Failure of either control invalidates the claim for this setup.

Flatten  (transport control): with  present but  uniform, the packet must not focus sharply nor meet the  gate. Failure of either control invalidates the claim for this setup.

Repro bundle. Figure assets and recorder outputs (.npz/.csv with , energy tallies) for Figure 2 are archived with engine commit < hash> and bundle ID < ID>; see Data and Code Availability.

FIGURE 2

Collapse/sink from dissipative focusing. A compact packet encounters a focusing transport field  and a central damping well  (both -smoothed). Timeline of  showing inward focusing and core formation. Final frame. Collapse is certified when the core energy fraction  within radius ​ exceeds 0.80 for  frames; this run passes the gate with collapse time ​ (reported in the data bundle). The total energy decays monotonically; Rayleigh loss  and boundary flux together account for the decrease, satisfying the discrete identity from §4 (budget drift ≤ 1–3% post-transient). Falsifiers: (i) with  focusing but , collapse must not persist (rebound/breathing control); (ii) with  present but  flattened, focusing must not achieve the gate. Config grav_5_2_collapse_sink.yml; 2-D 2562;  auto (CFL); sponge parameters as in §4. Grid refinement: a 5122 repeat reproduces  within error (Supplementary Appendix D).

### 5.3 Ray-like bending in a graded medium (geodesic-analog convergence)

#### 5.3.1 Objective

Show that a compact packet follows a ray-like path through a smoothly graded , consistent with the geometric-optics/eikonal prediction derived from §4. The observable is a centerline trajectory whose bending is sign-correct and whose path residual against the eikonal ray stays within a small tolerance (operational “geodesic-analog” behavior).

#### 5.3.2 Minimal setup

* Domain and BCs: 2-D grid (2562), absorbing boundaries with graded sponges; boundary flux tallied (§4).

Domain and BCs: 2-D grid (2562), absorbing boundaries with graded sponges; boundary flux tallied (§4).

* Profiles: Smooth, -graded transport  (monotone across one axis or radially focusing/defocusing, as noted in caption). .

Profiles: Smooth, -graded transport  (monotone across one axis or radially focusing/defocusing, as noted in caption). .

* Seed: Compact packet launched to traverse the gradient at a shallow incidence (quasi-ray).

Seed: Compact packet launched to traverse the gradient at a shallow incidence (quasi-ray).

* Config: grav_5_3_geodesic_convergence.yml (commit/hash in caption).

Config: grav_5_3_geodesic_convergence.yml (commit/hash in caption).

#### 5.3.3 Primary metric and gates

* Metric (ray agreement). Extract the packet centerline  (centroid path) and compare to the eikonal ray ​ computed from the  profile (procedure in §4). Report the RMS path residual

Metric (ray agreement). Extract the packet centerline  (centroid path) and compare to the eikonal ray ​ computed from the  profile (procedure in §4). Report the RMS path residual

Normalized by the path length LLL, and verify sign-correct bending when the gradient is reversed.

* •Acceptance gates (this subsection): within a pre-registered tolerance (small fraction of domain width);Energy budget closure within 1%–3% after transients; with , loss arises from boundary flux only;Spectral safety (sub-Nyquist content);Robustness: unchanged within error under seed-shape variant and modest apron changes; 5122 confirmation provided in Supplementary Appendix E reproduces  within error.

•Acceptance gates (this subsection):

* within a pre-registered tolerance (small fraction of domain width);

within a pre-registered tolerance (small fraction of domain width);

* Energy budget closure within 1%–3% after transients; with , loss arises from boundary flux only;

Energy budget closure within 1%–3% after transients; with , loss arises from boundary flux only;

* Spectral safety (sub-Nyquist content);

Spectral safety (sub-Nyquist content);

* Robustness: unchanged within error under seed-shape variant and modest apron changes; 5122 confirmation provided in Supplementary Appendix E reproduces  within error.

Robustness: unchanged within error under seed-shape variant and modest apron changes; 5122 confirmation provided in Supplementary Appendix E reproduces  within error.

#### 5.3.4 Results (2562 main run)

The packet bends toward decreasing effective transport as it crosses the gradient, and the measured centerline closely tracks the eikonal prediction from §4. The RMS path residual ​ remains within the acceptance tolerance; reversing the gradient (control, not shown) flips the bending sign. Total energy decays smoothly due to absorbing boundaries; Rayleigh loss = 0, and the discrete budget closes within tolerance.

Interpretation. The observed path is a media analog of a geodesic: bending emerges from spatial variation of  via ray refraction in the geometric-optics limit, not from forces or curvature (scope/limits in §5 intro; derivation in §4).

#### 5.3.5 Falsification route

* Gradient reversal: bending must flip sign.

Gradient reversal: bending must flip sign.

* Null profile: with , , the path must be straight within the null noise band.

Null profile: with , , the path must be straight within the null noise band.

* Ray mismatch: ​ exceeding tolerance falsifies ray agreement for this setup.

Ray mismatch: ​ exceeding tolerance falsifies ray agreement for this setup.

Repro bundle. Figure assets and recorder outputs (.npz/.csv with centerline and eikonal-ray data) for Figure 3 are archived with engine commit < hash> and bundle ID < ID>; see Data and Code Availability.

FIGURE 3

Ray-like bending in a graded medium (geodesic-analog convergence). A compact packet traverses a smooth  gradient in  with  and absorbing boundaries. Timeline of . Final frame. The centerline follows the eikonal ray predicted from the  profile (procedure in §4); the RMS path residual  remains within tolerance, and bending is sign-correct. Energy declines monotonically due to boundary absorption; Rayleigh loss = 0, and the discrete budget closes within 1%–3% post-transient (§4 identity). Falsifiers: reversing the gradient must flip the bending sign; the null profile ,  must yield a straight path within noise. Config grav_5_3_geodesic_convergence.yml; 2-D 2562;  auto (CFL); sponge parameters as in §4. Grid refinement: a 5122 repeat reproduces ​ within error (Supplementary Appendix E).

### 5.4 Transit delay and escape threshold (“redshift-like” analog)

#### 5.4.1 Objective

Show that a compact packet experiences a deterministic transit delay when crossing a damping slab  in an otherwise near-uniform transport field , and characterize an escape threshold when the lossy region is thick/deep enough to extinguish the packet before exit. This is an analog of “gravitational redshift/delay,” arising from propagation in loss (not potential or curvature).

#### 5.4.2 Minimal setup

* •Domain and BCs: 2-D grid (2562), absorbing boundaries with graded sponges; boundary flux tallied (§4).

•Domain and BCs: 2-D grid (2562), absorbing boundaries with graded sponges; boundary flux tallied (§4).

* Profiles: Near-uniform  (SPD, constant to within a small tolerance). Damping slab  with a smooth  ramp-in/ramp-out, finite width , peak height .

Profiles: Near-uniform  (SPD, constant to within a small tolerance). Damping slab  with a smooth  ramp-in/ramp-out, finite width , peak height .

* Seed: Compact packet launched normal to the slab; reference run uses the same setup with .

Seed: Compact packet launched normal to the slab; reference run uses the same setup with .

* Config: grav_5_4_escape_redshift.yml (commit/hash in caption).

Config: grav_5_4_escape_redshift.yml (commit/hash in caption).

#### 5.4.3 Primary metric and gates

* Transit delay . Define entry/exit planes bracketing the slab; measure the packet’s arrival times (centroid crossing or envelope peak).

Transit delay . Define entry/exit planes bracketing the slab; measure the packet’s arrival times (centroid crossing or envelope peak).

Gate:

and monotone in the slab’s path integral

across small thickness/height variations (when provided).

* Escape threshold. For larger ​ or , report escape vs. collapse (no exit) within a fixed observation window.

Escape threshold. For larger ​ or , report escape vs. collapse (no exit) within a fixed observation window.

* Acceptance (this subsection):1.  vs. reference; monotone trend when a small sweep is included;2. Energy budget closure within 1%–3% post-transient, with Rayleigh loss  and boundary flux accounting for the full decline (identity in §4);3. Spectral safety (sub-Nyquist content);4. Robustness: unchanged within error under seed-shape variant and modest apron changes.

Acceptance (this subsection):

* 1.  vs. reference; monotone trend when a small sweep is included;

1.  vs. reference; monotone trend when a small sweep is included;

* 2. Energy budget closure within 1%–3% post-transient, with Rayleigh loss  and boundary flux accounting for the full decline (identity in §4);

2. Energy budget closure within 1%–3% post-transient, with Rayleigh loss  and boundary flux accounting for the full decline (identity in §4);

* 3. Spectral safety (sub-Nyquist content);

3. Spectral safety (sub-Nyquist content);

* 4. Robustness: unchanged within error under seed-shape variant and modest apron changes.

4. Robustness: unchanged within error under seed-shape variant and modest apron changes.

#### 5.4.4 Results (2562 main run)

Crossing the lossy slab introduces a measurable positive delay  relative to the uniform-medium reference. Total energy decays monotonically; the Rayleigh tally is positive, and together with boundary flux explains the budget drop within tolerance. For thicker/deeper slabs (when run), the packet fails to exit within the observation window, indicating an escape threshold consistent with increasing .

Interpretation. The delay arises from propagation in a lossy region; it is an operational analog to redshift/time delay but does not imply potential energy or spacetime curvature. Here,  is a loss channel, and  remains nearly uniform (scope/limits in §5 intro; energy identity in §4).

#### 5.4.5 Falsification route

* Remove loss: With , the measured  must be zero within the null band.

Remove loss: With , the measured  must be zero within the null band.

* Thin the slab: Reducing  or ​ must reduce ; a non-monotone trend falsifies the claim.

Thin the slab: Reducing  or ​ must reduce ; a non-monotone trend falsifies the claim.

* Uniform control: With  and the same seed, any observed delay must track only ; if  persists when , the effect is spurious.

Uniform control: With  and the same seed, any observed delay must track only ; if  persists when , the effect is spurious.

Repro bundle. Figure assets and recorder outputs (.npz/.csv with entry/exit times and energy tallies) for Figure 4 are archived with engine commit < hash> and bundle ID < ID>; see Data and Code Availability.

FIGURE 4

Transit delay and escape threshold in a damping slab (“redshift-like” analog). A compact packet crosses a smooth  slab in an otherwise near-uniform  (absorbing BCs). Timeline of  through the slab. Final frame. We define the transit delay  from centroid crossings at fixed entry/exit planes (method §4). This run exhibits ; total energy decays monotonically, with Rayleigh loss  and boundary flux accounting for the decline; the discrete budget closes within 1%–3% post-transient (§4 identity). For thicker/deeper slabs (when run), the packet fails to exit, marking an escape threshold consistent with increasing . Falsifiers:  must yield ; small decreases in  or ​ must reduce . Config grav_5_4_escape_redshift.yml; 2-D 2562;  auto (CFL); sponge parameters as in §4.

### 5.5 Orbital containment (limit-cycle)

#### 5.5.1 Objective

Demonstrate sustained, bounded circulation (an orbit-like limit cycle) emerging from anisotropic transport  plus a radially graded damping ring . The observable is a circulating centroid with bounded radius and stable period while the energy budget closes (loss = Rayleigh + boundary flux; no forces or curvature).

#### 5.5.2 Minimal setup

* Domain and BCs: 2-D grid (2562), reflective basin for the core region with a thin absorbing apron outside to remove far-field clutter (§4).

Domain and BCs: 2-D grid (2562), reflective basin for the core region with a thin absorbing apron outside to remove far-field clutter (§4).

* Profiles: Disk-shaped basin where  supports tangential transport (mild radial anisotropy; optional small off-diagonal near the rim). A smooth  annulus attenuates radial motion more than tangential.

Profiles: Disk-shaped basin where  supports tangential transport (mild radial anisotropy; optional small off-diagonal near the rim). A smooth  annulus attenuates radial motion more than tangential.

* Seed/IC: Compact packet placed off-center with a tangential bias (initial speed tuned inside the capture band).

Seed/IC: Compact packet placed off-center with a tangential bias (initial speed tuned inside the capture band).

* Config: grav_5_5_orbital_containment.yml (commit/hash in caption).

Config: grav_5_5_orbital_containment.yml (commit/hash in caption).

#### 5.5.3 Primary metric and gates

* Metric (bounded orbit). From the centroid path , compute radius . After transients, measure pericenter  and apocenter  over many cycles and require

Metric (bounded orbit). From the centroid path , compute radius . After transients, measure pericenter  and apocenter  over many cycles and require

(boundedness gate). Track the circulation period

(stability within a tight band) and the rectification ratio

(flat, no secular drift)

* •Acceptance (this subsection):Bounded radius (gate above) over ≥5–10 periods;Energy budget closure within 1%–3% post-transient with Rayleigh loss + boundary flux accounting for decay (§4 identity);Spectral safety (sub-Nyquist);Robustness: capture persists across a finite tangential-speed interval (capture band); 5122 confirmation (Supplementary Appendix E) reproduces the metrics within error.

•Acceptance (this subsection):

* Bounded radius (gate above) over ≥5–10 periods;

Bounded radius (gate above) over ≥5–10 periods;

* Energy budget closure within 1%–3% post-transient with Rayleigh loss + boundary flux accounting for decay (§4 identity);

Energy budget closure within 1%–3% post-transient with Rayleigh loss + boundary flux accounting for decay (§4 identity);

* Spectral safety (sub-Nyquist);

Spectral safety (sub-Nyquist);

* Robustness: capture persists across a finite tangential-speed interval (capture band); 5122 confirmation (Supplementary Appendix E) reproduces the metrics within error.

Robustness: capture persists across a finite tangential-speed interval (capture band); 5122 confirmation (Supplementary Appendix E) reproduces the metrics within error.

#### 5.5.4 Results (2562 main run)

The packet curves into the annulus, sheds radial energy in the  ring, and locks into a steady circulation. Over many periods the radius remains bounded (), the period  is stable to small jitter, and  is flat within measurement noise. Budgets close within tolerance; Rayleigh loss is concentrated where  peaks, and boundary flux is small and steady. Varying the initial tangential speed within a narrow window preserves containment (capture band); outside it the packet escapes or collapses (mapped in the supplement when included).

Interpretation. The containment is a deterministic limit cycle of the  dynamics:  supports tangential transport while  selectively damps radial components, producing an effective annular “well” without introducing forces or curvature.

#### 5.5.5 Falsification route

* Remove  (negative control): with the damping ring off, no bounded orbit should persist (capture band vanishes).

Remove  (negative control): with the damping ring off, no bounded orbit should persist (capture band vanishes).

* Disrupt  support: flattening  or removing its tangential preference should eliminate sustained circulation.

Disrupt  support: flattening  or removing its tangential preference should eliminate sustained circulation.

* Leakage/closure: large per-period boundary leakage or budget non-closure falsifies containment for this setup.

Leakage/closure: large per-period boundary leakage or budget non-closure falsifies containment for this setup.

Repro bundle. Figure assets and recorder outputs (.npz/.csv with , peri/apo markers, , and energy tallies) for Figure 5 are archived with engine commit < hash> and bundle ID < ID>; see Data and Code Availability.

FIGURE 5

Orbital containment (limit-cycle). A compact packet launched with tangential bias enters a basin where  supports tangential transport and a smooth  annulus damps radial motion (reflective core; absorbing apron). Timeline of  showing capture and steady circulation. Final frame. The orbit-like state satisfies the boundedness gate  over many periods; the period  is stable and the rectification ratio  is flat (methods §4). The energy budget closes within 1%–3% post-transient, with Rayleigh loss localized to the annulus and small, steady boundary flux. Falsifier: with  the capture band disappears and no sustained orbit is observed; flattening  likewise removes containment. Config grav_5_5_orbital_containment.yml; 2-D 2562;  auto (CFL); sponge parameters as in §4. Grid refinement: a 5122 repeat reproduces , , and budget closure within error (Supplementary Appendix E).

### 5.6 Equivalence-/inertial-like response

#### 5.6.1 Objective

Test an equivalence-like property of the medium: packets with different internal properties (amplitude/width) but the same launch kinematics traverse the same path through a given  (and near-zero ) to within a small tolerance. Operationally, the bending/deflection depends on the field structure  and launch conditions, not on packet “mass-like” details.

#### 5.6.2 Minimal setup

* •Domain and BCs: 2-D grid (2562), absorbing boundaries with graded sponges; boundary flux tallied (§4).

•Domain and BCs: 2-D grid (2562), absorbing boundaries with graded sponges; boundary flux tallied (§4).

* Profiles: Uniform or weakly graded  (as specified in the caption);  (exact value noted; used only for numerical hygiene if present).

Profiles: Uniform or weakly graded  (as specified in the caption);  (exact value noted; used only for numerical hygiene if present).

* Seeds: Two (or more) compact packets, A and B, launched from the same point with the same initial velocity; they differ only in amplitude  and/or width  (e.g., A: ​; B: ​).

Seeds: Two (or more) compact packets, A and B, launched from the same point with the same initial velocity; they differ only in amplitude  and/or width  (e.g., A: ​; B: ​).

* Config: grav_5_6_equivalence_inertial.yml (commit/hash in caption).

Config: grav_5_6_equivalence_inertial.yml (commit/hash in caption).

#### 5.6.3 Primary metric and gates

* •Path congruence. Extract centroid paths  and report the normalized RMS path mismatch

•Path congruence. Extract centroid paths  and report the normalized RMS path mismatch

with

the path length. Gate:

below a pre-registered threshold (small fraction of domain width).

* Arrival congruence. Difference in arrival time at a fixed exit plane  within tolerance.

Arrival congruence. Difference in arrival time at a fixed exit plane  within tolerance.

* Acceptance (this subsection):

Acceptance (this subsection):

* and  within tolerance; sign-correct bending if a weak gradient is present;

and  within tolerance; sign-correct bending if a weak gradient is present;

* Energy budget closure within 1%–3% post-transient; with , energy decline (if any) is accounted for by boundary flux;

Energy budget closure within 1%–3% post-transient; with , energy decline (if any) is accounted for by boundary flux;

* Spectral safety (sub-Nyquist content);

Spectral safety (sub-Nyquist content);

* Robustness: same verdict under a modest change of () and an alternate seed shape (Gaussian ↔ top-hat).

Robustness: same verdict under a modest change of () and an alternate seed shape (Gaussian ↔ top-hat).

#### 5.6.4 Results (2562 main run)

Packets A and B co-propagate along the same centerline within the measurement band; εpath\varepsilon_{\text{path}}εpath and  both satisfy the gates. When a weak gradient in  is present, both packets deflect with the same sign and magnitude (within error). Energy traces are smooth; with , the observed decay is explained by boundary absorption, and the discrete budget closes within tolerance.

Interpretation. In this regime the update law (linear transport + Rayleigh-type damping) makes the ray geometry depend on  and the launch kinematics, not on amplitude/width—an inertial-like or equivalence-like behavior of the analog medium. This is not a statement about gravitational mass; it is an operational analog confined to structured propagation (scope/limits in §5 intro; derivation cues in §4).

#### 5.6.5 Falsification route

* Amplitude/width sensitivity: if changing () at fixed launch kinematics produces a path mismatch ​ above the gate or a significant , the equivalence-like claim fails.

Amplitude/width sensitivity: if changing () at fixed launch kinematics produces a path mismatch ​ above the gate or a significant , the equivalence-like claim fails.

* Uniform control: with , , both packets must follow a straight, coincident path within the null band.

Uniform control: with , , both packets must follow a straight, coincident path within the null band.

* Strong loss: if modest  breaks congruence (beyond gate) while boundary accounting still closes, the effect is not equivalence-like in this setup.

Strong loss: if modest  breaks congruence (beyond gate) while boundary accounting still closes, the effect is not equivalence-like in this setup.

Repro bundle. Figure assets and recorder outputs (.npz/.csv with , and energy tallies) for Figure 6 are archived with engine commit < hash> and bundle ID < ID>; see Data and Code Availability.

FIGURE 6

Equivalence-/inertial-like response. Two packets with different amplitude/width but the same launch kinematics traverse the same  (absorbing BCs; ). Timeline of  showing co-propagation. Final frame. The normalized RMS path mismatch  and arrival-time difference  are both within pre-registered tolerances; if a weak gradient in  is present, both packets bend with the same sign and magnitude within error. Energy evolves smoothly; with , decline is explained by boundary flux, and the discrete budget closes within 1%–3% after transients (§4 identity). Falsifiers: varying () at fixed launch should not change the path beyond tolerance; with , paths must be straight and coincident within noise. Config grav_5_6_equivalence_inertial.yml; 2-D 2562;  auto (CFL); sponge parameters as in §4.

### 5.7 Directional drift from anisotropy (frame-drag–like analog)

#### 5.7.1 Objective

Show that a compact packet develops a steady lateral drift when propagating through a medium with anisotropic transport featuring a controlled off-diagonal component . The observable is a non-zero, sign-controlled drift rate transverse to the nominal travel direction, produced by the orientation of  (no forces, no curvature).

#### 5.7.2 Minimal setup

* •Domain and BCs: 2-D grid (2562), absorbing boundaries with graded sponges; boundary flux tallied (§4).

•Domain and BCs: 2-D grid (2562), absorbing boundaries with graded sponges; boundary flux tallied (§4).

* Profiles: Spatially uniform magnitude of transport but with a tilted principal frame:

Profiles: Spatially uniform magnitude of transport but with a tilted principal frame:

with

and small fixed rotation

so that

. Unless otherwise noted

(or very small, only for numerical hygiene).

* Seed: Compact packet launched along the nominal  direction (zero initial lateral velocity).

Seed: Compact packet launched along the nominal  direction (zero initial lateral velocity).

* Config: grav_5_7_directional_drift.yml (commit/hash in caption).

Config: grav_5_7_directional_drift.yml (commit/hash in caption).

#### 5.7.3 Primary metric and gates

* Transverse drift rate. From the centroid path , estimate the signed lateral drift

Transverse drift rate. From the centroid path , estimate the signed lateral drift

using a mid-segment linear fit to avoid entrance/exit transients (method §4).

* •Acceptance (this subsection):Non-zero  (or slope sss) with the correct sign set by ;Energy budget closure within 1%–3% post-transient; with , any energy decline is explained by boundary flux (§4 identity);Spectral safety (sub-Nyquist);Robustness: same verdict under seed-shape variant (Gaussian ↔ top-hat) and modest apron changes; zero drift when  (null control).

•Acceptance (this subsection):

* Non-zero  (or slope sss) with the correct sign set by ;

Non-zero  (or slope sss) with the correct sign set by ;

* Energy budget closure within 1%–3% post-transient; with , any energy decline is explained by boundary flux (§4 identity);

Energy budget closure within 1%–3% post-transient; with , any energy decline is explained by boundary flux (§4 identity);

* Spectral safety (sub-Nyquist);

Spectral safety (sub-Nyquist);

* Robustness: same verdict under seed-shape variant (Gaussian ↔ top-hat) and modest apron changes; zero drift when  (null control).

Robustness: same verdict under seed-shape variant (Gaussian ↔ top-hat) and modest apron changes; zero drift when  (null control).

#### 5.7.4 Results (2562 main run)

The centroid accumulates a steady transverse offset while advancing along ​; the fitted mid-segment drift rate is non-zero and sign-correct for the chosen tilt . Repeating with  yields drift consistent with zero (null control). Energy decreases smoothly due to absorbing boundaries; with , Rayleigh loss = 0, and the discrete budget closes within tolerance.

Interpretation. Drift arises from principal-axis rotation of the anisotropic transport tensor: rays preferentially align to the faster direction, producing a lateral bias set by  (geometric-optics view in §4). This is a media analog—not a claim of force or spacetime curvature.

#### 5.7.5 Falsification route

* Turn off the tilt: with , the measured drift must vanish within the null band.

Turn off the tilt: with , the measured drift must vanish within the null band.

* Flip the sign: ​ must flip the drift sign.

Flip the sign: ​ must flip the drift sign.

* Over-damp test: introducing moderate  that suppresses the signal without changing signs would falsify the “transport-induced” mechanism for this setup.

Over-damp test: introducing moderate  that suppresses the signal without changing signs would falsify the “transport-induced” mechanism for this setup.

Repro bundle. Figure assets and recorder outputs (.npz/.csv with centroid path and drift estimate, plus energy tallies) for Figure 7 are archived with engine commit < hash> and bundle ID < ID>; see Data and Code Availability.

FIGURE 7

Directional drift from anisotropy (frame-drag–like analog). A compact packet traverses a medium with tilted anisotropic transport  (off-diagonal ; ; absorbing boundaries. Timeline of  exhibiting steady lateral offset. Final frame. The mid-segment drift rate  (or slope ) is non-zero and sign-correct for the chosen tilt; with  the drift is within the null band (control). Energy declines monotonically due to boundary absorption; Rayleigh loss = 0, and the discrete budget closes within 1%–3% after transients (§4 identity). Falsifiers: zero-tilt  must yield zero drift; flipping  must flip the measured drift. Config grav_5_7_directional_drift.yml; 2-D 2562;  auto (CFL); sponge parameters as in §4.

### 5.8 Curvature without coordinates (ray-shaping via )

#### 5.8.1 Objective

Show that we can produce a curved, ray-like trajectory purely by shaping the transport tensor  in a Cartesian grid—i.e., without using curvilinear coordinates or external forcing. The observable is a centerline whose signed curvature  matches the eikonal prediction computed from the designed  (see §4).

#### 5.8.2 Minimal setup

* •Domain and BCs: 2-D grid (2562), absorbing boundaries with graded sponges; boundary flux tallied (§4).

•Domain and BCs: 2-D grid (2562), absorbing boundaries with graded sponges; boundary flux tallied (§4).

* Profiles: Smooth,  patterned  (SPD everywhere) that bends rays along a target arc/guide; .

Profiles: Smooth,  patterned  (SPD everywhere) that bends rays along a target arc/guide; .

* Seed: Compact packet launched to enter the guide at shallow incidence (quasi-ray).

Seed: Compact packet launched to enter the guide at shallow incidence (quasi-ray).

* Config: grav_5_8_curvature_without_coords.yml (commit/hash in caption).

Config: grav_5_8_curvature_without_coords.yml (commit/hash in caption).

#### 5.8.3 Primary metric and gates

#### 5.8.3.1 Curvature agreement

Extract the packet centerline  and compute its signed curvature  (mid-segment, finite-difference estimate). Compute the RMS residual to the eikonal prediction  derived from the designed  (§4):

Gate:  below a pre-registered tolerance (small fraction of the mean ); sign-correct curvature throughout the guided segment.

Acceptance (this subsection):

* within tolerance; sign-correct bending;

within tolerance; sign-correct bending;

* Energy budget closure within 1%–3% post-transient; with , loss is boundary flux only (identity in §4);

Energy budget closure within 1%–3% post-transient; with , loss is boundary flux only (identity in §4);

* Spectral safety (sub-Nyquist);

Spectral safety (sub-Nyquist);

* Robustness: unchanged within error under seed-shape swap (Gaussian ↔ top-hat) and modest apron changes; null control with  yields .

Robustness: unchanged within error under seed-shape swap (Gaussian ↔ top-hat) and modest apron changes; null control with  yields .

#### 5.8.4 Results (2562 main run)

The packet follows the designed guide, producing a smooth, sign-consistent curvature. The measured  tracks the eikonal prediction with a small ​ (within the acceptance band). Total energy decreases smoothly due to absorbing boundaries; Rayleigh loss = 0, and the discrete budget closes within tolerance. A null run with uniform  yields a straight path within the noise band.

Interpretation. The “curvature” here is a media analog arising from spatial variation of  that refracts rays—no coordinate transformation, forces, or spacetime curvature are invoked (scope/limits in §5 intro; derivation in §4).

#### 5.8.5 Falsification route

* Uniform control: ,  must yield  along the path.

Uniform control: ,  must yield  along the path.

* Pattern reversal/mirroring: flipping the designed guide’s orientation must flip the sign of .

Pattern reversal/mirroring: flipping the designed guide’s orientation must flip the sign of .

* Tolerance breach: ​ exceeding the pre-registered bound falsifies ray-shaping for this setup.

Tolerance breach: ​ exceeding the pre-registered bound falsifies ray-shaping for this setup.

Repro bundle. Figure assets and recorder outputs (.npz/.csv with centerline and , plus energy tallies) for Figure 8 are archived with engine commit < hash> and bundle ID < ID>; see Data and Code Availability.

FIGURE 8

Curvature without coordinates (ray-shaping via ). A compact packet traverses a smooth patterned  (SPD; ; absorbing BCs) that bends rays along a target arc. Timeline of . Final frame. The centerline curvature  follows the eikonal prediction from the designed  (procedure §4); the RMS curvature residual  remains within tolerance and the curvature is sign-correct along the guide. Energy declines monotonically due to boundary absorption; Rayleigh loss = 0, and the discrete budget closes within 1%–3% after transients (§4 identity). Falsifiers: with  the path must be straight within noise; mirroring the guide must flip the curvature sign. Config grav_5_8_curvature_without_coords.yml; 2-D 2562;  auto (CFL); sponge parameters as in §4.

### 5.9 Local collapse trap

#### 5.9.1 Objective

Show localized trapping: a compact packet enters a finite  well embedded in an otherwise smooth , sheds radial motion, and remains confined in the well region without re-emergence. This is a dissipative analog of a potential “trap”: the mechanism is focusing in  plus loss in  (no forces or curvature).

#### 5.9.2 Minimal setup

* •Domain and BCs: 2-D grid (2562), absorbing boundaries with graded sponges; boundary flux tallied (§4).

•Domain and BCs: 2-D grid (2562), absorbing boundaries with graded sponges; boundary flux tallied (§4).

* Profiles: Smooth background  (near-uniform or mildly focusing), with a localized  well centered at ; all profiles -smoothed over 3–5 .

Profiles: Smooth background  (near-uniform or mildly focusing), with a localized  well centered at ; all profiles -smoothed over 3–5 .

* Seed: Compact packet launched toward  (zero initial angular momentum unless noted).

Seed: Compact packet launched toward  (zero initial angular momentum unless noted).

* Config: grav_5_9_local_collapse_trap.yml (commit/hash in caption).

Config: grav_5_9_local_collapse_trap.yml (commit/hash in caption).

#### 5.9.3 Primary metric and gates

* Capture decision + time. Define an inner mask . Let  be the fraction of total energy inside .

Capture decision + time. Define an inner mask . Let  be the fraction of total energy inside .

* Capture gate:  for  consecutive frames (e.g., ) and no subsequent exit within the observation window.

Capture gate:  for  consecutive frames (e.g., ) and no subsequent exit within the observation window.

* Capture time: .

Capture time: .

* Acceptance (this subsection):1. Gate satisfied (capture) and no re-emergence;2. Energy budget closure within 1%–3% post-transient; Rayleigh loss  (localized in the well) + boundary flux account for the decrease (identity in §4);3. Spectral safety (sub-Nyquist);4. Robustness: verdict unchanged under small changes of ​ and seed shape; negative control with  does not capture.

Acceptance (this subsection):

* 1. Gate satisfied (capture) and no re-emergence;

1. Gate satisfied (capture) and no re-emergence;

* 2. Energy budget closure within 1%–3% post-transient; Rayleigh loss  (localized in the well) + boundary flux account for the decrease (identity in §4);

2. Energy budget closure within 1%–3% post-transient; Rayleigh loss  (localized in the well) + boundary flux account for the decrease (identity in §4);

* 3. Spectral safety (sub-Nyquist);

3. Spectral safety (sub-Nyquist);

* 4. Robustness: verdict unchanged under small changes of ​ and seed shape; negative control with  does not capture.

4. Robustness: verdict unchanged under small changes of ​ and seed shape; negative control with  does not capture.

#### 5.9.4 Results (2562 main run)

On entering the  well the packet focuses and stalls;  rises above the 0.80 gate at ​ and stays high for the remainder of the run. The total energy decays monotonically; the Rayleigh tally is positive and concentrated within the well, and together with boundary flux explains the drop; the discrete budget closes within tolerance. A matched negative control with  shows no sustained capture (rebound/dispersion).

Interpretation. Trapping here is an operational analog produced by directional transport + dissipation.  is a loss channel, not potential energy; confinement is the limit behavior of the () dynamics (scope/limits in §5 intro; energy identity in §4).

#### 5.9.5 Falsification route

* Remove loss (control): with , the capture gate must fail.

Remove loss (control): with , the capture gate must fail.

* Shift the well: moving  off the traversed path must remove capture.

Shift the well: moving  off the traversed path must remove capture.

* Thin the well: reducing  or  must increase ​ and can eliminate capture; a non-monotone trend falsifies the claim.

Thin the well: reducing  or  must increase ​ and can eliminate capture; a non-monotone trend falsifies the claim.

Repro bundle. Figure assets and recorder outputs (.npz/.csv with , , and energy tallies) for Figure 9 are archived with engine commit < hash> and bundle ID < ID>; see Data and Code Availability.

FIGURE 9

Local collapse trap. A compact packet encounters a localized damping well  embedded in a smooth  (absorbing BCs). Timeline of  showing inward focusing and stall. Final frame. Capture is certified when the core energy fraction  within radius ​ exceeds 0.80 for  frames without re-emergence; this run passes the gate with capture time ​ (reported in the data bundle). Total energy decays monotonically; Rayleigh loss > 0 (localized in the well) and boundary flux account for the decline; the discrete budget closes within 1%–3% post-transient (§4 identity). Falsifiers:  must remove capture; shifting/weakening the well must delay or eliminate capture. Config grav_5_9_local_collapse_trap.yml; 2-D 2562;  auto (CFL); smoothing/apron parameters as in §4.

### 5.10 Reversible rebound (conservative basin)

#### 5.10.1 Objective

Demonstrate reversible, near-elastic rebound when a packet encounters a conservative transport basin (structured , no damping). The observable is a collision-like interaction where the packet exits with the same speed (within tolerance) and mirrored angle as it entered—i.e., a high restitution and repeatable geometry without energy injection or loss.

#### 5.10.2 Minimal setup

* Domain and BCs: 2-D grid (2562); reflective basin walls that define the conservative region; thin absorbing apron outside to quench far-field clutter (flux tallied; §4).

Domain and BCs: 2-D grid (2562); reflective basin walls that define the conservative region; thin absorbing apron outside to quench far-field clutter (flux tallied; §4).

* Profiles: Conservative  basin shaped to steer rays specularly (SPD everywhere; -smoothed). .

Profiles: Conservative  basin shaped to steer rays specularly (SPD everywhere; -smoothed). .

* Seed: Compact packet aimed to strike the basin at a set incidence angle.

Seed: Compact packet aimed to strike the basin at a set incidence angle.

* Config: grav_5_10_reversible_rebound.yml (commit/hash in caption).

Config: grav_5_10_reversible_rebound.yml (commit/hash in caption).

#### 5.10.3 Primary metric and gates

* Restitution (speed/energy). Let  be centroid speed just before impact and  after exit.

Restitution (speed/energy). Let  be centroid speed just before impact and  after exit.

Gate:

(and/or

) ≥ pre-registered threshold (near-unity).

* Specular repeatability. Incidence vs. exit angles obey  within a small tolerance; successive rebounds (when run) reproduce geometry within tolerance.

Specular repeatability. Incidence vs. exit angles obey  within a small tolerance; successive rebounds (when run) reproduce geometry within tolerance.

* •Acceptance (this subsection):Restitution above threshold and specular repeatability satisfied;Energy budget closure within 1%–3% post-transient, with Rayleigh loss = 0 and boundary flux ≈ 0 during the interaction (reflective core; any apron flux is negligible and tallied);Spectral safety (sub-Nyquist);Robustness: verdict unchanged under small incidence-angle and seed-shape variations.

•Acceptance (this subsection):

* Restitution above threshold and specular repeatability satisfied;

Restitution above threshold and specular repeatability satisfied;

* Energy budget closure within 1%–3% post-transient, with Rayleigh loss = 0 and boundary flux ≈ 0 during the interaction (reflective core; any apron flux is negligible and tallied);

Energy budget closure within 1%–3% post-transient, with Rayleigh loss = 0 and boundary flux ≈ 0 during the interaction (reflective core; any apron flux is negligible and tallied);

* Spectral safety (sub-Nyquist);

Spectral safety (sub-Nyquist);

* Robustness: verdict unchanged under small incidence-angle and seed-shape variations.

Robustness: verdict unchanged under small incidence-angle and seed-shape variations.

#### 5.10.4 Results (2562 main run)

The packet strikes the conservative -basin, undergoes a clean specular-like turn, and exits along the mirrored direction. Measured speed restitution ​ is near unity; the angle condition holds within the gate. Energy traces are flat over the interaction window; Rayleigh tally = 0, and boundary-band flux remains at the noise floor (reflective core). Repeating the shot with a slightly different incidence angle or an alternate seed (Gaussian ↔ top-hat) preserves restitution and geometry within tolerance.

Interpretation. With  and specularly shaped , the dynamics are conservative: the update law reduces to anisotropic transport where the basin acts as a geometric mirror. The result is a reversible rebound—an operational analog of elastic reflection—without invoking forces or curvature (scope/limits in §5 intro; energy identity in §4).

#### 5.10.5 Falsification route

* Introduce loss: adding  in the basin should lower , ​ below the gate (inelastic rebound).

Introduce loss: adding  in the basin should lower , ​ below the gate (inelastic rebound).

* Flatten : removing the specular shaping should eliminate controlled rebound (no mirrored exit).

Flatten : removing the specular shaping should eliminate controlled rebound (no mirrored exit).

* Leakage/closure: detectable apron leakage during the interaction or budget non-closure falsifies conservativity for this setup.

Leakage/closure: detectable apron leakage during the interaction or budget non-closure falsifies conservativity for this setup.

Repro bundle. Figure assets and recorder outputs (.npz/.csv with , ​, angles, and energy tallies) for Figure 10 are archived with engine commit < hash> and bundle ID < ID>; see Data and Code Availability.

FIGURE 10

Reversible rebound in a conservative transport basin. A compact packet impinges on a specularly shaped, lossless  basin (reflective core; absorbing apron; ). Timeline of  through approach and rebound. Final frame. Restitution  (and ) is near-unity; the exit angle mirrors incidence within tolerance (methods §4). The energy budget closes within 1%–3% post-transient: Rayleigh loss = 0; boundary-band flux ≈0 during the interaction. Falsifiers: introducing  should reduce restitution; flattening  should remove the specular exit; measurable leakage violates conservativity. Config grav_5_10_reversible_rebound.yml; 2-D 2562;  auto (CFL); smoothing/apron parameters as in §4.

### 5.11 Inverse-square–like radial bias (attraction analog)

#### 5.11.1 Objective

Show a central, inward bias consistent with an inverse-square–like trend when a packet traverses a domain whose transport field  is radially graded so that the gradient magnitude scales approximately as . The observable is a sign-correct inward drift and a mid-track power-law relation between radial drift and radius.

#### 5.11.2 Minimal setup

* •Domain and BCs: 2-D grid (2562), absorbing boundaries with graded sponges; boundary flux tallied (§4).

•Domain and BCs: 2-D grid (2562), absorbing boundaries with graded sponges; boundary flux tallied (§4).

* Profiles: Radially symmetric, -smoothed transport  with a central strengthening such that  over the measurement annulus; off-diagonals are zero (or small) so the effect is purely radial.  (exact value noted; only used for numerical hygiene if present).

Profiles: Radially symmetric, -smoothed transport  with a central strengthening such that  over the measurement annulus; off-diagonals are zero (or small) so the effect is purely radial.  (exact value noted; only used for numerical hygiene if present).

* Seed: Compact packet launched from ​ with near-tangential motion (low initial radial component).

Seed: Compact packet launched from ​ with near-tangential motion (low initial radial component).

* Config: grav_5_11_inverse_square_drift.yml (commit/hash in caption).

Config: grav_5_11_inverse_square_drift.yml (commit/hash in caption).

#### 5.11.3 Primary metric and gates

* Radial drift exponent. From the centroid path , compute  and the signed mid-segment radial speed . Fit a power law  over the mid-track window (excluding entrance/exit).

Radial drift exponent. From the centroid path , compute  and the signed mid-segment radial speed . Fit a power law  over the mid-track window (excluding entrance/exit).

* Gate: inward drift (correct sign) and  within a pre-registered band around two (e.g., ).

Gate: inward drift (correct sign) and  within a pre-registered band around two (e.g., ).

* •Acceptance (this subsection):Sign-correct inward drift and α\alphaα within band;Energy budget closure within 1%–3% post-transient; with , any decline is explained by boundary flux only (identity in §4);Spectral safety (sub-Nyquist);Robustness: unchanged within error under modest seed-shape change (Gaussian ↔ top-hat) and gradient-strength perturbation; null control with uniform  yields drift consistent with zero.

•Acceptance (this subsection):

* Sign-correct inward drift and α\alphaα within band;

Sign-correct inward drift and α\alphaα within band;

* Energy budget closure within 1%–3% post-transient; with , any decline is explained by boundary flux only (identity in §4);

Energy budget closure within 1%–3% post-transient; with , any decline is explained by boundary flux only (identity in §4);

* Spectral safety (sub-Nyquist);

Spectral safety (sub-Nyquist);

* Robustness: unchanged within error under modest seed-shape change (Gaussian ↔ top-hat) and gradient-strength perturbation; null control with uniform  yields drift consistent with zero.

Robustness: unchanged within error under modest seed-shape change (Gaussian ↔ top-hat) and gradient-strength perturbation; null control with uniform  yields drift consistent with zero.

#### 5.11.4 Results (2562 main run)

The centroid acquires a steady inward bias while advancing around the center. The mid-track log–log fit of  vs.  yields an exponent  within the acceptance band (value reported in the data bundle), and the drift is sign-correct. Energy decreases smoothly due to the absorbing apron; with , Rayleigh loss ≈ 0, and the discrete budget closes within tolerance.

Interpretation. The inverse-square–like behavior is a media analog: a radial strengthening of  refracts rays toward the center so that the radial component of transport grows roughly like  over the measurement annulus. No forces, mass, or spacetime curvature are invoked (scope/limits in §5 intro; geometric-optics view in §4).

#### 5.11.5 Falsification route

* Reverse the gradient: flipping the sign of  must produce outward drift (sign flip).

Reverse the gradient: flipping the sign of  must produce outward drift (sign flip).

* Flatten the profile: with  uniform, radial drift must lie within the null band.

Flatten the profile: with  uniform, radial drift must lie within the null band.

* Exponent check: a mid-track fit with α\alphaα far outside the band falsifies the inverse-square-like claim for this setup.

Exponent check: a mid-track fit with α\alphaα far outside the band falsifies the inverse-square-like claim for this setup.

Repro bundle. Figure assets and recorder outputs (.npz/.csv with , , and the log–log fit, plus energy tallies) for Figure 11 are archived with engine commit < hash> and bundle ID < ID>; see Data and Code Availability.

FIGURE 11

Inverse-square–like radial bias (attraction analog). A compact packet traverses a domain with radially strengthened transport  such that  (absorbing BCs; ). Timeline of . Final frame. From the centroid path we compute  and  over the mid-track window and fit ; the measured α\alphaα lies within the pre-registered band around 2, and drift is inward (sign-correct). Energy declines monotonically due to boundary absorption; Rayleigh loss ≈ 0, and the discrete budget closes within 1%–3% after transients (§4 identity). Falsifiers: reversing the radial gradient must produce outward drift; with uniform  the radial drift must be within the null band. Config grav_5_11_inverse_square_drift.yml; 2-D 2562;  auto (CFL); smoothing/apron parameters as in §4.

### 5.12 Repulsion via curvature inversion (defocusing analog)

#### 5.12.1 Objective

Demonstrate defocusing/outward divergence when the transport gradient is sign-inverted relative to the focusing cases: a compact packet launched across a region with  oriented to increase effective transport along the approach should develop a sign-correct outward drift and nearby trajectories should separate. This is a media analog (ray refraction from ), not a force or curvature claim.

#### 5.12.2 Minimal setup

* Domain and BCs: 2-D grid (2562), absorbing boundaries with graded sponges; boundary flux tallied (§4).

Domain and BCs: 2-D grid (2562), absorbing boundaries with graded sponges; boundary flux tallied (§4).

* Profiles: Smooth,  gradient in  with the opposite sign to §5.11/§5.3 so that rays are pushed outward (defocusing). .

Profiles: Smooth,  gradient in  with the opposite sign to §5.11/§5.3 so that rays are pushed outward (defocusing). .

* Seed(s): (i) A single compact packet for centerline measurement; (ii) an optional two-ray probe: two packets launched with small transverse offset ​ to quantify divergence.

Seed(s): (i) A single compact packet for centerline measurement; (ii) an optional two-ray probe: two packets launched with small transverse offset ​ to quantify divergence.

* Config: grav_5_12_repulsion_inversion.yml (commit/hash in caption).

Config: grav_5_12_repulsion_inversion.yml (commit/hash in caption).

#### 5.12.3 Primary metric and gates

* Outward bias (single-ray). From the centroid path, compute the signed radial slope  over the mid-track window; gate:  (outward) with the correct sign under gradient reversal.

Outward bias (single-ray). From the centroid path, compute the signed radial slope  over the mid-track window; gate:  (outward) with the correct sign under gradient reversal.

* Divergence (two-ray). Track the transverse separation  between the two probes; fit  or, for short windows, . Gate:  (or ) and monotone growth over the window.

Divergence (two-ray). Track the transverse separation  between the two probes; fit  or, for short windows, . Gate:  (or ) and monotone growth over the window.

* •Acceptance (this subsection):Outward bias (single-ray) and positive divergence rate (two-ray) within tolerance;Energy budget closure within 1%–3% post-transient; with  the decline (if any) is boundary flux only (identity in §4);Spectral safety (sub-Nyquist);Robustness: verdict unchanged under seed-shape swap (Gaussian ↔ top-hat) and modest apron changes; null control with  yields  and no measurable divergence.

•Acceptance (this subsection):

* Outward bias (single-ray) and positive divergence rate (two-ray) within tolerance;

Outward bias (single-ray) and positive divergence rate (two-ray) within tolerance;

* Energy budget closure within 1%–3% post-transient; with  the decline (if any) is boundary flux only (identity in §4);

Energy budget closure within 1%–3% post-transient; with  the decline (if any) is boundary flux only (identity in §4);

* Spectral safety (sub-Nyquist);

Spectral safety (sub-Nyquist);

* Robustness: verdict unchanged under seed-shape swap (Gaussian ↔ top-hat) and modest apron changes; null control with  yields  and no measurable divergence.

Robustness: verdict unchanged under seed-shape swap (Gaussian ↔ top-hat) and modest apron changes; null control with  yields  and no measurable divergence.

#### 5.12.4 Results (2562 main run)

The centerline exhibits a clear outward drift across the graded region (positive mid-track slope ); the two-ray probe shows monotone separation with a positive fitted growth parameter (reported in the data bundle). Energy decays smoothly due to the absorbing sponge; with , Rayleigh loss = 0, and the discrete budget closes within tolerance. Reversing the gradient flips the drift sign and removes the divergence trend (control), while a uniform- null shows straight propagation with ​.

Interpretation. Defocusing here is a transport effect: rays refract away from regions of increasing transport (opposite of the focusing cases). The observable outward bias and separation follow from the geometric-optics limit of  (see §4), not from forces or spacetime curvature.

#### 5.12.5 Falsification route

* Gradient reversal: must flip the sign of outward bias (to inward) and suppress divergence.

Gradient reversal: must flip the sign of outward bias (to inward) and suppress divergence.

* Uniform control: with ,  both  and the growth rate must sit within the null band.

Uniform control: with ,  both  and the growth rate must sit within the null band.

* Over-strong  (if added): introducing damping that changes the verdict while boundary accounting still closes would falsify a pure transport explanation for this setup.

Over-strong  (if added): introducing damping that changes the verdict while boundary accounting still closes would falsify a pure transport explanation for this setup.

Repro bundle. Figure assets and recorder outputs (.npz/.csv with centerline, two-ray separation, and energy tallies) for Figure 12 are archived with engine commit < hash> and bundle ID < ID>; see Data and Code Availability.

FIGURE 12

Repulsion via curvature inversion (defocusing analog). A compact packet traverses a smooth  gradient in  with the opposite sign of the focusing cases (absorbing BCs; ). Timeline of . Final frame. The centerline shows a sign-correct outward bias (positive mid-track radial slope), and (when used) a two-ray probe exhibits monotone separation with a positive growth parameter (methods §4). Energy declines monotonically due to boundary absorption; Rayleigh loss = 0, and the discrete budget closes within 1%–3% post-transient (§4 identity). Falsifiers: reversing the gradient must flip the bias and suppress separation; with uniform  the path must be straight and the two-ray separation flat. Config grav_5_12_repulsion_inversion.yml; 2-D 2562;  auto (CFL); sponge parameters as in §4.

### 5.13 Interference and stacking

#### 5.13.1 Objective

Demonstrate phase-sensitive superposition in the scalar medium: two coherent packets launched to overlap in a region of (nearly) uniform  and negligible  exhibit constructive or destructive outcomes depending on the relative phase . The observables are the interference visibility in the overlap zone and the constructive gain (“stacking”) relative to a single-packet baseline.

#### 5.13.2 Minimal setup

* •Domain and BCs: 2-D grid (2562), absorbing boundaries with graded sponges; boundary flux tallied (§4).

•Domain and BCs: 2-D grid (2562), absorbing boundaries with graded sponges; boundary flux tallied (§4).

* Profiles:  uniform (or very weakly graded, noted in the caption);  (small only for numerical hygiene if present).

Profiles:  uniform (or very weakly graded, noted in the caption);  (small only for numerical hygiene if present).

* Seeds: Two equal-envelope compact packets launched from opposite sides to overlap in a fixed region; relative phase  set at initialization.

Seeds: Two equal-envelope compact packets launched from opposite sides to overlap in a fixed region; relative phase  set at initialization.

* Config: grav_5_13_interference_stacking.yml (commit/hash in caption).

Config: grav_5_13_interference_stacking.yml (commit/hash in caption).

#### 5.13.3 Primary metrics and gates

* Visibility (contrast) at overlap. In a small ROI centered on the overlap, measure peak and trough of  (or ) at the overlap time t∗t_\astt∗​ and report

Visibility (contrast) at overlap. In a small ROI centered on the overlap, measure peak and trough of  (or ) at the overlap time t∗t_\astt∗​ and report

with

the ROI statistic. Gate:

follows a cosine law within tolerance (high near

, low near

).

* Constructive gain (“stacking”). Compare the ROI peak at  to the single-packet baseline:

Constructive gain (“stacking”). Compare the ROI peak at  to the single-packet baseline:

Gate:

near the linear superposition prediction (≈2 in amplitude; tolerance specified).

* •Acceptance (this subsection): trend (cosine-like) and constructive gain  within gates;Energy budget closure within 1%–3% post-transient; with , any decline is boundary flux only (identity in §4);Spectral safety (sub-Nyquist content);Robustness: verdict unchanged under small seed-shape swaps (Gaussian ↔ top-hat) and modest timing offsets; incoherent control (random ) shows reduced/vanishing contrast

•Acceptance (this subsection):

* trend (cosine-like) and constructive gain  within gates;

trend (cosine-like) and constructive gain  within gates;

* Energy budget closure within 1%–3% post-transient; with , any decline is boundary flux only (identity in §4);

Energy budget closure within 1%–3% post-transient; with , any decline is boundary flux only (identity in §4);

* Spectral safety (sub-Nyquist content);

Spectral safety (sub-Nyquist content);

* Robustness: verdict unchanged under small seed-shape swaps (Gaussian ↔ top-hat) and modest timing offsets; incoherent control (random ) shows reduced/vanishing contrast

Robustness: verdict unchanged under small seed-shape swaps (Gaussian ↔ top-hat) and modest timing offsets; incoherent control (random ) shows reduced/vanishing contrast

#### 5.13.4 Results (2562 main run)

At the programmed overlap, the field exhibits phase-dependent contrast: near  the ROI amplitude increases (stacking), while near  it shows a strong notch (destructive). The measured visibility follows the expected cosine trend within tolerance, and the constructive gain  is close to the linear-superposition prediction. Total energy evolves smoothly; with , Rayleigh loss ≈ 0, and the discrete budget closes within the 1%–3% gate (decline, if any, is boundary absorption). Incoherent/phase-scrambled control runs reduce contrast as expected.

Interpretation. Interference and stacking are wave-propagation features of the scalar medium under the linear transport law; they are not gravitational claims. Here,  is (nearly) uniform and  is negligible, so outcomes track coherence and phase rather than curvature or forces (scope/limits in §5 intro; update/energy identities in §4).

#### 5.13.5 Falsification route

* Phase scramble: randomizing  must collapse visibility.

Phase scramble: randomizing  must collapse visibility.

* Single-packet control: with one seed removed, the ROI peak must match the baseline (no stacking).

Single-packet control: with one seed removed, the ROI peak must match the baseline (no stacking).

* Loss sensitivity: increasing  should lower visibility and ; if  remains high under strong loss while budgets still close, the superposition claim fails for this setup.

Loss sensitivity: increasing  should lower visibility and ; if  remains high under strong loss while budgets still close, the superposition claim fails for this setup.

Repro bundle. Figure assets and recorder outputs (.npz/.csv with ROI metrics , , and energy tallies) for Figure 13 are archived with engine commit < hash> and bundle ID < ID>; see Data and Code Availability.

FIGURE 13

Interference and stacking. Two coherent packets meet in a region of uniform  with  (absorbing BCs). Timeline of  through the overlap. Final frame. The interference visibility  measured in a small ROI follows the expected cosine-like trend (high near , low near ), and the constructive gain at  approaches the linear-superposition prediction (methods §4). Energy evolves smoothly; Rayleigh loss ≈ 0, and the discrete budget closes within 1%–3% after transients (§4 identity). Falsifiers: phase scramble must reduce ; removing one seed must eliminate stacking; stronger  should suppress contrast. Config grav_5_13_interference_stacking.yml; 2-D 2562;  auto (CFL); sponge parameters as in §4.

## 6 Discussion

### 6.1 What we demonstrated

Structured propagation in (, ) yields reproducible operational analogs of gravitational-like behavior. Across §5 we reported deflection and ray bending (§§5.1, 5.3, 5.8), dissipative collapse/containment and transit delay (§§5.2, 5.4, 5.5, 5.9), anisotropy-driven drift (§5.7), repulsion via curvature inversion (§5.12), and phase-sensitive interference (§5.13). Each subsection declares a primary metric with a paired falsifier, and accepted runs meet the acceptance gates: (i) primary metric passes; (ii) energy budget closure within 1%–3% post-transient; (iii) spectral safety; and (iv) robustness checks as specified. As noted in §5, the source term is off for all experiments (); dynamics arise from the initial condition under , .

### 6.2 Transport-only vs. transport + loss

Transport-only (). Smooth gradients in  refract rays, producing sign-correct bending consistent with the eikonal construction in §4; rotated principal axes (non-zero ) induce directional drift with sign set by the tilt (§5.7).

Transport + loss () Focusing  combined with positive  yields collapse/trapping (§§5.2, 5.9) and orbital containment (§5.5); a lossy slab produces transit delay and escape thresholds (§5.4). Throughout,  acts as a loss channel (Rayleigh dissipation), not a potential; containment emerges from focusing + selective dissipation, not forces or curvature.

### 6.3 Predictions and falsifiers (operational, testable)

* Linear deflection:  in the weak-gradient regime; sign flips when the gradient is reversed (§§5.1, 5.3).

Linear deflection:  in the weak-gradient regime; sign flips when the gradient is reversed (§§5.1, 5.3).

* Delay monotonicity and escape:  increases with the path integral ; sufficiently large thickness/height prevents exit within the window (§5.4).

Delay monotonicity and escape:  increases with the path integral ; sufficiently large thickness/height prevents exit within the window (§5.4).

* Bounded orbit: with tangentially supportive  and an annular , the orbit gate  holds over a finite capture band in initial tangential speed (§5.5).

Bounded orbit: with tangentially supportive  and an annular , the orbit gate  holds over a finite capture band in initial tangential speed (§5.5).

* Anisotropy drift: the transverse drift sign matches ; setting  removes the drift (§5.7).

Anisotropy drift: the transverse drift sign matches ; setting  removes the drift (§5.7).

* Inverse-square–like trend: in a radial profile with , mid-track  with ; flipping the radial gradient reverses the bias (§5.11).

Inverse-square–like trend: in a radial profile with , mid-track  with ; flipping the radial gradient reverses the bias (§5.11).

* Interference control: visibility follows a cosine law in relative phase; constructive gain at  approaches linear superposition (§5.13). Each prediction carries a falsifier (null , remove , sign reversal, or control geometry) and is reported alongside budget accounting.

Interference control: visibility follows a cosine law in relative phase; constructive gain at  approaches linear superposition (§5.13). Each prediction carries a falsifier (null , remove , sign reversal, or control geometry) and is reported alongside budget accounting.

### 6.4 Numerical integrity and robustness

Main figures use 2562 grids; representative 5122 confirmations for deflection (§5.1) and containment (§5.5) reproduce primary metrics within error (Supplementary Appendix D,E). Profiles are -smoothed over 3–5 , unless an interface is intentionally sharp—in which case measured reflections are reported. Recorder spectra remain sub-Nyquist [36]; CFL and stability bounds are enforced per §4. The discrete energy identity closes by construction: declines are explained by Rayleigh loss (when ) and/or boundary flux (absorbing aprons), with post-transient drift within the stated tolerance.

### 6.5 Positioning relative to prior work (cf. §2)

Our results align with graded-index and anisotropic transport intuition and intersect the analog-gravity literature at the level of observables: we recover ray-like paths, delays, and capture behaviors via structured propagation in (). We do not model mass, forces, or spacetime curvature; agreement with eikonal predictions is treated as an observable mapping to , not a geometric equivalence. This stance clarifies scope while preserving predictive content and reproducibility.

## 7 Limitations and scope

Operational analogs, not GR. The claims in §5 are about observables produced by structured propagation in (). We do not model mass, forces, or spacetime curvature, and we make no attempt to solve Poisson/Einstein equations. Agreement with eikonal rays is treated as a mapping to , not a geometric equivalence.

Model class. Results use a linear scalar evolution with static fields  and ; no back-reaction ( do not depend on ). Unless noted, the source term is off (), so dynamics arise from the initial condition.

Damping ≠ potential.  is a loss channel (Rayleigh dissipation). With  the energy functional is not conserved and time-reversal symmetry is broken; “collapse/containment” reflect focusing + dissipation, not bound potentials.

Regime of validity (eikonal/gradients). Predictions for bending/curvature (§§5.1, 5.3, 5.8, 5.12) assume smooth  and weak gradients (slowly varying envelope). Strong gradients or sharp interfaces can introduce reflections and deviations; we typically smooth features over 3–5  and report measured reflections where sharp transitions are intentional.

Boundaries. Absorbing sponges approximate open domains and are not reflection-free; reflective basins are idealized. Small boundary effects can bias long-time energy tallies and late-stage trajectories; we mitigate by flux accounting and apron sweeps but cannot eliminate them entirely.

Discretization and stability. Results depend on finite ,  constrained by CFL; spectra approaching Nyquist may incur dispersion/aliasing. Main figures use 2562 grids; representative 5122 confirmations (5.1, 5.5; Appx D/E) reproduce primary metrics within error, but we do not claim full continuum extrapolation for every case.

Parameter sensitivity. Quantities such as capture bands, collapse/escape thresholds, and the inverse-square-like exponent depend on  focusing strength,  depth/width, and smoothing length. Reported slopes/exponents are extracted over mid-track windows; outside those windows the scaling may not hold.

Dimensionality. Baseline demonstrations are 2-D; selected 3-D confirmations are provided only where noted. We do not assume qualitative invariance of every effect under 3-D geometry.

Out of scope. Nonlinear self-interaction, time-dependent , , strongly dispersive/viscoelastic media, stochastic heterogeneity at scales comparable to , and hardware imperfections are not modeled here.

Mitigations and outlook. We partially address these limits via energy-budget closure, null/negative controls, and grid refinement on representative cases. Future work targets hardware validation, broader parameter sweeps (with uncertainty bands), heterogeneous , , and selective 3-D studies.

3D implications. Our demonstrations are 2D for clarity/efficiency; the framework and code generalize to 3D (Supplementary Appendix C). We expect quantitative shifts in stability/containment: e.g., different scaling of drift and radial “breathing” with basin curvature, and modified far-field decay rates from the 3D Green’s-function structure. The design-forward predictions (deflection vs. ; , ​ vs. basin shape) remain valid in 3D, but the acceptance gates will need 3D-specific calibration. A full 3D convergence/robustness study is slated as follow-on work.

## 8 Implications and predictions

### 8.1 Implications

The §5 suite shows that shaping () yields reproducible, falsifiable media analogs of gravitational-like observables. Practically, this enables: (i) benchmarking of transport solvers with pre-registered metrics and budget checks (bend , eikonal residuals, delay , capture time ​, orbit ratio , radial exponent , interference visibility ); (ii) design of graded media by steering with  and stabilizing/quenching with ; and (iii) inverse-design targets that translate desired paths/containment into constraints on  (transport) and  (loss).

### 8.2 Predictions (testable, with falsifiers)

* Linear deflection (weak gradients). ; sign flips under gradient reversal. (Falsifiers: ; reversal  flip; cf. Figures 1, 3).

Linear deflection (weak gradients). ; sign flips under gradient reversal. (Falsifiers: ; reversal  flip; cf. Figures 1, 3).

* Transit delay and escape.  increases monotonically with ; sufficiently large , , prevents exit within the window. (Falsifier: ; Figure 4).

Transit delay and escape.  increases monotonically with ; sufficiently large , , prevents exit within the window. (Falsifier: ; Figure 4).

* Bounded orbit (limit cycle). With tangentially supportive  and annular ,  over a finite capture band in initial tangential speed. (Falsifiers:  no sustained orbit; flatten  no containment; Figures 5, 12 confirm: E).

Bounded orbit (limit cycle). With tangentially supportive  and annular ,  over a finite capture band in initial tangential speed. (Falsifiers:  no sustained orbit; flatten  no containment; Figures 5, 12 confirm: E).

* Anisotropy drift. ;  removes drift. (Figure 7).

Anisotropy drift. ;  removes drift. (Figure 7).

* Inverse-square–like trend. For , mid-track  with ; flipping the radial gradient reverses bias. (Figure 11).

Inverse-square–like trend. For , mid-track  with ; flipping the radial gradient reverses bias. (Figure 11).

* Phase control.  follows a cosine law; constructive gain at  approaches linear superposition. (Figure 13).

Phase control.  follows a cosine law; constructive gain at  approaches linear superposition. (Figure 13).

Each prediction is paired with an explicit falsifier and is reported with energy-budget closure (Rayleigh loss and/or boundary flux; §4). Representative 5122 confirmations appear for deflection and containment (Appx D, E).

#### 8.2.1 Validation pathways

* Deflection/drift: graded-index or tilted-anisotropy plates/waveguides (null , sign reversal controls).

Deflection/drift: graded-index or tilted-anisotropy plates/waveguides (null , sign reversal controls).

* Delay/escape: programmable lossy slab with  ramp-in/out ( control).

Delay/escape: programmable lossy slab with  ramp-in/out ( control).

* Containment: annular  ring for limit cycles ( negative control).

Containment: annular  ring for limit cycles ( negative control).

* Interference: coherent pair with set  (phase scramble control).

Interference: coherent pair with set  (phase scramble control).

Experimental pathways. The transport tensor  and loss field  map naturally to engineered media: anisotropic acoustic metamaterials (spatially varying stiffness/density; off-diagonal couplings), photonic crystals/GRIN optics (index gradients as an optical transport analog), and loss-engineered layers (controlled attenuation as ). In such platforms, the predictions in §5 translate to design-forward tests: (i) deflection vs. local  (free-fall/bending), and (ii) drift rate  and radial period  vs. basin shape/smoothness (containment). The archived YAMLs (§9) provide exact fields and observables for bench replication; Supplementary Appendix D,E document grid-refinement checks.

### 8.3 Outlook

Near-term priorities: (i) hardware-in-the-loop confirmations for deflection (5.1/D) and containment (5.5/E); (ii) parameter-swept capture maps with uncertainty bands; (iii) robustness under heterogeneous/noisy , ; and (iv) selective 3-D demonstrations where geometry matters.

## 9 Data, code, and reproducibility

### 9.1 Dataset (all figures/results)

Record. Simulating Gravitational Dynamics via Scalar Field Propagation: Dataset—Zenodo, version DOI 10.5281/zenodo.17080017; license CC BY 4.0.

Contents. Per-phenomenon bundles (grav_5_1_* … grav_5_13_*) with raw arrays, summary. json, observables. csv, exact YAML configs, figures, and SHA-256 checksums.

Direct pointers for grid-refinement checks.

* §5.1 (Free-fall) 5122 repeat → Supplementary Appendix D. Dataset bundle: grav_5_1_free_fall_512.

§5.1 (Free-fall) 5122 repeat → Supplementary Appendix D. Dataset bundle: grav_5_1_free_fall_512.

* §5.5 (Orbital containment) 5122 repeat → Supplementary Appendix D. Dataset bundle: grav_5_5_orbital_containment_512.

§5.5 (Orbital containment) 5122 repeat → Supplementary Appendix D. Dataset bundle: grav_5_5_orbital_containment_512.

Cite this dataset as:

Toupin, B. (2025). Simulating Gravitational Dynamics via Scalar Field Propagation: Dataset. Zenodo. https://doi.org/10.5281/zenodo.17080017.

#### 9.1.1 Software (URFTSim engine and scripts)

Record. URFTSim (V6-IR) — Zenodo, version DOI 10.5281/zenodo.17088949; license MIT. Includes the simulator, batch/figure scripts, YAML configs, environment files, and CITATION. cff.

Reproducing this paper.

* Install from the software record (env files provided).

Install from the software record (env files provided).

* Run the exact YAML in the corresponding dataset bundle (configs are mirrored in both records).

Run the exact YAML in the corresponding dataset bundle (configs are mirrored in both records).

* Generate timelines/exposures with the included scripts and compare metrics to those reported in §5 and Supplementary Appendix D,E.

Generate timelines/exposures with the included scripts and compare metrics to those reported in §5 and Supplementary Appendix D,E.

Cite this software as:

Toupin, B. (2025). URFTSim (V6-IR) [Computer software]. Zenodo. https://doi.org/10.5281/zenodo.17088949.

### 9.2 Reproduction checklist (what to verify where)

* §5.1 Free-fall: Recompute bend angle and early-time quadratic fit , ​ from observables. csv. Expected values are listed in Supplementary Appendix D (table row + paragraph).

§5.1 Free-fall: Recompute bend angle and early-time quadratic fit , ​ from observables. csv. Expected values are listed in Supplementary Appendix D (table row + paragraph).

* §5.5 Orbital containment: Recompute , , , , ​ from observables. csv using the definitions in Supplementary Appendix C (methods) and compare to Supplementary Appendix E.

§5.5 Orbital containment: Recompute , , , , ​ from observables. csv using the definitions in Supplementary Appendix C (methods) and compare to Supplementary Appendix E.

* Acceptance gates: Each §5 case specifies its metric and pass criteria; reproduced values should fall within the gates given in the figure caption or corresponding appendix.

Acceptance gates: Each §5 case specifies its metric and pass criteria; reproduced values should fall within the gates given in the figure caption or corresponding appendix.

### 9.3 Provenance and integrity

* Determinism: All runs specify seeds; results are repeatable under the stated precision.

Determinism: All runs specify seeds; results are repeatable under the stated precision.

* Integrity checks: Verify downloads using the SHA-256 checksums shipped alongside each bundle.

Integrity checks: Verify downloads using the SHA-256 checksums shipped alongside each bundle.

* Energy proxy: Definition and caveats are in Supplementary Appendix C.1; raw  series are included for every run.

Energy proxy: Definition and caveats are in Supplementary Appendix C.1; raw  series are included for every run.

### 9.4 Licensing and reuse

* Data and figures: CC BY 4.0 (attribute the dataset record).

Data and figures: CC BY 4.0 (attribute the dataset record).

* Code: MIT (retain copyright notice).

Code: MIT (retain copyright notice).

## 10 Conclusion

We presented a unified scalar-propagation framework in which structured () produces reproducible media analogs of gravitational-like observables. The §5 suite covers deflection and ray bending, dissipative collapse/containment and delay, anisotropy-driven drift, repulsion via curvature inversion, and phase-sensitive interference. Each phenomenon is stated as a primary metric with an explicit falsifier, and accepted runs satisfy pre-registered acceptance gates (metric pass, energy-budget closure within 1%–3% post-transient, spectral safety, robustness).

Our contribution is practical and falsifiable. (i) We make the update rules and discrete energy identity operational by tallying Rayleigh loss and boundary flux in every experiment. (ii) We separate transport effects (from ) from loss (from ), showing that collapse/containment arise from focusing + dissipation, not from bound potentials. (iii) We package end-to-end reproducibility: configs, code, outputs, and figure scripts (see §9), with representative 5122 grid-refinement checks for deflection (D) and orbital containment (E).

Scope is explicit: these are media analogs, not statements about mass, forces, or spacetime curvature. Agreement with eikonal predictions is treated as an observable mapping to , not a geometric equivalence; damping is a loss channel, not stored energy.

The framework carries predictive value: linear deflection vs. , monotone transit delay vs.  with escape thresholds, bounded orbits with  over a capture band, anisotropy-set drift, an inverse-square–like radial trend, and phase-controlled interference (see §8). Each prediction has a built-in null/negative control.

Looking ahead, we target (i) hardware-in-the-loop confirmations for deflection and containment; (ii) parameter-swept capture maps with uncertainty bands; (iii) robustness under heterogeneous/noisy , ; and (iv) selective 3-D validations where geometry matters. We also see immediate use as benchmarks for transport solvers and as design cues for graded media via inverse constraints on  (steering) and  (stabilization).

All materials needed to replicate and extend these results are archived (DOI, commit, bundles in §9).

## Statements

### Data availability statement

The original contributions presented in the study are included in the article/Supplementary Material, further inquiries can be directed to the corresponding author.

### Author contributions

BT: Conceptualization, Methodology, Software, Validation, Formal analysis, Investigation, Data curation, Visualization, Writing – original draft, Writing – review and editing.

### Funding

The author(s) declare that no financial support was received for the research and/or publication of this article.

### Conflict of interest

Author BT was employed by DIRECTV LLC.

### Generative AI statement

The author(s) declare that Generative AI was used in the creation of this manuscript. The author affirms that all scientific content including theoretical models, derivations, simulations, and interpretations was independently developed. Generative AI (OpenAI ChatGPT) was used solely for technical assistance in simulation rendering, language refinement, and formatting. No scientific results or theoretical frameworks were generated by AI. All final content was authored, reviewed, and verified by the human researcher.

Any alternative text (alt text) provided alongside figures in this article has been generated by Frontiers with the support of artificial intelligence and reasonable efforts have been made to ensure accuracy, including review by the authors wherever possible. If you identify any issues, please contact us.

### Publisher’s note

All claims expressed in this article are solely those of the authors and do not necessarily represent those of their affiliated organizations, or those of the publisher, the editors and the reviewers. Any product that may be evaluated in this article, or claim that may be made by its manufacturer, is not guaranteed or endorsed by the publisher.

### Supplementary material

The Supplementary Material for this article can be found online at: https://www.frontiersin.org/articles/10.3389/fphy.2025.1672745/full#supplementary-material

## References

* 1.MisnerCWThorneKSWheelerJA. Gravitation. San Francisco: W. H. Freeman (1973).Google Scholar

1.

MisnerCWThorneKSWheelerJA. Gravitation. San Francisco: W. H. Freeman (1973).

* Google Scholar
* 2.WaldRM. General relativity. Chicago: University of Chicago Press (1984).Google Scholar

2.

WaldRM. General relativity. Chicago: University of Chicago Press (1984).

* Google Scholar
* 3.CarrollSM. Spacetime and geometry: an introduction to general relativity. San Francisco: Addison-Wesley (2004).Google Scholar

3.

CarrollSM. Spacetime and geometry: an introduction to general relativity. San Francisco: Addison-Wesley (2004).

* Google Scholar
* 4.UnruhWG. Experimental black-hole evaporation?Phys Rev Lett (1981) 46:1351–3. 10.1103/PhysRevLett.46.1351CrossRef Google Scholar

4.

UnruhWG. Experimental black-hole evaporation?Phys Rev Lett (1981) 46:1351–3. 10.1103/PhysRevLett.46.1351

* CrossRef
* Google Scholar
* 5.VisserM. Acoustic Black holes: horizons, ergospheres and hawking radiation. Class Quan Grav (1998) 15:1767–91. 10.1088/0264-9381/15/6/024CrossRef Google Scholar

5.

VisserM. Acoustic Black holes: horizons, ergospheres and hawking radiation. Class Quan Grav (1998) 15:1767–91. 10.1088/0264-9381/15/6/024

* CrossRef
* Google Scholar
* 6.BarcelóCLiberatiSVisserM. Analogue gravity. Living Rev Relativ (2011) 14:3. 10.12942/lrr-201-3Pubmed AbstractCrossRef Google Scholar

6.

BarcelóCLiberatiSVisserM. Analogue gravity. Living Rev Relativ (2011) 14:3. 10.12942/lrr-201-3

* Pubmed Abstract
* CrossRef
* Google Scholar
* 7.LeonhardtU. Optical conformal mapping. Science (2006) 312(5781):1777–80. 10.1126/science.1126493Pubmed AbstractCrossRef Google Scholar

7.

LeonhardtU. Optical conformal mapping. Science (2006) 312(5781):1777–80. 10.1126/science.1126493

* Pubmed Abstract
* CrossRef
* Google Scholar
* 8.PhilbinTGKuklewiczCRobertsonSHillSKönigFLeonhardtU. Fiber-optic analogue of the event horizon. Science (2008) 319(5868):1367–70. 10.1126/science.1153625Pubmed AbstractCrossRef Google Scholar

8.

PhilbinTGKuklewiczCRobertsonSHillSKönigFLeonhardtU. Fiber-optic analogue of the event horizon. Science (2008) 319(5868):1367–70. 10.1126/science.1153625

* Pubmed Abstract
* CrossRef
* Google Scholar
* 9.PretoriusF. Evolution of binary black-hole spacetimes. Phys Rev Lett (2005) 95:121101. 10.1103/PhysRevLett.95.121101Pubmed AbstractCrossRef Google Scholar

9.

PretoriusF. Evolution of binary black-hole spacetimes. Phys Rev Lett (2005) 95:121101. 10.1103/PhysRevLett.95.121101

* Pubmed Abstract
* CrossRef
* Google Scholar
* 10.AlcubierreM. Introduction to 3+1 numerical relativity. Oxford: Oxford University Press (2008).Google Scholar

10.

AlcubierreM. Introduction to 3+1 numerical relativity. Oxford: Oxford University Press (2008).

* Google Scholar
* 11.BaumgarteTWShapiroSL. Numerical relativity: solving einstein’s equations on the computer. Cambridge: Cambridge University Press (2010).Google Scholar

11.

BaumgarteTWShapiroSL. Numerical relativity: solving einstein’s equations on the computer. Cambridge: Cambridge University Press (2010).

* Google Scholar
* 12.DysonFWEddingtonASDavidsonC. A determination of the deflection of light by the Sun’s gravitational field. Philos Trans R Soc A (1920) 220:291–333. 10.1098/rsta.1920.0009CrossRef Google Scholar

12.

DysonFWEddingtonASDavidsonC. A determination of the deflection of light by the Sun’s gravitational field. Philos Trans R Soc A (1920) 220:291–333. 10.1098/rsta.1920.0009

* CrossRef
* Google Scholar
* 13.ShapiroII. Fourth test of general relativity. Phys Rev Lett (1964) 13:789–91. 10.1103/PhysRevLett.13.789CrossRef Google Scholar

13.

ShapiroII. Fourth test of general relativity. Phys Rev Lett (1964) 13:789–91. 10.1103/PhysRevLett.13.789

* CrossRef
* Google Scholar
* 14.PoundRVRebkaGAJr. Apparent weight of photons. Phys Rev Lett (1960) 4:337–41. 10.1103/PhysRevLett.4.337CrossRef Google Scholar

14.

PoundRVRebkaGAJr. Apparent weight of photons. Phys Rev Lett (1960) 4:337–41. 10.1103/PhysRevLett.4.337

* CrossRef
* Google Scholar
* 15.PendryJBSchurigDSmithDR. Controlling electromagnetic fields. Science (2006) 312(5781):1780–2. 10.1126/science.1125907Pubmed AbstractCrossRef Google Scholar

15.

PendryJBSchurigDSmithDR. Controlling electromagnetic fields. Science (2006) 312(5781):1780–2. 10.1126/science.1125907

* Pubmed Abstract
* CrossRef
* Google Scholar
* 16.LuneburgRK. Mathematical theory of optics. Berkeley: University of California Press (1964).Google Scholar

16.

LuneburgRK. Mathematical theory of optics. Berkeley: University of California Press (1964).

* Google Scholar
* 17.BornMWolfE. Principles of optics. 7th ed. Cambridge: Cambridge University Press (1999).Google Scholar

17.

BornMWolfE. Principles of optics. 7th ed. Cambridge: Cambridge University Press (1999).

* Google Scholar
* 18.CerjanCKosloffDKosloffRReshefM. A nonreflecting boundary condition for discrete acoustic and elastic wave equations. Geophysics (1985) 50(4):705–8. 10.1190/1.1441945CrossRef Google Scholar

18.

CerjanCKosloffDKosloffRReshefM. A nonreflecting boundary condition for discrete acoustic and elastic wave equations. Geophysics (1985) 50(4):705–8. 10.1190/1.1441945

* CrossRef
* Google Scholar
* 19.BerengerJ-P. A perfectly matched layer for the absorption of electromagnetic waves. J Comput Phys (1994) 114(2):185–200. 10.1006/jcph.1994.1159CrossRef Google Scholar

19.

BerengerJ-P. A perfectly matched layer for the absorption of electromagnetic waves. J Comput Phys (1994) 114(2):185–200. 10.1006/jcph.1994.1159

* CrossRef
* Google Scholar
* 20.EinsteinA. Die Grundlage der allgemeinen Relativitätstheorie. Annalen der Physik (1916) 49(7):769–822. 10.1002/andp.19163540702CrossRef Google Scholar

20.

EinsteinA. Die Grundlage der allgemeinen Relativitätstheorie. Annalen der Physik (1916) 49(7):769–822. 10.1002/andp.19163540702

* CrossRef
* Google Scholar
* 21.BransCDickeRH. Mach's principle and a relativistic theory of gravitation. Phys Rev (1961) 124:925–35. 10.1103/physrev.124.925CrossRef Google Scholar

21.

BransCDickeRH. Mach's principle and a relativistic theory of gravitation. Phys Rev (1961) 124:925–35. 10.1103/physrev.124.925

* CrossRef
* Google Scholar
* 22.SakharovAD. Vacuum quantum fluctuations in curved space and the theory of gravitation. Sov Phys Dokl.* (1968) 12:1040–1.Google Scholar

22.

SakharovAD. Vacuum quantum fluctuations in curved space and the theory of gravitation. Sov Phys Dokl.* (1968) 12:1040–1.

* Google Scholar
* 23.JacobsonT. Thermodynamics of spacetime: the einstein equation of state. Phys Rev Lett (1995) 75:1260–3. 10.1103/physrevlett.75.1260Pubmed AbstractCrossRef Google Scholar

23.

JacobsonT. Thermodynamics of spacetime: the einstein equation of state. Phys Rev Lett (1995) 75:1260–3. 10.1103/physrevlett.75.1260

* Pubmed Abstract
* CrossRef
* Google Scholar
* 24.VerlindeE. On the origin of gravity and the laws of newton. J High Energ Phys (2011) 2011(4):29. 10.1007/jhep04(2011)029CrossRef Google Scholar

24.

VerlindeE. On the origin of gravity and the laws of newton. J High Energ Phys (2011) 2011(4):29. 10.1007/jhep04(2011)029

* CrossRef
* Google Scholar
* 25.VolovikG. the universe in a helium droplet. Oxford University Press (2003).Google Scholar

25.

VolovikG. the universe in a helium droplet. Oxford University Press (2003).

* Google Scholar
* 26.LeonhardtUPiwnickiP. Relativistic effects of light in moving media with extremely low group velocity. Phys Rev Lett (2000) 84:822–5. 10.1103/physrevlett.84.822Pubmed AbstractCrossRef Google Scholar

26.

LeonhardtUPiwnickiP. Relativistic effects of light in moving media with extremely low group velocity. Phys Rev Lett (2000) 84:822–5. 10.1103/physrevlett.84.822

* Pubmed Abstract
* CrossRef
* Google Scholar
* 27.BarcelóCLiberatiSVisserM. Analogue gravity. Living Rev Relativity (2011) 14:3. 10.12942/lrr-2011-3Pubmed AbstractCrossRef Google Scholar

27.

BarcelóCLiberatiSVisserM. Analogue gravity. Living Rev Relativity (2011) 14:3. 10.12942/lrr-2011-3

* Pubmed Abstract
* CrossRef
* Google Scholar
* 28.TuringAM. The chemical basis of morphogenesis. Philosophical Trans R Soc B (1952) 237:37–72. 10.1098/rstb.1952.0012CrossRef Google Scholar

28.

TuringAM. The chemical basis of morphogenesis. Philosophical Trans R Soc B (1952) 237:37–72. 10.1098/rstb.1952.0012

* CrossRef
* Google Scholar
* 29.MeinhardtH. Models of biological pattern formation. Academic Press (1982).Google Scholar

29.

MeinhardtH. Models of biological pattern formation. Academic Press (1982).

* Google Scholar
* 30.AldrovandiRPereiraJG. Teleparallel gravity: an introduction. Springer (2013).Google Scholar

30.

AldrovandiRPereiraJG. Teleparallel gravity: an introduction. Springer (2013).

* Google Scholar
* 31.GomesHGrybSKoslowskiT. Einstein gravity as a 3D conformally invariant theory. Quan Grav (2011) 28:045005. 10.1088/0264-9381/28/4/045005CrossRef Google Scholar

31.

GomesHGrybSKoslowskiT. Einstein gravity as a 3D conformally invariant theory. Quan Grav (2011) 28:045005. 10.1088/0264-9381/28/4/045005

* CrossRef
* Google Scholar
* 32.RovelliCSmolinL. Loop space representation of quantum general relativity. Nucl Phys B (1990) 331(1):80–152. 10.1016/0550-3213(90)90019-aCrossRef Google Scholar

32.

RovelliCSmolinL. Loop space representation of quantum general relativity. Nucl Phys B (1990) 331(1):80–152. 10.1016/0550-3213(90)90019-a

* CrossRef
* Google Scholar
* 33.PolchinskiJ, I and II. Cambridge University Press (1998).Google Scholar

33.

PolchinskiJ, I and II. Cambridge University Press (1998).

* Google Scholar
* 34.CourantRFriedrichsKLewyH. On the partial difference equations of mathematical physics. IBM J (1967) 11(2):215–34. 10.1147/rd.112.0215CrossRef Google Scholar

34.

CourantRFriedrichsKLewyH. On the partial difference equations of mathematical physics. IBM J (1967) 11(2):215–34. 10.1147/rd.112.0215

* CrossRef
* Google Scholar
* 35.MorsePMIngardKU. Theoretical acoustics. Princeton University Press (1986).Google Scholar

35.

MorsePMIngardKU. Theoretical acoustics. Princeton University Press (1986).

* Google Scholar
* 36.LoganJD. Applied partial differential equations. Springer (2015).Google Scholar

36.

LoganJD. Applied partial differential equations. Springer (2015).

* Google Scholar
* 37.MurrayJD. Mathematical biology I: an introduction. Springer (2002).Google Scholar

37.

MurrayJD. Mathematical biology I: an introduction. Springer (2002).

* Google Scholar
* 38.WhithamGB. Linear and nonlinear waves. Wiley (1974).Google Scholar

38.

WhithamGB. Linear and nonlinear waves. Wiley (1974).

* Google Scholar
* 39.JacksonJD. Classical electrodynamics. Wiley (1998).Google Scholar

39.

JacksonJD. Classical electrodynamics. Wiley (1998).

* Google Scholar
* 40.LandauLDLifshitzEM. The classical theory of fields. Pergamon (1975).Google Scholar

40.

LandauLDLifshitzEM. The classical theory of fields. Pergamon (1975).

* Google Scholar
* 41.BatemanH. On dissipative systems and related variational principles. Phys Rev (1931) 38(4):815–9. 10.1103/physrev.38.815CrossRef Google Scholar

41.

BatemanH. On dissipative systems and related variational principles. Phys Rev (1931) 38(4):815–9. 10.1103/physrev.38.815

* CrossRef
* Google Scholar
* 42.EringenAC. Mechanics of continua. Malabar, FL, USA: R. E. Krieger Publishing Company (1980).Google Scholar

42.

EringenAC. Mechanics of continua. Malabar, FL, USA: R. E. Krieger Publishing Company (1980).

* Google Scholar
* 43.HossenfelderSMisteleT. A bimetric theory with exchange symmetry. Phys. Rev. (2008) 78. 10.1103/PhysRevD.78.044015CrossRef Google Scholar

43.

HossenfelderSMisteleT. A bimetric theory with exchange symmetry. Phys. Rev. (2008) 78. 10.1103/PhysRevD.78.044015

* CrossRef
* Google Scholar
* 44.de RhamC. Massive gravity. Living Rev Relativ (2021) 17:7. 10.12942/lrr-2014-7Pubmed AbstractCrossRef Google Scholar

44.

de RhamC. Massive gravity. Living Rev Relativ (2021) 17:7. 10.12942/lrr-2014-7

* Pubmed Abstract
* CrossRef
* Google Scholar
* 45.PadmanabhanT. Thermodynamical aspects of gravity: new insights. Rep Prog Phys (2010) 73:046901. 10.1088/0034-4885/73/4/046901CrossRef Google Scholar

45.

PadmanabhanT. Thermodynamical aspects of gravity: new insights. Rep Prog Phys (2010) 73:046901. 10.1088/0034-4885/73/4/046901

* CrossRef
* Google Scholar
* 46.PoissonE. The motion of point particles in curved spacetime. Living Rev Relativity (2004) 7(1):6. 10.12942/lrr-2004-6Pubmed AbstractCrossRef Google Scholar

46.

PoissonE. The motion of point particles in curved spacetime. Living Rev Relativity (2004) 7(1):6. 10.12942/lrr-2004-6

* Pubmed Abstract
* CrossRef
* Google Scholar
* 47.JacobsonT. Trans-planckian redshifts and the substance of the space-time river. Prog Theor Phys Suppl (1999) 136:1–17. 10.1143/ptps.136.1CrossRef Google Scholar

47.

JacobsonT. Trans-planckian redshifts and the substance of the space-time river. Prog Theor Phys Suppl (1999) 136:1–17. 10.1143/ptps.136.1

* CrossRef
* Google Scholar
* 48.BarcelóL. Living rev. Relativity (canonical survey) (2000).Google Scholar

48.

BarcelóL. Living rev. Relativity (canonical survey) (2000).

* Google Scholar
* 49.CliffordW. The confrontation between general relativity and experiment. Living Rev Relativity (2001). 10.12942/lrr-2014-4CrossRef Google Scholar

49.

CliffordW. The confrontation between general relativity and experiment. Living Rev Relativity (2001). 10.12942/lrr-2014-4

* CrossRef
* Google Scholar

## Summary

Keywords

analogue gravity, scalar-field propagation, anisotropic wave equation, graded-index media, damping, geodesic analogue, orbital containment, reproducible simulations

Citation

Toupin B (2025) Simulating gravitational dynamics via scalar field propagation. Front. Phys. 13:1672745. doi: 10.3389/fphy.2025.1672745

Received

24 July 2025

Revised

16 September 2025

Accepted

26 September 2025

Published

11 November 2025

Volume

13 - 2025

Edited by

Jisheng Kou, Shaoxing University, China

Reviewed by

Saravana Prakash Thirumuruganandham, SIT Health, Ecuador

Saken Toktarbay, Al-Farabi Kazakh National University Institute of Experimental and Theoretical Physics, Kazakhstan

Updates

Check for updates

Copyright

© 2025 Toupin.

This is an open-access article distributed under the terms of the Creative Commons Attribution License (CC BY). The use, distribution or reproduction in other forums is permitted, provided the original author(s) and the copyright owner(s) are credited and that the original publication in this journal is cited, in accordance with accepted academic practice. No use, distribution or reproduction is permitted which does not comply with these terms.

*Correspondence: Brendan Toupin, bbtoupin@gmail.com

Disclaimer

All claims expressed in this article are solely those of the authors and do not necessarily represent those of their affiliated organizations, or those of the publisher, the editors and the reviewers. Any product that may be evaluated in this article or claim that may be made by its manufacturer is not guaranteed or endorsed by the publisher.

## Outline

## Figures

## Cite article

Copy to clipboard

Export citation file

* BibTex
* EndNote
* Reference Manager
* Simple Text file

## Share article

* Facebook
* X
* LinkedIn
* Email
* WeChat

Share on WeChat

Scan with WeChat to share this article

## Article metrics