# Strouhal number - Wikipedia
Source URL: https://en.wikipedia.org/wiki/Strouhal_number

In dimensional analysis, the Strouhal number (St, or sometimes Sr to avoid the conflict with the Stanton number) is a dimensionless number describing oscillating flow mechanisms. The parameter is named after Vincenc Strouhal, a Czech physicist who experimented in 1878 with wires experiencing vortex shedding and singing in the wind.[1][2] The Strouhal number is an integral part of the fundamentals of fluid mechanics.

The Strouhal number is often given as

St

=

f
L

U

,

{\displaystyle {\text{St}}={\frac {fL}{U}},}

where f is the frequency of vortex shedding in Hertz,[3] L is the characteristic length (for example, hydraulic diameter or airfoil thickness), and U is the average flow speed in meters per second. In certain cases, like heaving (plunging) flight, this characteristic length is the amplitude of oscillation. This selection of characteristic length can be used to present a distinction between Strouhal number and reduced frequency:

St

=

k
A

π
c

,

{\displaystyle {\text{St}}={\frac {kA}{\pi c}},}

where k is the reduced frequency, and A is amplitude of the heaving oscillation.

In the case of uniform flow past a fixed cylinder, the cylinder's diameter is the characteristic length. In that case, the Strouhal number is a function of the Reynolds number based on diameter, 

R
e

D

=
ρ
V
D

/

μ

{\displaystyle \mathrm {Re} _{D}=\rho VD/\mu }

,[5] where 

ρ

{\displaystyle \rho }

 is the fluid's density (kg/m3) and 

μ

{\displaystyle \mu }

 [kg-m/s] is the fluid's dynamic viscosity.  Over four orders of magnitude in Reynolds number, from 102 to 105, the value of the Strouhal number remains close to 0.2 (see figure).

For spheres in uniform flow in the Reynolds number range of 8×102 < Re < 2×105 there co-exist two values of the Strouhal number.  The lower frequency is attributed to the large-scale instability of the wake, is independent of the Reynolds number Re, and is approximately equal to 0.2.  The higher-frequency Strouhal number is caused by small-scale instabilities from the separation of the shear layer.[6][7]

For large Strouhal numbers (order of 1), viscosity dominates fluid flow, resulting in a collective oscillating movement of the fluid "plug". For low Strouhal numbers (order of 10−4 and below), the high-speed, quasi-steady-state portion of the movement dominates the oscillation. Oscillation at intermediate Strouhal numbers is characterized by the buildup and rapidly subsequent shedding of vortices.[8][clarification needed]

## Derivation

Knowing Newton's second law stating force is equivalent to mass times acceleration, or 

F
=
m
a

{\displaystyle F=ma}

, and that acceleration is the derivative of velocity, or 

U
t

{\displaystyle {\tfrac {U}{t}}}

 (characteristic speed/time) in the case of fluid mechanics, we see

Since characteristic speed can be represented as length per unit time, 

L
t

{\displaystyle {\tfrac {L}{t}}}

, we get

where,

Dividing both sides by 

m

U

2

L

{\displaystyle {\tfrac {mU^{2}}{L}}}

, we get

where,

This provides a dimensionless basis for a relationship between mass, characteristic speed, net external forces, and length (size) which can be used to analyze the effects of fluid mechanics on a body with mass.

If the net external forces are predominantly elastic, we can use Hooke's law to see

where,

Assuming 

Δ
L
∝
L

{\displaystyle \Delta L\propto L}

, then 

F
≈
k
L

{\displaystyle F\approx kL}

. With the natural resonant frequency of the elastic system, 

ω

0

2

{\displaystyle \omega _{0}^{2}}

, being equal to 

k
m

{\displaystyle {\tfrac {k}{m}}}

, we get

where,

Given that cyclic motion frequency can be represented by 

f
=

ω

0

2

L

U

{\displaystyle f={\tfrac {\omega _{0}^{2}L}{U}}}

 we get,

where,

## Applications

### Micro/Nanorobotics

In the field of micro and nanorobotics, the Strouhal number is used alongside the Reynolds number in analyzing the impact of an external oscillatory fluidic flow on the body of a microrobot. When considering a microrobot with cyclic motion, the Strouhal number can be evaluated as

where,

The analysis of a microrobot using the Strouhal number allows one to assess the impact that the motion of the fluid it is in has on its motion in relation to the inertial forces acting on the robot–regardless of the dominant forces being elastic or not.[9]

### Medical

In the medical field, microrobots that use swimming motions to move may make micromanipulations in unreachable environments.

The equation used for a blood vessel:[10]

where,

The Strouhal number is used as a ratio of the Deborah number (De) and Weissenberg number (Wi):[10]

The Strouhal number may also be used to obtain the Womersley number (Wo). The case for blood flow can be categorized as an unsteady viscoelastic flow, therefore the Womersley number is[10]

Or considering both equations,

### Metrology

In metrology, specifically axial-flow turbine meters, the Strouhal number is used in combination with the Roshko number to give a correlation between flow rate and frequency.  The advantage of this method over the frequency/viscosity versus K-factor method is that it takes into account temperature effects on the meter.

where,

This relationship leaves Strouhal dimensionless, although a dimensionless approximation is often used for C3, resulting in units of pulses/volume (same as K-factor).

This relationship between flow and frequency can also be found in the aeronautical field. Considering pulsating methane-air coflow jet diffusion flames, we get

where,

For a small Strouhal number (St=0.1) the modulation forms a deviation in the flow that travels very far downstream. As the Strouhal number grows, the non-dimensional frequency approaches the natural frequency of a flickering flame, and eventually will have greater pulsation than the flame.[11]

### Animal locomotion

In swimming or flying animals, Strouhal number is defined as

where,

In animal flight or swimming, propulsive efficiency is high over a narrow range of Strouhal constants, generally peaking in the 0.2 < St < 0.4 range.[12] This range is used in the swimming of dolphins, sharks, and bony fish, and in the cruising flight of birds, bats and insects.[12] However, in other forms of flight other values are found.[12] Intuitively the ratio measures the steepness of the strokes, viewed from the side (e.g., assuming movement through a stationary fluid) – f is the stroke frequency, A is the amplitude, so the numerator fA is half the vertical speed of the wing tip, while the denominator V is the horizontal speed. Thus the graph of the wing tip forms an approximate sinusoid with aspect (maximal slope) twice the Strouhal constant.[13]

#### Efficient motion

The Strouhal number is most commonly used for assessing oscillating flow as a result of an object's motion through a fluid. The Strouhal number reflects the difficulty for animals to travel efficiently through a fluid with their cyclic propelling motions. The number relates to propulsive efficiency, which peaks between 70%–80% when within the optimal Strouhal number range of 0.2 to 0.4. Through the use of factors such as the stroke frequency, the amplitude of each stroke, and velocity, the Strouhal number is able to analyze the efficiency and impact of an animal's propulsive forces through a fluid, such as those from swimming or flying. For instance, the value represents the constraints to achieve greater propulsive efficiency, which affects motion when cruising and aerodynamic forces when hovering.[14]

Greater reactive forces and properties that act against the object, such as viscosity and density, reduce the ability of an animal's motion to fall within the ideal Strouhal number range when swimming. Through the assessment of different species that fly or swim, it was found that the motion of many species of birds and fish falls within the optimal Strouhal range.[14] However, the Strouhal number varies more within the same species than other species based on the method of how they move in a constrained manner in response to aerodynamic forces.[14]

#### Example: Alcid

The Strouhal number has significant importance in analyzing the flight of animals since it is based on the streamlines and the animal's velocity as it travels through the fluid. Its significance is demonstrated through the motion of alcids as it passes through different mediums (air to water). The assessment of alcids determined the peculiarity of being able to fly under the efficient Strouhal number range in air and water despite a high mass relative to their wing area.[15] The alcid's efficient dual-medium motion developed through natural selection where the environment played a role in the evolution of animals over time to fall under a certain efficient range. The dual-medium motion demonstrates how alcids had two different flight patterns based on the stroke velocities as it moved through each fluid.[15] However, as the bird travels through a different medium, it has to face the influence of the fluid's density and viscosity. Furthermore, the alcid also has to resist the upward-acting buoyancy as it moves horizontally.

## Scaling of the Strouhal number

### Scale Analysis

In order to determine significance of the Strouhal number at varying scales, one may perform scale analysis–a simplification method to analyze the impact of factors as they change with respect to some scale. When considered in the context of microrobotics and nanorobotics, size is the factor of interest when performing scale analysis.

Scale analysis of the Strouhal number allows for analysis of the relationship between mass and inertial forces as both change with respect to size. Taking its original underived form, 

m

U

2

F
L

{\displaystyle {\tfrac {mU^{2}}{FL}}}

, we can then relate each term to size and see how the ratio changes as size changes.

Given 

m
=
V
ρ

{\displaystyle m=V\rho }

 where m is mass, V is volume, and 

ρ

{\displaystyle \rho }

 is density, we can see mass is directly related to size as volume scales with length (L). Taking the volume to be 

L

3

{\displaystyle L^{3}}

, we can directly relate mass and size as

Characteristic speed (U) is in terms of 

distance
time

{\displaystyle {\tfrac {\text{distance}}{\text{time}}}}

, and relative distance scales with size, therefore

The net external forces (F) scales in relation to mass and acceleration, given by 

F
=
m
⋅
a

{\displaystyle F=m\cdot a}

. Acceleration is in terms of 

distance

time

2

{\displaystyle {\tfrac {\text{distance}}{{\text{time}}^{2}}}}

, therefore 

a
≈
L

{\displaystyle a\approx L}

. The mass-size relationship was established to be 

m
≈

L

3

{\displaystyle m\approx L^{3}}

, so considering all three relationships, we get

Length (L) already denotes size and remains L.

Taking all of this together, we get

With the Strouhal number relating the mass to inertial forces, this can be expected as these two factors will scale proportionately with size and neither will increase nor decrease in significance with respect to their contribution to the body's behavior in the cyclic motion of the fluid.

### Relationship with the Richardson number

The scaling relationship between the Richardson number and the Strouhal number is represented by the equation:[16]

where a and b are constants depending on the condition.

For round helium buoyant jets and plumes:[16]

When 

Ri

<
100

{\displaystyle {\text{Ri}}<100}

,

When 

100
<

Ri

<
500

{\displaystyle 100<{\text{Ri}}<500}

,

For planar buoyant jets and plumes:[16]

For shape-independent scaling:[16]

### Relationship with Reynolds number

The Strouhal number and Reynolds number must be considered when addressing the ideal method to develop a body made to move through a fluid. Furthermore, the relationship for these values is expressed through Lighthill's elongated-body theory, which relates the reactive forces experienced by a body moving through a fluid with its inertial forces.[17] The Strouhal number was determined to depend upon the dimensionless Lighthill number, which in turn relates to the Reynolds number. The value of the Strouhal number can then be seen to decrease with an increasing Reynolds number, and to increase with an increasing Lighthill number.[17]

## See also

* Aeroelastic flutter – Interactions among inertial, elastic, and aerodynamic forcesPages displaying short descriptions of redirect targets
* Froude number – Dimensionless number; ratio of a fluid's flow inertia to the external field
* Kármán vortex street – Repeating pattern of swirling vortices
* Mach number – Dimensionless quantity in fluid dynamics
* Reynolds number – Ratio of inertial to viscous forces acting on a liquid
* Rossby number – Ratio of inertial force to Coriolis force
* Weber number – Dimensionless number in fluid mechanics
* Womersley number – Dimensionless expression of the pulsatile flow frequency in relation to viscous effects
* Weissenberg number – Dimensionless parameter in fluid mechanics
* Deborah number – Dimensionless number in rheology
* Richardson number – Dimensionless metric in fluid dynamics

## References

* ^ Strouhal, V. (1878) "Ueber eine besondere Art der Tonerregung" (On an unusual sort of sound excitation), Annalen der Physik und Chemie, 3rd series, 5 (10) : 216–251.
* ^ White, Frank M. (1999). Fluid Mechanics (4th ed.). McGraw Hill. ISBN 978-0-07-116848-9.
* ^ Triantafyllou, M. S.; Triantafyllou, G. S.; Gopalkrishnan, R. (8 August 1991). "Wake mechanics for thrust generation in oscillating foils" (PDF). Physics of Fluids A: Fluid Dynamics. 3 (12): 2835. Bibcode:1991PhFlA...3.2835T. doi:10.1063/1.858173. Retrieved 13 August 2024.[permanent dead link]
* ^ Lienhard, John H (1966). Synopsis of Lift, Drag, and Vortex Frequency Data for Rigid Circular Cylinders (PDF). College of Engineering, Research Division (Report). Bulletin 300. Pullman, WA: Washington State University. Retrieved February 7, 2025.
* ^ "Strouhal number". Archived from the original on 2010-05-01. Retrieved 2020-02-14.
* ^ Kim, K. J.; Durbin, P. A. (1988). "Observations of the frequencies in a sphere wake and drag increase by acoustic excitation". Physics of Fluids. 31 (11): 3260–3265. Bibcode:1988PhFl...31.3260K. doi:10.1063/1.866937.
* ^ Sakamoto, H.; Haniu, H. (1990). "A study on vortex shedding from spheres in uniform flow". Journal of Fluids Engineering. 112 (December): 386–392. Bibcode:1990ATJFE.112..386S. doi:10.1115/1.2909415. S2CID 15578514.
* ^ Sobey, Ian J. (1982). "Oscillatory flows at intermediate Strouhal number in asymmetry channels". Journal of Fluid Mechanics. 125: 359–373. Bibcode:1982JFM...125..359S. doi:10.1017/S0022112082003371. S2CID 122167909.
* ^ Sitti, Metin (2017). Mobile Microrobotics. The MIT Press. pp. 13–24. ISBN 9780262036436.
* ^ a b c Doutel, E.; Galindo-Rosales, F. J.; Campo-Deaño, L. (December 2, 2021). "Hemodynamics Challenges for the Navigation of Medical Microbots for the Treatment of CVDs". Materials. 14 (23): 7402. Bibcode:2021Mate...14.7402D. doi:10.3390/ma14237402. PMC 8658690. PMID 34885556.
* ^ Sanchez-Sanz, M.; Liñan, A.; Smoke, M. D.; Bennett, B. A. V. (July 16, 2009). "Influence of Strouhal number on pulsating methane–air coflow jet diffusion flames". Combustion Theory and Modelling. 14 (3): 453–478. doi:10.1080/13647830.2010.490048. S2CID 53640323.
* ^ a b c Taylor, Graham K.; Nudds, Robert L.; Thomas, Adrian L. R. (2003). "Flying and swimming animals cruise at a Strouhal number tuned for high power efficiency". Nature. 425 (6959): 707–711. Bibcode:2003Natur.425..707T. doi:10.1038/nature02000. PMID 14562101. S2CID 4431906.
* ^ Corum, Jonathan (2003). "The Strouhal Number in Cruising Flight". Retrieved 2012-11-13– depiction of Strouhal number for flying and swimming animals{{cite web}}:  CS1 maint: postscript (link)

```
{{cite web}}
```

* ^ a b c Taylor, G. K.; Nudds, R. L.; Thomas, A. L. R. (October 16, 2003). "Flying and swimming animals cruise at a Strouhal number tuned for high power efficiency". Nature. 425 (6959): 707–711. Bibcode:2003Natur.425..707T. doi:10.1038/nature02000. PMID 14562101. S2CID 4431906. ProQuest 204520869.
* ^ a b Lapsansky, Anthony B.; Zatz, Daniel; Tobalske, Bret W. (June 30, 2020). "Alcids 'fly' at efficient Strouhal numbers in both air and water but vary stroke velocity and angle". eLife. 9 e55774. doi:10.7554/eLife.55774. PMC 7332295. PMID 32602463.
* ^ a b c d Wimer, N. T.; Lapointe, C.; Christopher, J. D.; Nigam, S. P.; Hayden, T. R. S.; Upadhye, A.; Strobel, M.; Rieker, G. B.; Hamlington, P. E. (May 21, 2020). "Scaling of the Puffing Strouhal Number for Buoyant Jets and Plumes". Journal of Fluid Mechanics. 895 A26. arXiv:1904.01580. Bibcode:2020JFM...895A..26W. doi:10.1017/jfm.2020.271. S2CID 96428731.
* ^ a b Eloy, Cristophe (March 5, 2012). "Optimal Strouhal number for swimming animals". Journal of Fluids and Structures. 30: 205–218. arXiv:1102.0223. Bibcode:2012JFS....30..205E. doi:10.1016/j.jfluidstructs.2012.02.008. S2CID 56221298.

## External links

* Vincenc Strouhal, Ueber eine besondere Art der Tonerregung[permanent dead link]

vteFluid mechanics
Fluid statics | Hydraulics
Archimedes' principle
Fluid dynamics | Computational fluid dynamics
Aerodynamics
Navier–Stokes equations
Boundary layer
Entrance length
Dimensionless numbers | Alfvén Mach
Archimedes
Atwood
Bagnold
Bejan
Biot
Bodenstein
Bond
Brinkman
Capillary
Cauchy
Chandrasekhar
Damköhler
Darcy
Dean
Deborah
Dukhin
Eckert
Ekman
Eötvös
Euler
Froude
Galilei
Graetz
Grashof
Görtler
Hagen
Iribarren
Kapitza
Keulegan–Carpenter
Knudsen
Laplace
Lewis
Mach
 Marangoni
Morton
Nusselt
Ohnesorge
Péclet
Prandtl
magnetic
turbulent
Rayleigh
Reynolds
magnetic
Richardson
Roshko
Rossby
Rouse
Schmidt
Scruton
Sherwood
Shields
Stanton
Stokes
Strouhal
Stuart
Suratman
Taylor
Ursell
Weber
Weissenberg
Womersley

* v
* t
* e
* Hydraulics
* Archimedes' principle
* Computational fluid dynamics
* Aerodynamics
* Navier–Stokes equations
* Boundary layer
Entrance length
* Entrance length
* Alfvén Mach
* Archimedes
* Atwood
* Bagnold
* Bejan
* Biot
* Bodenstein
* Bond
* Brinkman
* Capillary
* Cauchy
* Chandrasekhar
* Damköhler
* Darcy
* Dean
* Deborah
* Dukhin
* Eckert
* Ekman
* Eötvös
* Euler
* Froude
* Galilei
* Graetz
* Grashof
* Görtler
* Hagen
* Iribarren
* Kapitza
* Keulegan–Carpenter
* Knudsen
* Laplace
* Lewis
* Mach
* Marangoni
* Morton
* Nusselt
* Ohnesorge
* Péclet
* Prandtl
magnetic
turbulent
* magnetic
* turbulent
* Rayleigh
* Reynolds
magnetic
* magnetic
* Richardson
* Roshko
* Rossby
* Rouse
* Schmidt
* Scruton
* Sherwood
* Shields
* Stanton
* Stokes
* Strouhal
* Stuart
* Suratman
* Taylor
* Ursell
* Weber
* Weissenberg
* Womersley

Authority control databases | GND

* GND
* Dimensionless numbers of fluid mechanics
* Fluid dynamics
* All articles with dead external links
* Articles with dead external links from August 2024
* Articles with permanently dead external links
* CS1 maint: postscript
* Articles with short description
* Short description matches Wikidata
* Wikipedia articles needing clarification from December 2025
* Pages displaying short descriptions of redirect targets via Module:Annotated link
* Articles with dead external links from January 2018