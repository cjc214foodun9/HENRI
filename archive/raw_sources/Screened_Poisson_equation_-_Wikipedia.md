# Screened Poisson equation - Wikipedia
Source URL: https://en.wikipedia.org/wiki/Screened_Poisson_equation

| This article needs additional citations for verification. Please help improve this article by adding citations to reliable sources. Unsourced material may be challenged and removed.Find sources: "Screened Poisson equation" – news · newspapers · books · scholar · JSTOR (July 2017) (Learn how and when to remove this message)

In physics, the screened Poisson equation is a Poisson equation, which arises in (for example) the Klein–Gordon equation, electric field screening in plasmas, and nonlocal granular fluidity[1] in granular flow.

## Statement of the equation

The equation is

[

Δ
−

λ

2

]

u
(

r

)
=
−
f
(

r

)
,

{\displaystyle \left[\Delta -\lambda ^{2}\right]u(\mathbf {r} )=-f(\mathbf {r} ),}

where 

Δ

{\displaystyle \Delta }

 is the Laplace operator, λ is a constant that expresses the "screening", f is an arbitrary function of position (known as the "source function") and u is the function to be determined.

In the homogeneous case (f=0), the screened Poisson equation is the same as the time-independent Klein–Gordon equation.  In the inhomogeneous case, the screened Poisson equation is very similar to the inhomogeneous Helmholtz equation, the only difference being the sign within the brackets.

### Electrostatics

In electric-field screening, screened Poisson equation for the electric potential 

ϕ
(

r

)

{\displaystyle \phi (\mathbf {r} )}

 is usually written as (SI units)

[

Δ
−

k

0

2

]

ϕ
(

r

)
=
−

ρ

e
x
t

(

r

)

ϵ

0

,

{\displaystyle \left[\Delta -k_{0}^{2}\right]\phi (\mathbf {r} )=-{\frac {\rho _{\rm {ext}}(\mathbf {r} )}{\epsilon _{0}}},}

where 

k

0

−
1

{\displaystyle k_{0}^{-1}}

 is the screening length, 

ρ

e
x
t

(

r

)

{\displaystyle \rho _{\rm {ext}}(\mathbf {r} )}

 is the charge density produced by an external field in the absence of screening and 

ϵ

0

{\displaystyle \epsilon _{0}}

 is the vacuum permittivity. This equation can be derived in several screening models like Thomas–Fermi screening in solid-state physics and Debye screening in plasmas.

## Solutions

### Three dimensions

Without loss of generality, we will take λ to be non-negative. When λ is zero, the equation reduces to Poisson's equation. Therefore, when λ is very small, the solution approaches that of the unscreened Poisson equation, which, in dimension 

n
=
3

{\displaystyle n=3}

, is a superposition of 1/r functions weighted by the source function f:

u
(

r

)

(

Poisson

)

=
∭

d

3

r

′

f
(

r

′

)

4
π

|

r

−

r

′

|

.

{\displaystyle u(\mathbf {r} )_{({\text{Poisson}})}=\iiint \mathrm {d} ^{3}\mathbf {r} '{\frac {f(\mathbf {r} ')}{4\pi |\mathbf {r} -\mathbf {r} '|}}.}

On the other hand, when λ is extremely large, u approaches the value f/λ2, which goes to zero as λ goes to infinity. As we shall see, the solution for intermediate values of λ behaves as a superposition of screened (or damped) 1/r functions, with λ behaving as the strength of the screening.

The screened Poisson equation can be solved for general f using the method of Green's functions. The Green's function G is defined by

[

Δ
−

λ

2

]

G
(

r

)
=
−

δ

3

(

r

)
,

{\displaystyle \left[\Delta -\lambda ^{2}\right]G(\mathbf {r} )=-\delta ^{3}(\mathbf {r} ),}

where δ3 is a delta function with unit mass concentrated at the origin of R3.

Assuming u and its derivatives vanish at large r, we may perform a continuous Fourier transform in spatial coordinates:

G
~

(

k

)
=
∭

d

3

r

G
(

r

)

e

−
i

k

⋅

r

{\displaystyle {\tilde {G}}(\mathbf {k} )=\iiint \mathrm {d} ^{3}\mathbf {r} \;G(\mathbf {r} )e^{-i\mathbf {k} \cdot \mathbf {r} }}

where the integral is taken over all space. It is then straightforward to show that

[

k

2

+

λ

2

]

G
~

(

k

)
=
1.

{\displaystyle \left[k^{2}+\lambda ^{2}\right]{\tilde {G}}(\mathbf {k} )=1.}

The Green's function in r is therefore given by the inverse Fourier transform,

G
(

r

)
=

1

(
2
π

)

3

∭

d

3

k

e

i

k

⋅

r

k

2

+

λ

2

.

{\displaystyle G(\mathbf {r} )={\frac {1}{(2\pi )^{3}}}\;\iiint \mathrm {d} ^{3}\!\mathbf {k} \;{\frac {e^{i\mathbf {k} \cdot \mathbf {r} }}{k^{2}+\lambda ^{2}}}.}

This integral may be evaluated using spherical coordinates in k-space. The integration over the angular coordinates is straightforward, and the integral reduces to one over the radial wavenumber 

k

r

{\displaystyle k_{r}}

:

G
(

r

)
=

1

2

π

2

r

∫

0

∞

d

k

r

k

r

sin
⁡

k

r

r

k

r

2

+

λ

2

.

{\displaystyle G(\mathbf {r} )={\frac {1}{2\pi ^{2}r}}\;\int _{0}^{\infty }\mathrm {d} k_{r}\;{\frac {k_{r}\,\sin k_{r}r}{k_{r}^{2}+\lambda ^{2}}}.}

This may be evaluated using contour integration. The result is:

G
(

r

)
=

e

−
λ
r

4
π
r

.

{\displaystyle G(\mathbf {r} )={\frac {e^{-\lambda r}}{4\pi r}}.}

The solution to the full problem is then given by

u
(

r

)
=
∫

d

3

r

′

G
(

r

−

r

′

)
f
(

r

′

)
=
∫

d

3

r

′

e

−
λ

|

r

−

r

′

|

4
π

|

r

−

r

′

|

f
(

r

′

)
.

{\displaystyle u(\mathbf {r} )=\int \mathrm {d} ^{3}\mathbf {r} 'G(\mathbf {r} -\mathbf {r} ')f(\mathbf {r} ')=\int \mathrm {d} ^{3}\mathbf {r} '{\frac {e^{-\lambda |\mathbf {r} -\mathbf {r} '|}}{4\pi |\mathbf {r} -\mathbf {r} '|}}f(\mathbf {r} ').}

As stated above, this is a superposition of screened 1/r functions, weighted by the source function f and with λ acting as the strength of the screening. The screened 1/r function is often encountered in physics as a screened Coulomb potential, also called a "Yukawa potential".

### Two dimensions

In two dimensions:
In the case of a magnetized plasma, the screened Poisson equation is quasi-2D:

(

Δ

⊥

−

1

ρ

2

)

u
(

r

⊥

)
=
−
f
(

r

⊥

)

{\displaystyle \left(\Delta _{\perp }-{\frac {1}{\rho ^{2}}}\right)u(\mathbf {r} _{\perp })=-f(\mathbf {r} _{\perp })}

with 

Δ

⊥

=
∇
⋅

∇

⊥

{\displaystyle \Delta _{\perp }=\nabla \cdot \nabla _{\perp }}

 and 

∇

⊥

=
∇
−

B

B

⋅
∇

{\displaystyle \nabla _{\perp }=\nabla -{\frac {\mathbf {B} }{B}}\cdot \nabla }

, with 

B

{\displaystyle \mathbf {B} }

 the magnetic field and 

ρ

{\displaystyle \rho }

 is the (ion) Larmor radius.
The two-dimensional Fourier Transform of the associated Green's function is:

G
~

(

k

⊥

)
=
∬

d

2

r

 
G
(

r

⊥

)

e

−
i

k

⊥

⋅

r

⊥

.

{\displaystyle {\tilde {G}}(\mathbf {k_{\perp }} )=\iint d^{2}\mathbf {r} ~G(\mathbf {r} _{\perp })e^{-i\mathbf {k} _{\perp }\cdot \mathbf {r} _{\perp }}.}

The 2D screened Poisson equation yields:

(

k

⊥

2

+

1

ρ

2

)

G
~

(

k

⊥

)
=
1.

{\displaystyle \left(k_{\perp }^{2}+{\frac {1}{\rho ^{2}}}\right){\tilde {G}}(\mathbf {k} _{\perp })=1.}

The Green's function is therefore given by the inverse Fourier transform:

G
(

r

⊥

)
=

1

4

π

2

∬

d

2

k

e

i

k

⊥

⋅

r

⊥

k

⊥

2

+
1

/

ρ

2

.

{\displaystyle G(\mathbf {r} _{\perp })={\frac {1}{4\pi ^{2}}}\;\iint \mathrm {d} ^{2}\!\mathbf {k} \;{\frac {e^{i\mathbf {k} _{\perp }\cdot \mathbf {r} _{\perp }}}{k_{\perp }^{2}+1/\rho ^{2}}}.}

This integral can be calculated using polar coordinates in k-space:

k

⊥

=
(

k

r

cos
⁡
(
θ
)
,

k

r

sin
⁡
(
θ
)
)

{\displaystyle \mathbf {k} _{\perp }=(k_{r}\cos(\theta ),k_{r}\sin(\theta ))}

The integration over the angular coordinate gives a Bessel function, and the integral reduces to one over the radial wavenumber 

k

r

{\displaystyle k_{r}}

:

G
(

r

⊥

)
=

1

2
π

∫

0

∞

d

k

r

k

r

J

0

(

k

r

r

⊥

)

k

r

2

+
1

/

ρ

2

=

1

2
π

K

0

(

r

⊥

/

ρ
)
.

{\displaystyle G(\mathbf {r} _{\perp })={\frac {1}{2\pi }}\;\int _{0}^{\infty }\mathrm {d} k_{r}\;{\frac {k_{r}\,J_{0}(k_{r}r_{\perp })}{k_{r}^{2}+1/\rho ^{2}}}={\frac {1}{2\pi }}K_{0}(r_{\perp }\,/\,\rho ).}

## Connection to the Laplace distribution

The Green's functions in both 2D and 3D are identical to the probability density function of the multivariate Laplace distribution for two and three dimensions respectively.

## Application in differential geometry

The homogeneous case, studied in the context of differential geometry, involving Einstein warped product manifolds, explores cases where the warped function satisfies the homogeneous version of the screened Poisson equation. Under specific conditions, the manifold dimension, Ricci curvature, and screening parameter are interconnected via a quadratic relationship.[2]

## See also

* Yukawa interaction

## References

* ^ Kamrin, Ken; Koval, Georg (26 April 2012). "Nonlocal Constitutive Relation for Steady Granular Flow" (PDF). Physical Review Letters. 108 (17) 178301. Bibcode:2012PhRvL.108q8301K. doi:10.1103/PhysRevLett.108.178301. hdl:1721.1/71598. PMID 22680912.
* ^ Pigazzini, Alexander; Lussardi, Luca; Toda, Magdalena; DeBenedictis, Andrew (29 July 2024). "Einstein warped-product manifolds and the screened Poisson equation". Contemporary Mathematics series of the American Mathematical Society (AMS) - Book entitled: "Recent Advances in Differential Geometry and Related Areas" (2025). 821: 173-179.
* Partial differential equations
* Plasma physics equations
* Electrostatics
* Articles with short description
* Short description matches Wikidata
* Articles needing additional references from July 2017
* All articles needing additional references