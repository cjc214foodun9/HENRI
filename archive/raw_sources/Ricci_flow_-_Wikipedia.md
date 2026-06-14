# Ricci flow - Wikipedia
Source URL: https://en.wikipedia.org/wiki/Ricci_flow

In differential geometry and geometric analysis, the Ricci flow (/ˈriːtʃi/ REE-chee, Italian: [ˈrittʃi]), sometimes also referred to as Hamilton's Ricci flow, is a certain partial differential equation for a Riemannian metric. It is often said to be analogous to the diffusion of heat and the heat equation, due to formal similarities in the mathematical structure of the equation. However, it is nonlinear and exhibits many phenomena not present in the study of the heat equation.

The Ricci flow, so named for the presence of the Ricci tensor in its definition, was introduced by Richard Hamilton, who used it through the 1980s to prove striking new results in Riemannian geometry. Later extensions of Hamilton's methods by various authors resulted in new applications to geometry, including the resolution of the differentiable sphere conjecture by Simon Brendle and Richard Schoen.

Following the possibility that the singularities of solutions of the Ricci flow could identify the topological data predicted by William Thurston's geometrization conjecture, Hamilton produced a number of results in the 1990s which were directed towards the conjecture's resolution. In 2002 and 2003, Grigori Perelman presented a number of fundamental new results about the Ricci flow, including a novel variant of some technical aspects of Hamilton's program. Perelman's work is now widely regarded as forming the proof of the Thurston conjecture and the Poincaré conjecture, regarded as a special case of the former. It should be emphasized that the Poincaré conjecture has been a well-known open problem in the field of geometric topology since 1904. These results by Hamilton and Perelman are considered as a milestone in the fields of geometry and topology.

## Mathematical definition

On a smooth manifold M, a smooth Riemannian metric g automatically determines the Ricci tensor Ricg. For each element p of M, by definition gp is a positive-definite inner product on the tangent space TpM at p. If given a one-parameter family of Riemannian metrics gt, one may then consider the derivative 

∂

∂
t

g

t

{\displaystyle {\frac {\partial }{\partial t}}g_{t}}

, which then assigns to each particular value of t and p a symmetric bilinear form on TpM. Since the Ricci tensor of a Riemannian metric also assigns to each p a symmetric bilinear form on TpM, the following definition is meaningful.

* Given a smooth manifold M and an open real interval (a, b), a Ricci flow assigns, to each t in the interval (a,b), a Riemannian metric gt on M such that 

∂

∂
t

g

t

=
−
2
 

R
i
c

g

t

{\displaystyle {\frac {\partial }{\partial t}}g_{t}=-2\ \mathrm {Ric} ^{g_{t}}}

.

The Ricci tensor is often thought of as an average value of the sectional curvatures, or as an algebraic trace of the Riemann curvature tensor. However, for the analysis of existence and uniqueness of Ricci flows, it is extremely significant that the Ricci tensor can be defined, in local coordinates, by a formula involving the first and second derivatives of the metric tensor. This makes the Ricci flow into a geometrically defined partial differential equation. The analysis of the ellipticity of the local coordinate formula provides the foundation for the existence of Ricci flows; see the following section for the corresponding result.

Let k be a nonzero number. Given a Ricci flow gt on an interval (a,b), consider Gt = gkt for t between ⁠a/k⁠ and ⁠b/k⁠. Then ⁠∂/∂t⁠ Gt = −2k RicGt. So, with this very trivial change of parameters, the number −2 appearing in the definition of the Ricci flow could be replaced by any other nonzero number. For this reason, the use of −2 can be regarded as an arbitrary convention, albeit one which essentially every paper and exposition on Ricci flow follows. The only significant difference is that if −2 were replaced by a positive number, then the existence theorem discussed in the following section would become a theorem which produces a Ricci flow that moves backwards (rather than forwards) in parameter values from initial data.

The parameter t is usually called time, although this is only as part of standard informal terminology in the mathematical field of partial differential equations. It is not physically meaningful terminology. In fact, in the standard quantum field theoretic interpretation of the Ricci flow in terms of the renormalization group, the parameter t corresponds to length or energy, rather than time.[1]

### Normalized Ricci flow

Suppose that M is a compact smooth manifold, and let gt be a Ricci flow for t in the interval (a, b). Define Ψ:(a, b) → (0, ∞) so that each of the Riemannian metrics Ψ(t)gt has volume 1; this is possible since M is compact. (More generally, it would be possible if each Riemannian metric gt had finite volume.) Then define F:(a, b) → (0, ∞) to be the antiderivative of Ψ which vanishes at a. Since Ψ is positive-valued, F is a bijection onto its image (0, S). Now the Riemannian metrics Gs  =  Ψ(F −1(s))gF −1(s), defined for parameters s ∈ (0, S), satisfy

Here R denotes scalar curvature. This is called the normalized Ricci flow equation. Thus, with an explicitly defined change of scale Ψ and a reparametrization of the parameter values, a Ricci flow can be converted into a normalized Ricci flow. The converse also holds, by reversing the above calculations.

The primary reason for considering the normalized Ricci flow is that it allows a convenient statement of the major convergence theorems for Ricci flow. However, it is not essential to do so, and for virtually all purposes it suffices to consider Ricci flow in its standard form. Moreover, the normalized Ricci flow is not generally meaningful on noncompact manifolds.

## Existence and uniqueness

Let 

M

{\displaystyle M}

 be a smooth closed manifold, and let 

g

0

{\displaystyle g_{0}}

 be any smooth Riemannian metric on 

M

{\displaystyle M}

. Making use of the Nash–Moser implicit function theorem, Hamilton (1982) showed the following existence theorem:

* There exists a positive number 

T

{\displaystyle T}

 and a Ricci flow 

g

t

{\displaystyle g_{t}}

 parametrized by 

t
∈
(
0
,
T
)

{\displaystyle t\in (0,T)}

 such that 

g

t

{\displaystyle g_{t}}

 converges to 

g

0

{\displaystyle g_{0}}

 in the 

C

∞

{\displaystyle C^{\infty }}

 topology as 

t

{\displaystyle t}

 decreases to 0.

He showed the following uniqueness theorem:

* If 

{

g

t

:
t
∈
(
0
,
T
)
}

{\displaystyle \{g_{t}:t\in (0,T)\}}

 and 

{

g
~

t

:
t
∈
(
0
,

T
~

)
}

{\displaystyle \{{\widetilde {g}}_{t}:t\in (0,{\widetilde {T}})\}}

 are two Ricci flows as in the above existence theorem, then 

g

t

=

g
~

t

{\displaystyle g_{t}={\widetilde {g}}_{t}}

 for all 

t
∈
(
0
,
min
{
T
,

T
~

}
)
.

{\displaystyle t\in (0,\min\{T,{\widetilde {T}}\}).}

The existence theorem provides a one-parameter family of smooth Riemannian metrics. In fact, any such one-parameter family also depends smoothly on the parameter. Precisely, this says that relative to any smooth coordinate chart 

(
U
,
ϕ
)

{\displaystyle (U,\phi )}

 on 

M

{\displaystyle M}

, the function 

g

i
j

:
U
×
(
0
,
T
)
→

R

{\displaystyle g_{ij}:U\times (0,T)\to \mathbb {R} }

 is smooth for any 

i
,
j
=
1
,
…
,
n

{\displaystyle i,j=1,\dots ,n}

.

Dennis DeTurck subsequently gave a proof of the above results which uses the Banach implicit function theorem instead.[2] His work is essentially a simpler Riemannian version of Yvonne Choquet-Bruhat's well-known proof and interpretation of well-posedness for the Einstein equations in Lorentzian geometry.

As a consequence of Hamilton's existence and uniqueness theorem, when given the data 

(
M
,

g

0

)

{\displaystyle (M,g_{0})}

, one may speak unambiguously of the Ricci flow on 

M

{\displaystyle M}

 with initial data 

g

0

{\displaystyle g_{0}}

, and one may select 

T

{\displaystyle T}

 to take on its maximal possible value, which could be infinite. The principle behind virtually all major applications of Ricci flow, in particular in the proof of the Poincaré conjecture and geometrization conjecture, is that, as 

t

{\displaystyle t}

 approaches this maximal value, the behavior of the metrics 

g

t

{\displaystyle g_{t}}

 can reveal and reflect deep information about 

M

{\displaystyle M}

.

## Convergence theorems

Complete expositions of the following convergence theorems are given in Andrews & Hopper (2011) and Brendle (2010).

Let (M, g0) be a smooth closed Riemannian manifold. Under any of the following three conditions:

* M is two-dimensional
* M is three-dimensional and g0 has positive Ricci curvature
* M has dimension greater than three and the product metric on (M, g0) × ℝ has positive isotropic curvature

the normalized Ricci flow with initial data g0 exists for all positive time and converges smoothly, as t goes to infinity, to a metric of constant curvature.

The three-dimensional result is due to Hamilton (1982). Hamilton's proof, inspired by and loosely modeled upon James Eells and Joseph Sampson's epochal 1964 paper on convergence of the harmonic map heat flow,[3] included many novel features, such as an extension of the maximum principle to the setting of symmetric 2-tensors. His paper (together with that of Eells−Sampson) is among the most widely cited in the field of differential geometry. There is an exposition of his result in Chow, Lu & Ni (2006, Chapter 3).

In terms of the proof, the two-dimensional case is properly viewed as a collection of three different results, one for each of the cases in which the Euler characteristic of M is positive, zero, or negative. As demonstrated by Hamilton (1988), the negative case is handled by the maximum principle, while the zero case is handled by integral estimates; the positive case is more subtle, and Hamilton dealt with the subcase in which g0 has positive curvature by combining a straightforward adaptation of Peter Li and Shing-Tung Yau's gradient estimate to the Ricci flow together with an innovative "entropy estimate". The full positive case was demonstrated by Bennett Chow (1991), in an extension of Hamilton's techniques. Since any Ricci flow on a two-dimensional manifold is confined to a single conformal class, it can be recast as a partial differential equation for a scalar function on the fixed Riemannian manifold (M, g0). As such, the Ricci flow in this setting can also be studied by purely analytic methods; correspondingly, there are alternative non-geometric proofs of the two-dimensional convergence theorem.

The higher-dimensional case has a longer history. Soon after Hamilton's breakthrough result, Gerhard Huisken extended his methods to higher dimensions, showing that if g0 almost has constant positive curvature (in the sense of smallness of certain components of the Ricci decomposition), then the normalized Ricci flow converges smoothly to constant curvature. Hamilton (1986) found a novel formulation of the maximum principle in terms of trapping by convex sets, which led to a general criterion relating convergence of the Ricci flow of positively curved metrics to the existence of "pinching sets" for a certain multidimensional ordinary differential equation. As a consequence, he was able to settle the case in which M is four-dimensional and g0 has positive curvature operator. Twenty years later, Christoph Böhm and Burkhard Wilking found a new algebraic method of constructing "pinching sets", thereby removing the assumption of four-dimensionality from Hamilton's result (Böhm & Wilking 2008). Simon Brendle and Richard Schoen showed that positivity of the isotropic curvature is preserved by the Ricci flow on a closed manifold; by applying Böhm and Wilking's method, they were able to derive a new Ricci flow convergence theorem (Brendle & Schoen 2009). Their convergence theorem included as a special case the resolution of the differentiable sphere theorem, which at the time had been a long-standing conjecture. The convergence theorem given above is due to Brendle (2008), which subsumes the earlier higher-dimensional convergence results of Huisken, Hamilton, Böhm & Wilking, and Brendle & Schoen.

### Corollaries

The results in dimensions three and higher show that any smooth closed manifold M which admits a metric g0 of the given type must be a space form of positive curvature. Since these space forms are largely understood by work of Élie Cartan and others, one may draw corollaries such as

* Suppose that M is a smooth closed 3-dimensional manifold which admits a smooth Riemannian metric of positive Ricci curvature. If M is simply-connected then it must be diffeomorphic to the 3-sphere.

So if one could show directly that any smooth closed simply-connected 3-dimensional manifold admits a smooth Riemannian metric of positive Ricci curvature, then the Poincaré conjecture would immediately follow. However, as matters are understood at present, this result is only known as a (trivial) corollary of the Poincaré conjecture, rather than vice versa.

### Possible extensions

Given any n larger than two, there exist many closed n-dimensional smooth manifolds which do not have any smooth Riemannian metrics of constant curvature. So one cannot hope to be able to simply drop the curvature conditions from the above convergence theorems. It could be possible to replace the curvature conditions by some alternatives, but the existence of compact manifolds such as complex projective space, which has a metric of nonnegative curvature operator (the Fubini-Study metric) but no metric of constant curvature, makes it unclear how much these conditions could be pushed. Likewise, the possibility of formulating analogous convergence results for negatively curved Riemannian metrics is complicated by the existence of closed Riemannian manifolds whose curvature is arbitrarily close to constant and yet admit no metrics of constant curvature.[4]

## Li–Yau inequalities

Making use of a technique pioneered by Peter Li and Shing-Tung Yau for parabolic differential equations on Riemannian manifolds, Hamilton (1993a) proved the following "Li–Yau inequality".[5]

* Let 

M

{\displaystyle M}

 be a smooth manifold, and let 

g

t

{\displaystyle g_{t}}

 be a solution of the Ricci flow with 

t
∈
(
0
,
T
)

{\displaystyle t\in (0,T)}

 such that each 

g

t

{\displaystyle g_{t}}

 is complete with bounded curvature. Furthermore, suppose that each 

g

t

{\displaystyle g_{t}}

 has nonnegative curvature operator. Then, for any curve 

γ
:
[

t

1

,

t

2

]
→
M

{\displaystyle \gamma :[t_{1},t_{2}]\to M}

 with 

[

t

1

,

t

2

]
⊂
(
0
,
T
)

{\displaystyle [t_{1},t_{2}]\subset (0,T)}

, one has 

d

d
t

(

R

g
(
t
)

(
γ
(
t
)
)

)

+

R

g
(
t
)

(
γ
(
t
)
)

t

+

1
2

Ric

g
(
t
)

⁡
(

γ
′

(
t
)
,

γ
′

(
t
)
)
≥
0.

{\displaystyle {\frac {d}{dt}}{\big (}R^{g(t)}(\gamma (t)){\big )}+{\frac {R^{g(t)}(\gamma (t))}{t}}+{\frac {1}{2}}\operatorname {Ric} ^{g(t)}(\gamma '(t),\gamma '(t))\geq 0.}

Perelman (2002) showed the following alternative Li–Yau inequality.

* Let 

M

{\displaystyle M}

 be a smooth closed 

n

{\displaystyle n}

-manifold, and let 

g

t

{\displaystyle g_{t}}

 be a solution of the Ricci flow. Consider the backwards heat equation for 

n

{\displaystyle n}

-forms, i.e. 

∂

∂
t

ω
+

Δ

g
(
t
)

ω
=
0

{\displaystyle {\tfrac {\partial }{\partial t}}\omega +\Delta ^{g(t)}\omega =0}

; given 

p
∈
M

{\displaystyle p\in M}

 and 

t

0

∈
(
0
,
T
)

{\displaystyle t_{0}\in (0,T)}

, consider the particular solution which, upon integration, converges weakly to the Dirac delta measure as 

t

{\displaystyle t}

 increases to 

t

0

{\displaystyle t_{0}}

. Then, for any curve 

γ
:
[

t

1

,

t

2

]
→
M

{\displaystyle \gamma :[t_{1},t_{2}]\to M}

 with 

[

t

1

,

t

2

]
⊂
(
0
,
T
)

{\displaystyle [t_{1},t_{2}]\subset (0,T)}

, one has 

d

d
t

(

f
(
γ
(
t
)
,
t
)

)

+

f

(

γ
(
t
)
,
t

)

2
(

t

0

−
t
)

≤

R

g
(
t
)

(
γ
(
t
)
)
+

|

γ
′

(
t
)

|

g
(
t
)

2

2

.

{\displaystyle {\frac {d}{dt}}{\big (}f(\gamma (t),t){\big )}+{\frac {f{\big (}\gamma (t),t{\big )}}{2(t_{0}-t)}}\leq {\frac {R^{g(t)}(\gamma (t))+|\gamma '(t)|_{g(t)}^{2}}{2}}.}

 where 

ω
=
(
4
π
(

t

0

−
t
)

)

−
n

/

2

e

−
f

d

μ

g
(
t
)

{\displaystyle \omega =(4\pi (t_{0}-t))^{-n/2}e^{-f}{\text{d}}\mu _{g(t)}}

.

Both of these remarkable inequalities are of profound importance for the proof of the Poincaré conjecture and geometrization conjecture. The terms on the right hand side of Perelman's Li–Yau inequality motivates the definition of his "reduced length" functional, the analysis of which leads to his "noncollapsing theorem". The noncollapsing theorem allows application of Hamilton's compactness theorem (Hamilton 1995) to construct "singularity models", which are Ricci flows on new three-dimensional manifolds. Owing to the Hamilton–Ivey estimate, these new Ricci flows have nonnegative curvature. Hamilton's Li–Yau inequality can then be applied to see that the scalar curvature is, at each point, a nondecreasing (nonnegative) function of time. This is a powerful result that allows many further arguments to go through. In the end, Perelman shows that any of his singularity models is asymptotically like a complete gradient shrinking Ricci soliton, which are completely classified; see the previous section.

See Chow, Lu & Ni (2006, Chapters 10 and 11) for details on Hamilton's Li–Yau inequality; the books Chow et al. (2008) and Müller (2006) contain expositions of both inequalities above.

## Examples

### Constant-curvature and Einstein metrics

Let 

(
M
,
g
)

{\displaystyle (M,g)}

 be a Riemannian manifold which is Einstein, meaning that there is a number 

λ

{\displaystyle \lambda }

 such that 

Ric

g

=
λ
g

{\displaystyle {\text{Ric}}^{g}=\lambda g}

. Then 

g

t

=
(
1
−
2
λ
t
)
g

{\displaystyle g_{t}=(1-2\lambda t)g}

 is a Ricci flow with 

g

0

=
g

{\displaystyle g_{0}=g}

, since then

If 

M

{\displaystyle M}

 is closed, then according to Hamilton's uniqueness theorem above, this is the only Ricci flow with initial data 

g

{\displaystyle g}

. One sees, in particular, that:

* if 

λ

{\displaystyle \lambda }

 is positive, then the Ricci flow "contracts" 

g

{\displaystyle g}

 since the scale factor 

1
−
2
λ
t

{\displaystyle 1-2\lambda t}

 is less than 1 for positive 

t

{\displaystyle t}

; furthermore, one sees that 

t

{\displaystyle t}

 can only be less than 

1

/

2
λ

{\displaystyle 1/2\lambda }

, in order that 

g

t

{\displaystyle g_{t}}

 is a Riemannian metric. This is the simplest examples of a "finite-time singularity".
* if 

λ

{\displaystyle \lambda }

 is zero, which is synonymous with 

g

{\displaystyle g}

 being Ricci-flat, then 

g

t

{\displaystyle g_{t}}

 is independent of time, and so the maximal interval of existence is the entire real line.
* if 

λ

{\displaystyle \lambda }

 is negative, then the Ricci flow "expands" 

g

{\displaystyle g}

 since the scale factor 

1
−
2
λ
t

{\displaystyle 1-2\lambda t}

 is greater than 1 for all positive 

t

{\displaystyle t}

; furthermore one sees that 

t

{\displaystyle t}

 can be taken arbitrarily large. One says that the Ricci flow, for this initial metric, is "immortal".

In each case, since the Riemannian metrics assigned to different values of 

t

{\displaystyle t}

 differ only by a constant scale factor, one can see that the normalized Ricci flow 

G

s

{\displaystyle G_{s}}

 exists for all time and is constant in 

s

{\displaystyle s}

; in particular, it converges smoothly (to its constant value) as 

s
→
∞

{\displaystyle s\to \infty }

.

The Einstein condition has as a special case that of constant curvature; hence the particular examples of the sphere (with its standard metric) and hyperbolic space appear as special cases of the above.

### Ricci solitons

Ricci solitons are Ricci flows that may change their size but not their shape up to diffeomorphisms.

* Cylinders Sk × Rl (for k ≥ 2) shrink self similarly under the Ricci flow up to diffeomorphisms
* A significant 2-dimensional example is the cigar soliton, which is given by the metric (dx2 + dy2)/(e4t + x2 + y2) on the Euclidean plane. Although this metric shrinks under the Ricci flow, its geometry remains the same. Such solutions are called steady Ricci solitons.
* An example of a 3-dimensional steady Ricci soliton is the Bryant soliton, which is rotationally symmetric, has positive curvature, and is obtained by solving a system of ordinary differential equations.  A similar construction works in arbitrary dimension.
* There exist numerous families of Kähler manifolds, invariant under a U(n) action and birational to Cn, which are Ricci solitons. These examples were constructed by Cao and Feldman-Ilmanen-Knopf. (Chow-Knopf 2004)
* A 4-dimensional example exhibiting only torus symmetry was recently discovered by Bamler-Cifarelli-Conlon-Deruelle.

A gradient shrinking Ricci soliton consists of a smooth Riemannian manifold (M,g) and f ∈ C∞(M) such that

One of the major achievements of Perelman (2002) was to show that, if M is a closed three-dimensional smooth manifold, then finite-time singularities of the Ricci flow on M are modeled on complete gradient shrinking Ricci solitons (possibly on underlying manifolds distinct from M). In 2008, Huai-Dong Cao, Bing-Long Chen, and Xi-Ping Zhu completed the classification of these solitons, showing:

* Suppose (M,g,f) is a complete gradient shrinking Ricci soliton with dim(M) = 3. If M is simply-connected then the Riemannian manifold (M,g) is isometric to 

R

3

{\displaystyle \mathbb {R} ^{3}}

, 

S

3

{\displaystyle S^{3}}

, or 

S

2

×

R

{\displaystyle S^{2}\times \mathbb {R} }

, each with their standard Riemannian metrics. This was originally shown by Perelman (2003a) with some extra conditional assumptions. Note that if M is not simply-connected, then one may consider the universal cover 

π
:

M
′

→
M
,

{\displaystyle \pi :M'\to M,}

 and then the above theorem applies to 

(

M
′

,

π

∗

g
,
f
∘
π
)
.

{\displaystyle (M',\pi ^{\ast }g,f\circ \pi ).}

There is not yet a good understanding of gradient shrinking Ricci solitons in any higher dimensions.

## Relationship to uniformization and geometrization

Hamilton's first work on Ricci flow was published at the same time as William Thurston's geometrization conjecture, which concerns the topological classification of three-dimensional smooth manifolds.[6] Hamilton's idea was to define a kind of nonlinear diffusion equation which would tend to smooth out irregularities in the metric. Suitable canonical forms had already been identified by Thurston; the possibilities, called Thurston model geometries, include the three-sphere S3, three-dimensional Euclidean space E3, three-dimensional hyperbolic space H3, which are homogeneous and isotropic, and five slightly more exotic Riemannian manifolds, which are homogeneous but not isotropic.  (This list is closely related to, but not identical with, the Bianchi classification of the three-dimensional real Lie algebras into nine classes.)

Hamilton succeeded in proving that any smooth closed three-manifold which admits a metric of positive Ricci curvature also admits a unique Thurston geometry, namely a spherical metric, which does indeed act like an attracting fixed point under the Ricci flow, renormalized to preserve volume. (Under the unrenormalized Ricci flow, the manifold collapses to a point in finite time.) However, this doesn't prove the full geometrization conjecture, because of the restrictive assumption on curvature.

Indeed, a triumph of nineteenth-century geometry was the proof of the uniformization theorem, the analogous topological classification of smooth two-manifolds, where Hamilton showed that the Ricci flow does indeed evolve a negatively curved two-manifold into a two-dimensional multi-holed torus which is locally isometric to the hyperbolic plane.  This topic is closely related to important topics in analysis, number theory, dynamical systems, mathematical physics, and even cosmology.

Note that the term "uniformization" suggests a kind of smoothing away of irregularities in the geometry, while the term "geometrization" suggests placing a geometry on a smooth manifold.  Geometry is being used here in a precise manner akin to Klein's notion of geometry (see Geometrization conjecture for further details).  In particular, the result of geometrization may be a geometry that is not isotropic.  In most cases including the cases of constant curvature, the geometry is unique.  An important theme in this area is the interplay between real and complex formulations.  In particular, many discussions of uniformization speak of complex curves rather than real two-manifolds.

## Singularities

Hamilton showed that a compact Riemannian manifold always admits a short-time Ricci flow solution. Later Shi generalized the short-time existence result to complete manifolds of bounded curvature.[7] In general, however, due to the highly non-linear nature of the Ricci flow equation, singularities form in finite time. These singularities are curvature singularities, which means that as one approaches the singular time the norm of the curvature tensor 

|

Rm
⁡

|

{\displaystyle |\operatorname {Rm} |}

 blows up to infinity in the region of the singularity. A fundamental problem in Ricci flow is to understand all the possible geometries of singularities. When successful, this can lead to insights into the topology of manifolds. For instance, analyzing the geometry of singular regions that may develop in 3-D Ricci flow, is the crucial ingredient in Perelman's proof of the Poincaré and Geometrization conjectures.

### Blow-up limits of singularities

To study the formation of singularities it is useful, as in the study of other non-linear differential equations, to consider blow-ups limits. Intuitively speaking, one zooms into the singular region of the Ricci flow by rescaling time and space. Under certain assumptions, the zoomed in flow tends to a limiting Ricci flow 

(

M

∞

,

g

∞

(
t
)
)
,
t
∈
(
−
∞
,
0
]

{\displaystyle (M_{\infty },g_{\infty }(t)),t\in (-\infty ,0]}

, called a singularity model. Singularity models are ancient Ricci flows, i.e. they can be extended infinitely into the past. Understanding the possible singularity models in Ricci flow is an active research endeavor.

Below, we sketch the blow-up procedure in more detail: Let 

(
M
,

g

t

)
,

t
∈
[
0
,
T
)
,

{\displaystyle (M,g_{t}),\,t\in [0,T),}

 be a Ricci flow that develops a singularity as 

t
→
T

{\displaystyle t\rightarrow T}

. Let 

(

p

i

,

t

i

)
∈
M
×
[
0
,
T
)

{\displaystyle (p_{i},t_{i})\in M\times [0,T)}

 be a sequence of points in spacetime such that

as 

i
→
∞

{\displaystyle i\rightarrow \infty }

. Then one considers the parabolically rescaled metrics

Due to the symmetry of the Ricci flow equation under parabolic dilations, the metrics 

g

i

(
t
)

{\displaystyle g_{i}(t)}

 are also solutions to the Ricci flow equation. In the case that

i.e. up to time 

t

i

{\displaystyle t_{i}}

 the maximum of the curvature is attained at 

p

i

{\displaystyle p_{i}}

, then the pointed sequence of Ricci flows 

(
M
,

g

i

(
t
)
,

p

i

)

{\displaystyle (M,g_{i}(t),p_{i})}

 subsequentially converges smoothly to a limiting ancient Ricci flow 

(

M

∞

,

g

∞

(
t
)
,

p

∞

)

{\displaystyle (M_{\infty },g_{\infty }(t),p_{\infty })}

. Note that in general 

M

∞

{\displaystyle M_{\infty }}

 is not diffeomorphic to 

M

{\displaystyle M}

.

### Type I and Type II singularities

Hamilton distinguishes between Type I and Type II singularities in Ricci flow. In particular, one says a Ricci flow 

(
M
,

g

t

)
,

t
∈
[
0
,
T
)

{\displaystyle (M,g_{t}),\,t\in [0,T)}

, encountering a singularity a time 

T

{\displaystyle T}

 is of Type I if

Otherwise the singularity is of Type II. It is known that the blow-up limits of Type I singularities are gradient shrinking Ricci solitons.[8] In the Type II case it is an open question whether the singularity model must be a steady Ricci soliton—so far all known examples are.

### Singularities in 3d Ricci flow

In 3d the possible blow-up limits of Ricci flow singularities are well-understood. From the work of Hamilton, Perelman and Brendle, blowing up at points of maximum curvature leads to one of the following three singularity models:

* The shrinking round spherical space form 

S

3

/

Γ

{\displaystyle S^{3}/\Gamma }
* The shrinking round cylinder 

S

2

×

R

{\displaystyle S^{2}\times \mathbb {R} }
* The Bryant soliton

The first two singularity models arise from Type I singularities, whereas the last one arises from a Type II singularity.

### Singularities in 4d Ricci flow

In four dimensions very little is known about the possible singularities, other than that the possibilities are far more numerous than in three dimensions. To date the following singularity models are known

* S

3

×

R

{\displaystyle S^{3}\times \mathbb {R} }
* S

2

×

R

2

{\displaystyle S^{2}\times \mathbb {R} ^{2}}
* The 4d Bryant soliton
* Compact Einstein manifold of positive scalar curvature
* Compact gradient Kahler–Ricci shrinking soliton
* The FIK shrinker (discovered by M. Feldman, T. Ilmanen, D. Knopf) [9]
* The BCCD shrinker (discovered by Richard Bamler, Charles Cifarelli, Ronan Conlon, and Alix Deruelle)[10]

Note that the first three examples are generalizations of 3d singularity models. The FIK shrinker models the collapse of an embedded sphere with self-intersection number −1.

## Relation to diffusion

To see why the evolution equation defining the Ricci flow is indeed a kind of nonlinear diffusion equation, we can consider the special case of (real) two-manifolds in more detail.  Any metric tensor on a two-manifold can be written with respect to an exponential isothermal coordinate chart in the form

(These coordinates provide an example of a conformal coordinate chart, because angles, but not distances, are correctly represented.)

The easiest way to compute the Ricci tensor and Laplace–Beltrami operator for our Riemannian two-manifold is to use the differential forms method of Élie Cartan.  Take the coframe field

so that metric tensor becomes

Next, given an arbitrary smooth function 

h
(
x
,
y
)

{\displaystyle h(x,y)}

, compute the exterior derivative

Take the Hodge dual

Take another exterior derivative

(where we used the anti-commutative property of the exterior product).  That is,

Taking another Hodge dual gives

which gives the desired expression for the Laplace/Beltrami operator

To compute the curvature tensor, we take the exterior derivative of the covector fields making up our coframe:

From these expressions, we can read off the only independent spin connection one-form

where we have taken advantage of the anti-symmetric property of the connection (

ω

2

1

=
−

ω

1

2

{\displaystyle {\omega ^{2}}_{1}=-{\omega ^{1}}_{2}}

). Take another exterior derivative

This gives the curvature two-form

from which we can read off the only linearly independent component of the Riemann tensor using

Namely

from which the only nonzero components of the Ricci tensor are

From this, we find components with respect to the coordinate cobasis, namely

But the metric tensor is also diagonal, with

and after some elementary manipulation, we obtain an elegant expression for the Ricci flow:

This is manifestly analogous to the best known of all diffusion equations, the heat equation

where now 

Δ
=

D

x

2

+

D

y

2

{\displaystyle \Delta =D_{x}^{2}+D_{y}^{2}}

 is the usual Laplacian on the Euclidean plane.
The reader may object that the heat equation is of course a linear partial differential equation—where is the promised nonlinearity in the p.d.e. defining the Ricci flow?

The answer is that nonlinearity enters because the Laplace-Beltrami operator depends upon the same function p which we used to define the metric.  But notice that the flat Euclidean plane is given by taking 

p
(
x
,
y
)
=
0

{\displaystyle p(x,y)=0}

.  So if 

p

{\displaystyle p}

 is small in magnitude, we can consider it to define small deviations from the geometry of a flat plane, and if we retain only first order terms in computing the exponential, the Ricci flow on our two-dimensional almost flat Riemannian manifold becomes the usual two dimensional heat equation.  This computation suggests that, just as (according to the heat equation) an irregular temperature distribution in a hot plate tends to become more homogeneous over time, so too (according to the Ricci flow) an almost flat Riemannian manifold will tend to flatten out the same way that heat can be carried off "to infinity" in an infinite flat plate.  But if our hot plate is finite in size, and has no boundary where heat can be carried off, we can expect to homogenize the temperature, but clearly we cannot expect to reduce it to zero.  In the same way, we expect that the Ricci flow, applied to a distorted round sphere, will tend to round out the geometry over time, but not to turn it into a flat Euclidean geometry.

## Recent developments

The Ricci flow has been intensively studied since 1981. Some recent work has focused on the question of precisely how higher-dimensional Riemannian manifolds evolve under the Ricci flow, and in particular, what types of parametric singularities may form.  For instance, a certain class of solutions to the Ricci flow demonstrates that neckpinch singularities will form on an evolving 

n

{\displaystyle n}

-dimensional metric Riemannian manifold having a certain topological property (positive Euler characteristic), as the flow approaches some characteristic time 

t

0

{\displaystyle t_{0}}

.  In certain cases, such neckpinches will produce manifolds called Ricci solitons.

For a 3-dimensional manifold, Perelman showed how to continue past the singularities using surgery on the manifold.

Kähler metrics remain Kähler under Ricci flow, and so Ricci flow has also been studied in this setting, where it is called Kähler–Ricci flow.

### Ricci flow on manifolds with boundary

The study of the Ricci flow on manifolds with boundary was started by Ying Shen.[11] Shen introduced a boundary value problem for  manifolds with weakly umbilic boundaries, that is, the Second Fundamental Form of the boundary is a constant multiple of the metric, and then he proved that when the initial metric has positive Ricci curvature and the boundary is totally geodesic, the solution to the flow converges to a metric of constant positive curvature and totally geodesic boundary. Simon Brendle[12] showed that Shen's theorem is also valid for surfaces with totally geodesic boundary, and also introduced dynamic boundary conditions coupled to the Ricci flow.[13] The first results for boundaries that are not totally geodesic, which include convergence results, were given by Jean Cortissoz[14] in the case of 3-manifolds with convex weakly umbilic boundary, with subsequent developments, together with Alexander Murcia[15] and César Reyes,[16] to metrics on a disk and a cylinder. Artem Pulemotov[17] and then Panagiotis Gianniotis[18] introduced an interesting boundary value problem for the Ricci flow, and proved short time existence and uniqueness. Recently, Gang Li[19] showed that in the case of surfaces with boundary, there can be existence of a solution for all time without convergence. Tsz-Kiu Aaron Chow[20] has introduced a family of solutions to the Ricci flow which preserves some geometric properties of the initial data, and has used them to provide geometric applications to manifolds with boundary.

## Notes

* ^ Friedan, D. (1980). "Nonlinear models in 2+ε dimensions". Physical Review Letters (Submitted manuscript). 45 (13): 1057–1060. Bibcode:1980PhRvL..45.1057F. doi:10.1103/PhysRevLett.45.1057.
* ^ DeTurck, Dennis M. (1983). "Deforming metrics in the direction of their Ricci tensors". J. Differential Geom. 18 (1): 157–162. doi:10.4310/jdg/1214509286.
* ^ Eells, James Jr.; Sampson, J.H. (1964). "Harmonic mappings of Riemannian manifolds". Amer. J. Math. 86 (1): 109–160. doi:10.2307/2373037. JSTOR 2373037.
* ^ Gromov, M.; Thurston, W. (1987). "Pinching constants for hyperbolic manifolds". Invent. Math. 89 (1): 1–12. Bibcode:1987InMat..89....1G. doi:10.1007/BF01404671. S2CID 119850633.
* ^ Li, Peter; Yau, Shing-Tung (1986). "On the parabolic kernel of the Schrödinger operator". Acta Math. 156 (3–4): 153–201. doi:10.1007/BF02399203. S2CID 120354778.
* ^ Weeks, Jeffrey R. (1985). The Shape of Space: how to visualize surfaces and three-dimensional manifolds. New York: Marcel Dekker. ISBN 978-0-8247-7437-0..  A popular book that explains the background for the Thurston classification program.
* ^ Shi, W.-X. (1989). "Deforming the metric on complete Riemannian manifolds". Journal of Differential Geometry. 30: 223–301. doi:10.4310/jdg/1214443292.
* ^ Enders, J.; Mueller, R.; Topping, P. (2011). "On Type I Singularities in Ricci flow". Communications in Analysis and Geometry. 19 (5): 905–922. arXiv:1005.1624. doi:10.4310/CAG.2011.v19.n5.a4. S2CID 968534.
* ^ Maximo, D. (2014). "On the blow-up of four-dimensional Ricci flow singularities". J. Reine Angew. Math. 2014 (692): 153–171. arXiv:1204.5967. doi:10.1515/crelle-2012-0080. S2CID 17651053.
* ^ Bamler, R.; Cifarelli, C.; Conlon, R.; Deruelle, A. (2022). "A new complete two-dimensional shrinking gradient Kähler-Ricci soliton". arXiv:2206.10785 [math.DG].
* ^ Shen, Ying (1996-03-01). "On Ricci deformation of a Riemannian metric on manifold with boundary". Pacific Journal of Mathematics. 173 (1): 203–221. doi:10.2140/pjm.1996.173.203. ISSN 0030-8730.
* ^ Brendle, S. (2002-11-01). "Curvature flows on surfaces with boundary". Mathematische Annalen. 324 (3): 491–519. doi:10.1007/s00208-002-0350-4. ISSN 1432-1807.
* ^ Brendle, S. (2002-12-01). "A family of curvature flows on surfaces with boundary". Mathematische Zeitschrift. 241 (4): 829–869. doi:10.1007/s00209-002-0439-1. ISSN 1432-1823.
* ^ Cortissoz, Jean C. (2009-02-01). "Three-manifolds of positive curvature and convex weakly umbilic boundary". Geometriae Dedicata. 138 (1): 83–98. doi:10.1007/s10711-008-9300-y. ISSN 1572-9168.
* ^ Cortissoz, Jean C.; Murcia, Alexander (2019-08-23). "The Ricci flow on surfaces with boundary". Communications in Analysis and Geometry. 27 (2): 377–420. arXiv:1209.2386. doi:10.4310/CAG.2019.v27.n2.a5. ISSN 1944-9992.
* ^ Cortissoz, Jean C.; Reyes, César (2023). "Classical solutions to the one-dimensional logarithmic diffusion equation with nonlinear Robin boundary conditions". Mathematische Nachrichten. 296 (9): 4086–4107. doi:10.1002/mana.202100415. ISSN 1522-2616.
* ^ Pulemotov, Artem (2013-10-01). "Quasilinear parabolic equations and the Ricci flow on manifolds with boundary". Journal für die reine und angewandte Mathematik (Crelles Journal). 2013 (683): 97–118. arXiv:1012.2941. doi:10.1515/crelle-2012-0004. ISSN 1435-5345.
* ^ Gianniotis, Panagiotis (2016). "The Ricci flow on manifolds with boundary". Journal of Differential Geometry. 104 (2): 291–324. arXiv:1210.0813. doi:10.4310/jdg/1476367059. MR 3557306.
* ^ Li, Gang (2025-03-14), A normalized Ricci flow on surfaces with boundary towards the complete hyperbolic metric, arXiv:2502.00660
* ^ Chow, Tsz-Kiu Aaron (2022-02-01). "Ricci flow on manifolds with boundary with arbitrary initial metric". Journal für die reine und angewandte Mathematik (Crelles Journal). 2022 (783): 159–216. arXiv:2012.04430. doi:10.1515/crelle-2021-0060. ISSN 1435-5345.

## References

Articles for a popular mathematical audience.

* Anderson, Michael T. (2004). "Geometrization of 3-manifolds via the Ricci flow" (PDF). Notices Amer. Math. Soc. 51 (2): 184–193. MR 2026939.
* Milnor, John (2003). "Towards the Poincaré Conjecture and the classification of 3-manifolds" (PDF). Notices Amer. Math. Soc. 50 (10): 1226–1233. MR 2009455.
* Morgan, John W. (2005). "Recent progress on the Poincaré conjecture and the classification of 3-manifolds". Bull. Amer. Math. Soc. (N.S.). 42 (1): 57–78. doi:10.1090/S0273-0979-04-01045-6. MR 2115067.
* Tao, T. (2008). "Ricci flow" (PDF). In Gowers, Timothy; Barrow-Green, June; Leader, Imre (eds.). The Princeton Companion to Mathematics. Princeton University Press. pp. 279–281. ISBN 978-0-691-11880-2.

Research articles.

* Böhm, Christoph; Wilking, Burkhard (2008). "Manifolds with positive curvature operators are space forms". Ann. of Math. (2). 167 (3): 1079–1097. arXiv:math/0606187. doi:10.4007/annals.2008.167.1079. JSTOR 40345372. MR 2415394. S2CID 15521923.
* Brendle, Simon (2008). "A general convergence result for the Ricci flow in higher dimensions". Duke Math. J. 145 (3): 585–601. arXiv:0706.1218. doi:10.1215/00127094-2008-059. MR 2462114. S2CID 438716. Zbl 1161.53052.
* Brendle, Simon; Schoen, Richard (2009). "Manifolds with 1/4-pinched curvature are space forms". J. Amer. Math. Soc. 22 (1): 287–307. arXiv:0705.0766. Bibcode:2009JAMS...22..287B. doi:10.1090/S0894-0347-08-00613-9. JSTOR 40587231. MR 2449060. S2CID 2901565.
* Cao, Huai-Dong; Xi-Ping Zhu (June 2006). "A Complete Proof of the Poincaré and Geometrization Conjectures — application of the Hamilton-Perelman theory of the Ricci flow" (PDF). Asian Journal of Mathematics. 10 (2). MR 2488948. Erratum.
Revised version: Huai-Dong Cao; Xi-Ping Zhu (2006). "Hamilton-Perelman's Proof of the Poincaré Conjecture and the Geometrization Conjecture". arXiv:math.DG/0612069.
* Revised version: Huai-Dong Cao; Xi-Ping Zhu (2006). "Hamilton-Perelman's Proof of the Poincaré Conjecture and the Geometrization Conjecture". arXiv:math.DG/0612069.
* Chow, Bennett (1991). "The Ricci flow on the 2-sphere". J. Differential Geom. 33 (2): 325–334. doi:10.4310/jdg/1214446319. MR 1094458. Zbl 0734.53033.
* Colding, Tobias H.; Minicozzi, William P. II (2005). "Estimates for the extinction time for the Ricci flow on certain 3-manifolds and a question of Perelman" (PDF). J. Amer. Math. Soc. 18 (3): 561–569. arXiv:math/0308090. doi:10.1090/S0894-0347-05-00486-8. JSTOR 20161247. MR 2138137. S2CID 2810043.
* Hamilton, Richard S. (1982). "Three-manifolds with positive Ricci curvature". Journal of Differential Geometry. 17 (2): 255–306. doi:10.4310/jdg/1214436922. MR 0664497. Zbl 0504.53034.
* Hamilton, Richard S. (1986). "Four-manifolds with positive curvature operator". J. Differential Geom. 24 (2): 153–179. doi:10.4310/jdg/1214440433. MR 0862046. Zbl 0628.53042.
* Hamilton, Richard S. (1988). "The Ricci flow on surfaces". Mathematics and general relativity (Santa Cruz, CA, 1986). Contemp. Math. Vol. 71. Amer. Math. Soc., Providence, RI. pp. 237–262. doi:10.1090/conm/071/954419. MR 0954419.
* Hamilton, Richard S. (1993a). "The Harnack estimate for the Ricci flow". J. Differential Geom. 37 (1): 225–243. doi:10.4310/jdg/1214453430. MR 1198607. Zbl 0804.53023.
* Hamilton, Richard S. (1993b). "Eternal solutions to the Ricci flow". J. Differential Geom. 38 (1): 1–11. doi:10.4310/jdg/1214454093. MR 1231700. Zbl 0792.53041.
* Hamilton, Richard S. (1995a). "A compactness property for solutions of the Ricci flow". Amer. J. Math. 117 (3): 545–572. doi:10.2307/2375080. JSTOR 2375080. MR 1333936.
* Hamilton, Richard S. (1995b). "The formation of singularities in the Ricci flow". Surveys in differential geometry, Vol. II (Cambridge, MA, 1993). Int. Press, Cambridge, MA. pp. 7–136. doi:10.4310/SDG.1993.v2.n1.a2. MR 1375255.
* Hamilton, Richard S. (1997). "Four-manifolds with positive isotropic curvature". Comm. Anal. Geom. 5 (1): 1–92. doi:10.4310/CAG.1997.v5.n1.a1. MR 1456308. Zbl 0892.53018.
* Hamilton, Richard S. (1999). "Non-singular solutions of the Ricci flow on three-manifolds". Comm. Anal. Geom. 7 (4): 695–729. doi:10.4310/CAG.1999.v7.n4.a2. MR 1714939.
* Bruce Kleiner; John Lott (2008). "Notes on Perelman's papers". Geometry & Topology. 12 (5): 2587–2855. arXiv:math.DG/0605667. doi:10.2140/gt.2008.12.2587. MR 2460872. S2CID 119133773.
* Perelman, Grisha (2002). "The entropy formula for the Ricci flow and its geometric applications". arXiv:math/0211159.
* Perelman, Grisha (2003a). "Ricci flow with surgery on three-manifolds". arXiv:math/0303109.
* Perelman, Grisha (2003b). "Finite extinction time for the solutions to the Ricci flow on certain three-manifolds". arXiv:math/0307245.

## Textbooks

* Andrews, Ben; Hopper, Christopher (2011). The Ricci Flow in Riemannian Geometry: A Complete Proof of the Differentiable 1/4-Pinching Sphere Theorem. Lecture Notes in Mathematics. Vol. 2011. Heidelberg: Springer. doi:10.1007/978-3-642-16286-2. ISBN 978-3-642-16285-5.
* Brendle, Simon (2010). Ricci Flow and the Sphere Theorem. Graduate Studies in Mathematics. Vol. 111. Providence, RI: American Mathematical Society. doi:10.1090/gsm/111. ISBN 978-0-8218-4938-5.
* Cao, H.D.; Chow, B.; Chu, S.C.; Yau, S.T., eds. (2003). Collected Papers on Ricci Flow. Series in Geometry and Topology. Vol. 37. Somerville, MA: International Press. ISBN 1-57146-110-8.
* Chow, Bennett; Chu, Sun-Chin; Glickenstein, David; Guenther, Christine; Isenberg, James; Ivey, Tom; Knopf, Dan; Lu, Peng; Luo, Feng; Ni, Lei (2007). The Ricci Flow: Techniques and Applications. Part I. Geometric Aspects. Mathematical Surveys and Monographs. Vol. 135. Providence, RI: American Mathematical Society. doi:10.1090/surv/135. ISBN 978-0-8218-3946-1.
* Chow, Bennett; Chu, Sun-Chin; Glickenstein, David; Guenther, Christine; Isenberg, James; Ivey, Tom; Knopf, Dan; Lu, Peng; Luo, Feng; Ni, Lei (2008). The Ricci Flow: Techniques and Applications. Part II. Analytic Aspects. Mathematical Surveys and Monographs. Vol. 144. Providence, RI: American Mathematical Society. doi:10.1090/surv/144. ISBN 978-0-8218-4429-8.
* Chow, Bennett; Chu, Sun-Chin; Glickenstein, David; Guenther, Christine; Isenberg, James; Ivey, Tom; Knopf, Dan; Lu, Peng; Luo, Feng; Ni, Lei (2010). The Ricci Flow: Techniques and Applications. Part III. Geometric-Analytic Aspects. Mathematical Surveys and Monographs. Vol. 163. Providence, RI: American Mathematical Society. doi:10.1090/surv/163. ISBN 978-0-8218-4661-2.
* Chow, Bennett; Chu, Sun-Chin; Glickenstein, David; Guenther, Christine; Isenberg, James; Ivey, Tom; Knopf, Dan; Lu, Peng; Luo, Feng; Ni, Lei (2015). The Ricci Flow: Techniques and Applications. Part IV. Long-Time Solutions and Related Topics. Mathematical Surveys and Monographs. Vol. 206. Providence, RI: American Mathematical Society. doi:10.1090/surv/206. ISBN 978-0-8218-4991-0.
* Chow, Bennett; Knopf, Dan (2004). The Ricci Flow: An Introduction. Mathematical Surveys and Monographs. Vol. 110. Providence, RI: American Mathematical Society. doi:10.1090/surv/110. ISBN 0-8218-3515-7.
* Chow, Bennett; Lu, Peng; Ni, Lei (2006). Hamilton's Ricci Flow. Graduate Studies in Mathematics. Vol. 77. Beijing, New York: American Mathematical Society, Providence, RI; Science Press. doi:10.1090/gsm/077. ISBN 978-0-8218-4231-7.
* Morgan, John W.; Fong, Frederick Tsz-Ho (2010). Ricci Flow and Geometrization of 3-Manifolds. University Lecture Series. Vol. 53. Providence, RI: American Mathematical Society. doi:10.1090/ulect/053. ISBN 978-0-8218-4963-7.
* Morgan, John; Tian, Gang (2007). Ricci Flow and the Poincaré Conjecture. Clay Mathematics Monographs. Vol. 3. Providence, RI and Cambridge, MA: American Mathematical Society and Clay Mathematics Institute. ISBN 978-0-8218-4328-4.
* Müller, Reto (2006). Differential Harnack inequalities and the Ricci flow. EMS Series of Lectures in Mathematics. Vol. 5. Zürich: European Mathematical Society (EMS). doi:10.4171/030. hdl:2318/1701023. ISBN 978-3-03719-030-2.
* Topping, Peter (2006). Lectures on the Ricci Flow. London Mathematical Society Lecture Note Series. Vol. 325. Cambridge: Cambridge University Press. doi:10.1017/CBO9780511721465. ISBN 0-521-68947-3.
* Zhang, Qi S. (2011). Sobolev Inequalities, Heat Kernels under Ricci Flow, and the Poincaré Conjecture. Boca Raton, FL: CRC Press. ISBN 978-1-4398-3459-6.

## External links

* Isenberg, James A (23 April 2014). "Ricci Flow" (video). Brady Haran. Archived from the original on 2021-12-12. Retrieved 23 April 2014.

vteRiemannian geometry (Glossary)
Basic concepts | Curvature
tensor
Scalar
Ricci
Sectional
Exponential map
Geodesic
Inner product
Metric tensor
Levi-Civita connection
Covariant derivative
Signature
Raising and lowering indices/Musical isomorphism
Parallel transport
Riemannian manifold
Pseudo-Riemannian manifold
Riemannian volume form
Types of manifolds | Hermitian
Hyperbolic
Kähler
Kenmotsu
Main results | Fundamental theorem of Riemannian geometry
Gauss's lemma
Gauss–Bonnet theorem
Hopf–Rinow theorem
Nash embedding theorem
Ricci flow
Schur's lemma
Generalizations | Finsler
Hilbert
Sub-Riemannian
Applications | General relativity
Geometrization conjecture
Poincaré conjecture
Uniformization theorem

* v
* t
* e
* Curvature
tensor
Scalar
Ricci
Sectional
* tensor
* Scalar
* Ricci
* Sectional
* Exponential map
* Geodesic
* Inner product
* Metric tensor
Levi-Civita connection
Covariant derivative
Signature
Raising and lowering indices/Musical isomorphism
* Levi-Civita connection
* Covariant derivative
* Signature
* Raising and lowering indices/Musical isomorphism
* Parallel transport
* Riemannian manifold
* Pseudo-Riemannian manifold
* Riemannian volume form
* Hermitian
* Hyperbolic
* Kähler
* Kenmotsu
* Fundamental theorem of Riemannian geometry
* Gauss's lemma
* Gauss–Bonnet theorem
* Hopf–Rinow theorem
* Nash embedding theorem
* Ricci flow
* Schur's lemma
* Finsler
* Hilbert
* Sub-Riemannian
* General relativity
* Geometrization conjecture
* Poincaré conjecture
* Uniformization theorem

vteManifolds (Glossary, List, Category)
Basic concepts | Topological manifold
Atlas
Differentiable/Smooth manifold
Differential structure
Smooth atlas
Submanifold
Riemannian manifold
Smooth map
Submersion
Pushforward
Tangent space
Differential form
Vector field
Main theorems (list) | Atiyah–Singer index
Darboux's
De Rham's
Frobenius
Generalized Stokes
Hopf–Rinow
Noether's
Sard's
Whitney embedding
Maps | Curve
Diffeomorphism
Local
Geodesic
Exponential map
in Lie theory
Foliation
Immersion
Integral curve
Lie derivative
Section
Submersion
Types ofmanifolds | Calabi–Yau
Closed
Collapsing
Complete
(Almost) Complex
(Almost) Contact
Einstein
Fibered
Finsler
(Almost, Ricci-) Flat
G-structure
Hadamard
Hermitian
Hyperbolic
(Hyper) Kähler
Kenmotsu
Lie group
Lie algebra
Manifold with boundary
Nilmanifold
Oriented
Parallelizable
Poisson
Prime
Quaternionic
Hypercomplex
(Pseudo-, Sub-) Riemannian
Rizza
Sasakian
Stein
(Almost) Symplectic
Tame
Tensors | Vectors
Distribution
Lie bracket
Pushforward
Tangent space
bundle
Torsion
Vector field
Vector flow
Covectors
Closed/Exact
Covariant derivative
Cotangent space
bundle
De Rham cohomology
Differential form
Complex
Vector-valued
One-form
Exterior derivative
Interior product
Pullback
Ricci curvature
flow
Riemann curvature tensor
Tensor field
density
Volume form
Wedge product
Bundles
Adjoint
Affine
Associated
Cotangent
Dual
Fiber
(Co-) Fibration
Jet
Lie algebra
(Stable) Normal
Principal
Spinor
Subbundle
Tangent
Tensor
Vector
Connections
Affine
Cartan
Ehresmann
Form
Generalized
Koszul
Levi-Civita
Principal
Vector
Parallel transport | Vectors | Distribution
Lie bracket
Pushforward
Tangent space
bundle
Torsion
Vector field
Vector flow | Covectors | Closed/Exact
Covariant derivative
Cotangent space
bundle
De Rham cohomology
Differential form
Complex
Vector-valued
One-form
Exterior derivative
Interior product
Pullback
Ricci curvature
flow
Riemann curvature tensor
Tensor field
density
Volume form
Wedge product | Bundles | Adjoint
Affine
Associated
Cotangent
Dual
Fiber
(Co-) Fibration
Jet
Lie algebra
(Stable) Normal
Principal
Spinor
Subbundle
Tangent
Tensor
Vector | Connections | Affine
Cartan
Ehresmann
Form
Generalized
Koszul
Levi-Civita
Principal
Vector
Parallel transport
Vectors | Distribution
Lie bracket
Pushforward
Tangent space
bundle
Torsion
Vector field
Vector flow
Covectors | Closed/Exact
Covariant derivative
Cotangent space
bundle
De Rham cohomology
Differential form
Complex
Vector-valued
One-form
Exterior derivative
Interior product
Pullback
Ricci curvature
flow
Riemann curvature tensor
Tensor field
density
Volume form
Wedge product
Bundles | Adjoint
Affine
Associated
Cotangent
Dual
Fiber
(Co-) Fibration
Jet
Lie algebra
(Stable) Normal
Principal
Spinor
Subbundle
Tangent
Tensor
Vector
Connections | Affine
Cartan
Ehresmann
Form
Generalized
Koszul
Levi-Civita
Principal
Vector
Parallel transport
Related | Classification of manifolds
Gauge theory
History
Morse theory
Moving frame
Singularity theory
Generalizations | Banach
Diffeology
Diffiety
Fréchet
Hilbert
K-theory
Non-Hausdorff
Orbifold
Secondary calculus
over commutative algebras
Sheaf
Stratifold
Supermanifold
Stratified space

* v
* t
* e
* Topological manifold
Atlas
* Atlas
* Differentiable/Smooth manifold
Differential structure
Smooth atlas
* Differential structure
* Smooth atlas
* Submanifold
* Riemannian manifold
* Smooth map
* Submersion
* Pushforward
* Tangent space
* Differential form
* Vector field
* Atiyah–Singer index
* Darboux's
* De Rham's
* Frobenius
* Generalized Stokes
* Hopf–Rinow
* Noether's
* Sard's
* Whitney embedding
* Curve
* Diffeomorphism
Local
* Local
* Geodesic
* Exponential map
in Lie theory
* in Lie theory
* Foliation
* Immersion
* Integral curve
* Lie derivative
* Section
* Submersion
* Calabi–Yau
* Closed
* Collapsing
* Complete
* (Almost) Complex
* (Almost) Contact
* Einstein
* Fibered
* Finsler
* (Almost, Ricci-) Flat
* G-structure
* Hadamard
* Hermitian
* Hyperbolic
* (Hyper) Kähler
* Kenmotsu
* Lie group
Lie algebra
* Lie algebra
* Manifold with boundary
* Nilmanifold
* Oriented
* Parallelizable
* Poisson
* Prime
* Quaternionic
* Hypercomplex
* (Pseudo-, Sub-) Riemannian
* Rizza
* Sasakian
* Stein
* (Almost) Symplectic
* Tame

Vectors | Distribution
Lie bracket
Pushforward
Tangent space
bundle
Torsion
Vector field
Vector flow
Covectors | Closed/Exact
Covariant derivative
Cotangent space
bundle
De Rham cohomology
Differential form
Complex
Vector-valued
One-form
Exterior derivative
Interior product
Pullback
Ricci curvature
flow
Riemann curvature tensor
Tensor field
density
Volume form
Wedge product
Bundles | Adjoint
Affine
Associated
Cotangent
Dual
Fiber
(Co-) Fibration
Jet
Lie algebra
(Stable) Normal
Principal
Spinor
Subbundle
Tangent
Tensor
Vector
Connections | Affine
Cartan
Ehresmann
Form
Generalized
Koszul
Levi-Civita
Principal
Vector
Parallel transport

* Distribution
* Lie bracket
* Pushforward
* Tangent space
bundle
* bundle
* Torsion
* Vector field
* Vector flow
* Closed/Exact
* Covariant derivative
* Cotangent space
bundle
* bundle
* De Rham cohomology
* Differential form
Complex
Vector-valued
One-form
* Complex
* Vector-valued
* One-form
* Exterior derivative
* Interior product
* Pullback
* Ricci curvature
flow
* flow
* Riemann curvature tensor
* Tensor field
density
* density
* Volume form
* Wedge product
* Adjoint
* Affine
* Associated
* Cotangent
* Dual
* Fiber
* (Co-) Fibration
* Jet
* Lie algebra
* (Stable) Normal
* Principal
* Spinor
* Subbundle
* Tangent
* Tensor
* Vector
* Affine
* Cartan
* Ehresmann
* Form
* Generalized
* Koszul
* Levi-Civita
* Principal
* Vector
* Parallel transport
* Classification of manifolds
* Gauge theory
* History
* Morse theory
* Moving frame
* Singularity theory
* Banach
* Diffeology
* Diffiety
* Fréchet
* Hilbert
* K-theory
* Non-Hausdorff
* Orbifold
* Secondary calculus
over commutative algebras
* over commutative algebras
* Sheaf
* Stratifold
* Supermanifold
* Stratified space

Authority control databases
International | GND
Other | Yale LUX

* GND
* Yale LUX
* 1981 introductions
* 3-manifolds
* Geometric flow
* Nonlinear partial differential equations
* Riemannian geometry
* Riemannian manifolds
* Articles with short description
* Short description is different from Wikidata
* Pages with Italian IPA