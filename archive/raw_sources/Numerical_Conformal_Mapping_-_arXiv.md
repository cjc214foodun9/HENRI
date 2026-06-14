# Numerical Conformal Mapping - arXiv
Source URL: https://arxiv.org/html/2507.14872v1

# Numerical Conformal Mapping

Conformal mapping may be the best-known topic in complex analysis.
Any simply connected nonempty domain ΩΩ\Omegaroman_Ω in the complex plane
ℂℂ{\mathbb{C}}blackboard_C (assuming Ω≠ℂΩℂ\Omega\neq{\mathbb{C}}roman_Ω ≠ blackboard_C) can be mapped bijectively
to the unit disk by an analytic function with nonvanishing
derivative, as in Figure 1. If ΩΩ\Omegaroman_Ω is doubly-connected, it can
be mapped to a circular annulus 1<|z|<R1𝑧𝑅1<|z|<R1 < | italic_z | < italic_R for some R𝑅Ritalic_R, called the
conformal modulus, which is uniquely determined by ΩΩ\Omegaroman_Ω,
as in Figure 2. If ΩΩ\Omegaroman_Ω has connectivity higher than 2222, it
can be mapped onto various canonical domains such as a disk with
exclusions in the form of slits or smaller disks, as in Figure 3.

Most conformal maps cannot be found analytically, so when
computers began to appear in the 1950s and 1960s, the field of numerical conformal mapping was born.
Many methods involve discretizations of integral equations, such
as those of Gerschgorin, Lichtenstein, Kerzmann-Stein-Trummer, Symm,
and Theodorsen. Dieter Gaier of the University of Giessen published
an important monograph in those early years, which unfortunately was
never translated to English [11]. A later reference is the book
of Henrici [14], and there is a major survey paper by Wegmann
[24].

Conformal mapping of polygons has always been a conspicuous special
case, thanks to the Schwarz-Christoffel transformation, and this
became readily available for applications with the appearance
of Driscoll’s SC Toolbox in Matlab in the 1990s
[7,9,21]. To this day, this Toolbox remains the most
widely-used software for conformal mapping, and Figure 4 illustrates
that its power extends to nontrivial geometries [10]. This figure
shows the mapping of a quadrilateral, namely a domain with four
distinguished boundary points, onto a rectangle, whose aspect ratio
μ≈18.20539𝜇18.20539\mu\approx 18.20539italic_μ ≈ 18.20539
(also called the conformal modulus) is uniquely determined [18].

Many other conformal mapping methods have been developed too.
For multiply-connected domains, for example, there are methods
based on generalized Schwarz-Christoffel mapping [5], the
Schottky-Klein prime function [4,6], and an integral equation
related to the Neumann kernel
[16]. In a field as old as this, there
are numerous further methods that have been explored, including
“circle packing” (introduced by Thurston) [20], the
“zipper algorithm” [15], the “charge simulation method”
(a version of the method of fundamental solutions) [1],
rational approximation [12,23], and many varieties of series and
iterations, sometimes accelerated by the Fast Multipole Method [2,17].
Commands conformal and conformal2 for smooth simply
and doubly connected conformal mapping can be found in
Chebfun [8]. Figure 5 shows another map of a quadrilateral,
now one with curved sides, computed by the “lightning”
rational function method [12].

Traditionally, a fundamental distinction in numerical conformal
mapping was between methods mapping from problem domain to
canonical domain and those going in the inverse direction, from
canonical domain to problem domain. In the former case, the map
can be reduced to a Laplace problem, so the integral equations
of Lichtenstein and Symm are linear, for example, whereas that of
Theodorsen in the other direction is nonlinear. However, the recent
appearance of fast numerical methods for rational approximation has
diminished the distinction between the forward and
inverse maps. Once one has the boundary correspondence function,
the homeomorphism between the domain and range boundaries (provided this
makes sense, as is always the case
for Jordan domains), it is an easy matter to use it to compute efficient and
accurate rational approximations in both directions [13].
For example, the conformal map of an irregular hexagon
to the unit disk, and its inverse map, can each typically be approximated
to 6-digit accuracy by rational functions of degrees of the order of 50 to
100.  Evaluation of such functions takes just microseconds per point.

As just mentioned, conformal mapping is a special case of a Laplace problem.
For a simply-connected domain ΩΩ\Omegaroman_Ω bounded
by a Jordan curve ΓΓ\Gammaroman_Γ enclosing the origin,
following Theorem 16.5a of [14] and p. 253 of [19],
suppose we seek the unique conformal map f𝑓fitalic_f onto the unit disk with
f⁢(0)=0𝑓00f(0)=0italic_f ( 0 ) = 0 and f′⁢(0)>0superscript𝑓′00f^{\prime}(0)>0italic_f start_POSTSUPERSCRIPT ′ end_POSTSUPERSCRIPT ( 0 ) > 0.
Then g⁢(z)=log⁡(f⁢(z)/z)𝑔𝑧𝑓𝑧𝑧g(z)=\log(f(z)/z)italic_g ( italic_z ) = roman_log ( italic_f ( italic_z ) / italic_z ) is a nonzero
analytic function on ΩΩ\Omegaroman_Ω that is continuous on Ω¯¯Ω\overline{\Omega}over¯ start_ARG roman_Ω end_ARG and
has real part −log⁡|z|𝑧-\log|z|- roman_log | italic_z | for z∈Γ𝑧Γz\in\Gammaitalic_z ∈ roman_Γ. If we write
g⁢(z)=u⁢(z)+i⁢v⁢(z)𝑔𝑧𝑢𝑧𝑖𝑣𝑧g(z)=u(z)+iv(z)italic_g ( italic_z ) = italic_u ( italic_z ) + italic_i italic_v ( italic_z ), where u𝑢uitalic_u and v𝑣vitalic_v are real harmonic functions,
then u𝑢uitalic_u is the solution of the Dirichlet problem

 | Δ⁢u=0;u⁢(z)=−log⁡|z|,z∈Γ,formulae-sequenceΔ𝑢0formulae-sequence𝑢𝑧𝑧𝑧Γ\Delta u=0\,;\quad u(z)=-\log|z|,~{}z\in\Gamma,roman_Δ italic_u = 0 ; italic_u ( italic_z ) = - roman_log | italic_z | , italic_z ∈ roman_Γ , |  | (1)

and v𝑣vitalic_v is its harmonic conjugate in ΩΩ\Omegaroman_Ω with v⁢(0)=0𝑣00v(0)=0italic_v ( 0 ) = 0.
Combining these elements, we see that f𝑓fitalic_f is given by the formula

 | f⁢(z)=z⁢eu⁢(z)+i⁢v⁢(z).𝑓𝑧𝑧superscript𝑒𝑢𝑧𝑖𝑣𝑧f(z)=z\kern 0.5pte^{u(z)+iv(z)}.italic_f ( italic_z ) = italic_z italic_e start_POSTSUPERSCRIPT italic_u ( italic_z ) + italic_i italic_v ( italic_z ) end_POSTSUPERSCRIPT . |  | (2)

Thus a solution to (1), provided it also produces the harmonic conjugate
v⁢(z)𝑣𝑧v(z)italic_v ( italic_z ),
solves the conformal mapping problem.
Note that u⁢(z)+log⁡|z|𝑢𝑧𝑧u(z)+\log|z|italic_u ( italic_z ) + roman_log | italic_z | is the Green’s function of ΩΩ\Omegaroman_Ω
with respect to the point z=0𝑧0z=0italic_z = 0, so f𝑓fitalic_f is essentially the
exponential of the Green’s function.
Domains of higher connectivity can also be reduced to Laplace problems.

The availability of so many tools for numerical conformal mapping
may suggest that the problem is easy, but in fact there are
challenges, of which two stand out. One is that most domains of
practical interest contain corners, where the mapping function
will usually be singular, and it is essential to treat these
specially if one wants more than a digit or two of accuracy.
In the case of a polygon, the corners define the whole problem, and
these are dealt with by the Schwarz-Christoffel formula and its
numerical realization e.g. with compound Gauss-Jacobi quadrature.
The other great challenge is that of exponential distortions, which
are referred to as
the phenomenon of “crowding”. Whenever a domain is elongated in
certain directions, even as mildly as in the example of Figure 1,
its conformal map onto a nonelongated
canonical region will involve
exponentially large distortions. As summarized in
Theorems 2–5 of [13], the distortion scales as exp⁡(π⁢L)𝜋𝐿\exp(\pi L)roman_exp ( italic_π italic_L ),
where L𝐿Litalic_L is the aspect ratio of the elongation. As a consequence,
it is usually not a good idea to attempt to
map an elongated region onto, say, a disk or a half-plane.
Other targets such as rectangles or
infinite strips may come into play, for example to treat
a domain like that of Figure 4.

I have saved the most philosophical question for last.
What is the use of numerical conformal mapping?
The following views are personal, and not all experts would
agree with them.

A great use of numerical conformal maps is to give us insight into
principles of complex analysis, harmonic functions, and their applications.
For example, the blue curves in
Figure 4 can be interpreted as flow lines of
electricity, heat, ideal fluid, or probability
from one end to the other. If the image rectangle has length-to-width
ratio L𝐿Litalic_L, for example, then a channel cut into this shape from a piece of
metal will have electrical resistance ρ⁢L𝜌𝐿\rho Litalic_ρ italic_L, where ρ𝜌\rhoitalic_ρ is the
resistance of a unit square. A good image, which will almost always
have to be numerically computed, can fix these ideas beautifully in
the mind.
Throughout my career I have drawn pleasure and insight from pictures like
these. I cannot imagine teaching complex variables without
showing some online demonstrations of conformal maps.

Specifically, two of the features that numerical conformal maps
illustrate compellingly are precisely the two
computational challenges mentioned above: behavior near singularities
(note how the blue curves in Figure 4
avoid salient corners while wrapping tightly
around reentrant ones), and exponential distortions (note the big white
region in Figure 1).

The use of conformal maps that is mentioned perhaps
more often is that they may be helpful for solving problems.
For example, every
complex analysis text tells the reader that a conformal map
may be used to solve the Laplace equation, since it reduces
a hard problem to an easy one.
I believe that the truth is not so simple.
In fact, computing a conformal map is essentially the same as solving
a Laplace problem, and whatever numerical method one employs to find
the map could probably be applied to the Laplace problem directly.
So in many cases, nothing is gained numerically from conformal mapping.
The lesson becomes even stronger for applications to other PDEs that
may not be conformally invariant.

For example, suppose one is given a Laplace Dirichlet problem on the
domain on the left side of Figure 1. One can solve it by mapping
to the disk and applying the Poisson integral formula—but where
does one get that map? Probably by solving an integral equation or expanding
in a series, and these techniques would work equally well for the Laplace
problem itself. In smooth multiply connected domains, for example,
series expansions work beautifully for solving
Laplace problems [22], and although a circular
annulus is a natural domain for doubly-connected geometries, the canonical
domains with connectivity
≥3absent3\geq 3≥ 3 are less natural and do not often lead to an easy solution of your
PDE.  In more extreme cases, conformal maps may not merely transplant
the difficulty but increase it, when corners or
elongations are present.

So for me, the glory of numerical conformal mapping is not in the
numbers it produces, but in the images and insights.
To finish with a fine image provided by Toby Driscoll,
Figure 6 shows Emma’s maze again, but now with the solution
indicated by a color spectrum blending from red at one end to blue at
the other. The lightness of the colors is scaled by
1−y21superscript𝑦21-y^{2}1 - italic_y start_POSTSUPERSCRIPT 2 end_POSTSUPERSCRIPT if the target rectangle has its smaller dimension −1<y<11𝑦1-1<y<1- 1 < italic_y < 1.
As a mark of exponential distortion or “crowding,”
a particle beginning Brownian motion at the center of this royal road
(marked by a black dot)
would have exponentially small probability very close to
(8/π)⁢exp⁡(−μ⁢π/2)≈9.692555×10−138𝜋𝜇𝜋29.692555superscript1013(8/\pi)\exp(-\mu\pi/2)\approx 9.692555\times 10^{-13}( 8 / italic_π ) roman_exp ( - italic_μ italic_π / 2 ) ≈ 9.692555 × 10 start_POSTSUPERSCRIPT - 13 end_POSTSUPERSCRIPT of hitting the boundary first at one of the ends
rather than along the sides [3, chapter 10].

Acknowledgments.
A number of colleagues have helped me considerably
in preparing this column, and I am particularly grateful
to Tom DeLillo and Toby Driscoll.

References

[1]
K. Amano, A charge simulation method for the numerical conformal
mapping of interior, exterior and doubly-connected domains, J. Comput. Appl. Math., 53 (1994), 353–370.

[2]
L. Banjai and L. N. Trefethen, A multipole method for
Schwarz-Christoffel mapping of polygons with thousands of sides,
SIAM J. Sci. Comput., 25 (2003), 1042–1065.

[3]
F. Bornemann, D. Laurie, S. Wagon, and J. Waldvogel,
The SIAM 100-digit Challenge: A Study in
High-Accuracy Numerical Computing, SIAM, 2004.

[4]
D. G. Crowdy, Solving Problems in Multiply Connected Domains,
SIAM (2020).

[5]
T. K. DeLillo, A. R. Elcrat, E. H. Kropf, and J. A. Pfaltzgraff,
Efficient calculation of Schwarz-Christoffel transformations for
multiply connected domains using Laurent series, Comput. Methods Funct. Theory, 13 (2013), pp. 307–336.

[6]
T. K. DeLillo and C. C. Green, Computation of plane potential flow
past multi-element airfoils using the Schottky-Klein prime function,
Physica D, 450 (2023), 133753.

[7] T. A. Driscoll,
Schwarz-Christoffel Toolbox in Matlab, https://github.com/tobydriscoll/sc-toolbox.

[8] T. A. Driscoll, N. Hale, and L. N. Trefethen,
Chebfun Guide, Pafnuty Publications, Oxford (2014);
see also www.chebfun.org.

[9] T. A. Driscoll and L. N. Trefethen,
Schwarz-Christoffel Mapping, Cambridge (2002).

[10]
T. A. Driscoll and S. A. Vavasis, Numerical conformal mapping
using cross-ratios and Delaunay triangulation,
SIAM J. Sci. Comput., 19 (1998), 1783–1803.

[11] D. Gaier,
Konstructive Methoden der konformen Abbildung, Springer, Berlin (1964).

[12] A. Gopal and L. N. Trefethen,
Solving Laplace problems with corner singularities via
rational functions, SIAM J. Numer. Anal., 57 (2019),
2074–2094; see also people.maths.ox.ac.uk/trefethen/laplace/.

[13] A. Gopal and L. N. Trefethen,
Representation of conformal maps by rational
functions, Numer. Math., 142 (2019), 359–382.

[14]
P. Henrici, Applied and Computational Complex Analysis, v. 3,
Wiley, New York (1986).

[15]
D. E. Marshall and S. Rohde, Convergence of a variant of the zipper
algorithm for conformal mapping, SIAM J. Numer. Anal., 45
(2007), 2577–2609.

[16]
M. M. S. Nasser, Numerical conformal mapping via a boundary integral
equation with the generalized Neumann kernel,
SIAM J. Sci. Comput., 31 (2009), 1695–1715.

[17]
S. T. O’Donnell and V. Rokhlin,
A fast algorithm for the numerical evaluation of conformal mappings,
SIAM J. Sci. Stat. Comput., 10 (1989), 475–487.

[18]
N. Papamichael and N. Stylianopoulos, Numerical Conformal
Mapping: Domain Decomposition and the Mapping of Quadrilaterals,
World Scientific (2010).

[19]
M. Schiffer, Some recent developments in the theory of conformal
mapping, appendix to R. Courant, Dirichlet’s Principle,
Conformal Mapping, and Minimal Surfaces, Interscience (1950).

[20]
K. Stephenson, Introduction to Circle Packing: The Theory of
Discrete Analytic Functions, Cambridge (2005).

[21]
L. N. Trefethen, Numerical computation of the Schwarz-Christoffel
transformation, SIAM J. Sci. Comput., 1 (1980), 82–102.

[22]
L. N. Trefethen, Series solution of Laplace problems, ANZIAM
J., 60 (2018), pp. 1–26.

[23]
L. N. Trefethen, Numerical conformal mapping with rational functions,
Comput. Methods Funct. Theory, 20 (2020), pp. 369–387.

[24]
R. Wegmann, Methods for numerical conformal mapping, in Handbook
of Complex Analysis: Geometric Function Theory, v. 2222, R. Kühnau,
ed., Elsevier, 2 (2005), pp. 351–477.