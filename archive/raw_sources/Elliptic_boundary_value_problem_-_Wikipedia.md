# Elliptic boundary value problem - Wikipedia
Source URL: https://en.wikipedia.org/wiki/Elliptic_boundary_value_problem

| This article needs additional citations for verification. Please help improve this article by adding citations to reliable sources. Unsourced material may be challenged and removed.Find sources: "Elliptic boundary value problem" – news · newspapers · books · scholar · JSTOR (June 2022) (Learn how and when to remove this message)

In mathematics, an elliptic boundary value problem is a special kind of boundary value problem which can be thought of as the steady state of an evolution problem. For example, the Dirichlet problem for the Laplacian gives the eventual distribution of heat in a room several hours after the heating is turned on.

Differential equations describe a large class of natural phenomena, from the heat equation describing the evolution of heat in (for instance) a metal plate, to the Navier-Stokes equation describing the movement of fluids, including Einstein's equations describing the physical universe in a relativistic way. Although all these equations are boundary value problems, they are further subdivided into categories. This is necessary because each category must be analyzed using different techniques. The present article deals with the category of boundary value problems known as linear elliptic problems.

Boundary value problems and partial differential equations specify relations between two or more quantities. For instance, in the heat equation, the rate of change of temperature at a point is related to the difference of temperature between that point and the nearby points so that, over time, the heat flows from hotter points to cooler points. Boundary value problems can involve space, time and other quantities such as temperature, velocity, pressure, magnetic field, etc.

Some problems do not involve time. For instance, if one hangs a clothesline between the house and a tree, then in the absence of wind, the clothesline will not move and will adopt a gentle hanging curved shape known as the catenary.[1] This curved shape can be computed as the solution of a differential equation relating position, tension, angle and gravity, but since the shape does not change over time, there is no time variable.

Elliptic boundary value problems are a class of problems which do not involve the time variable, and instead only depend on space variables.

## The main example

In two dimensions, let 

x
,
y

{\displaystyle x,y}

 be the coordinates. We will use the subscript notation 

u

x

,

u

x
x

{\displaystyle u_{x},u_{xx}}

 for the first and second partial derivatives of 

u

{\displaystyle u}

 with respect to 

x

{\displaystyle x}

, and a similar notation for 

y

{\displaystyle y}

. We define the gradient 

∇
u
=
(

u

x

,

u

y

)

{\displaystyle \nabla u=(u_{x},u_{y})}

, the Laplace operator 

Δ
u
=

u

x
x

+

u

y
y

{\displaystyle \Delta u=u_{xx}+u_{yy}}

 and the divergence 

∇
⋅
(
u
,
v
)
=

u

x

+

v

y

{\displaystyle \nabla \cdot (u,v)=u_{x}+v_{y}}

. Note from the definitions that 

Δ
u
=
∇
⋅
(
∇
u
)

{\displaystyle \Delta u=\nabla \cdot (\nabla u)}

.

The main example for boundary value problems is the Laplace operator,

where 

Ω

{\displaystyle \Omega }

 is a region in the plane and 

∂
Ω

{\displaystyle \partial \Omega }

 is the boundary of that region. The function 

f

{\displaystyle f}

 is known data and the solution 

u

{\displaystyle u}

 is what must be computed.

The solution 

u

{\displaystyle u}

 can be interpreted as the stationary or limit distribution of heat in a metal plate shaped like 

Ω

{\displaystyle \Omega }

 with its boundary 

∂
Ω

{\displaystyle \partial \Omega }

 kept at zero degrees. The function 

f

{\displaystyle f}

 represents the intensity of heat generation at each point in the plate. After waiting for a long time, the temperature distribution in the metal plate will approach 

u

{\displaystyle u}

.

### Second-order linear problems

In general, a boundary-value problem (BVP) consists of a partial differential equation (PDE) subject to a boundary condition. For now, second-order PDEs subject to a Dirichlet boundary condition will be considered.

Let 

U

{\displaystyle U}

 be an open, bounded subset of 

R

n

{\displaystyle \mathbb {R} ^{n}}

, denote its boundary as 

∂
U

{\displaystyle \partial U}

 and the variables as 

x
=
(

x

1

,
.
.
.
,

x

n

)

{\displaystyle x=(x_{1},...,x_{n})}

. Representing the PDE as a partial differential operator 

L

{\displaystyle L}

 acting on an unknown function 

u
=
u
(
x
)

{\displaystyle u=u(x)}

 of 

x
∈
U

{\displaystyle x\in U}

 results in a BVP of the form

{

L
u

=
f

in 

U

u

=
0

on 

∂
U
,

{\displaystyle \left\{{\begin{aligned}Lu&=f&&{\text{in }}U\\u&=0&&{\text{on }}\partial U,\end{aligned}}\right.}

where 

f
:
U
→

R

{\displaystyle f:U\rightarrow \mathbb {R} }

 is a given function 

f
=
f
(
x
)

{\displaystyle f=f(x)}

 and 

u
:
U
∪
∂
U
→

R

{\displaystyle u:U\cup \partial U\rightarrow \mathbb {R} }

 and the operator 

L

{\displaystyle L}

 is either of the form:

L
u
(
x
)
=
−

∑

i
,
j
=
1

n

(

a

i
j

(
x
)

u

x

i

)

x

j

+

∑

i
=
1

n

b

i

(
x
)

u

x

i

(
x
)
+
c
(
x
)
u
(
x
)
,

{\displaystyle Lu(x)=-\sum _{i,j=1}^{n}(a_{ij}(x)u_{x_{i}})_{x_{j}}+\sum _{i=1}^{n}b_{i}(x)u_{x_{i}}(x)+c(x)u(x),}

or

L
u
(
x
)
=
−

∑

i
,
j
=
1

n

a

i
j

(
x
)

u

x

i

x

j

+

∑

i
=
1

n

b
~

i

(
x
)

u

x

i

(
x
)
+
c
(
x
)
u
(
x
)
,

{\displaystyle Lu(x)=-\sum _{i,j=1}^{n}a_{ij}(x)u_{x_{i}x_{j}}+\sum _{i=1}^{n}{\tilde {b}}_{i}(x)u_{x_{i}}(x)+c(x)u(x),}

for given coefficient functions 

a

i
j

(
x
)
,

b

i

(
x
)
,
c
(
x
)

{\displaystyle a_{ij}(x),b_{i}(x),c(x)}

.

The PDE 

L
u
=
f

{\displaystyle Lu=f}

 is said to be in divergence form in case of the former and in nondivergence form in case of the latter. If the functions 

a

i
j

{\displaystyle a_{ij}}

 are continuously differentiable then both cases are equivalent for

b
~

i

(
x
)
=

b

i

(
x
)
+

∑

j

a

i
j
,

x

j

(
x
)
.

{\displaystyle {\tilde {b}}_{i}(x)=b_{i}(x)+\sum _{j}a_{ij,x_{j}}(x).}

In matrix notation, we can let 

a
(
x
)

{\displaystyle a(x)}

 be an 

n
×
n

{\displaystyle n\times n}

 matrix valued function of 

x

{\displaystyle x}

 and 

b
(
x
)

{\displaystyle b(x)}

 be a 

n

{\displaystyle n}

-dimensional column vector-valued function of 

x

{\displaystyle x}

, and then we may write (the divergence form as)

L
u
=
−
∇
⋅
(
a
∇
u
)
+

b

T

∇
u
+
c
u

{\displaystyle Lu=-\nabla \cdot (a\nabla u)+b^{T}\nabla u+cu}

One may assume, without loss of generality, that the matrix 

a

{\displaystyle a}

 is symmetric (that is, for all 

i
,
j
,
x

{\displaystyle i,j,x}

, 

a

i
j

(
x
)
=

a

j
i

(
x
)

{\displaystyle a_{ij}(x)=a_{ji}(x)}

. We make that assumption in the rest of this article.

We say that the operator 

L

{\displaystyle L}

 is elliptic if, for some constant 

α
>
0

{\displaystyle \alpha >0}

, any of the following equivalent conditions hold:

* λ

min

(
a
(
x
)
)
>
α

∀
x

{\displaystyle \lambda _{\min }(a(x))>\alpha \;\;\;\forall x}

 (see eigenvalue).
* u

T

a
(
x
)
u
>
α

u

T

u

∀
u
∈

R

n

{\displaystyle u^{T}a(x)u>\alpha u^{T}u\;\;\;\forall u\in \mathbb {R} ^{n}}

.
* ∑

i
,
j
=
1

n

a

i
j

u

i

u

j

>
α

∑

i
=
1

n

u

i

2

∀
u
∈

R

n

{\displaystyle \sum _{i,j=1}^{n}a_{ij}u_{i}u_{j}>\alpha \sum _{i=1}^{n}u_{i}^{2}\;\;\;\forall u\in \mathbb {R} ^{n}}

.

If the second-order partial differential operator 

L

{\displaystyle L}

 is elliptic, then the associated BVP is called an elliptic boundary-value problem.

### Boundary conditions

The above BVP is a particular example of a Dirichlet problem. The Neumann problem is

where 

u

ν

{\displaystyle u_{\nu }}

 is the derivative of 

u

{\displaystyle u}

 in the direction of the outwards pointing normal of 

∂
Ω

{\displaystyle \partial \Omega }

. In general, if 

B

{\displaystyle B}

 is any trace operator, one can construct the boundary value problem

In the rest of this article, we assume that 

L

{\displaystyle L}

 is elliptic and that the boundary condition is the Dirichlet condition 

u
=
0

 on 

∂
Ω

{\displaystyle u=0{\text{ on }}\partial \Omega }

.

## Sobolev spaces

The analysis of elliptic boundary value problems requires some fairly sophisticated tools of functional analysis. We require the space 

H

1

(
Ω
)

{\displaystyle H^{1}(\Omega )}

, the Sobolev space of "once-differentiable" functions on 

Ω

{\displaystyle \Omega }

, such that both the function 

u

{\displaystyle u}

 and its partial derivatives 

u

x

i

{\displaystyle u_{x_{i}}}

, 

i
=
1
,
…
,
n

{\displaystyle i=1,\dots ,n}

 are all square integrable. That is:

H

1

(
Ω
)
=

{

u
∈

L

2

(
Ω
)
,

u

x

j

∈

L

2

(
Ω
)
,

1
≤
i
≤
n

}

.

{\displaystyle H^{1}(\Omega )=\left\{u\in L^{2}(\Omega ),\;\;u_{x_{j}}\in L^{2}(\Omega ),\;\;1\leq i\leq n\right\}.}

 
There is a subtlety here in that the partial derivatives must be defined "in the weak sense" (see the article on Sobolev spaces for details.) The space 

H

1

{\displaystyle H^{1}}

 is a Hilbert space, which accounts for much of the ease with which these problems are analyzed.

Unless otherwise noted, all derivatives in this article are to be interpreted in the weak, Sobolev sense. We use the term "strong derivative" to refer to the classical derivative of calculus. We also specify that the spaces 

C

k

{\displaystyle C^{k}}

, 

k
=
0
,
1
,
…

{\displaystyle k=0,1,\dots }

 consist of functions that are 

k

{\displaystyle k}

 times strongly differentiable, and that the 

k

{\displaystyle k}

th derivative is continuous.

## Weak or variational formulation

The first step to cast the boundary value problem as in the language of Sobolev spaces is to rephrase it in its weak form. Consider the Laplace problem 

Δ
u
=
f

{\displaystyle \Delta u=f}

. Multiply each side of the equation by a "test function" 

φ

{\displaystyle \varphi }

 and integrate by parts using Green's theorem to obtain

We will be solving the Dirichlet problem, so that 

u
=
0

 on 

∂
Ω

{\displaystyle u=0{\text{ on }}\partial \Omega }

. For technical reasons, it is useful to assume that 

φ

{\displaystyle \varphi }

 is taken from the same space of functions as 

u

{\displaystyle u}

 is so we also assume that 

φ
=
0

 on 

∂
Ω

{\displaystyle \varphi =0{\text{ on }}\partial \Omega }

. This gets rid of the 

∫

∂
Ω

{\displaystyle \int _{\partial \Omega }}

 term, yielding

where

If 

L

{\displaystyle L}

 is a general elliptic operator, the same reasoning leads to the bilinear form

We do not discuss the Neumann problem but note that it is analyzed in a similar way.

### Continuous and coercive bilinear forms

The map 

A
(
u
,
φ
)

{\displaystyle A(u,\varphi )}

 is defined on the Sobolev space 

H

0

1

⊂

H

1

{\displaystyle H_{0}^{1}\subset H^{1}}

 of functions which are once differentiable and zero on the boundary 

∂
Ω

{\displaystyle \partial \Omega }

, provided we impose some conditions on 

a
,
b
,
c

{\displaystyle a,b,c}

 and 

Ω

{\displaystyle \Omega }

. There are many possible choices, but for the purpose of this article, we will assume that

* a

i
j

(
x
)

{\displaystyle a_{ij}(x)}

 is continuously differentiable on 

Ω
¯

{\displaystyle {\bar {\Omega }}}

 for 

i
,
j
=
1
,
…
,
n
,

{\displaystyle i,j=1,\dots ,n,}
* b

i

(
x
)

{\displaystyle b_{i}(x)}

 is continuous on 

Ω
¯

{\displaystyle {\bar {\Omega }}}

 for 

i
=
1
,
…
,
n
,

{\displaystyle i=1,\dots ,n,}
* c
(
x
)

{\displaystyle c(x)}

 is continuous on 

Ω
¯

{\displaystyle {\bar {\Omega }}}

 and
* Ω

{\displaystyle \Omega }

 is bounded.

The reader may verify that the map 

A
(
u
,
φ
)

{\displaystyle A(u,\varphi )}

 is furthermore bilinear and continuous, and that the map 

F
(
φ
)

{\displaystyle F(\varphi )}

 is linear in 

φ

{\displaystyle \varphi }

, and continuous if (for instance) 

f

{\displaystyle f}

 is square integrable.

We say that the map 

A

{\displaystyle A}

 is coercive if there is an 

α
>
0

{\displaystyle \alpha >0}

 for all 

u
,
φ
∈

H

0

1

(
Ω
)

{\displaystyle u,\varphi \in H_{0}^{1}(\Omega )}

,

This is trivially true for the Laplacian (with 

α
=
1

{\displaystyle \alpha =1}

) and is also true for an elliptic operator if we assume 

b
=
0

{\displaystyle b=0}

 and 

c
≤
0

{\displaystyle c\leq 0}

. (Recall that 

u

T

a
u
>
α

u

T

u

{\displaystyle u^{T}au>\alpha u^{T}u}

 when 

L

{\displaystyle L}

 is elliptic.)

### Existence and uniqueness of the weak solution

One may show, via the Lax–Milgram lemma, that whenever 

A
(
u
,
φ
)

{\displaystyle A(u,\varphi )}

 is coercive and 

F
(
φ
)

{\displaystyle F(\varphi )}

 is continuous, then there exists a unique solution 

u
∈

H

0

1

(
Ω
)

{\displaystyle u\in H_{0}^{1}(\Omega )}

 to the weak problem (*).

If further 

A
(
u
,
φ
)

{\displaystyle A(u,\varphi )}

 is symmetric (i.e., 

b
=
0

{\displaystyle b=0}

), one can show the same result using the Riesz representation theorem instead.

This relies on the fact that 

A
(
u
,
φ
)

{\displaystyle A(u,\varphi )}

 forms an inner product on 

H

0

1

(
Ω
)

{\displaystyle H_{0}^{1}(\Omega )}

, which itself depends on Poincaré's inequality.

## Strong solutions

We have shown that there is a 

u
∈

H

0

1

(
Ω
)

{\displaystyle u\in H_{0}^{1}(\Omega )}

 which solves the weak system, but we do not know if this 

u

{\displaystyle u}

 solves the strong system

Even more vexing is that we are not even sure that 

u

{\displaystyle u}

 is twice differentiable, rendering the expressions 

u

x

i

x

j

{\displaystyle u_{x_{i}x_{j}}}

 in 

L
u

{\displaystyle Lu}

 apparently meaningless. There are many ways to remedy the situation, the main one being regularity.

### Regularity

A regularity theorem for a linear elliptic boundary value problem of the second order takes the form

Theorem If (some condition), then the solution 

u

{\displaystyle u}

 is in 

H

2

(
Ω
)

{\displaystyle H^{2}(\Omega )}

, the space of "twice differentiable" functions whose second derivatives are square integrable.

There is no known simple condition necessary and sufficient for the conclusion of the theorem to hold, but the following conditions are known to be sufficient:

* The boundary of 

Ω

{\displaystyle \Omega }

 is 

C

2

{\displaystyle C^{2}}

, or
* Ω

{\displaystyle \Omega }

 is convex.

It may be tempting to infer that if 

∂
Ω

{\displaystyle \partial \Omega }

 is piecewise 

C

2

{\displaystyle C^{2}}

 then 

u

{\displaystyle u}

 is indeed in 

H

2

{\displaystyle H^{2}}

, but that is unfortunately false.

### Almost everywhere solutions

In the case that 

u
∈

H

2

(
Ω
)

{\displaystyle u\in H^{2}(\Omega )}

 then the second derivatives of 

u

{\displaystyle u}

 are defined almost everywhere, and in that case 

L
u
=
f

{\displaystyle Lu=f}

 almost everywhere.

### Strong solutions

One may further prove that if the boundary of 

Ω
⊂

R

n

{\displaystyle \Omega \subset \mathbb {R} ^{n}}

 is a smooth manifold and 

f

{\displaystyle f}

 is infinitely differentiable in the strong sense, then 

u

{\displaystyle u}

 is also infinitely differentiable in the strong sense. In this case, 

L
u
=
f

{\displaystyle Lu=f}

 with the strong definition of the derivative.

The proof of this relies upon an improved regularity theorem that says that if 

∂
Ω

{\displaystyle \partial \Omega }

 is 

C

k

{\displaystyle C^{k}}

 and 

f
∈

H

k
−
2

(
Ω
)

{\displaystyle f\in H^{k-2}(\Omega )}

, 

k
≥
2

{\displaystyle k\geq 2}

, then 

u
∈

H

k

(
Ω
)

{\displaystyle u\in H^{k}(\Omega )}

, together with a Sobolev imbedding theorem saying that functions in 

H

k

(
Ω
)

{\displaystyle H^{k}(\Omega )}

 are also in 

C

m

(

Ω
¯

)

{\displaystyle C^{m}({\bar {\Omega }})}

 whenever 

0
≤
m
<
k
−
n

/

2

{\displaystyle 0\leq m<k-n/2}

.

## Numerical solutions

While in exceptional circumstances, it is possible to solve elliptic problems explicitly, in general it is an impossible task. The natural solution is to approximate the elliptic problem with a simpler one and to solve this simpler problem on a computer.

Because of the good properties we have enumerated (as well as many we have not), there are extremely efficient numerical solvers for linear elliptic boundary value problems (see finite element method, finite difference method and spectral method for examples.)

## Eigenvalues and eigensolutions

Another Sobolev imbedding theorem states that the inclusion 

H

1

⊂

L

2

{\displaystyle H^{1}\subset L^{2}}

 is a compact linear map. Equipped with the spectral theorem for compact linear operators, one obtains the following result.

Theorem Assume that 

A
(
u
,
φ
)

{\displaystyle A(u,\varphi )}

 is coercive, continuous and symmetric. The map 

S
:
f
→
u

{\displaystyle S:f\rightarrow u}

 from 

L

2

(
Ω
)

{\displaystyle L^{2}(\Omega )}

 to 

L

2

(
Ω
)

{\displaystyle L^{2}(\Omega )}

 is a compact linear map. It has a basis of eigenvectors 

u

1

,

u

2

,
⋯
∈

H

1

(
Ω
)

{\displaystyle u_{1},u_{2},\dots \in H^{1}(\Omega )}

 and matching eigenvalues 

λ

1

,

λ

2

,
⋯
∈

R

{\displaystyle \lambda _{1},\lambda _{2},\dots \in \mathbb {R} }

 such that

* S

u

k

=

λ

k

u

k

,
k
=
1
,
2
,
…
,

{\displaystyle Su_{k}=\lambda _{k}u_{k},k=1,2,\dots ,}
* λ

k

→
0

{\displaystyle \lambda _{k}\rightarrow 0}

 as 

k
→
∞

{\displaystyle k\rightarrow \infty }

,
* λ

k

≩
0

∀
k

{\displaystyle \lambda _{k}\gneqq 0\;\;\forall k}

,
* ∫

Ω

u

j

u

k

=
0

{\displaystyle \int _{\Omega }u_{j}u_{k}=0}

 whenever 

j
≠
k

{\displaystyle j\neq k}

 and
* ∫

Ω

u

j

u

j

=
1

{\displaystyle \int _{\Omega }u_{j}u_{j}=1}

 for all 

j
=
1
,
2
,
…

.

{\displaystyle j=1,2,\dots \,.}

### Series solutions and the importance of eigensolutions

If one has computed the eigenvalues and eigenvectors, then one may find the "explicit" solution of 

L
u
=
f

{\displaystyle Lu=f}

,

via the formula

where

(See Fourier series.)

The series converges in 

L

2

{\displaystyle L^{2}}

. Implemented on a computer using numerical approximations, this is known as the spectral method.

### An example

Consider the problem

The reader may verify that the eigenvectors are exactly

with eigenvalues

The Fourier coefficients of 

g
(
x
)
=
x

{\displaystyle g(x)=x}

 can be looked up in a table, getting 

g
^

(
n
)
=

(
−
1

)

n
+
1

π
n

{\displaystyle {\hat {g}}(n)={(-1)^{n+1} \over \pi n}}

. Therefore,

yielding the solution

## Maximum principle

There are many variants of the maximum principle. We give a simple one.

Theorem. (Weak maximum principle.) Let 

u
∈

C

2

(
Ω
)
∩

C

1

(

Ω
¯

)

{\displaystyle u\in C^{2}(\Omega )\cap C^{1}({\bar {\Omega }})}

, and assume that 

c
(
x
)
=
0

∀
x
∈
Ω

{\displaystyle c(x)=0\;\forall x\in \Omega }

. Say that 

L
u
≤
0

{\displaystyle Lu\leq 0}

 in 

Ω

{\displaystyle \Omega }

. Then 

max

x
∈

Ω
¯

u
(
x
)
=

max

x
∈
∂
Ω

u
(
x
)

{\displaystyle \max _{x\in {\bar {\Omega }}}u(x)=\max _{x\in \partial \Omega }u(x)}

. In other words, the maximum is attained on the boundary.

A strong maximum principle would conclude that 

u
(
x
)
≨

max

y
∈
∂
Ω

u
(
y
)

{\displaystyle u(x)\lneqq \max _{y\in \partial \Omega }u(y)}

 for all 

x
∈
Ω

{\displaystyle x\in \Omega }

 unless 

u

{\displaystyle u}

 is constant.

## Notes

* ^ Swetz, Faauvel, Bekken, "Learn from the Masters", 1997, MAA ISBN 0-88385-703-0, pp. 128–29

## References

* Evans, Lawrence C. (2010). Partial differential equations (PDF). Graduate Studies in Mathematics. Vol. 19 (Second edition of 1998 original ed.). Providence, RI: American Mathematical Society. doi:10.1090/gsm/019. ISBN 978-0-8218-4974-3. MR 2597943.
* Mathematical analysis
* Partial differential equations
* Boundary value problems
* Articles needing additional references from June 2022
* All articles needing additional references
* Pages that use a deprecated format of the math tags