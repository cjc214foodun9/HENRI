# Solving the Poisson equation — FEniCSx tutorial - Jørgen S. Dokken
Source URL: https://jsdokken.com/dolfinx-tutorial/chapter1/fundamentals.html

* Repository
* Suggest edit
* Open issue
* .md
* .pdf

# Solving the Poisson equation

## Contents

# Solving the Poisson equation#

Authors: Hans Petter Langtangen, Anders Logg

Adapted to FEniCSx by Jørgen S. Dokken

The goal of this tutorial is to solve one of the most basic PDEs, the Poisson equation, with a few lines of code in FEniCSx.
We start by introducing some fundamental FEniCSx objects, such as Function,
functionspace, TrialFunction and TestFunction,
and learn how to write a basic PDE solver.
This will include:

```
Function
```

```
functionspace
```

```
TrialFunction
```

```
TestFunction
```

* How to formulate a mathematical variational problem

How to formulate a mathematical variational problem

* How to apply boundary conditions

How to apply boundary conditions

* How to solve the discrete linear system

How to solve the discrete linear system

* How to visualize the solution

How to visualize the solution

The Poisson equation is the following boundary-value problem

Here, \(u=u(\mathbf{x})\) is the unknown function, \(f=f(\mathbf{x})\) a prescribed function, \(\nabla^2\) the Laplace operator
(often written as \(\Delta\)), \(\Omega\) the spatial domain, and \(\partial\Omega\) the boundary of \(\Omega\).
The Poisson problem, including both the PDE, \(-\nabla^2 u = f\), and the boundary condition, \(u=u_D\) on \(\partial\Omega\), is an example of a boundary-value problem, which must be precisely stated before we can start solving it numerically with FEniCSx.

In the two-dimensional space with coordinates \(x\) and \(y\), we can expand the Poisson equation as

The unknown \(u\) is now a function of two variables, \(u=u(x,y)\), defined over the two-dimensional domain \(\Omega\).

The Poisson equation arises in numerous physical contexts, including
heat conduction, electrostatics, diffusion of substances, twisting of
elastic rods, inviscid fluid flow, and water waves. Moreover, the
equation appears in numerical splitting strategies for more complicated
systems of PDEs, in particular the Navier–Stokes equations.

Solving a boundary value problem in FEniCSx consists of the following steps:

* Identify the computational domain \(\Omega\), the PDE, and its corresponding boundary conditions and source terms \(f\).

Identify the computational domain \(\Omega\), the PDE, and its corresponding boundary conditions and source terms \(f\).

* Reformulate the PDE as a finite element variational problem.

Reformulate the PDE as a finite element variational problem.

* Write a Python program defining the computational domain, the boundary conditions, the variational problem, and the source terms, using FEniCSx.

Write a Python program defining the computational domain, the boundary conditions, the variational problem, and the source terms, using FEniCSx.

* Run the Python program to solve the boundary-value problem. Optionally, you can extend the program to derive quantities such as fluxes and averages,
and visualize the results.

Run the Python program to solve the boundary-value problem. Optionally, you can extend the program to derive quantities such as fluxes and averages,
and visualize the results.

As we have already covered step 1, we shall now cover steps 2-4.

## Finite element variational formulation#

FEniCSx is based on the finite element method, which is a general and
efficient mathematical technique for the numerical solution of
PDEs. The starting point for finite element methods is a PDE
expressed in variational form. For readers not familiar with variational problems, we suggest reading a proper treatment on the finite element method, as this tutorial is meant as a brief introduction to the subject. See the original tutorial [LL16] (Chapter 1.6.2).

The basic recipe for turning a PDE into a variational problem is:

* Multiply the PDE by a function \(v\)

Multiply the PDE by a function \(v\)

* Integrate the resulting equation over the domain \(\Omega\)

Integrate the resulting equation over the domain \(\Omega\)

* Perform integration by parts of those terms with second order derivatives

Perform integration by parts of those terms with second order derivatives

The function \(v\) which multiplies the PDE is called a test function. The unknown function \(u\) that is to be approximated is referred to as a trial function.
The terms trial and test functions are used in FEniCSx too. The test and trial functions belong to certain function spaces that specify the properties of the functions.

In the present case, we multiply the Poisson equation by a test function \(v\) and integrate over \(\Omega\):

Here \(\mathrm{d} x\) denotes the differential element for integration over the domain \(\Omega\). We will later let \(\mathrm{d} s\) denote the differential element for integration over \(\partial\Omega\), the boundary of \(\Omega\).

A rule of thumb when deriving variational formulations is that one tries to keep the order of derivatives of \(u\) and \(v\) as small as possible.
Here, we have a second-order differential of \(u\), which can be transformed to a first derivative by employing the technique of
integration by parts.
The formula reads

where \(\dfrac{\partial u}{\partial n}=\nabla u \cdot \vec{n}\) is the derivative of \(u\) in the outward normal direction \(\vec{n}\) on the boundary.

Another feature of variational formulations is that the test function \(v\) is required to vanish on the parts of the boundary where the solution \(u\) is known. See for instance [LM19].

In the present problem, this means that \(v\) is \(0\) on the whole boundary \(\partial\Omega\). Thus, the second term in the integration by parts formula vanishes, and we have that

If we require that this equation holds for all test functions \(v\) in some suitable space \(\hat{V}\), the so-called test space, we obtain a well-defined mathematical problem that uniquely determines the solution \(u\) which lies in some function space \(V\). Note that \(V\) does not have to be the same space as
\(\hat{V}\). We call the space \(V\) the trial space. We refer to the equation above as the weak form/variational form of the original boundary-value problem. We now properly state our variational problem:
Find \(u\in V\) such that

For the present problem, the trial and test spaces \(V\) and \(\hat{V}\) are defined as

In short, \(H^1(\Omega)\) is the Sobolev space containing functions \(v\) such that \(v^2\) and \(\vert \nabla v \vert ^2\) have finite integrals over \(\Omega\). The solution of the underlying
PDE must lie in a function space where the derivatives are
also continuous, but the Sobolev space \(H^1(\Omega)\) allows functions with discontinuous derivatives.
This weaker continuity requirement in our weak formulation (caused by the integration by parts) is of great importance when it comes to constructing the finite element function space. In particular, it allows the use of piecewise polynomial function spaces. This means that the function spaces are constructed
by stitching together polynomial functions on simple domains
such as intervals, triangles, quadrilaterals, tetrahedra and
hexahedra.

The variational problem is a continuous problem: it defines the solution \(u\) in the infinite-dimensional function space \(V\).
The finite element method for the Poisson equation finds an approximate solution of the variational problem by replacing the infinite-dimensional function spaces \(V\) and \(\hat{V}\) by discrete (finite dimensional) trial and test spaces \(V_h\subset V\) and \(\hat{V}_h \subset \hat{V}\). The discrete
variational problem reads: Find \(u_h\in V_h\) such that

This variational problem, together with suitable definitions of \(V_h\) and \(\hat{V}_h\) uniquely define our approximate numerical solution of the Poisson equation.
Note that the boundary condition is encoded as part of the test and trial spaces. This might seem complicated at first glance,
but means that the finite element variational problem and the continuous variational problem look the same.

## Abstract finite element variational formulation#

We will introduce the following notation for variational problems:
Find \(u\in V\) such that

For the Poisson equation, we have:

In the literature \(a(u,v)\) is known as the bilinear form and \(L(v)\) as a linear form.
For every linear problem, we will identify all terms with the unknown \(u\) and collect them in \(a(u,v)\), and collect all terms with only known functions in \(L(v)\).

To solve a linear PDE in FEniCSx, such as the Poisson equation, a user thus needs to perform two steps:

* Choose the finite element spaces \(V\) and \(\hat{V}\) by specifying the domain (the mesh) and the type of function space (polynomial degree and type).

Choose the finite element spaces \(V\) and \(\hat{V}\) by specifying the domain (the mesh) and the type of function space (polynomial degree and type).

* Express the PDE as a (discrete) variational problem: Find \(u\in V\) such that \(a(u,v)=L(v)\) for all \(v \in \hat{V}\).

Express the PDE as a (discrete) variational problem: Find \(u\in V\) such that \(a(u,v)=L(v)\) for all \(v \in \hat{V}\).

## References#

Hans Petter Langtangen and Anders Logg. Solving PDEs in Python: The FEniCS Tutorial I. Springer International Publishing, Cham, 2016. ISBN 978-3-319-52462-7. doi:10.1007/978-3-319-52462-7.

Hans Petter Langtangen and Kent-Andre Mardal. Introduction to Numerical Methods for Variational Problems. Springer International Publishing, Cham, 2019. ISBN 978-3-030-23788-2. doi:10.1007/978-3-030-23788-2_1.