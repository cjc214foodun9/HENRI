# Irreversible thermodynamics and Glansdorff-Prigogine principle derived from stochastic thermodynamics - arXiv
Source URL: https://arxiv.org/html/2501.16059v1

# Irreversible thermodynamics and
Glansdorff-Prigogine principle derived from
stochastic thermodynamics

#### Abstract

We derive the main equations of irreversible thermodynamic
including the expression for the Glansdorff-Prigogine extremal
principle from stochastic thermodynamics. To this end, we analyze
a system that is subject to gradients of temperature and external
forces that induce the appearance of fluxes of several sorts and
the creation of entropy. We show that the rate of entropy
production is a convex function of the fluxes, from which follows
that the excess entropy production is nonnegative, which is an
expression of the Glansdorff-Prigogine principle. We show that
the Lyapunov function associated with the excess entropy production
can be identified with a thermodynamic potential in the special
case where the gradients of temperature are absent.

## I Introduction

Irreversible thermodynamics prigogine1947 ; denbigh1951 ; prigogine1955 ; meixner1959 ; degroot1962 ; fitts1962 ; glansdorff1971 ; nicolis1977 ; kondepudi1998 ; lebon2008 
is a macroscopic theory that deals with systems in states out
of thermodynamic equilibrium. These states are maintained by
gradients of temperature that induce heat and entropy fluxes
and by external forces that cause the appearance of other types
of fluxes.
The change in energy of the system is due to the heat
flux and by the power of external forces.
The change in entropy of the system is not only due
to the entropy fluxes but also due to the creation of entropy
caused by irreversible processes occurring inside the system.

A system out of equilibrium is characterized within irreversible
thermodynamics by variables that include the fluxes of various
sorts and also by those variables that define the state of
thermodynamic equilibrium such as the energy and entropy of
the system. These quantities varies in time and eventually
approach a final value in the stationary state. The time variation
of the energy of the system is equal to its flux into the system
because energy is a conserved quantity.
However, this is not the case of entropy, which may be
created.
The time variation of the entropy of the system is thus equal to
the rate of entropy production minus the entropy flux to
the outside. The fundamental property of the production
of entropy is that it is nonnegative, which is an
expression of the second law of thermodynamics.

In a state of thermodynamic equilibrium, there is no
production of entropy. A system out of thermodynamic
equilibrium is characterized by a continuous production
of entropy. When the system approaches a stationary state
the production of entropy reaches a value which, according
to the extremal principle introduced by Prigogine in 1945
prigogine1945 , is a minimum. He based the principle
on the linear relation between forces and fluxes and on
the Onsager reciprocity relations onsager1931 .

When the condition of linearity between forces and fluxes
are not valid, as happens if the system is not close to
equilibrium, one does not expect the principle to be valid.
This lead Glansdorff and Prigogine to formulate in 1954
glansdorff1954  a more general extremal principle,
expressed in the following terms

 | ∑kδ⁢Xk⁢δ⁢Jk≥0,subscript𝑘𝛿subscript𝑋𝑘𝛿subscript𝐽𝑘0\sum_{k}\delta X_{k}\,\delta J_{k}\geq 0,∑ start_POSTSUBSCRIPT italic_k end_POSTSUBSCRIPT italic_δ italic_X start_POSTSUBSCRIPT italic_k end_POSTSUBSCRIPT italic_δ italic_J start_POSTSUBSCRIPT italic_k end_POSTSUBSCRIPT ≥ 0 , |  | (1)

where δ⁢Jk𝛿subscript𝐽𝑘\delta J_{k}italic_δ italic_J start_POSTSUBSCRIPT italic_k end_POSTSUBSCRIPT and δ⁢Xk𝛿subscript𝑋𝑘\delta X_{k}italic_δ italic_X start_POSTSUBSCRIPT italic_k end_POSTSUBSCRIPT are, respectively,
the deviations of the fluxes Jksubscript𝐽𝑘J_{k}italic_J start_POSTSUBSCRIPT italic_k end_POSTSUBSCRIPT and the forces Xksubscript𝑋𝑘X_{k}italic_X start_POSTSUBSCRIPT italic_k end_POSTSUBSCRIPT
from their values at the stationary state.

We aim here to derive the equations of irreversible
thermodynamics, including the Glandorff-Prigogine principle,
from stochastic thermodynamicstome2006 ; tome2010 ; esposito2010 ; vandenbroeck2010 ; vandenbroeck2013 ; tome2015 .
Our main result concerns the convexity property of the
rate of entropy production. We show that this quantity is
an upward convex function of the collection of fluxes.
This property allows us to defined an excess entropy
production 𝒫excsubscript𝒫exc{\cal P}_{\rm exc}caligraphic_P start_POSTSUBSCRIPT roman_exc end_POSTSUBSCRIPT which is also upward
convex and a minimum at the stationary state, which
is an equivalent statement of the Glansdorff-Prigogine
principle.

The Glansdorff-Prigogine principle has been understood as
a criterion for the stability of the stationary state,
and in this sense it has been regarded as connected to a
Lyapunov function schlogl1971a ; tomita1972 ; desobrino1975 ; schnakenberg1976 ; jiuli1984 ; maes2015 ; ito2022 . This
connection is expressed by the relation

 | d⁢Ld⁢t=𝒫exc,𝑑𝐿𝑑𝑡subscript𝒫exc\frac{dL}{dt}={\cal P}_{\rm exc},divide start_ARG italic_d italic_L end_ARG start_ARG italic_d italic_t end_ARG = caligraphic_P start_POSTSUBSCRIPT roman_exc end_POSTSUBSCRIPT , |  | (2)

where L𝐿Litalic_L is the Lyapunov function understood as a function
of probabilities of the microstates
schlogl1971a ; schnakenberg1976 .
In this sense L𝐿Litalic_L is not in general a function of the
macroscopic thermodynamic variables. However, as we will show here,
there is a special case, namely, when there is no temperature
gradients, that this is possible. In this case the Lyapunov
function is identified as a thermodynamic potential.

In the next section we formulate the irreversible thermodynamics
as a macroscopic theory and present its main results, including
the formulation of the Glansdorff-Prigogine principle in terms
of the excess entropy production. In the subsequent chapter we
demonstrate from stochastic thermodynamics the propositions
that were introduced as assumptions and postulates in the
present formulation of irreversible thermodynamics,
including the convexity of the entropy production.

## II Irreversible thermodynamics

### II.1 Fluxes and forces

A system out of thermodynamic equilibrium that is the object
of study of irreversible thermodynamics is described by the
fluxes of several sorts. In addition to the fluxes the
system is also described by those variables that define
the state of the system when in equilibrium.
These include the entropy S𝑆Sitalic_S, the internal energy U𝑈Uitalic_U,
and a set of complementary macroscopic variables
N1,N2,…,Ncsubscript𝑁1subscript𝑁2…subscript𝑁𝑐N_{1},N_{2},\ldots,N_{c}italic_N start_POSTSUBSCRIPT 1 end_POSTSUBSCRIPT , italic_N start_POSTSUBSCRIPT 2 end_POSTSUBSCRIPT , … , italic_N start_POSTSUBSCRIPT italic_c end_POSTSUBSCRIPT. These variables vary in time as a
consequence of two classes of processes. One class consists
of the internal processes and the other consists of the
processes caused by the external forces, which
we call external processes.

The variation in time of Nlsubscript𝑁𝑙N_{l}italic_N start_POSTSUBSCRIPT italic_l end_POSTSUBSCRIPT has two contributions.
One is the flux χksubscript𝜒𝑘\chi_{k}italic_χ start_POSTSUBSCRIPT italic_k end_POSTSUBSCRIPT from the outside induced by
the external process l𝑙litalic_l and the other is a
term Rlsubscript𝑅𝑙R_{l}italic_R start_POSTSUBSCRIPT italic_l end_POSTSUBSCRIPT describing its creation or annihilation
due to all internal processes. Therefore,

 | d⁢Nld⁢t=χl+Rl.𝑑subscript𝑁𝑙𝑑𝑡subscript𝜒𝑙subscript𝑅𝑙\frac{dN_{l}}{dt}=\chi_{l}+R_{l}.divide start_ARG italic_d italic_N start_POSTSUBSCRIPT italic_l end_POSTSUBSCRIPT end_ARG start_ARG italic_d italic_t end_ARG = italic_χ start_POSTSUBSCRIPT italic_l end_POSTSUBSCRIPT + italic_R start_POSTSUBSCRIPT italic_l end_POSTSUBSCRIPT . |  | (3)

Each external process l𝑙litalic_l ensues a work done on the system
per unit time, which is proportional to χlsubscript𝜒𝑙\chi_{l}italic_χ start_POSTSUBSCRIPT italic_l end_POSTSUBSCRIPT, that is,
equal to μl⁢χlsubscript𝜇𝑙subscript𝜒𝑙\mu_{l}\chi_{l}italic_μ start_POSTSUBSCRIPT italic_l end_POSTSUBSCRIPT italic_χ start_POSTSUBSCRIPT italic_l end_POSTSUBSCRIPT where μlsubscript𝜇𝑙\mu_{l}italic_μ start_POSTSUBSCRIPT italic_l end_POSTSUBSCRIPT is a parameter.
The total work done on the system per unit time by the
external forces is

 | Φw=∑lμl⁢χl,subscriptΦwsubscript𝑙subscript𝜇𝑙subscript𝜒𝑙\Phi_{\rm w}=\sum_{l}\mu_{l}\chi_{l},roman_Φ start_POSTSUBSCRIPT roman_w end_POSTSUBSCRIPT = ∑ start_POSTSUBSCRIPT italic_l end_POSTSUBSCRIPT italic_μ start_POSTSUBSCRIPT italic_l end_POSTSUBSCRIPT italic_χ start_POSTSUBSCRIPT italic_l end_POSTSUBSCRIPT , |  | (4)

where the summation is over the external processes.

The time variation of the internal energy is also due to
the internal and external processes. We denote by ϕksubscriptitalic-ϕ𝑘\phi_{k}italic_ϕ start_POSTSUBSCRIPT italic_k end_POSTSUBSCRIPT
the consumption of energy per unit time due to the process
k𝑘kitalic_k, either internal or external. Therefore, the time
derivative of the energy is given by

 | d⁢Ud⁢t=Φu,𝑑𝑈𝑑𝑡subscriptΦu\frac{dU}{dt}=\Phi_{\rm u},divide start_ARG italic_d italic_U end_ARG start_ARG italic_d italic_t end_ARG = roman_Φ start_POSTSUBSCRIPT roman_u end_POSTSUBSCRIPT , |  | (5)

where

 | Φu=∑kϕk,subscriptΦusubscript𝑘subscriptitalic-ϕ𝑘\Phi_{\rm u}=\sum_{k}\phi_{k},roman_Φ start_POSTSUBSCRIPT roman_u end_POSTSUBSCRIPT = ∑ start_POSTSUBSCRIPT italic_k end_POSTSUBSCRIPT italic_ϕ start_POSTSUBSCRIPT italic_k end_POSTSUBSCRIPT , |  | (6)

and the summation is over all processes, internal or external.

The flux of heat ΦΦ\Phiroman_Φ into the system is equal to the time
variation of the internal energy minus the work done on the system,

 | Φ=Φu−Φw,ΦsubscriptΦusubscriptΦw\Phi=\Phi_{\rm u}-\Phi_{\rm w},roman_Φ = roman_Φ start_POSTSUBSCRIPT roman_u end_POSTSUBSCRIPT - roman_Φ start_POSTSUBSCRIPT roman_w end_POSTSUBSCRIPT , |  | (7)

which is the expression of the first law, or

 | Φ=∑kϕk+∑l(ϕl−μl⁢χl),Φsubscript𝑘subscriptitalic-ϕ𝑘subscript𝑙subscriptitalic-ϕ𝑙subscript𝜇𝑙subscript𝜒𝑙\Phi=\sum_{k}\phi_{k}+\sum_{l}(\phi_{l}-\mu_{l}\chi_{l}),roman_Φ = ∑ start_POSTSUBSCRIPT italic_k end_POSTSUBSCRIPT italic_ϕ start_POSTSUBSCRIPT italic_k end_POSTSUBSCRIPT + ∑ start_POSTSUBSCRIPT italic_l end_POSTSUBSCRIPT ( italic_ϕ start_POSTSUBSCRIPT italic_l end_POSTSUBSCRIPT - italic_μ start_POSTSUBSCRIPT italic_l end_POSTSUBSCRIPT italic_χ start_POSTSUBSCRIPT italic_l end_POSTSUBSCRIPT ) , |  | (8)

where the first and the second summation run over
the internal and the external processes, respectively.

The entropy S𝑆Sitalic_S is not a conserved quantity, but it cannot be
annihilated. Thus its variation in time is equal to the rate
at which it is being created, denoted by 𝒫𝒫{\cal P}caligraphic_P, minus the
flux of entropy to the outside ΨΨ\Psiroman_Ψ, which is expressed by

 | d⁢Sd⁢t=𝒫−Ψ.𝑑𝑆𝑑𝑡𝒫Ψ\frac{dS}{dt}={\cal P}-\Psi.divide start_ARG italic_d italic_S end_ARG start_ARG italic_d italic_t end_ARG = caligraphic_P - roman_Ψ . |  | (9)

The rate of entropy production is nonnegative,

 | 𝒫≥0,𝒫0{\cal P}\geq 0,caligraphic_P ≥ 0 , |  | (10)

which is the expression of the second law and is
a postulate of irreversible thermodynamics.

The expression for the entropy flux ΨΨ\Psiroman_Ψ is set up as follows.
Each term of the first summation in (8) is understood as
a part of the total heat flux ΦΦ\Phiroman_Φ that is being introduced
into the system from a section of the environment that is
understood as a heat bath at a temperature Tksubscript𝑇𝑘T_{k}italic_T start_POSTSUBSCRIPT italic_k end_POSTSUBSCRIPT. This results
in a contribution to the entropy flux to the system which
we assume to be equal to ϕk/Tksubscriptitalic-ϕ𝑘subscript𝑇𝑘\phi_{k}/T_{k}italic_ϕ start_POSTSUBSCRIPT italic_k end_POSTSUBSCRIPT / italic_T start_POSTSUBSCRIPT italic_k end_POSTSUBSCRIPT.
In an analogous manner the contribution
to the entropy flux coming from the terms of the second
summation in (8) is (ϕk−μk⁢χk)/Tksubscriptitalic-ϕ𝑘subscript𝜇𝑘subscript𝜒𝑘subscript𝑇𝑘(\phi_{k}-\mu_{k}\chi_{k})/T_{k}( italic_ϕ start_POSTSUBSCRIPT italic_k end_POSTSUBSCRIPT - italic_μ start_POSTSUBSCRIPT italic_k end_POSTSUBSCRIPT italic_χ start_POSTSUBSCRIPT italic_k end_POSTSUBSCRIPT ) / italic_T start_POSTSUBSCRIPT italic_k end_POSTSUBSCRIPT.
Therefore, the entropy flux is written as

 | Ψ=−∑k1Tk⁢ϕk−∑l1Tl⁢(ϕl−μl⁢χl),Ψsubscript𝑘1subscript𝑇𝑘subscriptitalic-ϕ𝑘subscript𝑙1subscript𝑇𝑙subscriptitalic-ϕ𝑙subscript𝜇𝑙subscript𝜒𝑙\Psi=-\sum_{k}\frac{1}{T_{k}}\phi_{k}-\sum_{l}\frac{1}{T_{l}}(\phi_{l}-\mu_{l}%
\chi_{l}),roman_Ψ = - ∑ start_POSTSUBSCRIPT italic_k end_POSTSUBSCRIPT divide start_ARG 1 end_ARG start_ARG italic_T start_POSTSUBSCRIPT italic_k end_POSTSUBSCRIPT end_ARG italic_ϕ start_POSTSUBSCRIPT italic_k end_POSTSUBSCRIPT - ∑ start_POSTSUBSCRIPT italic_l end_POSTSUBSCRIPT divide start_ARG 1 end_ARG start_ARG italic_T start_POSTSUBSCRIPT italic_l end_POSTSUBSCRIPT end_ARG ( italic_ϕ start_POSTSUBSCRIPT italic_l end_POSTSUBSCRIPT - italic_μ start_POSTSUBSCRIPT italic_l end_POSTSUBSCRIPT italic_χ start_POSTSUBSCRIPT italic_l end_POSTSUBSCRIPT ) , |  | (11)

the minus sign being introduced because ΨΨ\Psiroman_Ψ is the flux
from the system to the outside. The first and
the second summation run over the internal and the external
processes, respectively.
The relation between entropy flux and entropy flux that we
have just assumed is a postulate of the
present formulation of irreversible
thermodynamics that we call Clausius relation.

We write equation (11) as

 | Ψ=−∑k1Tk⁢ϕk+∑lμlTl⁢χl,Ψsubscript𝑘1subscript𝑇𝑘subscriptitalic-ϕ𝑘subscript𝑙subscript𝜇𝑙subscript𝑇𝑙subscript𝜒𝑙\Psi=-\sum_{k}\frac{1}{T_{k}}\phi_{k}+\sum_{l}\frac{\mu_{l}}{T_{l}}\chi_{l},roman_Ψ = - ∑ start_POSTSUBSCRIPT italic_k end_POSTSUBSCRIPT divide start_ARG 1 end_ARG start_ARG italic_T start_POSTSUBSCRIPT italic_k end_POSTSUBSCRIPT end_ARG italic_ϕ start_POSTSUBSCRIPT italic_k end_POSTSUBSCRIPT + ∑ start_POSTSUBSCRIPT italic_l end_POSTSUBSCRIPT divide start_ARG italic_μ start_POSTSUBSCRIPT italic_l end_POSTSUBSCRIPT end_ARG start_ARG italic_T start_POSTSUBSCRIPT italic_l end_POSTSUBSCRIPT end_ARG italic_χ start_POSTSUBSCRIPT italic_l end_POSTSUBSCRIPT , |  | (12)

where now the first summation runs over all processes. This
expression allows us to introduce the following simplification.
We denote by Jksubscript𝐽𝑘J_{k}italic_J start_POSTSUBSCRIPT italic_k end_POSTSUBSCRIPT the negative of the fluxes, that is, Jksubscript𝐽𝑘J_{k}italic_J start_POSTSUBSCRIPT italic_k end_POSTSUBSCRIPT
can be either −ϕksubscriptitalic-ϕ𝑘-\phi_{k}- italic_ϕ start_POSTSUBSCRIPT italic_k end_POSTSUBSCRIPT or −χlsubscript𝜒𝑙-\chi_{l}- italic_χ start_POSTSUBSCRIPT italic_l end_POSTSUBSCRIPT. The quantities that
multiply these quantities in the expression (12) are
the thermodynamic forces, denoted by Fksubscript𝐹𝑘F_{k}italic_F start_POSTSUBSCRIPT italic_k end_POSTSUBSCRIPT. That is,
Fksubscript𝐹𝑘F_{k}italic_F start_POSTSUBSCRIPT italic_k end_POSTSUBSCRIPT can be either 1/Tk1subscript𝑇𝑘1/T_{k}1 / italic_T start_POSTSUBSCRIPT italic_k end_POSTSUBSCRIPT or −μl/Tlsubscript𝜇𝑙subscript𝑇𝑙-\mu_{l}/T_{l}- italic_μ start_POSTSUBSCRIPT italic_l end_POSTSUBSCRIPT / italic_T start_POSTSUBSCRIPT italic_l end_POSTSUBSCRIPT. Using this
notation, we write the entropy flux given by equation
(12) as the bilinear form

 | Ψ=∑kFν⁢Jk.Ψsubscript𝑘subscript𝐹𝜈subscript𝐽𝑘\Psi=\sum_{k}F_{\nu}J_{k}.roman_Ψ = ∑ start_POSTSUBSCRIPT italic_k end_POSTSUBSCRIPT italic_F start_POSTSUBSCRIPT italic_ν end_POSTSUBSCRIPT italic_J start_POSTSUBSCRIPT italic_k end_POSTSUBSCRIPT . |  | (13)

We remark that the thermodynamic forces Fksubscript𝐹𝑘F_{k}italic_F start_POSTSUBSCRIPT italic_k end_POSTSUBSCRIPT are understood
as parameters and should not be confused with the forces Xksubscript𝑋𝑘X_{k}italic_X start_POSTSUBSCRIPT italic_k end_POSTSUBSCRIPT
appearing in the expression (1). These quantities, that
we call conjugate forces, are defined by

 | Xk=∂𝒫∂Jk,subscript𝑋𝑘𝒫subscript𝐽𝑘X_{k}=\frac{\partial{\cal P}}{\partial J_{k}},italic_X start_POSTSUBSCRIPT italic_k end_POSTSUBSCRIPT = divide start_ARG ∂ caligraphic_P end_ARG start_ARG ∂ italic_J start_POSTSUBSCRIPT italic_k end_POSTSUBSCRIPT end_ARG , |  | (14)

where 𝒫𝒫{\cal P}caligraphic_P is understood
as a function of the fluxes Jksubscript𝐽𝑘J_{k}italic_J start_POSTSUBSCRIPT italic_k end_POSTSUBSCRIPT.

### II.2 Glansdorff-Prigogine principle

To reach expression (1) of the Glansdorff-Prigogine
principle, we assume that the production of entropy
𝒫𝒫{\cal P}caligraphic_P is an upward convex function of
the set of fluxes Jksubscript𝐽𝑘J_{k}italic_J start_POSTSUBSCRIPT italic_k end_POSTSUBSCRIPT. This assumption is also a
postulate of the present approach of irreversible
thermodynamics. Defining

 | Ak⁢l=∂Xk∂Jl=∂2𝒫∂Jk⁢∂Jl,subscript𝐴𝑘𝑙subscript𝑋𝑘subscript𝐽𝑙superscript2𝒫subscript𝐽𝑘subscript𝐽𝑙A_{kl}=\frac{\partial X_{k}}{\partial J_{l}}=\frac{\partial^{2}{\cal P}}{%
\partial J_{k}\partial J_{l}},italic_A start_POSTSUBSCRIPT italic_k italic_l end_POSTSUBSCRIPT = divide start_ARG ∂ italic_X start_POSTSUBSCRIPT italic_k end_POSTSUBSCRIPT end_ARG start_ARG ∂ italic_J start_POSTSUBSCRIPT italic_l end_POSTSUBSCRIPT end_ARG = divide start_ARG ∂ start_POSTSUPERSCRIPT 2 end_POSTSUPERSCRIPT caligraphic_P end_ARG start_ARG ∂ italic_J start_POSTSUBSCRIPT italic_k end_POSTSUBSCRIPT ∂ italic_J start_POSTSUBSCRIPT italic_l end_POSTSUBSCRIPT end_ARG , |  | (15)

the convexity of 𝒫𝒫{\cal P}caligraphic_P implies that the matrix
A𝐴Aitalic_A with elements Ak⁢lsubscript𝐴𝑘𝑙A_{kl}italic_A start_POSTSUBSCRIPT italic_k italic_l end_POSTSUBSCRIPT is semi-positive definite,
which is equivalent to say that

 | ∑k⁢lAk⁢l⁢δ⁢Jk⁢δ⁢Jl≥0,subscript𝑘𝑙subscript𝐴𝑘𝑙𝛿subscript𝐽𝑘𝛿subscript𝐽𝑙0\sum_{kl}A_{kl}\,\delta J_{k}\,\delta J_{l}\geq 0,∑ start_POSTSUBSCRIPT italic_k italic_l end_POSTSUBSCRIPT italic_A start_POSTSUBSCRIPT italic_k italic_l end_POSTSUBSCRIPT italic_δ italic_J start_POSTSUBSCRIPT italic_k end_POSTSUBSCRIPT italic_δ italic_J start_POSTSUBSCRIPT italic_l end_POSTSUBSCRIPT ≥ 0 , |  | (16)

where δ⁢Jk𝛿subscript𝐽𝑘\delta J_{k}italic_δ italic_J start_POSTSUBSCRIPT italic_k end_POSTSUBSCRIPT are deviations of the fluxes Jksubscript𝐽𝑘J_{k}italic_J start_POSTSUBSCRIPT italic_k end_POSTSUBSCRIPT.

Taking into account that the variation of Xksubscript𝑋𝑘X_{k}italic_X start_POSTSUBSCRIPT italic_k end_POSTSUBSCRIPT is

 | δ⁢Xk=∑lAk⁢l⁢δ⁢Jl,𝛿subscript𝑋𝑘subscript𝑙subscript𝐴𝑘𝑙𝛿subscript𝐽𝑙\delta X_{k}=\sum_{l}A_{kl}\delta J_{l},italic_δ italic_X start_POSTSUBSCRIPT italic_k end_POSTSUBSCRIPT = ∑ start_POSTSUBSCRIPT italic_l end_POSTSUBSCRIPT italic_A start_POSTSUBSCRIPT italic_k italic_l end_POSTSUBSCRIPT italic_δ italic_J start_POSTSUBSCRIPT italic_l end_POSTSUBSCRIPT , |  | (17)

which follows from (14), we reach the expression

 | ∑kδ⁢Xk⁢δ⁢Jk≥0.subscript𝑘𝛿subscript𝑋𝑘𝛿subscript𝐽𝑘0\sum_{k}\delta X_{k}\delta J_{k}\geq 0.∑ start_POSTSUBSCRIPT italic_k end_POSTSUBSCRIPT italic_δ italic_X start_POSTSUBSCRIPT italic_k end_POSTSUBSCRIPT italic_δ italic_J start_POSTSUBSCRIPT italic_k end_POSTSUBSCRIPT ≥ 0 . |  | (18)

This is the expression (1) of the Glansdorff-Prigogine
principle provided we interpret δ⁢Jμ𝛿subscript𝐽𝜇\delta J_{\mu}italic_δ italic_J start_POSTSUBSCRIPT italic_μ end_POSTSUBSCRIPT as the deviations
of the flux from their stationary values.

The Glansdorff-Prigogine principle can be formulated in an
equivalent manner in terms of the excess entropy production
defined by

 | 𝒫exc=𝒫−∑kXk0⁢(Jk−Jk0)−𝒫0,subscript𝒫exc𝒫subscript𝑘superscriptsubscript𝑋𝑘0subscript𝐽𝑘superscriptsubscript𝐽𝑘0subscript𝒫0{\cal P}_{\rm exc}={\cal P}-\sum_{k}X_{k}^{0}(J_{k}-J_{k}^{0})-{\cal P}_{0},caligraphic_P start_POSTSUBSCRIPT roman_exc end_POSTSUBSCRIPT = caligraphic_P - ∑ start_POSTSUBSCRIPT italic_k end_POSTSUBSCRIPT italic_X start_POSTSUBSCRIPT italic_k end_POSTSUBSCRIPT start_POSTSUPERSCRIPT 0 end_POSTSUPERSCRIPT ( italic_J start_POSTSUBSCRIPT italic_k end_POSTSUBSCRIPT - italic_J start_POSTSUBSCRIPT italic_k end_POSTSUBSCRIPT start_POSTSUPERSCRIPT 0 end_POSTSUPERSCRIPT ) - caligraphic_P start_POSTSUBSCRIPT 0 end_POSTSUBSCRIPT , |  | (19)

where Xk0superscriptsubscript𝑋𝑘0X_{k}^{0}italic_X start_POSTSUBSCRIPT italic_k end_POSTSUBSCRIPT start_POSTSUPERSCRIPT 0 end_POSTSUPERSCRIPT, Jk0superscriptsubscript𝐽𝑘0J_{k}^{0}italic_J start_POSTSUBSCRIPT italic_k end_POSTSUBSCRIPT start_POSTSUPERSCRIPT 0 end_POSTSUPERSCRIPT, and 𝒫0subscript𝒫0{\cal P}_{0}caligraphic_P start_POSTSUBSCRIPT 0 end_POSTSUBSCRIPT, are respectively
the values of the conjugate forces, the fluxes, and the
entropy production at the stationary state. From the
definition (19) it follows that the first order variation
δ⁢𝒫exc𝛿subscript𝒫exc\delta{\cal P}_{\rm exc}italic_δ caligraphic_P start_POSTSUBSCRIPT roman_exc end_POSTSUBSCRIPT from the stationary state
vanishes. Since 𝒫𝒫{\cal P}caligraphic_P is upward convex function so
is 𝒫excsubscript𝒫exc{\cal P}_{\rm exc}caligraphic_P start_POSTSUBSCRIPT roman_exc end_POSTSUBSCRIPT because it differs from 𝒫𝒫{\cal P}caligraphic_P
by linear terms in Jksubscript𝐽𝑘J_{k}italic_J start_POSTSUBSCRIPT italic_k end_POSTSUBSCRIPT, that is,
δ2⁢𝒫exc=δ2⁢𝒫≥0superscript𝛿2subscript𝒫excsuperscript𝛿2𝒫0\delta^{2}{\cal P}_{\rm exc}=\delta^{2}{\cal P}\geq 0italic_δ start_POSTSUPERSCRIPT 2 end_POSTSUPERSCRIPT caligraphic_P start_POSTSUBSCRIPT roman_exc end_POSTSUBSCRIPT = italic_δ start_POSTSUPERSCRIPT 2 end_POSTSUPERSCRIPT caligraphic_P ≥ 0.
Taking into account that δ⁢𝒫exc=0𝛿subscript𝒫exc0\delta{\cal P}_{\rm exc}=0italic_δ caligraphic_P start_POSTSUBSCRIPT roman_exc end_POSTSUBSCRIPT = 0,
then it follows that the excess entropy production is a
minimum at the stationary state, in fact an absolute minimum,
which is an equivalent form of the Glansdorff-Prigogine
principle. We may write

 | 𝒫exc≥0,subscript𝒫exc0{\cal P}_{\rm exc}\geq 0,caligraphic_P start_POSTSUBSCRIPT roman_exc end_POSTSUBSCRIPT ≥ 0 , |  | (20)

because from its definition, the value of the excess entropy
production at the stationary state is zero.

### II.3 Thermodynamic potential

Let us replace the expression for 𝒫𝒫{\cal P}caligraphic_P coming from
(19) and the expression for ΨΨ\Psiroman_Ψ given by (13)
in equation (9). The result is

 | d⁢Sd⁢t+∑k(Fk−Xk0)⁢(Jk−Jk0)=𝒫exc,𝑑𝑆𝑑𝑡subscript𝑘subscript𝐹𝑘superscriptsubscript𝑋𝑘0subscript𝐽𝑘superscriptsubscript𝐽𝑘0subscript𝒫exc\frac{dS}{dt}+\sum_{k}(F_{k}-X_{k}^{0})(J_{k}-J_{k}^{0})={\cal P}_{\rm exc},divide start_ARG italic_d italic_S end_ARG start_ARG italic_d italic_t end_ARG + ∑ start_POSTSUBSCRIPT italic_k end_POSTSUBSCRIPT ( italic_F start_POSTSUBSCRIPT italic_k end_POSTSUBSCRIPT - italic_X start_POSTSUBSCRIPT italic_k end_POSTSUBSCRIPT start_POSTSUPERSCRIPT 0 end_POSTSUPERSCRIPT ) ( italic_J start_POSTSUBSCRIPT italic_k end_POSTSUBSCRIPT - italic_J start_POSTSUBSCRIPT italic_k end_POSTSUBSCRIPT start_POSTSUPERSCRIPT 0 end_POSTSUPERSCRIPT ) = caligraphic_P start_POSTSUBSCRIPT roman_exc end_POSTSUBSCRIPT , |  | (21)

where we have taken into account that

 | 𝒫0=Ψ0=∑kFk⁢Jk0.subscript𝒫0subscriptΨ0subscript𝑘subscript𝐹𝑘superscriptsubscript𝐽𝑘0{\cal P}_{0}=\Psi_{0}=\sum_{k}F_{k}J_{k}^{0}.caligraphic_P start_POSTSUBSCRIPT 0 end_POSTSUBSCRIPT = roman_Ψ start_POSTSUBSCRIPT 0 end_POSTSUBSCRIPT = ∑ start_POSTSUBSCRIPT italic_k end_POSTSUBSCRIPT italic_F start_POSTSUBSCRIPT italic_k end_POSTSUBSCRIPT italic_J start_POSTSUBSCRIPT italic_k end_POSTSUBSCRIPT start_POSTSUPERSCRIPT 0 end_POSTSUPERSCRIPT . |  | (22)

The left-hand side of (21) is not in general the time
derivative of a thermodynamic potential, which here we are
defining as any linear combination of S𝑆Sitalic_S, U𝑈Uitalic_U, and Nlsubscript𝑁𝑙N_{l}italic_N start_POSTSUBSCRIPT italic_l end_POSTSUBSCRIPT.
However, this happens when all temperatures are equal,
as we show next.

When the temperatures Tksubscript𝑇𝑘T_{k}italic_T start_POSTSUBSCRIPT italic_k end_POSTSUBSCRIPT and Tlsubscript𝑇𝑙T_{l}italic_T start_POSTSUBSCRIPT italic_l end_POSTSUBSCRIPT are all the same,
the expression (11) for the entropy flux becomes

 | Ψ=−1T⁢ϕu+1T⁢∑kμk⁢χk,Ψ1𝑇subscriptitalic-ϕu1𝑇subscript𝑘subscript𝜇𝑘subscript𝜒𝑘\Psi=-\frac{1}{T}\phi_{\rm u}+\frac{1}{T}\sum_{k}\mu_{k}\chi_{k},roman_Ψ = - divide start_ARG 1 end_ARG start_ARG italic_T end_ARG italic_ϕ start_POSTSUBSCRIPT roman_u end_POSTSUBSCRIPT + divide start_ARG 1 end_ARG start_ARG italic_T end_ARG ∑ start_POSTSUBSCRIPT italic_k end_POSTSUBSCRIPT italic_μ start_POSTSUBSCRIPT italic_k end_POSTSUBSCRIPT italic_χ start_POSTSUBSCRIPT italic_k end_POSTSUBSCRIPT , |  | (23)

where T𝑇Titalic_T is the common temperature,
and the excess entropy production given by (19) becomes

 | 𝒫exc=𝒫−∑kXk0⁢(χk−χk0)−Y0⁢Φu−𝒫0,subscript𝒫exc𝒫subscript𝑘superscriptsubscript𝑋𝑘0subscript𝜒𝑘superscriptsubscript𝜒𝑘0superscript𝑌0subscriptΦusubscript𝒫0{\cal P}_{\rm exc}={\cal P}-\sum_{k}X_{k}^{0}(\chi_{k}-\chi_{k}^{0})-Y^{0}\Phi%
_{\rm u}-{\cal P}_{0},caligraphic_P start_POSTSUBSCRIPT roman_exc end_POSTSUBSCRIPT = caligraphic_P - ∑ start_POSTSUBSCRIPT italic_k end_POSTSUBSCRIPT italic_X start_POSTSUBSCRIPT italic_k end_POSTSUBSCRIPT start_POSTSUPERSCRIPT 0 end_POSTSUPERSCRIPT ( italic_χ start_POSTSUBSCRIPT italic_k end_POSTSUBSCRIPT - italic_χ start_POSTSUBSCRIPT italic_k end_POSTSUBSCRIPT start_POSTSUPERSCRIPT 0 end_POSTSUPERSCRIPT ) - italic_Y start_POSTSUPERSCRIPT 0 end_POSTSUPERSCRIPT roman_Φ start_POSTSUBSCRIPT roman_u end_POSTSUBSCRIPT - caligraphic_P start_POSTSUBSCRIPT 0 end_POSTSUBSCRIPT , |  | (24)

where Y=−∂𝒫/∂Φu𝑌𝒫subscriptΦuY=-\partial{\cal P}/\partial\Phi_{\rm u}italic_Y = - ∂ caligraphic_P / ∂ roman_Φ start_POSTSUBSCRIPT roman_u end_POSTSUBSCRIPT
and Xk=−∂𝒫/∂χksubscript𝑋𝑘𝒫subscript𝜒𝑘X_{k}=-\partial{\cal P}/\partial\chi_{k}italic_X start_POSTSUBSCRIPT italic_k end_POSTSUBSCRIPT = - ∂ caligraphic_P / ∂ italic_χ start_POSTSUBSCRIPT italic_k end_POSTSUBSCRIPT.
The equation (21) becomes

 | d⁢Sd⁢t+∑k(μkT+Xk0)⁢d⁢Nkd⁢t−(1T−Y0)⁢d⁢Ud⁢t=𝒫exc,𝑑𝑆𝑑𝑡subscript𝑘subscript𝜇𝑘𝑇superscriptsubscript𝑋𝑘0𝑑subscript𝑁𝑘𝑑𝑡1𝑇superscript𝑌0𝑑𝑈𝑑𝑡subscript𝒫exc\frac{dS}{dt}+\sum_{k}(\frac{\mu_{k}}{T}+X_{k}^{0})\frac{dN_{k}}{dt}-(\frac{1}%
{T}-Y^{0})\frac{dU}{dt}={\cal P}_{\rm exc},divide start_ARG italic_d italic_S end_ARG start_ARG italic_d italic_t end_ARG + ∑ start_POSTSUBSCRIPT italic_k end_POSTSUBSCRIPT ( divide start_ARG italic_μ start_POSTSUBSCRIPT italic_k end_POSTSUBSCRIPT end_ARG start_ARG italic_T end_ARG + italic_X start_POSTSUBSCRIPT italic_k end_POSTSUBSCRIPT start_POSTSUPERSCRIPT 0 end_POSTSUPERSCRIPT ) divide start_ARG italic_d italic_N start_POSTSUBSCRIPT italic_k end_POSTSUBSCRIPT end_ARG start_ARG italic_d italic_t end_ARG - ( divide start_ARG 1 end_ARG start_ARG italic_T end_ARG - italic_Y start_POSTSUPERSCRIPT 0 end_POSTSUPERSCRIPT ) divide start_ARG italic_d italic_U end_ARG start_ARG italic_d italic_t end_ARG = caligraphic_P start_POSTSUBSCRIPT roman_exc end_POSTSUBSCRIPT , |  | (25)

and we see that the left-hand side is the time derivative
of the thermodynamic potential

 | M=S+∑k(μkT+Xk0)⁢Nk−(1T−Y0)⁢U,𝑀𝑆subscript𝑘subscript𝜇𝑘𝑇superscriptsubscript𝑋𝑘0subscript𝑁𝑘1𝑇superscript𝑌0𝑈M=S+\sum_{k}(\frac{\mu_{k}}{T}+X_{k}^{0})N_{k}-(\frac{1}{T}-Y^{0})U,italic_M = italic_S + ∑ start_POSTSUBSCRIPT italic_k end_POSTSUBSCRIPT ( divide start_ARG italic_μ start_POSTSUBSCRIPT italic_k end_POSTSUBSCRIPT end_ARG start_ARG italic_T end_ARG + italic_X start_POSTSUBSCRIPT italic_k end_POSTSUBSCRIPT start_POSTSUPERSCRIPT 0 end_POSTSUPERSCRIPT ) italic_N start_POSTSUBSCRIPT italic_k end_POSTSUBSCRIPT - ( divide start_ARG 1 end_ARG start_ARG italic_T end_ARG - italic_Y start_POSTSUPERSCRIPT 0 end_POSTSUPERSCRIPT ) italic_U , |  | (26)

that is,

 | d⁢Md⁢t=𝒫exc.𝑑𝑀𝑑𝑡subscript𝒫exc\frac{dM}{dt}={\cal P}_{\rm exc}.divide start_ARG italic_d italic_M end_ARG start_ARG italic_d italic_t end_ARG = caligraphic_P start_POSTSUBSCRIPT roman_exc end_POSTSUBSCRIPT . |  | (27)

Recalling that 𝒫exc≥0subscript𝒫exc0{\cal P}_{\rm exc}\geq 0caligraphic_P start_POSTSUBSCRIPT roman_exc end_POSTSUBSCRIPT ≥ 0, we find

 | d⁢Md⁢t≥0,𝑑𝑀𝑑𝑡0\frac{dM}{dt}\geq 0,divide start_ARG italic_d italic_M end_ARG start_ARG italic_d italic_t end_ARG ≥ 0 , |  | (28)

and the thermodynamic potential M𝑀Mitalic_M increases with
time towards its value at the stationary state.

## III Stochastic thermodynamics

### III.1 Transition rates

We consider the same system studied in the previous section
but now we use a microscopic description provided by the
stochastic thermodynamics. The evolution of the system
follows a stochastic dynamics in continuous time governed
by a master equation. The probability distribution pisubscript𝑝𝑖p_{i}italic_p start_POSTSUBSCRIPT italic_i end_POSTSUBSCRIPT of
microstates i𝑖iitalic_i evolves in time according to the master equation

 | d⁢pid⁢t=∑j(≠i)(wi⁢j⁢pj−wj⁢i⁢pi),𝑑subscript𝑝𝑖𝑑𝑡subscriptannotated𝑗absent𝑖subscript𝑤𝑖𝑗subscript𝑝𝑗subscript𝑤𝑗𝑖subscript𝑝𝑖\frac{dp_{i}}{dt}=\sum_{j(\neq i)}(w_{ij}p_{j}-w_{ji}p_{i}),divide start_ARG italic_d italic_p start_POSTSUBSCRIPT italic_i end_POSTSUBSCRIPT end_ARG start_ARG italic_d italic_t end_ARG = ∑ start_POSTSUBSCRIPT italic_j ( ≠ italic_i ) end_POSTSUBSCRIPT ( italic_w start_POSTSUBSCRIPT italic_i italic_j end_POSTSUBSCRIPT italic_p start_POSTSUBSCRIPT italic_j end_POSTSUBSCRIPT - italic_w start_POSTSUBSCRIPT italic_j italic_i end_POSTSUBSCRIPT italic_p start_POSTSUBSCRIPT italic_i end_POSTSUBSCRIPT ) , |  | (29)

where wi⁢j≥0subscript𝑤𝑖𝑗0w_{ij}\geq 0italic_w start_POSTSUBSCRIPT italic_i italic_j end_POSTSUBSCRIPT ≥ 0 is the rate of the transition
j→i→𝑗𝑖j\to iitalic_j → italic_i.
Defining wi⁢isubscript𝑤𝑖𝑖w_{ii}italic_w start_POSTSUBSCRIPT italic_i italic_i end_POSTSUBSCRIPT, absent in (29), in such a way that

 | ∑iwi⁢j=0,subscript𝑖subscript𝑤𝑖𝑗0\sum_{i}w_{ij}=0,∑ start_POSTSUBSCRIPT italic_i end_POSTSUBSCRIPT italic_w start_POSTSUBSCRIPT italic_i italic_j end_POSTSUBSCRIPT = 0 , |  | (30)

then the master equation can be written in the form

 | d⁢pid⁢t=∑jwi⁢j⁢pj,𝑑subscript𝑝𝑖𝑑𝑡subscript𝑗subscript𝑤𝑖𝑗subscript𝑝𝑗\frac{dp_{i}}{dt}=\sum_{j}w_{ij}p_{j},divide start_ARG italic_d italic_p start_POSTSUBSCRIPT italic_i end_POSTSUBSCRIPT end_ARG start_ARG italic_d italic_t end_ARG = ∑ start_POSTSUBSCRIPT italic_j end_POSTSUBSCRIPT italic_w start_POSTSUBSCRIPT italic_i italic_j end_POSTSUBSCRIPT italic_p start_POSTSUBSCRIPT italic_j end_POSTSUBSCRIPT , |  | (31)

where wi⁢jsubscript𝑤𝑖𝑗w_{ij}italic_w start_POSTSUBSCRIPT italic_i italic_j end_POSTSUBSCRIPT are understood as the elements of a matrix,
the evolution matrix. From equation (30), it follows
that the diagonal elements of the evolution matrix are
negative or zero, wi⁢i≤0subscript𝑤𝑖𝑖0w_{ii}\leq 0italic_w start_POSTSUBSCRIPT italic_i italic_i end_POSTSUBSCRIPT ≤ 0. We will consider only
transitions that have the reverse. That is, if wi⁢jsubscript𝑤𝑖𝑗w_{ij}italic_w start_POSTSUBSCRIPT italic_i italic_j end_POSTSUBSCRIPT
is nonzero so is wj⁢isubscript𝑤𝑗𝑖w_{ji}italic_w start_POSTSUBSCRIPT italic_j italic_i end_POSTSUBSCRIPT.

We denote by qisubscript𝑞𝑖q_{i}italic_q start_POSTSUBSCRIPT italic_i end_POSTSUBSCRIPT the stationary solution of the master
equation. It fulfills the equation

 | ∑jwi⁢j⁢qj=0.subscript𝑗subscript𝑤𝑖𝑗subscript𝑞𝑗0\sum_{j}w_{ij}q_{j}=0.∑ start_POSTSUBSCRIPT italic_j end_POSTSUBSCRIPT italic_w start_POSTSUBSCRIPT italic_i italic_j end_POSTSUBSCRIPT italic_q start_POSTSUBSCRIPT italic_j end_POSTSUBSCRIPT = 0 . |  | (32)

The transition rates are set up according to several processes
that causes a change in the state of the system. We consider
first the processes associated to the contact of the system
with heat reservoirs at distinct temperatures. If we let Eisubscript𝐸𝑖E_{i}italic_E start_POSTSUBSCRIPT italic_i end_POSTSUBSCRIPT
be a state function representing the energy of the system
then the transition rate associated to the k𝑘kitalic_k reservoir
at a temperature Tksubscript𝑇𝑘T_{k}italic_T start_POSTSUBSCRIPT italic_k end_POSTSUBSCRIPT is

 | ai⁢jk=Ai⁢jk⁢e−βk⁢(Ei−Ej)/2,superscriptsubscript𝑎𝑖𝑗𝑘superscriptsubscript𝐴𝑖𝑗𝑘superscript𝑒subscript𝛽𝑘subscript𝐸𝑖subscript𝐸𝑗2a_{ij}^{k}=A_{ij}^{k}e^{-\beta_{k}(E_{i}-E_{j})/2},italic_a start_POSTSUBSCRIPT italic_i italic_j end_POSTSUBSCRIPT start_POSTSUPERSCRIPT italic_k end_POSTSUPERSCRIPT = italic_A start_POSTSUBSCRIPT italic_i italic_j end_POSTSUBSCRIPT start_POSTSUPERSCRIPT italic_k end_POSTSUPERSCRIPT italic_e start_POSTSUPERSCRIPT - italic_β start_POSTSUBSCRIPT italic_k end_POSTSUBSCRIPT ( italic_E start_POSTSUBSCRIPT italic_i end_POSTSUBSCRIPT - italic_E start_POSTSUBSCRIPT italic_j end_POSTSUBSCRIPT ) / 2 end_POSTSUPERSCRIPT , |  | (33)

where Aj⁢ik=Ai⁢jksuperscriptsubscript𝐴𝑗𝑖𝑘superscriptsubscript𝐴𝑖𝑗𝑘A_{ji}^{k}=A_{ij}^{k}italic_A start_POSTSUBSCRIPT italic_j italic_i end_POSTSUBSCRIPT start_POSTSUPERSCRIPT italic_k end_POSTSUPERSCRIPT = italic_A start_POSTSUBSCRIPT italic_i italic_j end_POSTSUBSCRIPT start_POSTSUPERSCRIPT italic_k end_POSTSUPERSCRIPT and βk=1/κ⁢Tksubscript𝛽𝑘1𝜅subscript𝑇𝑘\beta_{k}=1/\kappa T_{k}italic_β start_POSTSUBSCRIPT italic_k end_POSTSUBSCRIPT = 1 / italic_κ italic_T start_POSTSUBSCRIPT italic_k end_POSTSUBSCRIPT,
and κ𝜅\kappaitalic_κ is the Boltzmann constant.
From this relation it follows that the ratio between
the forward and backward transition rates is

 | aj⁢ikai⁢jk=e−βk⁢(Ej−Ei).superscriptsubscript𝑎𝑗𝑖𝑘superscriptsubscript𝑎𝑖𝑗𝑘superscript𝑒subscript𝛽𝑘subscript𝐸𝑗subscript𝐸𝑖\frac{a_{ji}^{k}}{a_{ij}^{k}}=e^{-\beta_{k}(E_{j}-E_{i})}.divide start_ARG italic_a start_POSTSUBSCRIPT italic_j italic_i end_POSTSUBSCRIPT start_POSTSUPERSCRIPT italic_k end_POSTSUPERSCRIPT end_ARG start_ARG italic_a start_POSTSUBSCRIPT italic_i italic_j end_POSTSUBSCRIPT start_POSTSUPERSCRIPT italic_k end_POSTSUPERSCRIPT end_ARG = italic_e start_POSTSUPERSCRIPT - italic_β start_POSTSUBSCRIPT italic_k end_POSTSUBSCRIPT ( italic_E start_POSTSUBSCRIPT italic_j end_POSTSUBSCRIPT - italic_E start_POSTSUBSCRIPT italic_i end_POSTSUBSCRIPT ) end_POSTSUPERSCRIPT . |  | (34)

We now consider the transitions associated to external
actions done on the system. To this end we suppose that
the system is acted by an external potential Visubscript𝑉𝑖V_{i}italic_V start_POSTSUBSCRIPT italic_i end_POSTSUBSCRIPT
due to external forces μlsubscript𝜇𝑙\mu_{l}italic_μ start_POSTSUBSCRIPT italic_l end_POSTSUBSCRIPT, that is,

 | Vi=−∑lμl⁢Nil,subscript𝑉𝑖subscript𝑙subscript𝜇𝑙superscriptsubscript𝑁𝑖𝑙V_{i}=-\sum_{l}\mu_{l}N_{i}^{l},italic_V start_POSTSUBSCRIPT italic_i end_POSTSUBSCRIPT = - ∑ start_POSTSUBSCRIPT italic_l end_POSTSUBSCRIPT italic_μ start_POSTSUBSCRIPT italic_l end_POSTSUBSCRIPT italic_N start_POSTSUBSCRIPT italic_i end_POSTSUBSCRIPT start_POSTSUPERSCRIPT italic_l end_POSTSUPERSCRIPT , |  | (35)

where Nilsuperscriptsubscript𝑁𝑖𝑙N_{i}^{l}italic_N start_POSTSUBSCRIPT italic_i end_POSTSUBSCRIPT start_POSTSUPERSCRIPT italic_l end_POSTSUPERSCRIPT are some state functions representing the
quantity that changes by the action of the force μlsubscript𝜇𝑙\mu_{l}italic_μ start_POSTSUBSCRIPT italic_l end_POSTSUBSCRIPT.
The transition rate associated to the change of Nilsuperscriptsubscript𝑁𝑖𝑙N_{i}^{l}italic_N start_POSTSUBSCRIPT italic_i end_POSTSUBSCRIPT start_POSTSUPERSCRIPT italic_l end_POSTSUPERSCRIPT is

 | bi⁢jl=Bi⁢jl⁢e−βl⁢(Ei−Ej)/2−βl⁢(Vi−Vj)/2,superscriptsubscript𝑏𝑖𝑗𝑙superscriptsubscript𝐵𝑖𝑗𝑙superscript𝑒subscript𝛽𝑙subscript𝐸𝑖subscript𝐸𝑗2subscript𝛽𝑙subscript𝑉𝑖subscript𝑉𝑗2b_{ij}^{l}=B_{ij}^{l}e^{-\beta_{l}(E_{i}-E_{j})/2-\beta_{l}(V_{i}-V_{j})/2},italic_b start_POSTSUBSCRIPT italic_i italic_j end_POSTSUBSCRIPT start_POSTSUPERSCRIPT italic_l end_POSTSUPERSCRIPT = italic_B start_POSTSUBSCRIPT italic_i italic_j end_POSTSUBSCRIPT start_POSTSUPERSCRIPT italic_l end_POSTSUPERSCRIPT italic_e start_POSTSUPERSCRIPT - italic_β start_POSTSUBSCRIPT italic_l end_POSTSUBSCRIPT ( italic_E start_POSTSUBSCRIPT italic_i end_POSTSUBSCRIPT - italic_E start_POSTSUBSCRIPT italic_j end_POSTSUBSCRIPT ) / 2 - italic_β start_POSTSUBSCRIPT italic_l end_POSTSUBSCRIPT ( italic_V start_POSTSUBSCRIPT italic_i end_POSTSUBSCRIPT - italic_V start_POSTSUBSCRIPT italic_j end_POSTSUBSCRIPT ) / 2 end_POSTSUPERSCRIPT , |  | (36)

where Bi⁢jl=Bj⁢ilsuperscriptsubscript𝐵𝑖𝑗𝑙superscriptsubscript𝐵𝑗𝑖𝑙B_{ij}^{l}=B_{ji}^{l}italic_B start_POSTSUBSCRIPT italic_i italic_j end_POSTSUBSCRIPT start_POSTSUPERSCRIPT italic_l end_POSTSUPERSCRIPT = italic_B start_POSTSUBSCRIPT italic_j italic_i end_POSTSUBSCRIPT start_POSTSUPERSCRIPT italic_l end_POSTSUPERSCRIPT.
We assume that this transition changes only Nilsuperscriptsubscript𝑁𝑖𝑙N_{i}^{l}italic_N start_POSTSUBSCRIPT italic_i end_POSTSUBSCRIPT start_POSTSUPERSCRIPT italic_l end_POSTSUPERSCRIPT whereas
the other variables Nil′superscriptsubscript𝑁𝑖superscript𝑙′N_{i}^{l^{\prime}}italic_N start_POSTSUBSCRIPT italic_i end_POSTSUBSCRIPT start_POSTSUPERSCRIPT italic_l start_POSTSUPERSCRIPT ′ end_POSTSUPERSCRIPT end_POSTSUPERSCRIPT, l′≠lsuperscript𝑙′𝑙l^{\prime}\neq litalic_l start_POSTSUPERSCRIPT ′ end_POSTSUPERSCRIPT ≠ italic_l, remain unchanged.
That is, Bi⁢jlsuperscriptsubscript𝐵𝑖𝑗𝑙B_{ij}^{l}italic_B start_POSTSUBSCRIPT italic_i italic_j end_POSTSUBSCRIPT start_POSTSUPERSCRIPT italic_l end_POSTSUPERSCRIPT vanishes whenever Nil′≠Nil′superscriptsubscript𝑁𝑖superscript𝑙′superscriptsubscript𝑁𝑖superscript𝑙′N_{i}^{l^{\prime}}\neq N_{i}^{l^{\prime}}italic_N start_POSTSUBSCRIPT italic_i end_POSTSUBSCRIPT start_POSTSUPERSCRIPT italic_l start_POSTSUPERSCRIPT ′ end_POSTSUPERSCRIPT end_POSTSUPERSCRIPT ≠ italic_N start_POSTSUBSCRIPT italic_i end_POSTSUBSCRIPT start_POSTSUPERSCRIPT italic_l start_POSTSUPERSCRIPT ′ end_POSTSUPERSCRIPT end_POSTSUPERSCRIPT
for l′≠lsuperscript𝑙′𝑙l^{\prime}\neq litalic_l start_POSTSUPERSCRIPT ′ end_POSTSUPERSCRIPT ≠ italic_l. In view of these restrictions, equation
(36) becomes

 | bi⁢jl=Bi⁢jl⁢e−βl⁢(Ei−Ej)/2+βl⁢μl⁢(Nil−Njl)/2.superscriptsubscript𝑏𝑖𝑗𝑙superscriptsubscript𝐵𝑖𝑗𝑙superscript𝑒subscript𝛽𝑙subscript𝐸𝑖subscript𝐸𝑗2subscript𝛽𝑙subscript𝜇𝑙superscriptsubscript𝑁𝑖𝑙superscriptsubscript𝑁𝑗𝑙2b_{ij}^{l}=B_{ij}^{l}e^{-\beta_{l}(E_{i}-E_{j})/2+\beta_{l}\mu_{l}(N_{i}^{l}-N%
_{j}^{l})/2}.italic_b start_POSTSUBSCRIPT italic_i italic_j end_POSTSUBSCRIPT start_POSTSUPERSCRIPT italic_l end_POSTSUPERSCRIPT = italic_B start_POSTSUBSCRIPT italic_i italic_j end_POSTSUBSCRIPT start_POSTSUPERSCRIPT italic_l end_POSTSUPERSCRIPT italic_e start_POSTSUPERSCRIPT - italic_β start_POSTSUBSCRIPT italic_l end_POSTSUBSCRIPT ( italic_E start_POSTSUBSCRIPT italic_i end_POSTSUBSCRIPT - italic_E start_POSTSUBSCRIPT italic_j end_POSTSUBSCRIPT ) / 2 + italic_β start_POSTSUBSCRIPT italic_l end_POSTSUBSCRIPT italic_μ start_POSTSUBSCRIPT italic_l end_POSTSUBSCRIPT ( italic_N start_POSTSUBSCRIPT italic_i end_POSTSUBSCRIPT start_POSTSUPERSCRIPT italic_l end_POSTSUPERSCRIPT - italic_N start_POSTSUBSCRIPT italic_j end_POSTSUBSCRIPT start_POSTSUPERSCRIPT italic_l end_POSTSUPERSCRIPT ) / 2 end_POSTSUPERSCRIPT . |  | (37)

From this equation, the ratio of
the forward and backward transition rate is

 | bj⁢ilbi⁢jl=e−βl⁢(Ej−Ei)+βl⁢μl⁢(Njl−Nil).superscriptsubscript𝑏𝑗𝑖𝑙superscriptsubscript𝑏𝑖𝑗𝑙superscript𝑒subscript𝛽𝑙subscript𝐸𝑗subscript𝐸𝑖subscript𝛽𝑙subscript𝜇𝑙superscriptsubscript𝑁𝑗𝑙superscriptsubscript𝑁𝑖𝑙\frac{b_{ji}^{l}}{b_{ij}^{l}}=e^{-\beta_{l}(E_{j}-E_{i})+\beta_{l}\mu_{l}(N_{j%
}^{l}-N_{i}^{l})}.divide start_ARG italic_b start_POSTSUBSCRIPT italic_j italic_i end_POSTSUBSCRIPT start_POSTSUPERSCRIPT italic_l end_POSTSUPERSCRIPT end_ARG start_ARG italic_b start_POSTSUBSCRIPT italic_i italic_j end_POSTSUBSCRIPT start_POSTSUPERSCRIPT italic_l end_POSTSUPERSCRIPT end_ARG = italic_e start_POSTSUPERSCRIPT - italic_β start_POSTSUBSCRIPT italic_l end_POSTSUBSCRIPT ( italic_E start_POSTSUBSCRIPT italic_j end_POSTSUBSCRIPT - italic_E start_POSTSUBSCRIPT italic_i end_POSTSUBSCRIPT ) + italic_β start_POSTSUBSCRIPT italic_l end_POSTSUBSCRIPT italic_μ start_POSTSUBSCRIPT italic_l end_POSTSUBSCRIPT ( italic_N start_POSTSUBSCRIPT italic_j end_POSTSUBSCRIPT start_POSTSUPERSCRIPT italic_l end_POSTSUPERSCRIPT - italic_N start_POSTSUBSCRIPT italic_i end_POSTSUBSCRIPT start_POSTSUPERSCRIPT italic_l end_POSTSUPERSCRIPT ) end_POSTSUPERSCRIPT . |  | (38)

The transition rate wi⁢jsubscript𝑤𝑖𝑗w_{ij}italic_w start_POSTSUBSCRIPT italic_i italic_j end_POSTSUBSCRIPT is the sum of the transition rates
just introduced,

 | wi⁢j=∑kai⁢jk+∑lbi⁢jlsubscript𝑤𝑖𝑗subscript𝑘superscriptsubscript𝑎𝑖𝑗𝑘subscript𝑙superscriptsubscript𝑏𝑖𝑗𝑙w_{ij}=\sum_{k}a_{ij}^{k}+\sum_{l}b_{ij}^{l}italic_w start_POSTSUBSCRIPT italic_i italic_j end_POSTSUBSCRIPT = ∑ start_POSTSUBSCRIPT italic_k end_POSTSUBSCRIPT italic_a start_POSTSUBSCRIPT italic_i italic_j end_POSTSUBSCRIPT start_POSTSUPERSCRIPT italic_k end_POSTSUPERSCRIPT + ∑ start_POSTSUBSCRIPT italic_l end_POSTSUBSCRIPT italic_b start_POSTSUBSCRIPT italic_i italic_j end_POSTSUBSCRIPT start_POSTSUPERSCRIPT italic_l end_POSTSUPERSCRIPT |  | (39)

and we point out that, given a transition j→i→𝑗𝑖j\to iitalic_j → italic_i, then just
one term on the right-hand side of (39) can be nonzero.
This assumption is accomplished by the partition of the whole
set of possible transitions j→i→𝑗𝑖j\to iitalic_j → italic_i in several disjoint subsets,
each one associated to a given process. In other words,
given a transition j→i→𝑗𝑖j\to iitalic_j → italic_i it is carried out by only one
of the processes.

### III.2 Heat flux

Let us determine the time derivative of the average
U=⟨Ei⟩𝑈delimited-⟨⟩subscript𝐸𝑖U=\langle E_{i}\rangleitalic_U = ⟨ italic_E start_POSTSUBSCRIPT italic_i end_POSTSUBSCRIPT ⟩. From the master equation, and using
(39), we find

 | d⁢Ud⁢t=Φu=∑kϕk+∑lϕl,𝑑𝑈𝑑𝑡subscriptΦusubscript𝑘subscriptitalic-ϕ𝑘subscript𝑙subscriptitalic-ϕ𝑙\frac{dU}{dt}=\Phi_{\rm u}=\sum_{k}\phi_{k}+\sum_{l}\phi_{l},divide start_ARG italic_d italic_U end_ARG start_ARG italic_d italic_t end_ARG = roman_Φ start_POSTSUBSCRIPT roman_u end_POSTSUBSCRIPT = ∑ start_POSTSUBSCRIPT italic_k end_POSTSUBSCRIPT italic_ϕ start_POSTSUBSCRIPT italic_k end_POSTSUBSCRIPT + ∑ start_POSTSUBSCRIPT italic_l end_POSTSUBSCRIPT italic_ϕ start_POSTSUBSCRIPT italic_l end_POSTSUBSCRIPT , |  | (40)

where

 | ϕk=∑i⁢j(Ei−Ej)⁢ai⁢jk⁢pj,subscriptitalic-ϕ𝑘subscript𝑖𝑗subscript𝐸𝑖subscript𝐸𝑗superscriptsubscript𝑎𝑖𝑗𝑘subscript𝑝𝑗\phi_{k}=\sum_{ij}(E_{i}-E_{j})a_{ij}^{k}p_{j},italic_ϕ start_POSTSUBSCRIPT italic_k end_POSTSUBSCRIPT = ∑ start_POSTSUBSCRIPT italic_i italic_j end_POSTSUBSCRIPT ( italic_E start_POSTSUBSCRIPT italic_i end_POSTSUBSCRIPT - italic_E start_POSTSUBSCRIPT italic_j end_POSTSUBSCRIPT ) italic_a start_POSTSUBSCRIPT italic_i italic_j end_POSTSUBSCRIPT start_POSTSUPERSCRIPT italic_k end_POSTSUPERSCRIPT italic_p start_POSTSUBSCRIPT italic_j end_POSTSUBSCRIPT , |  | (41)

and

 | ϕl=∑i⁢j(Ei−Ej)⁢bi⁢jl⁢pj.subscriptitalic-ϕ𝑙subscript𝑖𝑗subscript𝐸𝑖subscript𝐸𝑗superscriptsubscript𝑏𝑖𝑗𝑙subscript𝑝𝑗\phi_{l}=\sum_{ij}(E_{i}-E_{j})b_{ij}^{l}p_{j}.italic_ϕ start_POSTSUBSCRIPT italic_l end_POSTSUBSCRIPT = ∑ start_POSTSUBSCRIPT italic_i italic_j end_POSTSUBSCRIPT ( italic_E start_POSTSUBSCRIPT italic_i end_POSTSUBSCRIPT - italic_E start_POSTSUBSCRIPT italic_j end_POSTSUBSCRIPT ) italic_b start_POSTSUBSCRIPT italic_i italic_j end_POSTSUBSCRIPT start_POSTSUPERSCRIPT italic_l end_POSTSUPERSCRIPT italic_p start_POSTSUBSCRIPT italic_j end_POSTSUBSCRIPT . |  | (42)

Equation (40) is identified with equations
(5) and (6).

Let us now determine the time derivative of the average
Nl=⟨Nil⟩subscript𝑁𝑙delimited-⟨⟩superscriptsubscript𝑁𝑖𝑙N_{l}=\langle N_{i}^{l}\rangleitalic_N start_POSTSUBSCRIPT italic_l end_POSTSUBSCRIPT = ⟨ italic_N start_POSTSUBSCRIPT italic_i end_POSTSUBSCRIPT start_POSTSUPERSCRIPT italic_l end_POSTSUPERSCRIPT ⟩. From the master equation, and using
(39), we find

 | d⁢Nld⁢t=χl+Rl,𝑑subscript𝑁𝑙𝑑𝑡subscript𝜒𝑙subscript𝑅𝑙\frac{dN_{l}}{dt}=\chi_{l}+R_{l},divide start_ARG italic_d italic_N start_POSTSUBSCRIPT italic_l end_POSTSUBSCRIPT end_ARG start_ARG italic_d italic_t end_ARG = italic_χ start_POSTSUBSCRIPT italic_l end_POSTSUBSCRIPT + italic_R start_POSTSUBSCRIPT italic_l end_POSTSUBSCRIPT , |  | (43)

where

 | χl=∑i⁢j(Nil−Njl)⁢bi⁢jl⁢pjsubscript𝜒𝑙subscript𝑖𝑗superscriptsubscript𝑁𝑖𝑙superscriptsubscript𝑁𝑗𝑙superscriptsubscript𝑏𝑖𝑗𝑙subscript𝑝𝑗\chi_{l}=\sum_{ij}(N_{i}^{l}-N_{j}^{l})b_{ij}^{l}p_{j}italic_χ start_POSTSUBSCRIPT italic_l end_POSTSUBSCRIPT = ∑ start_POSTSUBSCRIPT italic_i italic_j end_POSTSUBSCRIPT ( italic_N start_POSTSUBSCRIPT italic_i end_POSTSUBSCRIPT start_POSTSUPERSCRIPT italic_l end_POSTSUPERSCRIPT - italic_N start_POSTSUBSCRIPT italic_j end_POSTSUBSCRIPT start_POSTSUPERSCRIPT italic_l end_POSTSUPERSCRIPT ) italic_b start_POSTSUBSCRIPT italic_i italic_j end_POSTSUBSCRIPT start_POSTSUPERSCRIPT italic_l end_POSTSUPERSCRIPT italic_p start_POSTSUBSCRIPT italic_j end_POSTSUBSCRIPT |  | (44)

is understood as the flux of Nlsubscript𝑁𝑙N_{l}italic_N start_POSTSUBSCRIPT italic_l end_POSTSUBSCRIPT into the system, and

 | Rl=∑k∑i⁢j(Nil−Njl)⁢ai⁢jk⁢pjsubscript𝑅𝑙subscript𝑘subscript𝑖𝑗superscriptsubscript𝑁𝑖𝑙superscriptsubscript𝑁𝑗𝑙superscriptsubscript𝑎𝑖𝑗𝑘subscript𝑝𝑗R_{l}=\sum_{k}\sum_{ij}(N_{i}^{l}-N_{j}^{l})a_{ij}^{k}p_{j}italic_R start_POSTSUBSCRIPT italic_l end_POSTSUBSCRIPT = ∑ start_POSTSUBSCRIPT italic_k end_POSTSUBSCRIPT ∑ start_POSTSUBSCRIPT italic_i italic_j end_POSTSUBSCRIPT ( italic_N start_POSTSUBSCRIPT italic_i end_POSTSUBSCRIPT start_POSTSUPERSCRIPT italic_l end_POSTSUPERSCRIPT - italic_N start_POSTSUBSCRIPT italic_j end_POSTSUBSCRIPT start_POSTSUPERSCRIPT italic_l end_POSTSUPERSCRIPT ) italic_a start_POSTSUBSCRIPT italic_i italic_j end_POSTSUBSCRIPT start_POSTSUPERSCRIPT italic_k end_POSTSUPERSCRIPT italic_p start_POSTSUBSCRIPT italic_j end_POSTSUBSCRIPT |  | (45)

is understood as the creation or annihilation of Nlsubscript𝑁𝑙N_{l}italic_N start_POSTSUBSCRIPT italic_l end_POSTSUBSCRIPT
by the internal processes represented by the rates
ai⁢jksuperscriptsubscript𝑎𝑖𝑗𝑘a_{ij}^{k}italic_a start_POSTSUBSCRIPT italic_i italic_j end_POSTSUBSCRIPT start_POSTSUPERSCRIPT italic_k end_POSTSUPERSCRIPT. This last formula was obtained using the condition
stated just below equation (36). Equation (43)
is identified with equation (3).

The total work done on the system per unit time is

 | Φw=∑lμl⁢χl,subscriptΦwsubscript𝑙subscript𝜇𝑙subscript𝜒𝑙\Phi_{\rm w}=\sum_{l}\mu_{l}\chi_{l},roman_Φ start_POSTSUBSCRIPT roman_w end_POSTSUBSCRIPT = ∑ start_POSTSUBSCRIPT italic_l end_POSTSUBSCRIPT italic_μ start_POSTSUBSCRIPT italic_l end_POSTSUBSCRIPT italic_χ start_POSTSUBSCRIPT italic_l end_POSTSUBSCRIPT , |  | (46)

which is identified as equation (4), and the total
heat flux Φ=Φu−ΦwΦsubscriptΦusubscriptΦw\Phi=\Phi_{\rm u}-\Phi_{\rm w}roman_Φ = roman_Φ start_POSTSUBSCRIPT roman_u end_POSTSUBSCRIPT - roman_Φ start_POSTSUBSCRIPT roman_w end_POSTSUBSCRIPT is

 | Φ=∑kϕk+∑l(ϕl−μl⁢χl),Φsubscript𝑘subscriptitalic-ϕ𝑘subscript𝑙subscriptitalic-ϕ𝑙subscript𝜇𝑙subscript𝜒𝑙\Phi=\sum_{k}\phi_{k}+\sum_{l}(\phi_{l}-\mu_{l}\chi_{l}),roman_Φ = ∑ start_POSTSUBSCRIPT italic_k end_POSTSUBSCRIPT italic_ϕ start_POSTSUBSCRIPT italic_k end_POSTSUBSCRIPT + ∑ start_POSTSUBSCRIPT italic_l end_POSTSUBSCRIPT ( italic_ϕ start_POSTSUBSCRIPT italic_l end_POSTSUBSCRIPT - italic_μ start_POSTSUBSCRIPT italic_l end_POSTSUBSCRIPT italic_χ start_POSTSUBSCRIPT italic_l end_POSTSUBSCRIPT ) , |  | (47)

which is identified with equation (8).

### III.3 Entropy production and entropy flux

The entropy of the system is assumed to be given by
the Gibbs formula,

 | S=−κ⁢∑ipi⁢ln⁡pi.𝑆𝜅subscript𝑖subscript𝑝𝑖subscript𝑝𝑖S=-\kappa\sum_{i}p_{i}\ln p_{i}.italic_S = - italic_κ ∑ start_POSTSUBSCRIPT italic_i end_POSTSUBSCRIPT italic_p start_POSTSUBSCRIPT italic_i end_POSTSUBSCRIPT roman_ln italic_p start_POSTSUBSCRIPT italic_i end_POSTSUBSCRIPT . |  | (48)

Its time derivative is

 | d⁢Sd⁢t=−κ⁢∑i⁢jwi⁢j⁢pj⁢ln⁡pi.𝑑𝑆𝑑𝑡𝜅subscript𝑖𝑗subscript𝑤𝑖𝑗subscript𝑝𝑗subscript𝑝𝑖\frac{dS}{dt}=-\kappa\sum_{ij}w_{ij}p_{j}\ln p_{i}.divide start_ARG italic_d italic_S end_ARG start_ARG italic_d italic_t end_ARG = - italic_κ ∑ start_POSTSUBSCRIPT italic_i italic_j end_POSTSUBSCRIPT italic_w start_POSTSUBSCRIPT italic_i italic_j end_POSTSUBSCRIPT italic_p start_POSTSUBSCRIPT italic_j end_POSTSUBSCRIPT roman_ln italic_p start_POSTSUBSCRIPT italic_i end_POSTSUBSCRIPT . |  | (49)

Using property (30), it can be written in the
equivalent form,

 | d⁢Sd⁢t=κ⁢∑i⁢jwi⁢j⁢pj⁢ln⁡pjpi.𝑑𝑆𝑑𝑡𝜅subscript𝑖𝑗subscript𝑤𝑖𝑗subscript𝑝𝑗subscript𝑝𝑗subscript𝑝𝑖\frac{dS}{dt}=\kappa\sum_{ij}w_{ij}p_{j}\ln\frac{p_{j}}{p_{i}}.divide start_ARG italic_d italic_S end_ARG start_ARG italic_d italic_t end_ARG = italic_κ ∑ start_POSTSUBSCRIPT italic_i italic_j end_POSTSUBSCRIPT italic_w start_POSTSUBSCRIPT italic_i italic_j end_POSTSUBSCRIPT italic_p start_POSTSUBSCRIPT italic_j end_POSTSUBSCRIPT roman_ln divide start_ARG italic_p start_POSTSUBSCRIPT italic_j end_POSTSUBSCRIPT end_ARG start_ARG italic_p start_POSTSUBSCRIPT italic_i end_POSTSUBSCRIPT end_ARG . |  | (50)

The variation of the entropy with time is
split into two parts,

 | d⁢Sd⁢t=𝒫−Ψ,𝑑𝑆𝑑𝑡𝒫Ψ\frac{dS}{dt}={\cal P}-\Psi,divide start_ARG italic_d italic_S end_ARG start_ARG italic_d italic_t end_ARG = caligraphic_P - roman_Ψ , |  | (51)

where 𝒫𝒫{\cal P}caligraphic_P is the entropy production rate and ΨΨ\Psiroman_Ψ
is the entropy flux from the system to the outside. The
entropy production rate is postulated to be given by
the Schnakenberg formula schnakenberg1976

 | 𝒫=κ2⁢∑i⁢j(wi⁢j⁢pj−wj⁢i⁢pi)⁢ln⁡wi⁢j⁢pjwj⁢i⁢pi.𝒫𝜅2subscript𝑖𝑗subscript𝑤𝑖𝑗subscript𝑝𝑗subscript𝑤𝑗𝑖subscript𝑝𝑖subscript𝑤𝑖𝑗subscript𝑝𝑗subscript𝑤𝑗𝑖subscript𝑝𝑖{\cal P}=\frac{\kappa}{2}\sum_{ij}(w_{ij}p_{j}-w_{ji}p_{i})\ln\frac{w_{ij}p_{j%
}}{w_{ji}p_{i}}.caligraphic_P = divide start_ARG italic_κ end_ARG start_ARG 2 end_ARG ∑ start_POSTSUBSCRIPT italic_i italic_j end_POSTSUBSCRIPT ( italic_w start_POSTSUBSCRIPT italic_i italic_j end_POSTSUBSCRIPT italic_p start_POSTSUBSCRIPT italic_j end_POSTSUBSCRIPT - italic_w start_POSTSUBSCRIPT italic_j italic_i end_POSTSUBSCRIPT italic_p start_POSTSUBSCRIPT italic_i end_POSTSUBSCRIPT ) roman_ln divide start_ARG italic_w start_POSTSUBSCRIPT italic_i italic_j end_POSTSUBSCRIPT italic_p start_POSTSUBSCRIPT italic_j end_POSTSUBSCRIPT end_ARG start_ARG italic_w start_POSTSUBSCRIPT italic_j italic_i end_POSTSUBSCRIPT italic_p start_POSTSUBSCRIPT italic_i end_POSTSUBSCRIPT end_ARG . |  | (52)

We point out that each term of the summation in
(52) is nonnegative because it is of the type
(x−y)⁢ln⁡x/y≥0𝑥𝑦𝑥𝑦0(x-y)\ln x/y\geq 0( italic_x - italic_y ) roman_ln italic_x / italic_y ≥ 0. Therefore, 𝒫≥0𝒫0{\cal P}\geq 0caligraphic_P ≥ 0,
which justify the postulate of irreversible
thermodynamics given by equation (10).

The production of entropy can also be expressed in the form

 | 𝒫=κ⁢∑i⁢jwi⁢j⁢pj⁢ln⁡wi⁢j⁢pjwj⁢i⁢pi.𝒫𝜅subscript𝑖𝑗subscript𝑤𝑖𝑗subscript𝑝𝑗subscript𝑤𝑖𝑗subscript𝑝𝑗subscript𝑤𝑗𝑖subscript𝑝𝑖{\cal P}=\kappa\sum_{ij}w_{ij}p_{j}\ln\frac{w_{ij}p_{j}}{w_{ji}p_{i}}.caligraphic_P = italic_κ ∑ start_POSTSUBSCRIPT italic_i italic_j end_POSTSUBSCRIPT italic_w start_POSTSUBSCRIPT italic_i italic_j end_POSTSUBSCRIPT italic_p start_POSTSUBSCRIPT italic_j end_POSTSUBSCRIPT roman_ln divide start_ARG italic_w start_POSTSUBSCRIPT italic_i italic_j end_POSTSUBSCRIPT italic_p start_POSTSUBSCRIPT italic_j end_POSTSUBSCRIPT end_ARG start_ARG italic_w start_POSTSUBSCRIPT italic_j italic_i end_POSTSUBSCRIPT italic_p start_POSTSUBSCRIPT italic_i end_POSTSUBSCRIPT end_ARG . |  | (53)

From this expression of 𝒫𝒫\cal Pcaligraphic_P and from (50), we
obtain the expression for the entropy flux, which is

 | Ψ=κ⁢∑i⁢jwi⁢j⁢pj⁢ln⁡wi⁢jwj⁢i,Ψ𝜅subscript𝑖𝑗subscript𝑤𝑖𝑗subscript𝑝𝑗subscript𝑤𝑖𝑗subscript𝑤𝑗𝑖\Psi=\kappa\sum_{ij}w_{ij}p_{j}\ln\frac{w_{ij}}{w_{ji}},roman_Ψ = italic_κ ∑ start_POSTSUBSCRIPT italic_i italic_j end_POSTSUBSCRIPT italic_w start_POSTSUBSCRIPT italic_i italic_j end_POSTSUBSCRIPT italic_p start_POSTSUBSCRIPT italic_j end_POSTSUBSCRIPT roman_ln divide start_ARG italic_w start_POSTSUBSCRIPT italic_i italic_j end_POSTSUBSCRIPT end_ARG start_ARG italic_w start_POSTSUBSCRIPT italic_j italic_i end_POSTSUBSCRIPT end_ARG , |  | (54)

and we see that it holds the important property of being
linear in pisubscript𝑝𝑖p_{i}italic_p start_POSTSUBSCRIPT italic_i end_POSTSUBSCRIPT.

Replacing (39) in equation (54) we may write the
entropy flux as

 | Ψ=κ⁢∑k∑i⁢jai⁢jk⁢pj⁢ln⁡ai⁢jkaj⁢ik+κ⁢∑l∑i⁢jbi⁢jl⁢pj⁢ln⁡bi⁢jlbj⁢il.Ψ𝜅subscript𝑘subscript𝑖𝑗superscriptsubscript𝑎𝑖𝑗𝑘subscript𝑝𝑗superscriptsubscript𝑎𝑖𝑗𝑘superscriptsubscript𝑎𝑗𝑖𝑘𝜅subscript𝑙subscript𝑖𝑗superscriptsubscript𝑏𝑖𝑗𝑙subscript𝑝𝑗superscriptsubscript𝑏𝑖𝑗𝑙superscriptsubscript𝑏𝑗𝑖𝑙\Psi=\kappa\sum_{k}\sum_{ij}a_{ij}^{k}p_{j}\ln\frac{a_{ij}^{k}}{a_{ji}^{k}}+%
\kappa\sum_{l}\sum_{ij}b_{ij}^{l}p_{j}\ln\frac{b_{ij}^{l}}{b_{ji}^{l}}.roman_Ψ = italic_κ ∑ start_POSTSUBSCRIPT italic_k end_POSTSUBSCRIPT ∑ start_POSTSUBSCRIPT italic_i italic_j end_POSTSUBSCRIPT italic_a start_POSTSUBSCRIPT italic_i italic_j end_POSTSUBSCRIPT start_POSTSUPERSCRIPT italic_k end_POSTSUPERSCRIPT italic_p start_POSTSUBSCRIPT italic_j end_POSTSUBSCRIPT roman_ln divide start_ARG italic_a start_POSTSUBSCRIPT italic_i italic_j end_POSTSUBSCRIPT start_POSTSUPERSCRIPT italic_k end_POSTSUPERSCRIPT end_ARG start_ARG italic_a start_POSTSUBSCRIPT italic_j italic_i end_POSTSUBSCRIPT start_POSTSUPERSCRIPT italic_k end_POSTSUPERSCRIPT end_ARG + italic_κ ∑ start_POSTSUBSCRIPT italic_l end_POSTSUBSCRIPT ∑ start_POSTSUBSCRIPT italic_i italic_j end_POSTSUBSCRIPT italic_b start_POSTSUBSCRIPT italic_i italic_j end_POSTSUBSCRIPT start_POSTSUPERSCRIPT italic_l end_POSTSUPERSCRIPT italic_p start_POSTSUBSCRIPT italic_j end_POSTSUBSCRIPT roman_ln divide start_ARG italic_b start_POSTSUBSCRIPT italic_i italic_j end_POSTSUBSCRIPT start_POSTSUPERSCRIPT italic_l end_POSTSUPERSCRIPT end_ARG start_ARG italic_b start_POSTSUBSCRIPT italic_j italic_i end_POSTSUBSCRIPT start_POSTSUPERSCRIPT italic_l end_POSTSUPERSCRIPT end_ARG . |  | (55)

Using (34) and (38), we find

 | Ψ=−∑k∑i⁢jai⁢jkTk⁢(Ei−Ej)⁢pjΨsubscript𝑘subscript𝑖𝑗superscriptsubscript𝑎𝑖𝑗𝑘subscript𝑇𝑘subscript𝐸𝑖subscript𝐸𝑗subscript𝑝𝑗\Psi=-\sum_{k}\sum_{ij}\frac{a_{ij}^{k}}{T_{k}}(E_{i}-E_{j})p_{j}roman_Ψ = - ∑ start_POSTSUBSCRIPT italic_k end_POSTSUBSCRIPT ∑ start_POSTSUBSCRIPT italic_i italic_j end_POSTSUBSCRIPT divide start_ARG italic_a start_POSTSUBSCRIPT italic_i italic_j end_POSTSUBSCRIPT start_POSTSUPERSCRIPT italic_k end_POSTSUPERSCRIPT end_ARG start_ARG italic_T start_POSTSUBSCRIPT italic_k end_POSTSUBSCRIPT end_ARG ( italic_E start_POSTSUBSCRIPT italic_i end_POSTSUBSCRIPT - italic_E start_POSTSUBSCRIPT italic_j end_POSTSUBSCRIPT ) italic_p start_POSTSUBSCRIPT italic_j end_POSTSUBSCRIPT | 

 | −∑l∑i⁢jbi⁢jlTl⁢(Ei−Ej)⁢pj+∑l∑i⁢jbi⁢jlTl⁢μl⁢(Nil−Njl)⁢pj,subscript𝑙subscript𝑖𝑗superscriptsubscript𝑏𝑖𝑗𝑙subscript𝑇𝑙subscript𝐸𝑖subscript𝐸𝑗subscript𝑝𝑗subscript𝑙subscript𝑖𝑗superscriptsubscript𝑏𝑖𝑗𝑙subscript𝑇𝑙subscript𝜇𝑙superscriptsubscript𝑁𝑖𝑙superscriptsubscript𝑁𝑗𝑙subscript𝑝𝑗-\sum_{l}\sum_{ij}\frac{b_{ij}^{l}}{T_{l}}(E_{i}-E_{j})p_{j}+\sum_{l}\sum_{ij}%
\frac{b_{ij}^{l}}{T_{l}}\mu_{l}(N_{i}^{l}-N_{j}^{l})p_{j},- ∑ start_POSTSUBSCRIPT italic_l end_POSTSUBSCRIPT ∑ start_POSTSUBSCRIPT italic_i italic_j end_POSTSUBSCRIPT divide start_ARG italic_b start_POSTSUBSCRIPT italic_i italic_j end_POSTSUBSCRIPT start_POSTSUPERSCRIPT italic_l end_POSTSUPERSCRIPT end_ARG start_ARG italic_T start_POSTSUBSCRIPT italic_l end_POSTSUBSCRIPT end_ARG ( italic_E start_POSTSUBSCRIPT italic_i end_POSTSUBSCRIPT - italic_E start_POSTSUBSCRIPT italic_j end_POSTSUBSCRIPT ) italic_p start_POSTSUBSCRIPT italic_j end_POSTSUBSCRIPT + ∑ start_POSTSUBSCRIPT italic_l end_POSTSUBSCRIPT ∑ start_POSTSUBSCRIPT italic_i italic_j end_POSTSUBSCRIPT divide start_ARG italic_b start_POSTSUBSCRIPT italic_i italic_j end_POSTSUBSCRIPT start_POSTSUPERSCRIPT italic_l end_POSTSUPERSCRIPT end_ARG start_ARG italic_T start_POSTSUBSCRIPT italic_l end_POSTSUBSCRIPT end_ARG italic_μ start_POSTSUBSCRIPT italic_l end_POSTSUBSCRIPT ( italic_N start_POSTSUBSCRIPT italic_i end_POSTSUBSCRIPT start_POSTSUPERSCRIPT italic_l end_POSTSUPERSCRIPT - italic_N start_POSTSUBSCRIPT italic_j end_POSTSUBSCRIPT start_POSTSUPERSCRIPT italic_l end_POSTSUPERSCRIPT ) italic_p start_POSTSUBSCRIPT italic_j end_POSTSUBSCRIPT , |  | (56)

where we used again the condition stated just below equation
(36), and Tl=1/κ⁢βlsubscript𝑇𝑙1𝜅subscript𝛽𝑙T_{l}=1/\kappa\beta_{l}italic_T start_POSTSUBSCRIPT italic_l end_POSTSUBSCRIPT = 1 / italic_κ italic_β start_POSTSUBSCRIPT italic_l end_POSTSUBSCRIPT. Using (41),
(42), and (44), this equation can be written as

 | Ψ=−∑k1Tk⁢ϕk−∑l1Tl⁢(ϕk−μl⁢χl),Ψsubscript𝑘1subscript𝑇𝑘subscriptitalic-ϕ𝑘subscript𝑙1subscript𝑇𝑙subscriptitalic-ϕ𝑘subscript𝜇𝑙subscript𝜒𝑙\Psi=-\sum_{k}\frac{1}{T_{k}}\phi_{k}-\sum_{l}\frac{1}{T_{l}}(\phi_{k}-\mu_{l}%
\chi_{l}),roman_Ψ = - ∑ start_POSTSUBSCRIPT italic_k end_POSTSUBSCRIPT divide start_ARG 1 end_ARG start_ARG italic_T start_POSTSUBSCRIPT italic_k end_POSTSUBSCRIPT end_ARG italic_ϕ start_POSTSUBSCRIPT italic_k end_POSTSUBSCRIPT - ∑ start_POSTSUBSCRIPT italic_l end_POSTSUBSCRIPT divide start_ARG 1 end_ARG start_ARG italic_T start_POSTSUBSCRIPT italic_l end_POSTSUBSCRIPT end_ARG ( italic_ϕ start_POSTSUBSCRIPT italic_k end_POSTSUBSCRIPT - italic_μ start_POSTSUBSCRIPT italic_l end_POSTSUBSCRIPT italic_χ start_POSTSUBSCRIPT italic_l end_POSTSUBSCRIPT ) , |  | (57)

which is identified with equation (11). Therefore, we
may say that the Clausius relation introduced in II.1 as a postulate
of the present formulation of irreversible thermodynamics in order
to reach equation (11) is a direct consequence of the
form we have assumed for the transitions rates, namely that
given by equations (33) and (36).

### III.4 Convexity of 𝒫𝒫{\cal P}caligraphic_P in relation to pisubscript𝑝𝑖p_{i}italic_p start_POSTSUBSCRIPT italic_i end_POSTSUBSCRIPT

We show now that the production of entropy is an
upward convex function of the collection of pisubscript𝑝𝑖p_{i}italic_p start_POSTSUBSCRIPT italic_i end_POSTSUBSCRIPT.
That is, we show that the second order variation
of the entropy production in relation to variations
in pisubscript𝑝𝑖p_{i}italic_p start_POSTSUBSCRIPT italic_i end_POSTSUBSCRIPT is nonnegative, δ2⁢𝒫≥0superscript𝛿2𝒫0\delta^{2}{\cal P}\geq 0italic_δ start_POSTSUPERSCRIPT 2 end_POSTSUPERSCRIPT caligraphic_P ≥ 0.
To this end we first observe that ΨΨ\Psiroman_Ψ is linear
in pisubscript𝑝𝑖p_{i}italic_p start_POSTSUBSCRIPT italic_i end_POSTSUBSCRIPT from which follows that its second order
variation in relation to variations in
pisubscript𝑝𝑖p_{i}italic_p start_POSTSUBSCRIPT italic_i end_POSTSUBSCRIPT vanishes. Therefore

 | δ2⁢𝒫=δ2⁢Γ,superscript𝛿2𝒫superscript𝛿2Γ\delta^{2}{\cal P}=\delta^{2}\Gamma,italic_δ start_POSTSUPERSCRIPT 2 end_POSTSUPERSCRIPT caligraphic_P = italic_δ start_POSTSUPERSCRIPT 2 end_POSTSUPERSCRIPT roman_Γ , |  | (58)

where Γ=d⁢S/d⁢t=𝒫−ΨΓ𝑑𝑆𝑑𝑡𝒫Ψ\Gamma=dS/dt={\cal P}-\Psiroman_Γ = italic_d italic_S / italic_d italic_t = caligraphic_P - roman_Ψ is the expression
on the right-hand side of (49), that is,

 | Γ=−κ⁢∑i⁢jwi⁢j⁢pj⁢ln⁡pi.Γ𝜅subscript𝑖𝑗subscript𝑤𝑖𝑗subscript𝑝𝑗subscript𝑝𝑖\Gamma=-\kappa\sum_{ij}w_{ij}p_{j}\ln p_{i}.roman_Γ = - italic_κ ∑ start_POSTSUBSCRIPT italic_i italic_j end_POSTSUBSCRIPT italic_w start_POSTSUBSCRIPT italic_i italic_j end_POSTSUBSCRIPT italic_p start_POSTSUBSCRIPT italic_j end_POSTSUBSCRIPT roman_ln italic_p start_POSTSUBSCRIPT italic_i end_POSTSUBSCRIPT . |  | (59)

The second order variation of ΓΓ\Gammaroman_Γ is

 | δ2⁢Γ=κ2⁢∑i⁢j∂2Γ∂pi⁢∂pj⁢δ⁢pi⁢δ⁢pj.superscript𝛿2Γ𝜅2subscript𝑖𝑗superscript2Γsubscript𝑝𝑖subscript𝑝𝑗𝛿subscript𝑝𝑖𝛿subscript𝑝𝑗\delta^{2}\Gamma=\frac{\kappa}{2}\sum_{ij}\frac{\partial^{2}\Gamma}{\partial p%
_{i}\partial p_{j}}\delta p_{i}\delta p_{j}.italic_δ start_POSTSUPERSCRIPT 2 end_POSTSUPERSCRIPT roman_Γ = divide start_ARG italic_κ end_ARG start_ARG 2 end_ARG ∑ start_POSTSUBSCRIPT italic_i italic_j end_POSTSUBSCRIPT divide start_ARG ∂ start_POSTSUPERSCRIPT 2 end_POSTSUPERSCRIPT roman_Γ end_ARG start_ARG ∂ italic_p start_POSTSUBSCRIPT italic_i end_POSTSUBSCRIPT ∂ italic_p start_POSTSUBSCRIPT italic_j end_POSTSUBSCRIPT end_ARG italic_δ italic_p start_POSTSUBSCRIPT italic_i end_POSTSUBSCRIPT italic_δ italic_p start_POSTSUBSCRIPT italic_j end_POSTSUBSCRIPT . |  | (60)

From (59), we obtain

 | ∂Γ∂pj=−κ⁢∑kwk⁢j⁢ln⁡pk−κ⁢∑kwj⁢k⁢pkpj,Γsubscript𝑝𝑗𝜅subscript𝑘subscript𝑤𝑘𝑗subscript𝑝𝑘𝜅subscript𝑘subscript𝑤𝑗𝑘subscript𝑝𝑘subscript𝑝𝑗\frac{\partial\Gamma}{\partial p_{j}}=-\kappa\sum_{k}w_{kj}\ln p_{k}-\kappa%
\sum_{k}w_{jk}\frac{p_{k}}{p_{j}},divide start_ARG ∂ roman_Γ end_ARG start_ARG ∂ italic_p start_POSTSUBSCRIPT italic_j end_POSTSUBSCRIPT end_ARG = - italic_κ ∑ start_POSTSUBSCRIPT italic_k end_POSTSUBSCRIPT italic_w start_POSTSUBSCRIPT italic_k italic_j end_POSTSUBSCRIPT roman_ln italic_p start_POSTSUBSCRIPT italic_k end_POSTSUBSCRIPT - italic_κ ∑ start_POSTSUBSCRIPT italic_k end_POSTSUBSCRIPT italic_w start_POSTSUBSCRIPT italic_j italic_k end_POSTSUBSCRIPT divide start_ARG italic_p start_POSTSUBSCRIPT italic_k end_POSTSUBSCRIPT end_ARG start_ARG italic_p start_POSTSUBSCRIPT italic_j end_POSTSUBSCRIPT end_ARG , |  | (61)

and

 | ∂2Γ∂pi⁢∂pj=−κ⁢(wi⁢jpi+wj⁢ipj)+κ⁢δi⁢j⁢∑kwj⁢k⁢pkpj2.superscript2Γsubscript𝑝𝑖subscript𝑝𝑗𝜅subscript𝑤𝑖𝑗subscript𝑝𝑖subscript𝑤𝑗𝑖subscript𝑝𝑗𝜅subscript𝛿𝑖𝑗subscript𝑘subscript𝑤𝑗𝑘subscript𝑝𝑘superscriptsubscript𝑝𝑗2\frac{\partial^{2}\Gamma}{\partial p_{i}\partial p_{j}}=-\kappa(\frac{w_{ij}}{%
p_{i}}+\frac{w_{ji}}{p_{j}})+\kappa\,\delta_{ij}\sum_{k}w_{jk}\frac{p_{k}}{p_{%
j}^{2}}.divide start_ARG ∂ start_POSTSUPERSCRIPT 2 end_POSTSUPERSCRIPT roman_Γ end_ARG start_ARG ∂ italic_p start_POSTSUBSCRIPT italic_i end_POSTSUBSCRIPT ∂ italic_p start_POSTSUBSCRIPT italic_j end_POSTSUBSCRIPT end_ARG = - italic_κ ( divide start_ARG italic_w start_POSTSUBSCRIPT italic_i italic_j end_POSTSUBSCRIPT end_ARG start_ARG italic_p start_POSTSUBSCRIPT italic_i end_POSTSUBSCRIPT end_ARG + divide start_ARG italic_w start_POSTSUBSCRIPT italic_j italic_i end_POSTSUBSCRIPT end_ARG start_ARG italic_p start_POSTSUBSCRIPT italic_j end_POSTSUBSCRIPT end_ARG ) + italic_κ italic_δ start_POSTSUBSCRIPT italic_i italic_j end_POSTSUBSCRIPT ∑ start_POSTSUBSCRIPT italic_k end_POSTSUBSCRIPT italic_w start_POSTSUBSCRIPT italic_j italic_k end_POSTSUBSCRIPT divide start_ARG italic_p start_POSTSUBSCRIPT italic_k end_POSTSUBSCRIPT end_ARG start_ARG italic_p start_POSTSUBSCRIPT italic_j end_POSTSUBSCRIPT start_POSTSUPERSCRIPT 2 end_POSTSUPERSCRIPT end_ARG . |  | (62)

Replacing this result in (60), we find

 | δ2⁢𝒫=δ2⁢Γ=κ2⁢∑i⁢jwi⁢j⁢pj⁢(δ⁢pipi−δ⁢pjpj)2,superscript𝛿2𝒫superscript𝛿2Γ𝜅2subscript𝑖𝑗subscript𝑤𝑖𝑗subscript𝑝𝑗superscript𝛿subscript𝑝𝑖subscript𝑝𝑖𝛿subscript𝑝𝑗subscript𝑝𝑗2\delta^{2}{\cal P}=\delta^{2}\Gamma=\frac{\kappa}{2}\sum_{ij}w_{ij}p_{j}(\frac%
{\delta p_{i}}{p_{i}}-\frac{\delta p_{j}}{p_{j}})^{2},italic_δ start_POSTSUPERSCRIPT 2 end_POSTSUPERSCRIPT caligraphic_P = italic_δ start_POSTSUPERSCRIPT 2 end_POSTSUPERSCRIPT roman_Γ = divide start_ARG italic_κ end_ARG start_ARG 2 end_ARG ∑ start_POSTSUBSCRIPT italic_i italic_j end_POSTSUBSCRIPT italic_w start_POSTSUBSCRIPT italic_i italic_j end_POSTSUBSCRIPT italic_p start_POSTSUBSCRIPT italic_j end_POSTSUBSCRIPT ( divide start_ARG italic_δ italic_p start_POSTSUBSCRIPT italic_i end_POSTSUBSCRIPT end_ARG start_ARG italic_p start_POSTSUBSCRIPT italic_i end_POSTSUBSCRIPT end_ARG - divide start_ARG italic_δ italic_p start_POSTSUBSCRIPT italic_j end_POSTSUBSCRIPT end_ARG start_ARG italic_p start_POSTSUBSCRIPT italic_j end_POSTSUBSCRIPT end_ARG ) start_POSTSUPERSCRIPT 2 end_POSTSUPERSCRIPT , |  | (63)

where we used the property (30). Taking into account
that wi⁢j≥0subscript𝑤𝑖𝑗0w_{ij}\geq 0italic_w start_POSTSUBSCRIPT italic_i italic_j end_POSTSUBSCRIPT ≥ 0 for i≠j𝑖𝑗i\neq jitalic_i ≠ italic_j, we reach the desired result

 | δ2⁢𝒫=δ2⁢Γ≥0.superscript𝛿2𝒫superscript𝛿2Γ0\delta^{2}{\cal P}=\delta^{2}\Gamma\geq 0.italic_δ start_POSTSUPERSCRIPT 2 end_POSTSUPERSCRIPT caligraphic_P = italic_δ start_POSTSUPERSCRIPT 2 end_POSTSUPERSCRIPT roman_Γ ≥ 0 . |  | (64)

### III.5 Convexity of 𝒫𝒫{\cal P}caligraphic_P in relation to the fluxes

We have just proven that 𝒫𝒫{\cal P}caligraphic_P is convex in relation to
the probabilities pisubscript𝑝𝑖p_{i}italic_p start_POSTSUBSCRIPT italic_i end_POSTSUBSCRIPT. We now show that 𝒫𝒫{\cal P}caligraphic_P is also
convex in relation to the fluxes Jksubscript𝐽𝑘J_{k}italic_J start_POSTSUBSCRIPT italic_k end_POSTSUBSCRIPT. This property is
expected because the fluxes are linear in pisubscript𝑝𝑖p_{i}italic_p start_POSTSUBSCRIPT italic_i end_POSTSUBSCRIPT.

As before, we use the notation Jksubscript𝐽𝑘J_{k}italic_J start_POSTSUBSCRIPT italic_k end_POSTSUBSCRIPT for −ϕksubscriptitalic-ϕ𝑘-\phi_{k}- italic_ϕ start_POSTSUBSCRIPT italic_k end_POSTSUBSCRIPT or
−χlsubscript𝜒𝑙-\chi_{l}- italic_χ start_POSTSUBSCRIPT italic_l end_POSTSUBSCRIPT and the notation Fksubscript𝐹𝑘F_{k}italic_F start_POSTSUBSCRIPT italic_k end_POSTSUBSCRIPT for 1/Tk1subscript𝑇𝑘1/T_{k}1 / italic_T start_POSTSUBSCRIPT italic_k end_POSTSUBSCRIPT or −μk/Tksubscript𝜇𝑘subscript𝑇𝑘-\mu_{k}/T_{k}- italic_μ start_POSTSUBSCRIPT italic_k end_POSTSUBSCRIPT / italic_T start_POSTSUBSCRIPT italic_k end_POSTSUBSCRIPT,
already introduced above. Then the expression (57) for
ΨΨ\Psiroman_Ψ is written in the simplified form

 | Ψ=∑kFk⁢Jk.Ψsubscript𝑘subscript𝐹𝑘subscript𝐽𝑘\Psi=\sum_{k}F_{k}J_{k}.roman_Ψ = ∑ start_POSTSUBSCRIPT italic_k end_POSTSUBSCRIPT italic_F start_POSTSUBSCRIPT italic_k end_POSTSUBSCRIPT italic_J start_POSTSUBSCRIPT italic_k end_POSTSUBSCRIPT . |  | (65)

As the fluxes Jksubscript𝐽𝑘J_{k}italic_J start_POSTSUBSCRIPT italic_k end_POSTSUBSCRIPT are linear in pjsubscript𝑝𝑗p_{j}italic_p start_POSTSUBSCRIPT italic_j end_POSTSUBSCRIPT,
it can be written as

 | Jk=∑jfk⁢j⁢pj,subscript𝐽𝑘subscript𝑗subscript𝑓𝑘𝑗subscript𝑝𝑗J_{k}=\sum_{j}f_{kj}p_{j},italic_J start_POSTSUBSCRIPT italic_k end_POSTSUBSCRIPT = ∑ start_POSTSUBSCRIPT italic_j end_POSTSUBSCRIPT italic_f start_POSTSUBSCRIPT italic_k italic_j end_POSTSUBSCRIPT italic_p start_POSTSUBSCRIPT italic_j end_POSTSUBSCRIPT , |  | (66)

and the explicit expressions for the coefficients fk⁢jsubscript𝑓𝑘𝑗f_{kj}italic_f start_POSTSUBSCRIPT italic_k italic_j end_POSTSUBSCRIPT
are obtained from (41), (42), and (44)
and they are either

 | fk⁢j=−∑i(Ei−Ej)⁢ai⁢jk,subscript𝑓𝑘𝑗subscript𝑖subscript𝐸𝑖subscript𝐸𝑗superscriptsubscript𝑎𝑖𝑗𝑘f_{kj}=-\sum_{i}(E_{i}-E_{j})a_{ij}^{k},italic_f start_POSTSUBSCRIPT italic_k italic_j end_POSTSUBSCRIPT = - ∑ start_POSTSUBSCRIPT italic_i end_POSTSUBSCRIPT ( italic_E start_POSTSUBSCRIPT italic_i end_POSTSUBSCRIPT - italic_E start_POSTSUBSCRIPT italic_j end_POSTSUBSCRIPT ) italic_a start_POSTSUBSCRIPT italic_i italic_j end_POSTSUBSCRIPT start_POSTSUPERSCRIPT italic_k end_POSTSUPERSCRIPT , |  | (67)

or

 | fl⁢j=−∑i(Ei−Ej)⁢bi⁢jl,subscript𝑓𝑙𝑗subscript𝑖subscript𝐸𝑖subscript𝐸𝑗superscriptsubscript𝑏𝑖𝑗𝑙f_{lj}=-\sum_{i}(E_{i}-E_{j})b_{ij}^{l},italic_f start_POSTSUBSCRIPT italic_l italic_j end_POSTSUBSCRIPT = - ∑ start_POSTSUBSCRIPT italic_i end_POSTSUBSCRIPT ( italic_E start_POSTSUBSCRIPT italic_i end_POSTSUBSCRIPT - italic_E start_POSTSUBSCRIPT italic_j end_POSTSUBSCRIPT ) italic_b start_POSTSUBSCRIPT italic_i italic_j end_POSTSUBSCRIPT start_POSTSUPERSCRIPT italic_l end_POSTSUPERSCRIPT , |  | (68)

or

 | fl⁢j=−∑i(Nil−Njl)⁢bi⁢jl.subscript𝑓𝑙𝑗subscript𝑖superscriptsubscript𝑁𝑖𝑙superscriptsubscript𝑁𝑗𝑙superscriptsubscript𝑏𝑖𝑗𝑙f_{lj}=-\sum_{i}(N_{i}^{l}-N_{j}^{l})b_{ij}^{l}.italic_f start_POSTSUBSCRIPT italic_l italic_j end_POSTSUBSCRIPT = - ∑ start_POSTSUBSCRIPT italic_i end_POSTSUBSCRIPT ( italic_N start_POSTSUBSCRIPT italic_i end_POSTSUBSCRIPT start_POSTSUPERSCRIPT italic_l end_POSTSUPERSCRIPT - italic_N start_POSTSUBSCRIPT italic_j end_POSTSUBSCRIPT start_POSTSUPERSCRIPT italic_l end_POSTSUPERSCRIPT ) italic_b start_POSTSUBSCRIPT italic_i italic_j end_POSTSUBSCRIPT start_POSTSUPERSCRIPT italic_l end_POSTSUPERSCRIPT . |  | (69)

Let us define as before

 | Xk=∂𝒫∂Jk,subscript𝑋𝑘𝒫subscript𝐽𝑘X_{k}=\frac{\partial{\cal P}}{\partial J_{k}},italic_X start_POSTSUBSCRIPT italic_k end_POSTSUBSCRIPT = divide start_ARG ∂ caligraphic_P end_ARG start_ARG ∂ italic_J start_POSTSUBSCRIPT italic_k end_POSTSUBSCRIPT end_ARG , |  | (70)

and

 | Ak⁢l=∂2𝒫∂Jk⁢∂Jl=∂Xk∂Jl.subscript𝐴𝑘𝑙superscript2𝒫subscript𝐽𝑘subscript𝐽𝑙subscript𝑋𝑘subscript𝐽𝑙A_{kl}=\frac{\partial^{2}{\cal P}}{\partial J_{k}\partial J_{l}}=\frac{%
\partial X_{k}}{\partial J_{l}}.italic_A start_POSTSUBSCRIPT italic_k italic_l end_POSTSUBSCRIPT = divide start_ARG ∂ start_POSTSUPERSCRIPT 2 end_POSTSUPERSCRIPT caligraphic_P end_ARG start_ARG ∂ italic_J start_POSTSUBSCRIPT italic_k end_POSTSUBSCRIPT ∂ italic_J start_POSTSUBSCRIPT italic_l end_POSTSUBSCRIPT end_ARG = divide start_ARG ∂ italic_X start_POSTSUBSCRIPT italic_k end_POSTSUBSCRIPT end_ARG start_ARG ∂ italic_J start_POSTSUBSCRIPT italic_l end_POSTSUBSCRIPT end_ARG . |  | (71)

We also define

 | Di⁢j=∂2𝒫∂pi⁢∂pj=∂2Γ∂pi⁢∂pj,subscript𝐷𝑖𝑗superscript2𝒫subscript𝑝𝑖subscript𝑝𝑗superscript2Γsubscript𝑝𝑖subscript𝑝𝑗D_{ij}=\frac{\partial^{2}{\cal P}}{\partial p_{i}\partial p_{j}}=\frac{%
\partial^{2}\Gamma}{\partial p_{i}\partial p_{j}},italic_D start_POSTSUBSCRIPT italic_i italic_j end_POSTSUBSCRIPT = divide start_ARG ∂ start_POSTSUPERSCRIPT 2 end_POSTSUPERSCRIPT caligraphic_P end_ARG start_ARG ∂ italic_p start_POSTSUBSCRIPT italic_i end_POSTSUBSCRIPT ∂ italic_p start_POSTSUBSCRIPT italic_j end_POSTSUBSCRIPT end_ARG = divide start_ARG ∂ start_POSTSUPERSCRIPT 2 end_POSTSUPERSCRIPT roman_Γ end_ARG start_ARG ∂ italic_p start_POSTSUBSCRIPT italic_i end_POSTSUBSCRIPT ∂ italic_p start_POSTSUBSCRIPT italic_j end_POSTSUBSCRIPT end_ARG , |  | (72)

the explicit form of which is given by (62),
from which we obtain

 | δ2⁢𝒫=12⁢∑i⁢jDi⁢j⁢δ⁢pi⁢δ⁢pj.superscript𝛿2𝒫12subscript𝑖𝑗subscript𝐷𝑖𝑗𝛿subscript𝑝𝑖𝛿subscript𝑝𝑗\delta^{2}{\cal P}=\frac{1}{2}\sum_{ij}D_{ij}\delta p_{i}\delta p_{j}.italic_δ start_POSTSUPERSCRIPT 2 end_POSTSUPERSCRIPT caligraphic_P = divide start_ARG 1 end_ARG start_ARG 2 end_ARG ∑ start_POSTSUBSCRIPT italic_i italic_j end_POSTSUBSCRIPT italic_D start_POSTSUBSCRIPT italic_i italic_j end_POSTSUBSCRIPT italic_δ italic_p start_POSTSUBSCRIPT italic_i end_POSTSUBSCRIPT italic_δ italic_p start_POSTSUBSCRIPT italic_j end_POSTSUBSCRIPT . |  | (73)

The relation between Di⁢jsubscript𝐷𝑖𝑗D_{ij}italic_D start_POSTSUBSCRIPT italic_i italic_j end_POSTSUBSCRIPT and Ak⁢lsubscript𝐴𝑘𝑙A_{kl}italic_A start_POSTSUBSCRIPT italic_k italic_l end_POSTSUBSCRIPT is

 | Di⁢j=∑k⁢lAk⁢l⁢fk⁢i⁢fl⁢j,subscript𝐷𝑖𝑗subscript𝑘𝑙subscript𝐴𝑘𝑙subscript𝑓𝑘𝑖subscript𝑓𝑙𝑗D_{ij}=\sum_{kl}A_{kl}f_{ki}f_{lj},italic_D start_POSTSUBSCRIPT italic_i italic_j end_POSTSUBSCRIPT = ∑ start_POSTSUBSCRIPT italic_k italic_l end_POSTSUBSCRIPT italic_A start_POSTSUBSCRIPT italic_k italic_l end_POSTSUBSCRIPT italic_f start_POSTSUBSCRIPT italic_k italic_i end_POSTSUBSCRIPT italic_f start_POSTSUBSCRIPT italic_l italic_j end_POSTSUBSCRIPT , |  | (74)

which follows because Jksubscript𝐽𝑘J_{k}italic_J start_POSTSUBSCRIPT italic_k end_POSTSUBSCRIPT is linear in pisubscript𝑝𝑖p_{i}italic_p start_POSTSUBSCRIPT italic_i end_POSTSUBSCRIPT.
Replacing this relation in (73), we find

 | δ2⁢𝒫=12⁢∑k⁢lAk⁢l⁢δ⁢Jk⁢δ⁢Jl,superscript𝛿2𝒫12subscript𝑘𝑙subscript𝐴𝑘𝑙𝛿subscript𝐽𝑘𝛿subscript𝐽𝑙\delta^{2}{\cal P}=\frac{1}{2}\sum_{kl}A_{kl}\delta J_{k}\delta J_{l},italic_δ start_POSTSUPERSCRIPT 2 end_POSTSUPERSCRIPT caligraphic_P = divide start_ARG 1 end_ARG start_ARG 2 end_ARG ∑ start_POSTSUBSCRIPT italic_k italic_l end_POSTSUBSCRIPT italic_A start_POSTSUBSCRIPT italic_k italic_l end_POSTSUBSCRIPT italic_δ italic_J start_POSTSUBSCRIPT italic_k end_POSTSUBSCRIPT italic_δ italic_J start_POSTSUBSCRIPT italic_l end_POSTSUBSCRIPT , |  | (75)

where

 | δ⁢Jk=∑ifk⁢i⁢δ⁢pi.𝛿subscript𝐽𝑘subscript𝑖subscript𝑓𝑘𝑖𝛿subscript𝑝𝑖\delta J_{k}=\sum_{i}f_{ki}\delta p_{i}.italic_δ italic_J start_POSTSUBSCRIPT italic_k end_POSTSUBSCRIPT = ∑ start_POSTSUBSCRIPT italic_i end_POSTSUBSCRIPT italic_f start_POSTSUBSCRIPT italic_k italic_i end_POSTSUBSCRIPT italic_δ italic_p start_POSTSUBSCRIPT italic_i end_POSTSUBSCRIPT . |  | (76)

But δ2⁢𝒫≥0superscript𝛿2𝒫0\delta^{2}{\cal P}\geq 0italic_δ start_POSTSUPERSCRIPT 2 end_POSTSUPERSCRIPT caligraphic_P ≥ 0, as shown above in (64), and

 | ∑k⁢lAk⁢l⁢δ⁢Jk⁢δ⁢Jl≥0,subscript𝑘𝑙subscript𝐴𝑘𝑙𝛿subscript𝐽𝑘𝛿subscript𝐽𝑙0\sum_{kl}A_{kl}\delta J_{k}\delta J_{l}\geq 0,∑ start_POSTSUBSCRIPT italic_k italic_l end_POSTSUBSCRIPT italic_A start_POSTSUBSCRIPT italic_k italic_l end_POSTSUBSCRIPT italic_δ italic_J start_POSTSUBSCRIPT italic_k end_POSTSUBSCRIPT italic_δ italic_J start_POSTSUBSCRIPT italic_l end_POSTSUBSCRIPT ≥ 0 , |  | (77)

which proves that 𝒫𝒫{\cal P}caligraphic_P is an upward convex function of
the collection of variables Jksubscript𝐽𝑘J_{k}italic_J start_POSTSUBSCRIPT italic_k end_POSTSUBSCRIPT, which we have taken as a
postulate of the present approach to irreversible thermodynamics.
taken as an assumption just above equation (15).
From this property, follows the Glansdorff-Prigogine principle
shown in II.2.

### III.6 Excess entropy production

Let us define the quantity Cisubscript𝐶𝑖C_{i}italic_C start_POSTSUBSCRIPT italic_i end_POSTSUBSCRIPT by

 | Cj=∂𝒫∂pj.subscript𝐶𝑗𝒫subscript𝑝𝑗C_{j}=\frac{\partial{\cal P}}{\partial p_{j}}.italic_C start_POSTSUBSCRIPT italic_j end_POSTSUBSCRIPT = divide start_ARG ∂ caligraphic_P end_ARG start_ARG ∂ italic_p start_POSTSUBSCRIPT italic_j end_POSTSUBSCRIPT end_ARG . |  | (78)

From the definition of 𝒫𝒫{\cal P}caligraphic_P, given by (52), we find

 | Cj=κ⁢∑iwi⁢j⁢ln⁡wi⁢j⁢pjwj⁢i⁢pi−κ⁢∑iwj⁢i⁢pipj,subscript𝐶𝑗𝜅subscript𝑖subscript𝑤𝑖𝑗subscript𝑤𝑖𝑗subscript𝑝𝑗subscript𝑤𝑗𝑖subscript𝑝𝑖𝜅subscript𝑖subscript𝑤𝑗𝑖subscript𝑝𝑖subscript𝑝𝑗C_{j}=\kappa\sum_{i}w_{ij}\ln\frac{w_{ij}p_{j}}{w_{ji}p_{i}}-\kappa\sum_{i}w_{%
ji}\frac{p_{i}}{p_{j}},italic_C start_POSTSUBSCRIPT italic_j end_POSTSUBSCRIPT = italic_κ ∑ start_POSTSUBSCRIPT italic_i end_POSTSUBSCRIPT italic_w start_POSTSUBSCRIPT italic_i italic_j end_POSTSUBSCRIPT roman_ln divide start_ARG italic_w start_POSTSUBSCRIPT italic_i italic_j end_POSTSUBSCRIPT italic_p start_POSTSUBSCRIPT italic_j end_POSTSUBSCRIPT end_ARG start_ARG italic_w start_POSTSUBSCRIPT italic_j italic_i end_POSTSUBSCRIPT italic_p start_POSTSUBSCRIPT italic_i end_POSTSUBSCRIPT end_ARG - italic_κ ∑ start_POSTSUBSCRIPT italic_i end_POSTSUBSCRIPT italic_w start_POSTSUBSCRIPT italic_j italic_i end_POSTSUBSCRIPT divide start_ARG italic_p start_POSTSUBSCRIPT italic_i end_POSTSUBSCRIPT end_ARG start_ARG italic_p start_POSTSUBSCRIPT italic_j end_POSTSUBSCRIPT end_ARG , |  | (79)

where we used the property (30). In the
stationary state, pi=qisubscript𝑝𝑖subscript𝑞𝑖p_{i}=q_{i}italic_p start_POSTSUBSCRIPT italic_i end_POSTSUBSCRIPT = italic_q start_POSTSUBSCRIPT italic_i end_POSTSUBSCRIPT, the value of Cjsubscript𝐶𝑗C_{j}italic_C start_POSTSUBSCRIPT italic_j end_POSTSUBSCRIPT is

 | Cj0=κ⁢∑iwi⁢j⁢ln⁡wi⁢j⁢qjwj⁢i⁢qi,superscriptsubscript𝐶𝑗0𝜅subscript𝑖subscript𝑤𝑖𝑗subscript𝑤𝑖𝑗subscript𝑞𝑗subscript𝑤𝑗𝑖subscript𝑞𝑖C_{j}^{0}=\kappa\sum_{i}w_{ij}\ln\frac{w_{ij}q_{j}}{w_{ji}q_{i}},italic_C start_POSTSUBSCRIPT italic_j end_POSTSUBSCRIPT start_POSTSUPERSCRIPT 0 end_POSTSUPERSCRIPT = italic_κ ∑ start_POSTSUBSCRIPT italic_i end_POSTSUBSCRIPT italic_w start_POSTSUBSCRIPT italic_i italic_j end_POSTSUBSCRIPT roman_ln divide start_ARG italic_w start_POSTSUBSCRIPT italic_i italic_j end_POSTSUBSCRIPT italic_q start_POSTSUBSCRIPT italic_j end_POSTSUBSCRIPT end_ARG start_ARG italic_w start_POSTSUBSCRIPT italic_j italic_i end_POSTSUBSCRIPT italic_q start_POSTSUBSCRIPT italic_i end_POSTSUBSCRIPT end_ARG , |  | (80)

where we have used property (32).

The excess entropy production is defined by

 | 𝒫exc=𝒫−∑jCj0⁢(pj−qj)−𝒫0,subscript𝒫exc𝒫subscript𝑗superscriptsubscript𝐶𝑗0subscript𝑝𝑗subscript𝑞𝑗subscript𝒫0{\cal P}_{\rm exc}={\cal P}-\sum_{j}C_{j}^{0}(p_{j}-q_{j})-{\cal P}_{0},caligraphic_P start_POSTSUBSCRIPT roman_exc end_POSTSUBSCRIPT = caligraphic_P - ∑ start_POSTSUBSCRIPT italic_j end_POSTSUBSCRIPT italic_C start_POSTSUBSCRIPT italic_j end_POSTSUBSCRIPT start_POSTSUPERSCRIPT 0 end_POSTSUPERSCRIPT ( italic_p start_POSTSUBSCRIPT italic_j end_POSTSUBSCRIPT - italic_q start_POSTSUBSCRIPT italic_j end_POSTSUBSCRIPT ) - caligraphic_P start_POSTSUBSCRIPT 0 end_POSTSUBSCRIPT , |  | (81)

where 𝒫0subscript𝒫0{\cal P}_{0}caligraphic_P start_POSTSUBSCRIPT 0 end_POSTSUBSCRIPT is the stationary value of 𝒫𝒫{\cal P}caligraphic_P,

 | 𝒫0=κ⁢∑i⁢jwi⁢j⁢qj⁢ln⁡wi⁢j⁢qjwj⁢i⁢qi.subscript𝒫0𝜅subscript𝑖𝑗subscript𝑤𝑖𝑗subscript𝑞𝑗subscript𝑤𝑖𝑗subscript𝑞𝑗subscript𝑤𝑗𝑖subscript𝑞𝑖{\cal P}_{0}=\kappa\sum_{ij}w_{ij}q_{j}\ln\frac{w_{ij}q_{j}}{w_{ji}q_{i}}.caligraphic_P start_POSTSUBSCRIPT 0 end_POSTSUBSCRIPT = italic_κ ∑ start_POSTSUBSCRIPT italic_i italic_j end_POSTSUBSCRIPT italic_w start_POSTSUBSCRIPT italic_i italic_j end_POSTSUBSCRIPT italic_q start_POSTSUBSCRIPT italic_j end_POSTSUBSCRIPT roman_ln divide start_ARG italic_w start_POSTSUBSCRIPT italic_i italic_j end_POSTSUBSCRIPT italic_q start_POSTSUBSCRIPT italic_j end_POSTSUBSCRIPT end_ARG start_ARG italic_w start_POSTSUBSCRIPT italic_j italic_i end_POSTSUBSCRIPT italic_q start_POSTSUBSCRIPT italic_i end_POSTSUBSCRIPT end_ARG . |  | (82)

Taking into account that 𝒫excsubscript𝒫exc{\cal P}_{\rm exc}caligraphic_P start_POSTSUBSCRIPT roman_exc end_POSTSUBSCRIPT differs from
𝒫𝒫{\cal P}caligraphic_P by linear terms then

 | δ2⁢𝒫exc=δ2⁢𝒫,superscript𝛿2subscript𝒫excsuperscript𝛿2𝒫\delta^{2}{\cal P}_{\rm exc}=\delta^{2}{\cal P},italic_δ start_POSTSUPERSCRIPT 2 end_POSTSUPERSCRIPT caligraphic_P start_POSTSUBSCRIPT roman_exc end_POSTSUBSCRIPT = italic_δ start_POSTSUPERSCRIPT 2 end_POSTSUPERSCRIPT caligraphic_P , |  | (83)

and it is also an upward convex function of pisubscript𝑝𝑖p_{i}italic_p start_POSTSUBSCRIPT italic_i end_POSTSUBSCRIPT. Taking into
account that 𝒫excsubscript𝒫exc{\cal P}_{\rm exc}caligraphic_P start_POSTSUBSCRIPT roman_exc end_POSTSUBSCRIPT vanishes at the stationary
state and that its variation at the stationary state also
vanishes, δ⁢𝒫exc=0𝛿subscript𝒫exc0\delta{\cal P}_{\rm exc}=0italic_δ caligraphic_P start_POSTSUBSCRIPT roman_exc end_POSTSUBSCRIPT = 0, then we may write

 | 𝒫exc≥0.subscript𝒫exc0{\cal P}_{\rm exc}\geq 0.caligraphic_P start_POSTSUBSCRIPT roman_exc end_POSTSUBSCRIPT ≥ 0 . |  | (84)

Using the relation

 | ∂𝒫∂pj=∑k∂𝒫∂Jk⁢∂Jk∂pj,𝒫subscript𝑝𝑗subscript𝑘𝒫subscript𝐽𝑘subscript𝐽𝑘subscript𝑝𝑗\frac{\partial{\cal P}}{\partial p_{j}}=\sum_{k}\frac{\partial{\cal P}}{%
\partial J_{k}}\frac{\partial J_{k}}{\partial p_{j}},divide start_ARG ∂ caligraphic_P end_ARG start_ARG ∂ italic_p start_POSTSUBSCRIPT italic_j end_POSTSUBSCRIPT end_ARG = ∑ start_POSTSUBSCRIPT italic_k end_POSTSUBSCRIPT divide start_ARG ∂ caligraphic_P end_ARG start_ARG ∂ italic_J start_POSTSUBSCRIPT italic_k end_POSTSUBSCRIPT end_ARG divide start_ARG ∂ italic_J start_POSTSUBSCRIPT italic_k end_POSTSUBSCRIPT end_ARG start_ARG ∂ italic_p start_POSTSUBSCRIPT italic_j end_POSTSUBSCRIPT end_ARG , |  | (85)

we find

 | Cj=∑kXk⁢fk⁢j,subscript𝐶𝑗subscript𝑘subscript𝑋𝑘subscript𝑓𝑘𝑗C_{j}=\sum_{k}X_{k}f_{kj},italic_C start_POSTSUBSCRIPT italic_j end_POSTSUBSCRIPT = ∑ start_POSTSUBSCRIPT italic_k end_POSTSUBSCRIPT italic_X start_POSTSUBSCRIPT italic_k end_POSTSUBSCRIPT italic_f start_POSTSUBSCRIPT italic_k italic_j end_POSTSUBSCRIPT , |  | (86)

from which we get

 | ∑jCj0⁢(pj−qj)=∑kXk0⁢(Jk−Jk0),subscript𝑗superscriptsubscript𝐶𝑗0subscript𝑝𝑗subscript𝑞𝑗subscript𝑘superscriptsubscript𝑋𝑘0subscript𝐽𝑘superscriptsubscript𝐽𝑘0\sum_{j}C_{j}^{0}(p_{j}-q_{j})=\sum_{k}X_{k}^{0}(J_{k}-J_{k}^{0}),∑ start_POSTSUBSCRIPT italic_j end_POSTSUBSCRIPT italic_C start_POSTSUBSCRIPT italic_j end_POSTSUBSCRIPT start_POSTSUPERSCRIPT 0 end_POSTSUPERSCRIPT ( italic_p start_POSTSUBSCRIPT italic_j end_POSTSUBSCRIPT - italic_q start_POSTSUBSCRIPT italic_j end_POSTSUBSCRIPT ) = ∑ start_POSTSUBSCRIPT italic_k end_POSTSUBSCRIPT italic_X start_POSTSUBSCRIPT italic_k end_POSTSUBSCRIPT start_POSTSUPERSCRIPT 0 end_POSTSUPERSCRIPT ( italic_J start_POSTSUBSCRIPT italic_k end_POSTSUBSCRIPT - italic_J start_POSTSUBSCRIPT italic_k end_POSTSUBSCRIPT start_POSTSUPERSCRIPT 0 end_POSTSUPERSCRIPT ) , |  | (87)

which replaced in (81) gives

 | 𝒫exc=𝒫−∑kXk0⁢(Jk−Jk0)−𝒫0,subscript𝒫exc𝒫subscript𝑘superscriptsubscript𝑋𝑘0subscript𝐽𝑘superscriptsubscript𝐽𝑘0subscript𝒫0{\cal P}_{\rm exc}={\cal P}-\sum_{k}X_{k}^{0}(J_{k}-J_{k}^{0})-{\cal P}_{0},caligraphic_P start_POSTSUBSCRIPT roman_exc end_POSTSUBSCRIPT = caligraphic_P - ∑ start_POSTSUBSCRIPT italic_k end_POSTSUBSCRIPT italic_X start_POSTSUBSCRIPT italic_k end_POSTSUBSCRIPT start_POSTSUPERSCRIPT 0 end_POSTSUPERSCRIPT ( italic_J start_POSTSUBSCRIPT italic_k end_POSTSUBSCRIPT - italic_J start_POSTSUBSCRIPT italic_k end_POSTSUBSCRIPT start_POSTSUPERSCRIPT 0 end_POSTSUPERSCRIPT ) - caligraphic_P start_POSTSUBSCRIPT 0 end_POSTSUBSCRIPT , |  | (88)

which is identified as the expression (19),
and the excess entropy defined within our formulation of
irreversible thermodynamics coincides with the
excess entropy defined by expression (81).

### III.7 Lyapunov function

Replacing in equation (81) the expression for Cj0superscriptsubscript𝐶𝑗0C_{j}^{0}italic_C start_POSTSUBSCRIPT italic_j end_POSTSUBSCRIPT start_POSTSUPERSCRIPT 0 end_POSTSUPERSCRIPT,
given by (80), and the expressions for 𝒫𝒫{\cal P}caligraphic_P and
𝒫0subscript𝒫0{\cal P}_{0}caligraphic_P start_POSTSUBSCRIPT 0 end_POSTSUBSCRIPT, given by (53) and (82), we find

 | 𝒫exc=−κ⁢∑i⁢jwi⁢j⁢pj⁢ln⁡piqi,subscript𝒫exc𝜅subscript𝑖𝑗subscript𝑤𝑖𝑗subscript𝑝𝑗subscript𝑝𝑖subscript𝑞𝑖{\cal P}_{\rm exc}=-\kappa\sum_{ij}w_{ij}p_{j}\ln\frac{p_{i}}{q_{i}},caligraphic_P start_POSTSUBSCRIPT roman_exc end_POSTSUBSCRIPT = - italic_κ ∑ start_POSTSUBSCRIPT italic_i italic_j end_POSTSUBSCRIPT italic_w start_POSTSUBSCRIPT italic_i italic_j end_POSTSUBSCRIPT italic_p start_POSTSUBSCRIPT italic_j end_POSTSUBSCRIPT roman_ln divide start_ARG italic_p start_POSTSUBSCRIPT italic_i end_POSTSUBSCRIPT end_ARG start_ARG italic_q start_POSTSUBSCRIPT italic_i end_POSTSUBSCRIPT end_ARG , |  | (89)

where we used the property (32).
Using the master equation in the form (31), we see that
the right-hand side of this equation is the time derivative of

 | L=−κ⁢∑ipi⁢ln⁡piqi,𝐿𝜅subscript𝑖subscript𝑝𝑖subscript𝑝𝑖subscript𝑞𝑖L=-\kappa\sum_{i}p_{i}\ln\frac{p_{i}}{q_{i}},italic_L = - italic_κ ∑ start_POSTSUBSCRIPT italic_i end_POSTSUBSCRIPT italic_p start_POSTSUBSCRIPT italic_i end_POSTSUBSCRIPT roman_ln divide start_ARG italic_p start_POSTSUBSCRIPT italic_i end_POSTSUBSCRIPT end_ARG start_ARG italic_q start_POSTSUBSCRIPT italic_i end_POSTSUBSCRIPT end_ARG , |  | (90)

that is,

 | 𝒫exc=d⁢Ld⁢t,subscript𝒫exc𝑑𝐿𝑑𝑡{\cal P}_{\rm exc}=\frac{dL}{dt},caligraphic_P start_POSTSUBSCRIPT roman_exc end_POSTSUBSCRIPT = divide start_ARG italic_d italic_L end_ARG start_ARG italic_d italic_t end_ARG , |  | (91)

from which follows

 | d⁢Ld⁢t≥0,𝑑𝐿𝑑𝑡0\frac{dL}{dt}\geq 0,divide start_ARG italic_d italic_L end_ARG start_ARG italic_d italic_t end_ARG ≥ 0 , |  | (92)

because 𝒫exc≥0subscript𝒫exc0{\cal P}_{\rm exc}\geq 0caligraphic_P start_POSTSUBSCRIPT roman_exc end_POSTSUBSCRIPT ≥ 0.

The quantity L𝐿Litalic_L can yet be written in the form

 | L=−κ⁢∑i[pi⁢ln⁡piqi−(pi−qi)],𝐿𝜅subscript𝑖delimited-[]subscript𝑝𝑖subscript𝑝𝑖subscript𝑞𝑖subscript𝑝𝑖subscript𝑞𝑖L=-\kappa\sum_{i}[p_{i}\ln\frac{p_{i}}{q_{i}}-(p_{i}-q_{i})],italic_L = - italic_κ ∑ start_POSTSUBSCRIPT italic_i end_POSTSUBSCRIPT [ italic_p start_POSTSUBSCRIPT italic_i end_POSTSUBSCRIPT roman_ln divide start_ARG italic_p start_POSTSUBSCRIPT italic_i end_POSTSUBSCRIPT end_ARG start_ARG italic_q start_POSTSUBSCRIPT italic_i end_POSTSUBSCRIPT end_ARG - ( italic_p start_POSTSUBSCRIPT italic_i end_POSTSUBSCRIPT - italic_q start_POSTSUBSCRIPT italic_i end_POSTSUBSCRIPT ) ] , |  | (93)

from which follows that

 | L≤0,𝐿0L\leq 0,italic_L ≤ 0 , |  | (94)

because the expression inside
square brackets is greater or equal zero.
The two inequalities (92) and (94) show
that L𝐿Litalic_L is a Lyapunov function in relation to
the variables pisubscript𝑝𝑖p_{i}italic_p start_POSTSUBSCRIPT italic_i end_POSTSUBSCRIPT.

## IV Absence of temperature grandients

The equation (91) tell us that the excess entropy
production is a time derivative of L𝐿Litalic_L which is a function
of the probabilities pisubscript𝑝𝑖p_{i}italic_p start_POSTSUBSCRIPT italic_i end_POSTSUBSCRIPT. We may ask whether
it is possible to write the excess entropy production
as a time derivative of a thermodynamic potential, understood
as a linear combination of S𝑆Sitalic_S, U𝑈Uitalic_U and Nlsubscript𝑁𝑙N_{l}italic_N start_POSTSUBSCRIPT italic_l end_POSTSUBSCRIPT. This is indeed
possible if the processes are isothermal, that is, if the
heat introduced into the system comes from reservoirs that
have all the same temperature which means that the system
is subject to no gradients of temperature. In other words,
βksubscript𝛽𝑘\beta_{k}italic_β start_POSTSUBSCRIPT italic_k end_POSTSUBSCRIPT and βlsubscript𝛽𝑙\beta_{l}italic_β start_POSTSUBSCRIPT italic_l end_POSTSUBSCRIPT appearing in the rates (33)
and (37) should have the same value.

When βk=βsubscript𝛽𝑘𝛽\beta_{k}=\betaitalic_β start_POSTSUBSCRIPT italic_k end_POSTSUBSCRIPT = italic_β, independent of k𝑘kitalic_k, the several
transitions given by (33) can be gathered into
a single transition rate ai⁢jsubscript𝑎𝑖𝑗a_{ij}italic_a start_POSTSUBSCRIPT italic_i italic_j end_POSTSUBSCRIPT, given by

 | ai⁢j=Ai⁢j⁢e−β⁢(Ei−Ej)/2,subscript𝑎𝑖𝑗subscript𝐴𝑖𝑗superscript𝑒𝛽subscript𝐸𝑖subscript𝐸𝑗2a_{ij}=A_{ij}e^{-\beta(E_{i}-E_{j})/2},italic_a start_POSTSUBSCRIPT italic_i italic_j end_POSTSUBSCRIPT = italic_A start_POSTSUBSCRIPT italic_i italic_j end_POSTSUBSCRIPT italic_e start_POSTSUPERSCRIPT - italic_β ( italic_E start_POSTSUBSCRIPT italic_i end_POSTSUBSCRIPT - italic_E start_POSTSUBSCRIPT italic_j end_POSTSUBSCRIPT ) / 2 end_POSTSUPERSCRIPT , |  | (95)

where Ai⁢j=Aj⁢isubscript𝐴𝑖𝑗subscript𝐴𝑗𝑖A_{ij}=A_{ji}italic_A start_POSTSUBSCRIPT italic_i italic_j end_POSTSUBSCRIPT = italic_A start_POSTSUBSCRIPT italic_j italic_i end_POSTSUBSCRIPT. Considering that βl=βsubscript𝛽𝑙𝛽\beta_{l}=\betaitalic_β start_POSTSUBSCRIPT italic_l end_POSTSUBSCRIPT = italic_β
is also independent of l𝑙litalic_l, the transitions (37)
becomes

 | bi⁢jl=Bi⁢jl⁢e−β⁢(Ei−Ej)/2+β⁢μl⁢(Nil−Njl)/2,superscriptsubscript𝑏𝑖𝑗𝑙superscriptsubscript𝐵𝑖𝑗𝑙superscript𝑒𝛽subscript𝐸𝑖subscript𝐸𝑗2𝛽subscript𝜇𝑙superscriptsubscript𝑁𝑖𝑙superscriptsubscript𝑁𝑗𝑙2b_{ij}^{l}=B_{ij}^{l}e^{-\beta(E_{i}-E_{j})/2+\beta\mu_{l}(N_{i}^{l}-N_{j}^{l}%
)/2},italic_b start_POSTSUBSCRIPT italic_i italic_j end_POSTSUBSCRIPT start_POSTSUPERSCRIPT italic_l end_POSTSUPERSCRIPT = italic_B start_POSTSUBSCRIPT italic_i italic_j end_POSTSUBSCRIPT start_POSTSUPERSCRIPT italic_l end_POSTSUPERSCRIPT italic_e start_POSTSUPERSCRIPT - italic_β ( italic_E start_POSTSUBSCRIPT italic_i end_POSTSUBSCRIPT - italic_E start_POSTSUBSCRIPT italic_j end_POSTSUBSCRIPT ) / 2 + italic_β italic_μ start_POSTSUBSCRIPT italic_l end_POSTSUBSCRIPT ( italic_N start_POSTSUBSCRIPT italic_i end_POSTSUBSCRIPT start_POSTSUPERSCRIPT italic_l end_POSTSUPERSCRIPT - italic_N start_POSTSUBSCRIPT italic_j end_POSTSUBSCRIPT start_POSTSUPERSCRIPT italic_l end_POSTSUPERSCRIPT ) / 2 end_POSTSUPERSCRIPT , |  | (96)

where β=1/κ⁢T𝛽1𝜅𝑇\beta=1/\kappa Titalic_β = 1 / italic_κ italic_T, and T𝑇Titalic_T is the common temperature
of the reservoirs.

The ratio of the rates of the forward and backward transitions are

 | ai⁢jaj⁢i=e−β⁢(Ei−Ej),subscript𝑎𝑖𝑗subscript𝑎𝑗𝑖superscript𝑒𝛽subscript𝐸𝑖subscript𝐸𝑗\frac{a_{ij}}{a_{ji}}=e^{-\beta(E_{i}-E_{j})},divide start_ARG italic_a start_POSTSUBSCRIPT italic_i italic_j end_POSTSUBSCRIPT end_ARG start_ARG italic_a start_POSTSUBSCRIPT italic_j italic_i end_POSTSUBSCRIPT end_ARG = italic_e start_POSTSUPERSCRIPT - italic_β ( italic_E start_POSTSUBSCRIPT italic_i end_POSTSUBSCRIPT - italic_E start_POSTSUBSCRIPT italic_j end_POSTSUBSCRIPT ) end_POSTSUPERSCRIPT , |  | (97)

 | bi⁢jlbj⁢il=e−β⁢(Ei−Ej)+β⁢μl⁢(Nil−Njl),superscriptsubscript𝑏𝑖𝑗𝑙superscriptsubscript𝑏𝑗𝑖𝑙superscript𝑒𝛽subscript𝐸𝑖subscript𝐸𝑗𝛽subscript𝜇𝑙superscriptsubscript𝑁𝑖𝑙superscriptsubscript𝑁𝑗𝑙\frac{b_{ij}^{l}}{b_{ji}^{l}}=e^{-\beta(E_{i}-E_{j})+\beta\mu_{l}(N_{i}^{l}-N_%
{j}^{l})},divide start_ARG italic_b start_POSTSUBSCRIPT italic_i italic_j end_POSTSUBSCRIPT start_POSTSUPERSCRIPT italic_l end_POSTSUPERSCRIPT end_ARG start_ARG italic_b start_POSTSUBSCRIPT italic_j italic_i end_POSTSUBSCRIPT start_POSTSUPERSCRIPT italic_l end_POSTSUPERSCRIPT end_ARG = italic_e start_POSTSUPERSCRIPT - italic_β ( italic_E start_POSTSUBSCRIPT italic_i end_POSTSUBSCRIPT - italic_E start_POSTSUBSCRIPT italic_j end_POSTSUBSCRIPT ) + italic_β italic_μ start_POSTSUBSCRIPT italic_l end_POSTSUBSCRIPT ( italic_N start_POSTSUBSCRIPT italic_i end_POSTSUBSCRIPT start_POSTSUPERSCRIPT italic_l end_POSTSUPERSCRIPT - italic_N start_POSTSUBSCRIPT italic_j end_POSTSUBSCRIPT start_POSTSUPERSCRIPT italic_l end_POSTSUPERSCRIPT ) end_POSTSUPERSCRIPT , |  | (98)

and we remark that these relations are not the condition of
detailed balance, which means that the stationary state may
be a nonequilibrium stationary state, although the temperatures
of the reservoirs are all the same. The detailed balance
condition is satisfied if the transitions determined by the
rate ai⁢jsubscript𝑎𝑖𝑗a_{ij}italic_a start_POSTSUBSCRIPT italic_i italic_j end_POSTSUBSCRIPT connects states i𝑖iitalic_i and j𝑗jitalic_j such that the
external potential are equal, Vi=Vjsubscript𝑉𝑖subscript𝑉𝑗V_{i}=V_{j}italic_V start_POSTSUBSCRIPT italic_i end_POSTSUBSCRIPT = italic_V start_POSTSUBSCRIPT italic_j end_POSTSUBSCRIPT. In this case we
see that both ratios (97) and (98)
are the same and given by pie/pjesubscriptsuperscript𝑝e𝑖subscriptsuperscript𝑝e𝑗p^{\rm e}_{i}/p^{\rm e}_{j}italic_p start_POSTSUPERSCRIPT roman_e end_POSTSUPERSCRIPT start_POSTSUBSCRIPT italic_i end_POSTSUBSCRIPT / italic_p start_POSTSUPERSCRIPT roman_e end_POSTSUPERSCRIPT start_POSTSUBSCRIPT italic_j end_POSTSUBSCRIPT where
piesuperscriptsubscript𝑝𝑖ep_{i}^{\rm e}italic_p start_POSTSUBSCRIPT italic_i end_POSTSUBSCRIPT start_POSTSUPERSCRIPT roman_e end_POSTSUPERSCRIPT is proportional to e−β⁢(Ei+Vi)superscript𝑒𝛽subscript𝐸𝑖subscript𝑉𝑖e^{-\beta(E_{i}+V_{i})}italic_e start_POSTSUPERSCRIPT - italic_β ( italic_E start_POSTSUBSCRIPT italic_i end_POSTSUBSCRIPT + italic_V start_POSTSUBSCRIPT italic_i end_POSTSUBSCRIPT ) end_POSTSUPERSCRIPT
and understood as the equilibrium probability distribution.
In the case of a chemical system, the condition Vi=Vjsubscript𝑉𝑖subscript𝑉𝑗V_{i}=V_{j}italic_V start_POSTSUBSCRIPT italic_i end_POSTSUBSCRIPT = italic_V start_POSTSUBSCRIPT italic_j end_POSTSUBSCRIPT
for internal processes leads to the well known relation
between the chemical potentials of and the stoichiometric
coefficientsexpressing the equilibrium condition
tome2018 .

In the present case, the total transition rate is written as

 | wi⁢j=ai⁢j+∑lbi⁢jl,subscript𝑤𝑖𝑗subscript𝑎𝑖𝑗subscript𝑙superscriptsubscript𝑏𝑖𝑗𝑙w_{ij}=a_{ij}+\sum_{l}b_{ij}^{l},italic_w start_POSTSUBSCRIPT italic_i italic_j end_POSTSUBSCRIPT = italic_a start_POSTSUBSCRIPT italic_i italic_j end_POSTSUBSCRIPT + ∑ start_POSTSUBSCRIPT italic_l end_POSTSUBSCRIPT italic_b start_POSTSUBSCRIPT italic_i italic_j end_POSTSUBSCRIPT start_POSTSUPERSCRIPT italic_l end_POSTSUPERSCRIPT , |  | (99)

and again, given i𝑖iitalic_i and j𝑗jitalic_j only one term on the right-hand
side of this equation can be nonzero. The time variation of
the energy is

 | d⁢Ud⁢t=Φu,𝑑𝑈𝑑𝑡subscriptΦu\frac{dU}{dt}=\Phi_{\rm u},divide start_ARG italic_d italic_U end_ARG start_ARG italic_d italic_t end_ARG = roman_Φ start_POSTSUBSCRIPT roman_u end_POSTSUBSCRIPT , |  | (100)

where

 | Φu=∑i⁢j(Ei−Ej)⁢wi⁢j.subscriptΦusubscript𝑖𝑗subscript𝐸𝑖subscript𝐸𝑗subscript𝑤𝑖𝑗\Phi_{\rm u}=\sum_{ij}(E_{i}-E_{j})w_{ij}.roman_Φ start_POSTSUBSCRIPT roman_u end_POSTSUBSCRIPT = ∑ start_POSTSUBSCRIPT italic_i italic_j end_POSTSUBSCRIPT ( italic_E start_POSTSUBSCRIPT italic_i end_POSTSUBSCRIPT - italic_E start_POSTSUBSCRIPT italic_j end_POSTSUBSCRIPT ) italic_w start_POSTSUBSCRIPT italic_i italic_j end_POSTSUBSCRIPT . |  | (101)

The flux of heat is

 | Φ=Φu−∑lμl⁢χl.ΦsubscriptΦusubscript𝑙subscript𝜇𝑙subscript𝜒𝑙\Phi=\Phi_{\rm u}-\sum_{l}\mu_{l}\chi_{l}.roman_Φ = roman_Φ start_POSTSUBSCRIPT roman_u end_POSTSUBSCRIPT - ∑ start_POSTSUBSCRIPT italic_l end_POSTSUBSCRIPT italic_μ start_POSTSUBSCRIPT italic_l end_POSTSUBSCRIPT italic_χ start_POSTSUBSCRIPT italic_l end_POSTSUBSCRIPT . |  | (102)

From the formula (54) for the entropy flux
and using (97) and (98), we obtain

 | Ψ=−1T⁢∑i⁢jwi⁢j⁢pj⁢(Ei−Ej)Ψ1𝑇subscript𝑖𝑗subscript𝑤𝑖𝑗subscript𝑝𝑗subscript𝐸𝑖subscript𝐸𝑗\Psi=-\frac{1}{T}\sum_{ij}w_{ij}p_{j}(E_{i}-E_{j})roman_Ψ = - divide start_ARG 1 end_ARG start_ARG italic_T end_ARG ∑ start_POSTSUBSCRIPT italic_i italic_j end_POSTSUBSCRIPT italic_w start_POSTSUBSCRIPT italic_i italic_j end_POSTSUBSCRIPT italic_p start_POSTSUBSCRIPT italic_j end_POSTSUBSCRIPT ( italic_E start_POSTSUBSCRIPT italic_i end_POSTSUBSCRIPT - italic_E start_POSTSUBSCRIPT italic_j end_POSTSUBSCRIPT ) | 

 | +1T∑lμl∑i⁢jbi⁢jlpj(Nil−Njl)],+\frac{1}{T}\sum_{l}\mu_{l}\sum_{ij}b_{ij}^{l}p_{j}(N_{i}^{l}-N_{j}^{l})],+ divide start_ARG 1 end_ARG start_ARG italic_T end_ARG ∑ start_POSTSUBSCRIPT italic_l end_POSTSUBSCRIPT italic_μ start_POSTSUBSCRIPT italic_l end_POSTSUBSCRIPT ∑ start_POSTSUBSCRIPT italic_i italic_j end_POSTSUBSCRIPT italic_b start_POSTSUBSCRIPT italic_i italic_j end_POSTSUBSCRIPT start_POSTSUPERSCRIPT italic_l end_POSTSUPERSCRIPT italic_p start_POSTSUBSCRIPT italic_j end_POSTSUBSCRIPT ( italic_N start_POSTSUBSCRIPT italic_i end_POSTSUBSCRIPT start_POSTSUPERSCRIPT italic_l end_POSTSUPERSCRIPT - italic_N start_POSTSUBSCRIPT italic_j end_POSTSUBSCRIPT start_POSTSUPERSCRIPT italic_l end_POSTSUPERSCRIPT ) ] , |  | (103)

which can be written as

 | Ψ=−1T⁢Φu+1T⁢∑lμl⁢χl.Ψ1𝑇subscriptΦu1𝑇subscript𝑙subscript𝜇𝑙subscript𝜒𝑙\Psi=-\frac{1}{T}\Phi_{\rm u}+\frac{1}{T}\sum_{l}\mu_{l}\chi_{l}.roman_Ψ = - divide start_ARG 1 end_ARG start_ARG italic_T end_ARG roman_Φ start_POSTSUBSCRIPT roman_u end_POSTSUBSCRIPT + divide start_ARG 1 end_ARG start_ARG italic_T end_ARG ∑ start_POSTSUBSCRIPT italic_l end_POSTSUBSCRIPT italic_μ start_POSTSUBSCRIPT italic_l end_POSTSUBSCRIPT italic_χ start_POSTSUBSCRIPT italic_l end_POSTSUBSCRIPT . |  | (104)

Before we proceed to determine other quantities of interest,
we observe that the entropy flux at the stationary state is

 | Ψ0=−1T⁢Φu0+1T⁢∑lμl⁢χl0.subscriptΨ01𝑇superscriptsubscriptΦu01𝑇subscript𝑙subscript𝜇𝑙superscriptsubscript𝜒𝑙0\Psi_{0}=-\frac{1}{T}\Phi_{\rm u}^{0}+\frac{1}{T}\sum_{l}\mu_{l}\chi_{l}^{0}.roman_Ψ start_POSTSUBSCRIPT 0 end_POSTSUBSCRIPT = - divide start_ARG 1 end_ARG start_ARG italic_T end_ARG roman_Φ start_POSTSUBSCRIPT roman_u end_POSTSUBSCRIPT start_POSTSUPERSCRIPT 0 end_POSTSUPERSCRIPT + divide start_ARG 1 end_ARG start_ARG italic_T end_ARG ∑ start_POSTSUBSCRIPT italic_l end_POSTSUBSCRIPT italic_μ start_POSTSUBSCRIPT italic_l end_POSTSUBSCRIPT italic_χ start_POSTSUBSCRIPT italic_l end_POSTSUBSCRIPT start_POSTSUPERSCRIPT 0 end_POSTSUPERSCRIPT . |  | (105)

Subtracting these two equations, we find

 | Ψ−Ψ0=−1T⁢(Φu−Φu0)+1T⁢∑lμl⁢(χl−χl0).ΨsubscriptΨ01𝑇subscriptΦusuperscriptsubscriptΦu01𝑇subscript𝑙subscript𝜇𝑙subscript𝜒𝑙superscriptsubscript𝜒𝑙0\Psi-\Psi_{0}=-\frac{1}{T}(\Phi_{\rm u}-\Phi_{\rm u}^{0})+\frac{1}{T}\sum_{l}%
\mu_{l}(\chi_{l}-\chi_{l}^{0}).roman_Ψ - roman_Ψ start_POSTSUBSCRIPT 0 end_POSTSUBSCRIPT = - divide start_ARG 1 end_ARG start_ARG italic_T end_ARG ( roman_Φ start_POSTSUBSCRIPT roman_u end_POSTSUBSCRIPT - roman_Φ start_POSTSUBSCRIPT roman_u end_POSTSUBSCRIPT start_POSTSUPERSCRIPT 0 end_POSTSUPERSCRIPT ) + divide start_ARG 1 end_ARG start_ARG italic_T end_ARG ∑ start_POSTSUBSCRIPT italic_l end_POSTSUBSCRIPT italic_μ start_POSTSUBSCRIPT italic_l end_POSTSUBSCRIPT ( italic_χ start_POSTSUBSCRIPT italic_l end_POSTSUBSCRIPT - italic_χ start_POSTSUBSCRIPT italic_l end_POSTSUBSCRIPT start_POSTSUPERSCRIPT 0 end_POSTSUPERSCRIPT ) . |  | (106)

The excess entropy production 𝒫excsubscript𝒫exc{\cal P}_{\rm exc}caligraphic_P start_POSTSUBSCRIPT roman_exc end_POSTSUBSCRIPT is given by
(81), and is

 | 𝒫exc=𝒫−∑jCj0⁢(pj−qj)−𝒫0,subscript𝒫exc𝒫subscript𝑗superscriptsubscript𝐶𝑗0subscript𝑝𝑗subscript𝑞𝑗subscript𝒫0{\cal P}_{\rm exc}={\cal P}-\sum_{j}C_{j}^{0}(p_{j}-q_{j})-{\cal P}_{0},caligraphic_P start_POSTSUBSCRIPT roman_exc end_POSTSUBSCRIPT = caligraphic_P - ∑ start_POSTSUBSCRIPT italic_j end_POSTSUBSCRIPT italic_C start_POSTSUBSCRIPT italic_j end_POSTSUBSCRIPT start_POSTSUPERSCRIPT 0 end_POSTSUPERSCRIPT ( italic_p start_POSTSUBSCRIPT italic_j end_POSTSUBSCRIPT - italic_q start_POSTSUBSCRIPT italic_j end_POSTSUBSCRIPT ) - caligraphic_P start_POSTSUBSCRIPT 0 end_POSTSUBSCRIPT , |  | (107)

which we write as

 | 𝒫exc=d⁢Sd⁢t+Ψ−∑jCj0⁢(pj−qj)−Ψ0,subscript𝒫exc𝑑𝑆𝑑𝑡Ψsubscript𝑗superscriptsubscript𝐶𝑗0subscript𝑝𝑗subscript𝑞𝑗subscriptΨ0{\cal P}_{\rm exc}=\frac{dS}{dt}+\Psi-\sum_{j}C_{j}^{0}(p_{j}-q_{j})-\Psi_{0},caligraphic_P start_POSTSUBSCRIPT roman_exc end_POSTSUBSCRIPT = divide start_ARG italic_d italic_S end_ARG start_ARG italic_d italic_t end_ARG + roman_Ψ - ∑ start_POSTSUBSCRIPT italic_j end_POSTSUBSCRIPT italic_C start_POSTSUBSCRIPT italic_j end_POSTSUBSCRIPT start_POSTSUPERSCRIPT 0 end_POSTSUPERSCRIPT ( italic_p start_POSTSUBSCRIPT italic_j end_POSTSUBSCRIPT - italic_q start_POSTSUBSCRIPT italic_j end_POSTSUBSCRIPT ) - roman_Ψ start_POSTSUBSCRIPT 0 end_POSTSUBSCRIPT , |  | (108)

because 𝒫=d⁢S/d⁢t+Ψ𝒫𝑑𝑆𝑑𝑡Ψ{\cal P}=dS/dt+\Psicaligraphic_P = italic_d italic_S / italic_d italic_t + roman_Ψ and 𝒫0=Ψ0subscript𝒫0subscriptΨ0{\cal P}_{0}=\Psi_{0}caligraphic_P start_POSTSUBSCRIPT 0 end_POSTSUBSCRIPT = roman_Ψ start_POSTSUBSCRIPT 0 end_POSTSUBSCRIPT.
Now from the definition of Cjsubscript𝐶𝑗C_{j}italic_C start_POSTSUBSCRIPT italic_j end_POSTSUBSCRIPT given by (78)

 | Cj=x⁢gj+∑lyl⁢hjl,subscript𝐶𝑗𝑥subscript𝑔𝑗subscript𝑙subscript𝑦𝑙superscriptsubscriptℎ𝑗𝑙C_{j}=xg_{j}+\sum_{l}y_{l}h_{j}^{l},italic_C start_POSTSUBSCRIPT italic_j end_POSTSUBSCRIPT = italic_x italic_g start_POSTSUBSCRIPT italic_j end_POSTSUBSCRIPT + ∑ start_POSTSUBSCRIPT italic_l end_POSTSUBSCRIPT italic_y start_POSTSUBSCRIPT italic_l end_POSTSUBSCRIPT italic_h start_POSTSUBSCRIPT italic_j end_POSTSUBSCRIPT start_POSTSUPERSCRIPT italic_l end_POSTSUPERSCRIPT , |  | (109)

where

 | x=∂𝒫∂Φu,yl=∂𝒫∂χl,formulae-sequence𝑥𝒫subscriptΦusubscript𝑦𝑙𝒫subscript𝜒𝑙x=\frac{\partial{\cal P}}{\partial\Phi_{\rm u}},\qquad y_{l}=\frac{\partial{%
\cal P}}{\partial\chi_{l}},italic_x = divide start_ARG ∂ caligraphic_P end_ARG start_ARG ∂ roman_Φ start_POSTSUBSCRIPT roman_u end_POSTSUBSCRIPT end_ARG , italic_y start_POSTSUBSCRIPT italic_l end_POSTSUBSCRIPT = divide start_ARG ∂ caligraphic_P end_ARG start_ARG ∂ italic_χ start_POSTSUBSCRIPT italic_l end_POSTSUBSCRIPT end_ARG , |  | (110)

and

 | gj=∂Φu∂pj,hjl=∂χl∂pj.formulae-sequencesubscript𝑔𝑗subscriptΦusubscript𝑝𝑗superscriptsubscriptℎ𝑗𝑙subscript𝜒𝑙subscript𝑝𝑗g_{j}=\frac{\partial\Phi_{\rm u}}{\partial p_{j}},\qquad h_{j}^{l}=\frac{%
\partial\chi_{l}}{\partial p_{j}}.italic_g start_POSTSUBSCRIPT italic_j end_POSTSUBSCRIPT = divide start_ARG ∂ roman_Φ start_POSTSUBSCRIPT roman_u end_POSTSUBSCRIPT end_ARG start_ARG ∂ italic_p start_POSTSUBSCRIPT italic_j end_POSTSUBSCRIPT end_ARG , italic_h start_POSTSUBSCRIPT italic_j end_POSTSUBSCRIPT start_POSTSUPERSCRIPT italic_l end_POSTSUPERSCRIPT = divide start_ARG ∂ italic_χ start_POSTSUBSCRIPT italic_l end_POSTSUBSCRIPT end_ARG start_ARG ∂ italic_p start_POSTSUBSCRIPT italic_j end_POSTSUBSCRIPT end_ARG . |  | (111)

Considering that ΦusubscriptΦu\Phi_{\rm u}roman_Φ start_POSTSUBSCRIPT roman_u end_POSTSUBSCRIPT and χlsubscript𝜒𝑙\chi_{l}italic_χ start_POSTSUBSCRIPT italic_l end_POSTSUBSCRIPT are linear
in pisubscript𝑝𝑖p_{i}italic_p start_POSTSUBSCRIPT italic_i end_POSTSUBSCRIPT then the coefficients gjsubscript𝑔𝑗g_{j}italic_g start_POSTSUBSCRIPT italic_j end_POSTSUBSCRIPT and hjlsuperscriptsubscriptℎ𝑗𝑙h_{j}^{l}italic_h start_POSTSUBSCRIPT italic_j end_POSTSUBSCRIPT start_POSTSUPERSCRIPT italic_l end_POSTSUPERSCRIPT are
independent of pisubscript𝑝𝑖p_{i}italic_p start_POSTSUBSCRIPT italic_i end_POSTSUBSCRIPT. This property allow us to
write

 | Cj0=x0⁢gj+∑lyl0⁢hjl,superscriptsubscript𝐶𝑗0superscript𝑥0subscript𝑔𝑗subscript𝑙superscriptsubscript𝑦𝑙0superscriptsubscriptℎ𝑗𝑙C_{j}^{0}=x^{0}g_{j}+\sum_{l}y_{l}^{0}h_{j}^{l},italic_C start_POSTSUBSCRIPT italic_j end_POSTSUBSCRIPT start_POSTSUPERSCRIPT 0 end_POSTSUPERSCRIPT = italic_x start_POSTSUPERSCRIPT 0 end_POSTSUPERSCRIPT italic_g start_POSTSUBSCRIPT italic_j end_POSTSUBSCRIPT + ∑ start_POSTSUBSCRIPT italic_l end_POSTSUBSCRIPT italic_y start_POSTSUBSCRIPT italic_l end_POSTSUBSCRIPT start_POSTSUPERSCRIPT 0 end_POSTSUPERSCRIPT italic_h start_POSTSUBSCRIPT italic_j end_POSTSUBSCRIPT start_POSTSUPERSCRIPT italic_l end_POSTSUPERSCRIPT , |  | (112)

where x0superscript𝑥0x^{0}italic_x start_POSTSUPERSCRIPT 0 end_POSTSUPERSCRIPT and yl0superscriptsubscript𝑦𝑙0y_{l}^{0}italic_y start_POSTSUBSCRIPT italic_l end_POSTSUBSCRIPT start_POSTSUPERSCRIPT 0 end_POSTSUPERSCRIPT are the values of x𝑥xitalic_x and ylsubscript𝑦𝑙y_{l}italic_y start_POSTSUBSCRIPT italic_l end_POSTSUBSCRIPT
at the stationary state, that is, when pi=qisubscript𝑝𝑖subscript𝑞𝑖p_{i}=q_{i}italic_p start_POSTSUBSCRIPT italic_i end_POSTSUBSCRIPT = italic_q start_POSTSUBSCRIPT italic_i end_POSTSUBSCRIPT.
Using the linear property we may also write

 | Φu=∑jgj⁢pj,subscriptΦusubscript𝑗subscript𝑔𝑗subscript𝑝𝑗\Phi_{\rm u}=\sum_{j}g_{j}p_{j},roman_Φ start_POSTSUBSCRIPT roman_u end_POSTSUBSCRIPT = ∑ start_POSTSUBSCRIPT italic_j end_POSTSUBSCRIPT italic_g start_POSTSUBSCRIPT italic_j end_POSTSUBSCRIPT italic_p start_POSTSUBSCRIPT italic_j end_POSTSUBSCRIPT , |  | (113)

 | χl=∑jhjl⁢pj,subscript𝜒𝑙subscript𝑗superscriptsubscriptℎ𝑗𝑙subscript𝑝𝑗\chi_{l}=\sum_{j}h_{j}^{l}p_{j},italic_χ start_POSTSUBSCRIPT italic_l end_POSTSUBSCRIPT = ∑ start_POSTSUBSCRIPT italic_j end_POSTSUBSCRIPT italic_h start_POSTSUBSCRIPT italic_j end_POSTSUBSCRIPT start_POSTSUPERSCRIPT italic_l end_POSTSUPERSCRIPT italic_p start_POSTSUBSCRIPT italic_j end_POSTSUBSCRIPT , |  | (114)

which lead us to the following conclusion

 | ∑jCj0⁢(pj−qj)=x0⁢(Φu−Φu0)+∑lyl0⁢(χl−χl0).subscript𝑗superscriptsubscript𝐶𝑗0subscript𝑝𝑗subscript𝑞𝑗superscript𝑥0subscriptΦusuperscriptsubscriptΦu0subscript𝑙superscriptsubscript𝑦𝑙0subscript𝜒𝑙superscriptsubscript𝜒𝑙0\sum_{j}C_{j}^{0}(p_{j}-q_{j})=x^{0}(\Phi_{\rm u}-\Phi_{\rm u}^{0})+\sum_{l}y_%
{l}^{0}(\chi_{l}-\chi_{l}^{0}).∑ start_POSTSUBSCRIPT italic_j end_POSTSUBSCRIPT italic_C start_POSTSUBSCRIPT italic_j end_POSTSUBSCRIPT start_POSTSUPERSCRIPT 0 end_POSTSUPERSCRIPT ( italic_p start_POSTSUBSCRIPT italic_j end_POSTSUBSCRIPT - italic_q start_POSTSUBSCRIPT italic_j end_POSTSUBSCRIPT ) = italic_x start_POSTSUPERSCRIPT 0 end_POSTSUPERSCRIPT ( roman_Φ start_POSTSUBSCRIPT roman_u end_POSTSUBSCRIPT - roman_Φ start_POSTSUBSCRIPT roman_u end_POSTSUBSCRIPT start_POSTSUPERSCRIPT 0 end_POSTSUPERSCRIPT ) + ∑ start_POSTSUBSCRIPT italic_l end_POSTSUBSCRIPT italic_y start_POSTSUBSCRIPT italic_l end_POSTSUBSCRIPT start_POSTSUPERSCRIPT 0 end_POSTSUPERSCRIPT ( italic_χ start_POSTSUBSCRIPT italic_l end_POSTSUBSCRIPT - italic_χ start_POSTSUBSCRIPT italic_l end_POSTSUBSCRIPT start_POSTSUPERSCRIPT 0 end_POSTSUPERSCRIPT ) . |  | (115)

From this last result and from the expression (106)
for Ψ−Ψ0ΨsubscriptΨ0\Psi-\Psi_{0}roman_Ψ - roman_Ψ start_POSTSUBSCRIPT 0 end_POSTSUBSCRIPT we reach the following expression
for the excess entropy production

 | 𝒫exc=d⁢Sd⁢t−rT⁢(Φu−Φu0)+∑lαlT⁢(χl−χl0),subscript𝒫exc𝑑𝑆𝑑𝑡𝑟𝑇subscriptΦusuperscriptsubscriptΦu0subscript𝑙subscript𝛼𝑙𝑇subscript𝜒𝑙superscriptsubscript𝜒𝑙0{\cal P}_{\rm exc}=\frac{dS}{dt}-\frac{r}{T}(\Phi_{\rm u}-\Phi_{\rm u}^{0})+%
\sum_{l}\frac{\alpha_{l}}{T}(\chi_{l}-\chi_{l}^{0}),caligraphic_P start_POSTSUBSCRIPT roman_exc end_POSTSUBSCRIPT = divide start_ARG italic_d italic_S end_ARG start_ARG italic_d italic_t end_ARG - divide start_ARG italic_r end_ARG start_ARG italic_T end_ARG ( roman_Φ start_POSTSUBSCRIPT roman_u end_POSTSUBSCRIPT - roman_Φ start_POSTSUBSCRIPT roman_u end_POSTSUBSCRIPT start_POSTSUPERSCRIPT 0 end_POSTSUPERSCRIPT ) + ∑ start_POSTSUBSCRIPT italic_l end_POSTSUBSCRIPT divide start_ARG italic_α start_POSTSUBSCRIPT italic_l end_POSTSUBSCRIPT end_ARG start_ARG italic_T end_ARG ( italic_χ start_POSTSUBSCRIPT italic_l end_POSTSUBSCRIPT - italic_χ start_POSTSUBSCRIPT italic_l end_POSTSUBSCRIPT start_POSTSUPERSCRIPT 0 end_POSTSUPERSCRIPT ) , |  | (116)

where we are using the abbreviations

 | rT=1T+x0αlT=μlT−yl0.formulae-sequence𝑟𝑇1𝑇superscript𝑥0subscript𝛼𝑙𝑇subscript𝜇𝑙𝑇superscriptsubscript𝑦𝑙0\frac{r}{T}=\frac{1}{T}+x^{0}\qquad\frac{\alpha_{l}}{T}=\frac{\mu_{l}}{T}-y_{l%
}^{0}.divide start_ARG italic_r end_ARG start_ARG italic_T end_ARG = divide start_ARG 1 end_ARG start_ARG italic_T end_ARG + italic_x start_POSTSUPERSCRIPT 0 end_POSTSUPERSCRIPT divide start_ARG italic_α start_POSTSUBSCRIPT italic_l end_POSTSUBSCRIPT end_ARG start_ARG italic_T end_ARG = divide start_ARG italic_μ start_POSTSUBSCRIPT italic_l end_POSTSUBSCRIPT end_ARG start_ARG italic_T end_ARG - italic_y start_POSTSUBSCRIPT italic_l end_POSTSUBSCRIPT start_POSTSUPERSCRIPT 0 end_POSTSUPERSCRIPT . |  | (117)

In view of equations (100) and (43),
it can be written as

 | 𝒫exc=d⁢Sd⁢t−rT⁢d⁢Ud⁢t+∑lαlT⁢d⁢Nld⁢t,subscript𝒫exc𝑑𝑆𝑑𝑡𝑟𝑇𝑑𝑈𝑑𝑡subscript𝑙subscript𝛼𝑙𝑇𝑑subscript𝑁𝑙𝑑𝑡{\cal P}_{\rm exc}=\frac{dS}{dt}-\frac{r}{T}\frac{dU}{dt}+\sum_{l}\frac{\alpha%
_{l}}{T}\frac{dN_{l}}{dt},caligraphic_P start_POSTSUBSCRIPT roman_exc end_POSTSUBSCRIPT = divide start_ARG italic_d italic_S end_ARG start_ARG italic_d italic_t end_ARG - divide start_ARG italic_r end_ARG start_ARG italic_T end_ARG divide start_ARG italic_d italic_U end_ARG start_ARG italic_d italic_t end_ARG + ∑ start_POSTSUBSCRIPT italic_l end_POSTSUBSCRIPT divide start_ARG italic_α start_POSTSUBSCRIPT italic_l end_POSTSUBSCRIPT end_ARG start_ARG italic_T end_ARG divide start_ARG italic_d italic_N start_POSTSUBSCRIPT italic_l end_POSTSUBSCRIPT end_ARG start_ARG italic_d italic_t end_ARG , |  | (118)

bearing in mind that Φu0=0superscriptsubscriptΦu00\Phi_{\rm u}^{0}=0roman_Φ start_POSTSUBSCRIPT roman_u end_POSTSUBSCRIPT start_POSTSUPERSCRIPT 0 end_POSTSUPERSCRIPT = 0 and that
Rl=−χl0subscript𝑅𝑙superscriptsubscript𝜒𝑙0R_{l}=-\chi_{l}^{0}italic_R start_POSTSUBSCRIPT italic_l end_POSTSUBSCRIPT = - italic_χ start_POSTSUBSCRIPT italic_l end_POSTSUBSCRIPT start_POSTSUPERSCRIPT 0 end_POSTSUPERSCRIPT.

This last equality allows us to write

 | 𝒫exc=d⁢Md⁢t,subscript𝒫exc𝑑𝑀𝑑𝑡{\cal P}_{\rm exc}=\frac{dM}{dt},caligraphic_P start_POSTSUBSCRIPT roman_exc end_POSTSUBSCRIPT = divide start_ARG italic_d italic_M end_ARG start_ARG italic_d italic_t end_ARG , |  | (119)

where

 | M=S−rT⁢U+∑lαlT⁢Nl+K,𝑀𝑆𝑟𝑇𝑈subscript𝑙subscript𝛼𝑙𝑇subscript𝑁𝑙𝐾M=S-\frac{r}{T}U+\sum_{l}\frac{\alpha_{l}}{T}N_{l}+K,italic_M = italic_S - divide start_ARG italic_r end_ARG start_ARG italic_T end_ARG italic_U + ∑ start_POSTSUBSCRIPT italic_l end_POSTSUBSCRIPT divide start_ARG italic_α start_POSTSUBSCRIPT italic_l end_POSTSUBSCRIPT end_ARG start_ARG italic_T end_ARG italic_N start_POSTSUBSCRIPT italic_l end_POSTSUBSCRIPT + italic_K , |  | (120)

where K𝐾Kitalic_K is a constant.
That is, the excess entropy production is the
time derivative of M𝑀Mitalic_M which is a linear combination
of S𝑆Sitalic_S and U𝑈Uitalic_U, and the complementary variables Nlsubscript𝑁𝑙N_{l}italic_N start_POSTSUBSCRIPT italic_l end_POSTSUBSCRIPT,
and can then be understood as a thermodynamic potential.

Comparing equations (119) and (91), we see
that M𝑀Mitalic_M and L𝐿Litalic_L differ by a constant. Since L𝐿Litalic_L
vanishes at the stationary state, we conclude that
L=M−M0𝐿𝑀subscript𝑀0L=M-M_{0}italic_L = italic_M - italic_M start_POSTSUBSCRIPT 0 end_POSTSUBSCRIPT where M0subscript𝑀0M_{0}italic_M start_POSTSUBSCRIPT 0 end_POSTSUBSCRIPT is the value of M𝑀Mitalic_M at the
stationary state. Since L≥0𝐿0L\geq 0italic_L ≥ 0 then

 | M≥M0,𝑀subscript𝑀0M\geq M_{0},italic_M ≥ italic_M start_POSTSUBSCRIPT 0 end_POSTSUBSCRIPT , |  | (121)

and M−M0𝑀subscript𝑀0M-M_{0}italic_M - italic_M start_POSTSUBSCRIPT 0 end_POSTSUBSCRIPT can be understood
as a Lyapunov function in relation to the fluxes
because S𝑆Sitalic_S, U𝑈Uitalic_U, and Nlsubscript𝑁𝑙N_{l}italic_N start_POSTSUBSCRIPT italic_l end_POSTSUBSCRIPT are functions of the
fluxes.

## V Conclusion

We have derived the main equations of irreversible thermodynamics
from stochastic thermodynamics including the Glansdorff-Prigogine
extremal principle. To this end we used a Master equation
defined through transition rates that represent the various
processes that are induced by gradients of temperature and
external forces. The production of entropy was shown to be
an upward convex function of the probabilities of the microstates.
Considering that the fluxes are linear in these
probabilities we showed that the entropy production can
also be understood as a convex function of the fluxes.
The convexity property is then used to show that
the excess entropy production is a minimum at the stationary
state, which is a statement of the Glansdorff-Prigogine
principle.

The stability of the stationary state can be analyzed by
thinking of the master equation as a set of ordinary
differential equations in the variables pisubscript𝑝𝑖p_{i}italic_p start_POSTSUBSCRIPT italic_i end_POSTSUBSCRIPT and by
the construction of a Lyapunov function in the variables pisubscript𝑝𝑖p_{i}italic_p start_POSTSUBSCRIPT italic_i end_POSTSUBSCRIPT.
We have construct such an equation and showed that
its time derivative is equal to the excess entropy production.
The question we have raised is whether we can construct
a Lyapunov function in relation to the macroscopic variables
S𝑆Sitalic_S, U𝑈Uitalic_U and Nlsubscript𝑁𝑙N_{l}italic_N start_POSTSUBSCRIPT italic_l end_POSTSUBSCRIPT, that is, a Lyapunov function associated
to the set of ordinary differential equations in these variables.
We have shown that this is possible when the temperatures
associated to the various transitions
rates are the same, that is, when no gradients of temperature
are present. In this case the Lyapunov function is a thermodynamic
potential in the sense that it is a linear combination of
S𝑆Sitalic_S, U𝑈Uitalic_U and Nlsubscript𝑁𝑙N_{l}italic_N start_POSTSUBSCRIPT italic_l end_POSTSUBSCRIPT.

## References

* (1)
 I. Prigogine,
Étude Thermodynamique des Phénoménes Irréversibles,
Desoer, Liége, 1947.
* (2)
 K. G. Denbigh,
Thermodynamics of the Steady State
Methuen, London 1951.
* (3)
 I. Prigogine,
Thermodynamics of Irreversible Processes,
Charles C. Thomas, Springfield, 1955.
* (4)
 J. Meixner und H. G. Reik,
”Thermodynamik der irreversible Prozesse”, in
S. Flügge (ed.), Encyclopedia of Physics,
Springer-Verlag, Berlin, 1959, vol. III/2, p. 413.
* (5)
 S. R. de Groot and P. Mazur,
Non-Equilibrium Thermodynamics,
North Holland, Amsterdam, 1962.
* (6)
 D. D. Fitts,
Nonequilibrium Thermodynamics:
A Phenomenological Theory of Irreversible Processes in Fluid Systems,
McGraw-Hill, New York, 1962.
* (7)
 P. Glansdorff and I. Prigogine,
Thermodynamic Theory of Structure, Stability and Fluctuations,
Wiley, London, 1971.
* (8)
 G. Nicolis and I. Prigogine,
Self-Organization in Nonequilibrium Systems,
Wiley, New York, 1977.
* (9)
 D. Kondepudi and I. Prigogine,
Modern Thermodynamics, From Heat Engines to Dissipative
Structures, Wiley, New York, 1998.
* (10)
 G. Lebon, D. Jou, and J. Casas-Vázquez,
Understanding non-equilibrium thermodynamics,
Springer, 2008.
* (11)
 I. Prigogine,
Bulletin de la Classe des Sciences, Academie Royale de Belgique,
31, 600 (1945).
* (12)
 L. Onsager,
Phys. Rev. 37, 405; 38, 2265 (1931).
* (13)
 P. Glansdorff and I. Prigogine,
Physica 20, 773 (1954).
* (14)
 T Tomé,
Braz. J. Phys. 36, 1285 (2006).
* (15)
 T. Tomé and M. J. de Oliveira,
Phys. Rev. E 82, 021120 (2010).
* (16)
 M. Esposito and C. Van den Broeck,
Phys. Rev. 82, 011143 (2010).
* (17)
 C. Van den Broeck and M. Esposito,
Phys. Rev. 82, 011144 (2010).
* (18)

C. Van den Broeck,
”Stochastic thermodynamics: A brief introduction”,
in C. Bechinger, F. Sciortino, and
P. Ziherl (eds.), Proceedings of the International School
of Physics Enrico Fermi, Course 184, IOS, Amsterdam,
2013; p. 155.
* (19)
 T. Tomé and M. J. de Oliveira,
Phys. Rev. E 91, 042140 (2015).
* (20)
 F. Schlögl,
Z. Phyzik 243, 303 (1971).
* (21)
 H. Tomita,
Prog. Theor. Phys. 47, 1052 (1972).
* (22)
 L. de Sobrino,
J. Theor. Biol. 54, 323 (1975).
* (23)
 J. Schnakenberg,
Rev. Mod. Phys. 48, 571 (1976).
* (24)
 L. Jiu-li, C. Van den Broeck, and G. Nicolis,
Z. Phys. B 56, 165 (1984).
* (25)
 C. Maes and K. Netočný,
J. Stat. Phys. 159, 1286 (2015).
* (26)
 S. Ito,
J. Phys. A 55, 054001 (2022).
* (27)
 T. Tomé and M. J. de Oliveira,
J. Chem. Phys. 148, 224104 (2018).