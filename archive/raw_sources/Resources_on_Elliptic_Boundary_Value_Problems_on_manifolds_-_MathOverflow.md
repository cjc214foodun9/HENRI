# Resources on Elliptic Boundary Value Problems on manifolds - MathOverflow
Source URL: https://mathoverflow.net/questions/284701/resources-on-elliptic-boundary-value-problems-on-manifolds

Resources on Elliptic Boundary Value Problems on manifolds
Ask Question
Asked 8 years, 7 months ago
Modified 8 years, 1 month ago
Viewed 617 times
3

My situation: I am currently trying to understand Uhlenbecks results on the Yang Mills equation. One of the most common notions in this paper is that of an elliptic system or an elliptic boundary value problem. However, as far as my knowledge goes I only know ellipticity as a condition on the leading coefficients of a second order linear differential operator. Or, a bit more general, on pseudodifferential operators acting on 
R
n
𝑅
𝑛
. I already asked another question on principal symbols for non-linear operators, which (partly) explains what ellipticity should mean for these. Nevertheless, I do not see, what role the boundary plays in the definition of an elliptic BVP.

Question: Are there any resources which define the terminology of an elliptic boundary value problem for non-linear operators of arbitrary order? Best, even on general domains, such as manifolds with boundary?

Remarks: There are some references which touch the kind of exposition I am looking for: I found this set of lecture notes on Linear Analysis on Manifolds which comes close to what I am looking for, but leaves out non-linear problems as well as boundary conditions. Furthermore "Multiple Integrals in the Calculus of Variations" by Morrey Jr. seems to be an often cited reference as well. And last but not least, the work of Agmon, Douglas an Nirenberg seems to be quite foundational.

reference-requestap.analysis-of-pdeselliptic-pdegeometric-analysisonline-resources
Share
Cite
Improve this question
Follow
edited Jan 28, 2018 at 8:22
Martin Sleziak
asked Oct 29, 2017 at 12:06
Peter
1
The term "elliptic" in "elliptic BVP" usually refers principally to the operator - whenever you're trying to solve 
F(u)=0
𝐹
(
𝑢
)
=
0
 where 
F
𝐹
 is (nonlinear) elliptic and 
u
𝑢
 is constrained somehow on the boundary of the domain, you could call this an elliptic boundary value problem. Perhaps some authors would also require the boundary condition to make the problem well-posed to call it elliptic, or that the linearized operator 
A:D(A)⊂X→X
𝐴
:
𝐷
(
𝐴
)
⊂
𝑋
→
𝑋
 is elliptic in some functional-analytic sense (e.g. sectorial) when restricted to the domain 
D(A)
𝐷
(
𝐴
)
 defined by the boundary condition. – 
Anthony Carapetis
 
Commented
Oct 30, 2017 at 5:44 
Add a comment
2 Answers
Sorted by:
Highest score (default)
Date modified (newest first)
Date created (oldest first)
3

As an addition to Hadrian's answer, for boundary value problems there are as you mention some issues with the boundary and choosing the correct boundary conditions. As a rule of thumb, people want to have a theorem of the kind "ellipticity implies fredholmness". Thus, for bvps you have to add a notion of ellipticity at the boundary. This is also called the Shapiro-Lopatinskii condition. Roughly speaking you define an operator-valued symbol at the boundary and require that it is invertible.

Reference include: Hörmander 3 (Chapter XX.1) this is somehow the standard viewpoint, "A short introduction to Boutet de Monvel's calculus" by Elmar Schrohe (if you want to consider pseudodifferential boundary value problems), and the book by Egorov and Schulze (they have a readable introduction into elliptic bvp, but most of the book is really tough to read).

Share
Cite
Improve this answer
Follow
edited Apr 28, 2018 at 13:16
answered Apr 28, 2018 at 10:58
mcd
Add a comment
1

In your other question, I believe the response you got was a reasonable place to start: the principal symbol of a differential operator makes good (i.e. invariant) sense for a linear operator.

The lecture notes by Albin which you reference in your question have a wonderful explanation of the definition of symbol for a linear differential operator of integral order--you're not restricted to only considering operators of order 2. The definition you wrote down in your other question gives this expression for the symbol: given a function 
f
𝑓
 satisfying 
d
f
p
=ξ∈
T
∗
p
M
𝑑
𝑓
𝑝
=
𝜉
∈
𝑇
𝑝
∗
𝑀
, we define the principal symbol of 
D
𝐷
 at the point 
(p,ξ)∈
T
∗
p
M
(
𝑝
,
𝜉
)
∈
𝑇
𝑝
∗
𝑀
 to be
σ
m
(D)(x,ξ)=
i
m
lim
t→∞
1
t
m
e
−itf
∘D∘
e
itf
𝜎
𝑚
(
𝐷
)
(
𝑥
,
𝜉
)
=
𝑖
𝑚
lim
𝑡
→
∞
1
𝑡
𝑚
𝑒
−
𝑖
𝑡
𝑓
∘
𝐷
∘
𝑒
𝑖
𝑡
𝑓
where this becomes the symbol of second order operator by setting 
m=2
𝑚
=
2
.

This all makes sense for what is hopefully as a general a domain as you could want: any smooth manifold. The symbol defines a function on the cotangent bundle 
T
∗
M
𝑇
∗
𝑀
 of your manifold. If this function is non-vanishing over each point in the cotangent bundle (or invertible more generally), then your linear operator is called elliptic. With this in mind, the definition of ellipticity for a non-linear operator should be that the linearized operator is itself elliptic!

Share
Cite
Improve this answer
Follow
answered Oct 30, 2017 at 3:46
Hadrian Quan
Add a comment
You must log in to answer this question.

Start asking to get answers

Find the answer to your question by asking.

Ask question

Explore related questions

reference-requestap.analysis-of-pdeselliptic-pdegeometric-analysisonline-resources

See similar questions with these tags.

Featured on Meta
Native Ads Coming To Comments
Linked
4
Principal symbol for non-linear differential operators
Related
14
Applications of pseudodifferential operators to PDE
10
Elliptic operator on non compact manifolds with ends of the type 
Ω×(r,∞)×R
Ω
×
(
𝑟
,
∞
)
×
𝑅
2
References for non-zero boundary value problem
1
Sobolev regularity for systems of elliptic boundary value problems
7
Boundary value problems with 
L
2
𝐿
2
 boundary data
6
Boundary regularity for elliptic PDE in Lipschitz domains
 Question feed
By continuing to use this website, you agree Stack Exchange can store cookies on your device and disclose information in accordance with our Cookie Policy. By exiting this window, default cookies will be accepted. To reject cookies, select an option from below.
Necessary cookies only
Customize settings