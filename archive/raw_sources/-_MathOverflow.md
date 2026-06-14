# - MathOverflow
Source URL: https://mathoverflow.net/questions/376187/looking-for-a-reference-on-conformal-mapping-on-bbb-rn

Looking for a reference on conformal mapping on 
R
n
𝑅
𝑛
Ask Question
Asked 5 years, 7 months ago
Modified 5 years, 7 months ago
Viewed 833 times
5

A mapping 
T:
R
n
→
R
n
𝑇
:
𝑅
𝑛
→
𝑅
𝑛
 is said to be conformal if it is bijective and preserves angles, i.e., if 
x,y:[0,1]→
R
n
𝑥
,
𝑦
:
[
0
,
1
]
→
𝑅
𝑛
 are curves with 
x(
t
0
)=y(
t
0
)
𝑥
(
𝑡
0
)
=
𝑦
(
𝑡
0
)
 then

cos(Tx(
t
0
),Ty(
t
0
))=cos(x(
t
0
),y(
t
0
))=
x
′
(
t
0
)⋅
y
′
(
t
0
)
|
x
′
(
t
0
)||
y
′
(
t
0
)|
.
cos
⁡
(
𝑇
𝑥
(
𝑡
0
)
,
𝑇
𝑦
(
𝑡
0
)
)
=
cos
⁡
(
𝑥
(
𝑡
0
)
,
𝑦
(
𝑡
0
)
)
=
𝑥
′
(
𝑡
0
)
⋅
𝑦
′
(
𝑡
0
)
|
𝑥
′
(
𝑡
0
)
|
|
𝑦
′
(
𝑡
0
)
|
.

A typical example of conformal mapping is the inversion 
I:
R
n
→
R
n
𝐼
:
𝑅
𝑛
→
𝑅
𝑛
 
I(x)=
x
|x
|
2
𝐼
(
𝑥
)
=
𝑥
|
𝑥
|
2
, with the convention that 
I(0)=∞
𝐼
(
0
)
=
∞
 and 
I(∞)=0
𝐼
(
∞
)
=
0
.

Trivial examples are rigid motions i.e., a combination of orthogonal group, Scallings or homothety and/or translations.

I am barely looking for a proof or a reference for the following Theorem:

Theorem: Every conformal mapping is the composite of finely many rigid motions and the Inversion mapping.

There are several book complex analysis dealing with the case 
n=2
𝑛
=
2
 on the complex plane.

But I haven't seen any for the higher dimensional situation.

dg.differential-geometryreal-analysismg.metric-geometryconformal-geometry
Share
Cite
Improve this question
Follow
edited Nov 13, 2020 at 8:32
M. Winter
14.8k3
3 gold badges
33
33 silver badges
86
86 bronze badges
asked Nov 11, 2020 at 13:40
Guy Fsone
1,2778
8 silver badges
22
22 bronze badges
Liouville theorem: scholar.google.com/… – 
Guy Fsone
 
Commented
Nov 11, 2020 at 18:13
Add a comment
4 Answers
Sorted by:
Highest score (default)
Date modified (newest first)
Date created (oldest first)
7

My two cents: a proof for 
n=3
𝑛
=
3
 is given explicitly by Dubrovin, Fomenko and Novikov in [1], §15.2 pp. 138-142. The authors explain also how to extend their proof to the case 
n>3
𝑛
>
3
 and leave the details as an exercise.

An addendum

The answer by @Piotr Hajlasz triggered my curiosity and pushed me to go a little bit beyond reference [1], which requires a 
C
4
𝐶
4
 regularity on the conformal map considered ([1], §15.2 p. 138).
According to Caraman ([2] section 3, chapter 2, p. 358), the proof of Liouville's theorem requiring a minimal regularity on the mapping was given first by Reshetnyak in [3]. Reshetnyak assumes the mapping to be of class 
W
1
n
𝑊
𝑛
1
: while the paper is short, the offered proof is highly non trivial.

Reference

[1] Boris A. Dubrovin, Analtoly T. Fomenko and Sergey P. Novikov, Modern geometry - methods and applications. Part I. The geometry of surfaces, transformation groups, and fields, translated by Robert G. Burns. 2nd ed. (English) Graduate Texts in Mathematics, 93, Berlin-Heidelberg-New York: Springer-Verlag, pp. xv+468 (1992), MR1138462, Zbl 0751.53001

[2] Petru Caraman, 
n
𝑛
-Dimensional Quasiconformal (QCF) Mappings, revised, enlarged and translated from the Romanian by the Author (English), Tunbridge Wells, Kent: Abacus Press, pp. 551 (1974), ISBN 0-85626-005-3, MR0357782, Zbl 0342.30015.

[3] Yuriĭ G. Reshetnyak, "Liouville’s theorem on conformal mappings for minimal regularity assumptions", (English, translated from the Russian), Siberian Mathematical Journal 8 (1967), pp. 631-634 (1968), MR0218544, Zbl 0167.36102.

Share
Cite
Improve this answer
Follow
edited Nov 12, 2020 at 22:36
answered Nov 11, 2020 at 18:25
Daniele Tampieri
7,33711
11 gold badges
36
36 silver badges
53
53 bronze badges
Add a comment
9

See the following Wikipedia page: https://en.wikipedia.org/wiki/Liouville%27s_theorem_(conformal_mappings)

Share
Cite
Improve this answer
Follow
answered Nov 11, 2020 at 13:45
Robert Bryant
110k8
8 gold badges
360
360 silver badges
467
467 bronze badges
Add a comment
6

Theorem (Liouville). If 
Ω⊂
R
n
Ω
⊂
𝑅
𝑛
, 
n≥3
𝑛
≥
3
 is open and 
f:Ω→
R
n
𝑓
:
Ω
→
𝑅
𝑛
 is conformal, then 
f
𝑓
 is a Mobius transformation.

While the theorem is true for 
f∈
C
1
𝑓
∈
𝐶
1
, there is no easy proof in that case. Standard proofs assume that 
f∈
C
3
𝑓
∈
𝐶
3
 or even 
f∈
C
4
𝑓
∈
𝐶
4
. The classical and well know proof due to Nevanlinna can be found here (see page 265):

http://www.pitt.edu/~hajlasz/Notatki/Differential_Geometry_1.pdf

Share
Cite
Improve this answer
Follow
answered Nov 11, 2020 at 17:01
Piotr Hajlasz
29.1k5
5 gold badges
94
94 silver badges
202
202 bronze badges
Add a comment
1

Have you tried the first chapter of Riccardo Benedetti Carlo Petronio, Lectures on Hyperbolic Geometry? It contains a proof of Liouville's theorem from which you can easily deduce the result you are looking for.

Share
Cite
Improve this answer
Follow
answered Nov 13, 2020 at 10:17
Romain Gicquaud
1,3338
8 silver badges
19
19 bronze badges
Add a comment
You must log in to answer this question.

Start asking to get answers

Find the answer to your question by asking.

Ask question

Explore related questions

dg.differential-geometryreal-analysismg.metric-geometryconformal-geometry

See similar questions with these tags.

Featured on Meta
Native Ads Coming To Comments
Linked
4
Conformal maps between two given domains
7
On the "Collected Works" of Charles Bradfield Morrey, Jr
Related
20
Finding Constant Curvature Metrics on Surfaces without full power of Uniformization
7
Analogy of Liouville conformal mapping theorem with Mostow rigidity?
8
Uniqueness theorem for conformal mapping
11
Jacobi's elliptic functions and plane sections of a torus
12
Are conformal maps between Riemannian manifolds real-analytic?
7
Can we prove that Schwarz-Christoffel transform works for all polygons, without using the Riemann mapping theorem?
 Question feed
By continuing to use this website, you agree Stack Exchange can store cookies on your device and disclose information in accordance with our Cookie Policy. By exiting this window, default cookies will be accepted. To reject cookies, select an option from below.
Necessary cookies only
Customize settings