# Persistent homology group - Wikipedia
Source URL: https://en.wikipedia.org/wiki/Persistent_homology_group

In persistent homology, a persistent homology group is a multiscale analog of a homology group that captures information about the evolution of topological features across a filtration of spaces. While the ordinary homology group represents nontrivial homology classes of an individual topological space, the persistent homology group tracks only those classes that remain nontrivial across multiple parameters in the underlying filtration. Analogous to the ordinary Betti number, the ranks of the persistent homology groups are known as the persistent Betti numbers. Persistent homology groups were first introduced by Herbert Edelsbrunner, David Letscher, and Afra Zomorodian in a 2002 paper Topological Persistence and Simplification, one of the foundational papers in the fields of persistent homology and topological data analysis,[1]  based largely on the persistence barcodes and the persistence algorithm, that were first described by Serguei Barannikov in the 1994 paper.[2] Since then, the study of persistent homology groups has led to applications in data science,[3] machine learning,[4] materials science,[5] biology,[6][7] and economics.[8]

## Definition

Let 

K

{\displaystyle K}

 be a simplicial complex, and let 

f
:
K
→

R

{\displaystyle f:K\to \mathbb {R} }

 be a real-valued monotonic function. Then for some values 

a

0

<

a

1

<
⋯
<

a

n

∈

R

{\displaystyle a_{0}<a_{1}<\cdots <a_{n}\in \mathbb {R} }

 the sublevel-sets 

K
(
a
)
:=

f

−
1

(
−
∞
,
a
]

{\displaystyle K(a):=f^{-1}(-\infty ,a]}

 yield a sequence of nested subcomplexes 

∅
=

K

0

⊆

K

1

⊆
⋯
⊆

K

n

=
K

{\displaystyle \emptyset =K_{0}\subseteq K_{1}\subseteq \cdots \subseteq K_{n}=K}

 known as a filtration of 

K

{\displaystyle K}

.

Applying 

p

t
h

{\displaystyle p^{th}}

 homology to each complex yields a sequence of homology groups 

0
=

H

p

(

K

0

)
→

H

p

(

K

1

)
→
⋯
→

H

p

(

K

n

)
=

H

p

(
K
)

{\displaystyle 0=H_{p}(K_{0})\to H_{p}(K_{1})\to \cdots \to H_{p}(K_{n})=H_{p}(K)}

 connected by homomorphisms induced by the inclusion maps of the underlying filtration. When homology is taken over a field, we get a sequence of vector spaces and linear maps known as a persistence module.

Let 

f

p

i
,
j

:

H

p

(

K

i

)
→

H

p

(

K

j

)

{\displaystyle f_{p}^{i,j}:H_{p}(K_{i})\to H_{p}(K_{j})}

 be the homomorphism induced by the inclusion 

K

i

↪

K

j

{\displaystyle K_{i}\hookrightarrow K_{j}}

. Then the 

p

t
h

{\displaystyle p^{th}}

 persistent homology groups are defined as the images 

H

p

i
,
j

:=
im
⁡

f

p

i
,
j

{\displaystyle H_{p}^{i,j}:=\operatorname {im} f_{p}^{i,j}}

 for all 

1
≤
i
≤
j
≤
n

{\displaystyle 1\leq i\leq j\leq n}

. In particular, the persistent homology group 

H

p

i
,
i

=

H

p

(

K

i

)

{\displaystyle H_{p}^{i,i}=H_{p}(K_{i})}

.

More precisely, the 

p

t
h

{\displaystyle p^{th}}

 persistent homology group can be defined as 

H

p

i
,
j

=

Z

p

(

K

i

)

/

(

B

p

(

K

j

)
∩

Z

p

(

K

i

)

)

{\displaystyle H_{p}^{i,j}=Z_{p}(K_{i})/\left(B_{p}(K_{j})\cap Z_{p}(K_{i})\right)}

, where 

Z

p

(

K

∙

)

{\displaystyle Z_{p}(K_{\bullet })}

 and 

B

p

(

K

∙

)

{\displaystyle B_{p}(K_{\bullet })}

 are the standard p-cycle and p-boundary groups, respectively.[9]

## Birth and death of homology classes

Sometimes the elements of 

H

p

i
,
j

{\displaystyle H_{p}^{i,j}}

 are described as the homology classes that are "born" at or before 

K

i

{\displaystyle K_{i}}

 and that have not yet "died" entering 

K

j

{\displaystyle K_{j}}

. These notions can be made precise as follows. A homology class 

γ
∈

H

p

(

K

i

)

{\displaystyle \gamma \in H_{p}(K_{i})}

 is said to be born at 

K

i

{\displaystyle K_{i}}

 if it is not contained in the image of the previous persistent homology group, i.e., 

γ
∉

H

p

i
−
1
,
i

{\displaystyle \gamma \notin H_{p}^{i-1,i}}

. Conversely, 

γ

{\displaystyle \gamma }

 is said to die entering 

K

j

{\displaystyle K_{j}}

 if 

γ

{\displaystyle \gamma }

 is subsumed (i.e., merges with) another older class as the sequence proceeds from 

K

j
−
1

→

K

j

{\displaystyle K_{j-1}\to K_{j}}

. That is to say, 

f

p

i
,
j
−
1

(
γ
)
∉

H

p

i
−
1
,
j
−
1

{\displaystyle f_{p}^{i,j-1}(\gamma )\notin H_{p}^{i-1,j-1}}

 but 

f

p

i
,
j

(
γ
)
∈

H

p

i
−
1
,
j

{\displaystyle f_{p}^{i,j}(\gamma )\in H_{p}^{i-1,j}}

. The determination that an older class persists if it merges with a younger class, instead of the other way around, is sometimes known as the Elder Rule.[10][11]

The indices 

i
,
j

{\displaystyle i,j}

 at which a homology class 

γ

{\displaystyle \gamma }

 is born and dies entering are known as the birth and death indices of 

γ

{\displaystyle \gamma }

. The difference 

j
−
i

{\displaystyle j-i}

 is known as the index persistence of 

γ

{\displaystyle \gamma }

, while the corresponding difference 

a

j

−

a

i

{\displaystyle a_{j}-a_{i}}

 in function values corresponding to those indices is known as the persistence of 

γ

{\displaystyle \gamma }

 . If there exists no index at which 

γ

{\displaystyle \gamma }

 dies, it is assigned an infinite death index. Thus, the persistence of each class can be represented as an interval in the extended real line 

R

∪
{
±
∞
}

{\displaystyle \mathbb {R} \cup \{\pm \infty \}}

 of either the form 

[

a

i

,

a

j

)

{\displaystyle [a_{i},a_{j})}

 or 

[

a

i

′

,
∞
)

{\displaystyle [a_{i}',\infty )}

. Since, in the case of an infinite field, the infinite number of classes always have the same persistence,  the collection over all classes of such intervals does not give meaningful multiplicities  for a multiset of intervals. Instead, such multiplicities and a multiset of intervals in the extended real line are given by the structure theorem of persistence homology.[2] This multiset is known as the persistence barcode.[12]

## Canonical form

Concretely, the structure theorem states that for any filtered complex over a field 

F

{\displaystyle F}

, there exists a linear transformation that preserves the filtration and converts the filtered complex into so called canonical form, a canonically defined direct sum of filtered complexes of two types: two-dimensional complexes with trivial homology 

d
(

e

a

j

)
=

e

a

i

{\displaystyle d(e_{a_{j}})=e_{a_{i}}}

 and one-dimensional complexes with trivial differential 

d
(

e

a

i

′

)
=
0

{\displaystyle d(e_{a'_{i}})=0}

.[2]

## Persistence diagram

Geometrically, a barcode can be plotted as a multiset of points (with possibly infinite coordinates) 

(

a

i

,

a

j

)

{\displaystyle (a_{i},a_{j})}

 in the extended plane 

(

R

∪
{
±
∞
}

)

2

{\displaystyle \left(\mathbb {R} \cup \{\pm \infty \}\right)^{2}}

. By the above definitions, each point will lie above the diagonal, and the distance to the diagonal is exactly equal to the persistence of the corresponding class times 

1

2

{\displaystyle {\frac {1}{\sqrt {2}}}}

.  This construction is known as the persistence diagram, and it provides a way of visualizing the structure of the persistence of homology classes in the sequence of persistent homology groups.[1]

## References

* ^ a b Edelsbrunner; Letscher; Zomorodian (2002). "Topological Persistence and Simplification". Discrete & Computational Geometry. 28 (4): 511–533. doi:10.1007/s00454-002-2885-2. ISSN 0179-5376.
* ^ a b c Barannikov, Sergey (1994). "Framed Morse complex and its invariants" (PDF). Advances in Soviet Mathematics. ADVSOV. 21: 93–115. doi:10.1090/advsov/021/03. ISBN 9780821802373. S2CID 125829976.
* ^ Chen, Li M. (2015). Mathematical problems in data science : theoretical and practical methods. Zhixun Su, Bo Jiang. Cham. pp. 120–124. ISBN 978-3-319-25127-1. OCLC 932464024.{{cite book}}:  CS1 maint: location missing publisher (link)

```
{{cite book}}
```

* ^ Machine Learning and Knowledge Extraction : First IFIP TC 5, WG 8.4, 8.9, 12.9 International Cross-Domain Conference, CD-MAKE 2017, Reggio, Italy, August 29 - September 1, 2017, Proceedings. Andreas Holzinger, Peter Kieseberg, A. Min Tjoa, Edgar R. Weippl. Cham. 2017. pp. 23–24. ISBN 978-3-319-66808-6. OCLC 1005114370.{{cite book}}:  CS1 maint: location missing publisher (link) CS1 maint: others (link)

```
{{cite book}}
```

* ^ Hirata, Akihiko (2016). Structural analysis of metallic glasses with computational homology. Kaname Matsue, Mingwei Chen. Japan. pp. 63–65. ISBN 978-4-431-56056-2. OCLC 946084762.{{cite book}}:  CS1 maint: location missing publisher (link)

```
{{cite book}}
```

* ^ Moraleda, Rodrigo Rojas (2020). Computational topology for biomedical image and data analysis : theory and applications. Nektarios A. Valous, Wei Xiong, Niels Halama. Boca Raton, FL. ISBN 978-0-429-81099-2. OCLC 1108919429.{{cite book}}:  CS1 maint: location missing publisher (link)

```
{{cite book}}
```

* ^ Rabadán, Raúl (2020). Topological data analysis for genomics and evolution : topology in biology. Andrew J. Blumberg. Cambridge, United Kingdom. pp. 132–158. ISBN 978-1-316-67166-5. OCLC 1129044889.{{cite book}}:  CS1 maint: location missing publisher (link)

```
{{cite book}}
```

* ^ Yen, Peter Tsung-Wen; Cheong, Siew Ann (2021). "Using Topological Data Analysis (TDA) and Persistent Homology to Analyze the Stock Markets in Singapore and Taiwan". Frontiers in Physics. 9: 20. Bibcode:2021FrP.....9...20Y. doi:10.3389/fphy.2021.572216. hdl:10356/155402. ISSN 2296-424X.
* ^ Edelsbrunner, Herbert (2010). Computational topology : an introduction. J. Harer. Providence, R.I.: American Mathematical Society. pp. 149–153. ISBN 978-0-8218-4925-5. OCLC 427757156.
* ^ Nielsen, Frank, ed. (2021). Progress in information geometry : theory and applications. Cham. p. 224. ISBN 978-3-030-65459-7. OCLC 1243544872.{{cite book}}:  CS1 maint: location missing publisher (link)

```
{{cite book}}
```

* ^ Oudot, Steve Y. (2015). Persistence theory : from quiver representations to data analysis. Providence, Rhode Island. pp. 2–3. ISBN 978-1-4704-2545-6. OCLC 918149730.{{cite book}}:  CS1 maint: location missing publisher (link)

```
{{cite book}}
```

* ^ Ghrist, Robert (2008). "Barcodes: The persistent topology of data". Bulletin of the American Mathematical Society. 45 (1): 61–75. doi:10.1090/S0273-0979-07-01191-3. ISSN 0273-0979.
* Computational topology
* Data analysis
* Homology theory
* Algebraic topology
* CS1 maint: location missing publisher
* CS1 maint: others
* Articles with short description
* Short description matches Wikidata