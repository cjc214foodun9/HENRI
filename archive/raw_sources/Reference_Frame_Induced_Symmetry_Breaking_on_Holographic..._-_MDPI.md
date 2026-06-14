# Reference Frame Induced Symmetry Breaking on Holographic... - MDPI
Source URL: https://www.mdpi.com/2073-8994/13/3/408

Loading [MathJax]/jax/output/HTML-CSS/fonts/Gyre-Pagella/Operators/Regular/Main.js
Submit to this Journal Review for this Journal Propose a Special Issue
Article Menu
Academic Editor
Kazuharu Bamba
Recommended Articles
Related Info Link
More by Authors Links
Article Views
4328
Citations
25
Table of Contents
Abstract
Introduction
Instantaneous Interactions across B
Reference Frame Induced Decoherence
Reference Frame Induced Entanglement
Reference Frame Induced Contextuality
Writing and Reading Classical Memories
Conclusions
Author Contributions
Funding
Institutional Review Board Statement
Informed Consent Statement
Data Availability Statement
Acknowledgments
Conflicts of Interest
Abbreviations
Appendix A. The Basics of Channel Theory Information Flow and Context Dependency
References
Altmetric
share
Share
announcement
Help
format_quote
Cite
question_answer
Discuss in SciProfiles
first_page
settings
Order Article Reprints
Open AccessArticle
Reference Frame Induced Symmetry Breaking on Holographic Screens
by Chris Fields 1,*,James F. Glazebrook 2,† andAntonino Marcianò 3,‡
1
23 Rue des Lavandières, 11160 Caunes Minervois, France
2
Department of Mathematics and Computer Science, Eastern Illinois University, Charleston, IL 61920, USA
3
Center for Field Theory and Particle Physics, Department of Physics, Fudan University, Shanghai 200433, China
*
Author to whom correspondence should be addressed.
†
Adjunct Faculty: Department of Mathematics, University of Illinois at Urbana–Champaign, Urbana, IL 61820, USA.
‡
Also: Laboratori Nazionali di Frascati INFN, Frascati, Rome 00044, Italy.
Symmetry 2021, 13(3), 408; https://doi.org/10.3390/sym13030408
Submission received: 9 February 2021 / Revised: 25 February 2021 / Accepted: 27 February 2021 / Published: 3 March 2021
(This article belongs to the Special Issue Symmetry in Quantum Theory of Gravity)

Download
keyboard_arrow_down
 Browse Figures Versions Notes

Abstract
Any interaction between finite quantum systems in a separable joint state can be viewed as encoding classical information on an induced holographic screen. Here we show that when such an interaction is represented as a measurement, the quantum reference frames (QRFs) deployed to identify systems and pick out their pointer states induce decoherence, breaking the symmetry of the holographic encoding in an observer-relative way. Observable entanglement, contextuality, and classical memory are, in this representation, logical and temporal relations between QRFs. Sharing entanglement as a resource requires a priori shared QRFs.
Keywords: black hole; contextuality; decoherence; quantum error-correcting code; quantum reference frame; system identification; channel theory

1. Introduction
The holographic principle (HP) states, in its covariant formulation, that for any finite spacelike boundary 
𝐵
ℬ
, open or closed, the classical, thermodynamic entropy 
𝑆
(
𝐿
(
𝐵
)
)
𝑆(𝐿(ℬ))
 of any light-sheet 
𝐿
(
𝐵
)
𝐿(ℬ)
 of 
𝐵
ℬ
 satisfies:
𝑆
(
𝐿
(
𝐵
)
)
≤
𝐴
(
𝐵
)
/
4
,
𝑆(𝐿(ℬ))≤𝐴(ℬ)/4,
	
(1)
where 
𝐴
(
𝐵
)
𝐴(ℬ)
 is the area of 
𝐵
ℬ
 in Planck units [1]. The HP was motivated by the Bekenstein bound on the thermodynamic entropy of a black hole (BH), and has traditionally been interpreted as a bound on the thermodynamic entropy of, and hence the classical information encodable on, an independently-defined surface 
𝐵
ℬ
, e.g., the stretched horizon of a BH [2,3]; see [1,4] for reviews.
We can, however, also view (1) from a more general perspective, as a fundamental principle of information geometry that associates a (finite) minimal surface 
𝐵
ℬ
 with any bit string of (finite) entropy S, and hence with any classical channel of width S bits. Such a channel can be constructed, without loss of generality, as follows: Let 
𝑈
=
𝐴
𝐵
𝑈=𝐴𝐵
 be a finite, closed quantum system, assume separability, 
|
𝐴
𝐵
〉
=
|
𝐴
〉
|
𝐵
〉
 over any time interval of interest, and write the interaction:
𝐻
𝐴
𝐵
=
𝛽
𝑘
𝑘
𝐵
𝑇
𝑘
∑
𝑖
𝑁
𝛼
𝑖
𝑘
𝑀
𝑖
𝑘
,
	
(2)
where 
𝑘
=
𝐴
 or B, the 
𝑀
𝑖
𝑘
 are N Hermitian operators with eigenvalues in 
{
−
1
,
1
}
, the 
𝛼
𝑖
𝑘
∈
[
0
,
1
]
 are such that 
∑
𝑖
𝑁
𝛼
𝑖
𝑘
=
1
, 
𝑘
𝐵
 is Boltzmann’s constant, 
𝑇
𝑘
 is k’s temperature, and 
𝛽
𝑘
≥
 ln 2 is an inverse measure of k’s thermodynamic efficiency that depends on the internal dynamics 
𝐻
𝑘
. At each time step, A obtains exactly N bits of information about B from this channel and vice versa, entirely independently of the internal dynamics 
𝐻
𝐴
 and 
𝐻
𝐵
. With this construction, we can state the following generalized holographic principle (cf. [5] Thm. 1):
GHP: If but only if a pair of finite quantum systems A and B have a separable joint state 
|
𝐴
𝐵
〉
=
|
𝐴
〉
|
𝐵
〉
, there is a finite spacelike surface 
𝐵
, with area 
𝐴
(
𝐵
)
≥
𝐴
(
𝐵
)
𝑚
𝑖
𝑛
=
4
ln
2
𝑁
𝑙
𝑃
2
, N the dimension of 
𝐻
𝐴
𝐵
 and 
𝑙
𝑃
 the Planck length, that implements 
𝐻
𝐴
𝐵
 as a classical channel.
This GHP is a purely information-theoretic principle that makes no reference to any spatial embedding of A or B. We show in [5] that it holds for any spatial embedding of A and B allowed by special relativity. As 
𝐵
 is ancillary to the interaction 
𝐻
𝐴
𝐵
, we will be unconcerned with the spatial scale of 
𝐵
; in systems with low energy densities, we can expect 
𝐴
(
𝐵
)
≫
𝐴
(
𝐵
)
𝑚
𝑖
𝑛
.
The form of the Hamiltonian (2) has two immediately-apparent symmetries. First, the number N of transferred bits is fixed; hence Equation (2) is symmetric as a channel. The holographic screen 
𝐵
 “looks the same” and encodes the same information, N bits, from either A’s or B’s perspective. Second, the terms 
𝛼
𝑖
𝑘
𝑀
𝑖
𝑘
 of Equation (2) can be re-arranged in any order. If we view 
𝐵
 as implemented by an ancillary array of non-interacting qubits as in [5,6], these qubits can be permuted arbitrarily; hence the state of 
𝐵
 is invariant under the symmetric group 
𝑆
𝑁
.
Here we consider the system A to be an “observer” of B, and study apparent, observer-relative symmetry breaking on 
𝐵
 induced by the implementation of one or more quantum reference frames (QRFs) by the internal Hamiltonian 
𝐻
𝐴
. The role of reference frames in physical theory is to allow observations made at different times and/or places to be compared. While in classical physics reference frames are often treated in abstracto, in quantum theory they must be considered to be physically implemented, and hence as QRFs; meter sticks, clocks, and gyroscopes are canonical examples [7]. Sharing an external QRF, e.g., a Cartesian frame, across either space or time requires the observers involved to implement an equivalent internal QRF [8]; hence all QRFs deployed by A can, without loss of generality, be considered to be implemented by 
𝐻
𝐴
 (cf. [9,10]).
We begin in Section 2 below by briefly reviewing some consequences of separating systems A and B by a holographic screen 
𝐵
; such separation prevents, in particular, measurements by A of the entanglement entropy of B. We then introduce in Section 3 an explicit, fully general, and semantically-rich category-theoretic formalism with which to specify the QRFs deployed by any observer, focussing first on QRFs employed for system identification (Section 3.1) and then considering QRFs employed for pointer measurements (Section 3.2). We show that sequential pointer measurements break the 
𝑆
𝑁
 symmetry of the screen 
𝐵
, inducing decoherence (Section 3.3). For illustration, we turn to the particular case of measuring Hawking quanta from a BH, showing explicitly that no experiment can demonstrate entanglement between a BH and a distant free quantum of radiation (Section 3.4). We close this section by showing that the free-energy costs of deploying a QRF induce coarse-graining (Section 3.5). These results provide the background required to prove, in Section 4, that sharing entanglement as a resource requires a priori shared, entangled QRFs, and then to prove, in Section 5, that whether two observers share QRFs is finite Turing undecidable. We close in Section 6 by showing that writing classical memories to a screen 
𝐵
 creates phase correlations that further disrupt the 
𝑆
𝑁
 symmetry of the screen. These results together suggest that, far from being “an apparent law of physics that stands by itself” [1], the HP in its generalized GHP form is central to quantum information theory.
2. Instantaneous Interactions across 
𝐵
To write the Hamiltonian (2), we require the joint state 
|
𝐴
𝐵
〉
 to be separable; it is this separability that makes 
𝐵
 a classical channel. Distinguishing the classical entropy (S) from the entanglement entropy (
𝑆
), we can restate the GHP in summary form as:
Lemma 1. If systems A and B are separated by a finite holographic screen 
𝐵
, the entanglement entropy of the joint state 
𝑆
(
|
𝐴
𝐵
〉
)
=
0
.
Proof.  By the definition of 
𝐵
; see [5], Thm. 1 showing that any finite-bandwidth classical communication channel can be represented as a finite holographic screen for details. □
Lemma 1 immediately rules out transfers of quantum information across 
𝐵
; hence A has access to neither the internal Hamiltonian 
𝐻
𝐵
 or the entanglement entropy 
𝑆
(
𝐵
)
=
𝑑
𝑒
𝑓
𝑚
𝑎
𝑥
(
𝑆
(
|
𝐵
1
𝐵
2
〉
)
)
 over tensor-product decompositions 
𝐵
1
𝐵
2
=
𝐵
. It is clear, moreover, that Equation (2) can hold only if the Hilbert space dimensions 
𝑑
𝑖
𝑚
(
𝐻
𝐴
)
,
𝑑
𝑖
𝑚
(
𝐻
𝐵
)
≥
𝑁
, where equality holds only if A and B contain no “hidden” degrees of freedom that do not, over any time interval under consideration, contribute to 
𝐻
𝐴
𝐵
. Hence A cannot place upper limits on either the dimension 
𝑑
𝑖
𝑚
(
𝐻
𝐵
)
 or the entanglement entropy 
𝑆
(
𝐵
)
 of B. We will assume in what follows that B is “large,” 
𝑑
𝑖
𝑚
(
𝐻
𝐵
)
≫
𝑁
, and has near-maximal entanglement entropy 
𝑆
(
𝐵
)
≈
𝑑
𝑖
𝑚
(
𝐻
𝐵
)
/
2
; this is effectively the assumption that 
𝐻
𝐴
𝐵
 only minimally perturbs 
|
𝐵
〉
. As with this assumption the full state 
|
𝐵
〉
 is not observable by A, we will use the notation 
|
𝐵
〉
𝐴
 to indicate the “observed” (by A) partial state of B that is encoded on 
𝐵
. By (2), this observed state 
|
𝐵
〉
𝐴
 is an eigenstate of 
𝐻
𝐴
𝐵
 with dimension N.
2.1. Example: Scattering
Consider a scattering process mediated by a gauge boson, as shown in Figure 1a. Both incoming and outgoing joint states are asymptotically separable, so the exchanged information is encoded on a holographic screen 
𝐵
. Ignoring charge and spin, the encoded information specifies a classical momentum transfer 
Δ
𝑝
→
. No quantum information is transferred across 
𝐵
; in particular, the scattering processes transfers no information about the entanglement entropy 
𝑆
(
𝐵
)
 to A (Figure 1b).
Figure 1. (a) A gauge boson transfers asymptotically-classical momentum information across a holographic screen 
𝐵
. (b) The scattering process transfers no information about the entanglement entropy 
𝑆
(
𝐵
)
.
Writing the Hamiltonian as in (2) requires the dimension N of the observed state 
|
𝐵
〉
𝐴
 to be finite and hence the momentum transfer 
Δ
𝑝
→
 to be discrete. Discrete values of 
Δ
𝑝
→
 reflect the discrete cost of information in units of ℏ. In a experimental setting in which 
Δ
𝑝
→
 is measured at some asymptotically-distant location, the dimension N, and hence the number of (ancillary) qubits required to represent 
𝐵
 as a channel, is set operationally by the resolution of the detector. In this case 
Δ
𝑝
→
 is the measured pointer value, and a full description of the interaction requires specification of the QRF employed for system identification as discussed in Section 3 below.
2.2. Example: Hawking Radiation
For an asymptotic observer A, coupled pair annihilation and production near a holographic screen 
𝐵
, with one positive- and one negative-energy mode transiting the screen (Figure 2a) is indistinguishable from a scattering event at 
𝐵
 (Figure 2b). Hence far from a black hole (BH), Hawking radiation from the BH is indistinguishable from scattering from the stretched horizon of the BH. Lemma 1 forbids any modes detectable by A from carrying information about the entanglement entropy 
𝑆
(
𝐵
)
 as discussed above; hence the only observable entropy of B is the classical, thermodynamic entropy 
𝑆
(
𝐿
(
𝐵
)
)
 given by Equation (1).
Figure 2. (a) Hawking pair annihilation-production near a BH is asymptotically indistinguishable from (b) symmetric scattering from the stretched horizon.
The distinction between the thermodynamic entropy 
𝑆
(
𝐿
(
𝐵
)
)
 and the entanglement entropy 
𝑆
(
𝐵
)
 for B a BH, and hence 
𝐵
 the stretched horizon, has been recently clarified from a number of perspectives [11,12,13,14], showing in particular that preserving unitarity does not require a firewall [15] to prevent detection of excess entanglement. Considering the outgoing states to be measured by some observer requires specifying a QRF as noted above; we consider this issue in the particular case of Hawking quanta further in Section 3.4 below.
2.3. Symmetry across 
𝐵
 Corresponds to “Free Choice” of QRFs
A QRF deployed by A, i.e., implemented by 
𝐻
𝐴
, corresponds to a set of observables held fixed while other observables are allowed to vary [6,8] as discussed more explicitly in Section 3 below. It thus corresponds to a subset of the 
𝑀
𝑖
𝐴
. Associative groupings of the 
𝑀
𝑖
𝐴
 in Equation (2) are clearly independent of associative groupings of the 
𝑀
𝑖
𝐵
; hence choices of QRF by A have no bearing on choices of QRF by B or vice versa. Equivalently, swapping the labels A and B has no effect on Equation (2).
This “free choice” of QRFs corresponds to a the absence of superdeterminist correlations between A and B. Such correlations implement entanglement [16,17] so are forbidden if 
|
𝐴
𝐵
〉
=
|
𝐴
〉
|
𝐵
〉
. We discuss the effects of locally breaking this free-choice symmetry in Section 4 below.
3. Reference Frame Induced Decoherence
3.1. QRFs for System Identification
From A’s perspective, the partial state 
|
𝐵
〉
𝐴
 encoded on 
𝐵
 is pure: it encodes all of the information about B that is accessible, even in principle, to A. Mixed or decoherent states (we will use these terms interchangeably), in contrast, always indicate a lack of access to information that is in principle accessible: a state 
𝜌
𝑆
 of S is decoherent if there is some non-null system E such that 
𝜌
𝑆
=
Tr
𝐸
𝜌
𝑆
𝐸
=
Tr
𝐸
|
𝑆
𝐸
〉
〈
𝑆
𝐸
|
. In this case, E is the purifier or “environment” of the S and 
|
𝑆
𝐸
〉
 is the purification of 
𝜌
𝑆
 by E (see [18,19,20] for extended discussions). That such a purifying E exists physically, not just formally, for any mixed 
𝜌
𝑆
 is a fundamental assumption of quantum theory [21,22], sometimes stated as an explicit axiom [23]. From an operation perspective, E comprises degrees of freedom that interact with those of S but that cannot be, or at least are not observed when 
𝜌
𝑆
 is measured. If 
𝜌
𝑆
 is, for example, the state of a particle beam, E would include the degrees of freedom of the ion source, the magnetic fields, the ambient vacuum in the beam lines, etc.
The minimal setting employed here avoids the circularity that arises when a system-environment decomposition 
𝐵
=
𝑆
𝐸
 is stipulated a priori [24,25,26,27] by forcing S to be identified by some QRF implemented by A. As shown in [6], any QRF can be specified by a cocone diagram (CCD), a category-theoretic construction comprising a hierarchical arrangement of Barwise-Seligman Information Flow binary classifiers/classifications 
𝐴
𝛼
 [28] as depicted in Figure 3. These classifiers represent observables in context; namely, each classifier is a conceptual representation 
𝐴
𝛼
=
〈
𝑒
𝑣
𝑒
𝑛
𝑡
𝛼
,
(
𝑐
𝑜
𝑛
𝑑
𝑖
𝑡
𝑖
𝑜
𝑛
,
𝑐
𝑜
𝑛
𝑡
𝑒
𝑥
𝑡
)
𝛼
,
𝑣
𝑎
𝑙
𝑢
𝑎
𝑡
𝑖
𝑜
𝑛
𝛼
〉
 (essential details of the concepts and constructions are recalled in Appendix A, and in particular Appendix A.1 for the latter concept) (More generally, these classifiers can be seen as a triad of: (i) events (atomic, observed or experienced, imposition of boundaries, etc.); (ii) conditions/contents/influences as paired with contexts/measurements/detectors; and (iii) valuation. Again see Appendix A.1 for the formal details). Each classifier 
𝐴
𝛼
 is valued in 
{
−
1
,
1
}
 in accordance with its associated operator 
𝑀
𝛼
𝐴
 implementing “yes/no questions” as intrinsic to Equation (2) ([6], Section 3.2). In this sense, 
𝐴
𝛼
 may be alternatively regarded as an eigen-classifier for 
𝑀
𝛼
𝐴
.
Figure 3. A cocone diagram (CCD) is a commuting diagram depicting maps (infomorphisms) 
𝑓
𝑖
𝑗
 between (eigen-)classifiers 
𝐴
𝑖
 and 
𝐴
𝑗
, maps 
𝑔
𝑘
𝑙
 from the 
𝐴
𝑘
 to one or more channels 
𝐶
𝑙
 over subsets of the 
𝐴
𝑖
, and maps 
ℎ
𝑙
 from channels 
𝐶
𝑙
 to the colimit 
𝐶
 (cf. Equation (6.7) of [32]). Such a CCD can be associated (double-headed arrows) with any subset of binary operators 
𝑀
𝑘
𝐴
…
𝑀
𝑛
𝐴
 provided that these operators all mutually commute. The CCD specifies, in this case, a classical algorithm implemented by 
𝐻
𝐴
. The complete set of operators 
𝑀
𝑖
𝐴
 and 
𝑀
𝑖
𝐵
 in (2) together with the array of mutually noninteracting qubits 
𝑞
1
…
𝑞
𝑁
 (i.e., the screen 
𝐵
) implement the classical channel between A and B. Free choice of QRFs by A and B corresponds to independent, free choice of z axis by A and B at each qubit. Note that should the CCD fail to commute (in which case the colimit becomes undefined), then the 
𝐴
𝑖
 are considered as “non-co-deployable” (observables), and their corresponding distributed system exhibits intrinsic contextuality ([33], Section 7).
To construct the CCD, we select a subset 
{
𝑀
𝑘
𝐴
,
…
𝑀
𝑛
𝐴
}
 of measurement operators and assign to each a binary classifier 
𝐴
𝑘
,
…
,
𝐴
𝑛
, respectively, with each requiring a fixed value, 
+
1
 or 
−
1
, from the corresponding operator; the 
𝐴
𝑘
…
𝐴
𝑛
 thus specify a fixed bit string as input to the CCD. Further binary classifiers, each of which can be thought of as a classical logic gate, are added to form “hidden layers” 
𝐶
; the maps between classifiers are “infomorphisms” as defined in [28] that satisfy the diagram-commutativity requirements for a cocone (see [29,30] for general category-theoretic definitions, Reference [31] for discussion of the cocone as a general representation of complex conditionals, Reference [32] for applications, examples, and discussion of the obvious analogy with artificial neural networks, and [6] for summaries of the relevant definitions as they apply in the current context). As shown in [6], a CCD exists over a subset 
𝐴
𝑘
…
𝐴
𝑛
 of classifiers if and only if the corresponding subset 
{
𝑀
𝑘
𝐴
,
…
𝑀
𝑛
𝐴
}
 of binary-valued operators all mutually commute (see [33] for a formal proof). The colimit 
𝐶
 of the CCD encodes the classical “output” of the QRF as a bit string. As formulated in [33], the CCD then becomes manifestly a scale-free context-dependent architecture. Operationally, it can be thought of as a “deep” neural network with re-entrant connections [32]. In the general case these connections are implemented by quantum processes (i.e., by 
𝐻
𝐴
); the intermediate classifiers at each layer 
𝐶
 then implement measurements, with the general form of (2), of the outcomes of these processes.
The channel implemented by the qubits 
𝑞
1
…
𝑞
𝑁
 is free of classical noise by definition: there is no external system to provide a noise source [6,8]. It is evident in Figure 3, however, that this channel transmits a fixed classical bit string, e.g., (1, 1, 1, …, 1) from B to A if but only if A and B share a z axis. Hence we have:
Lemma 2. The channel implemented by a holographic screen between A and B is free of quantum noise if and only if A and B share QRFs.
Proof.  If A and B share QRFs, each channel qubit is prepared and then measured in the same basis. As the preparation—measurement cycle is effectively instantaneous, prepared and measured outcomes must be the same up to measurement resolution. If A and B do not share QRFs, the preparation and measurement bases for each channel qubit may be arbitrarily different, introducing arbitrary phase rotations, i.e., quantum noise, between B’s preparation and A’s measurement. □
Shared QRFs correspond to einselection of a preferred basis for decoherence [18,22] at 
𝐵
. We show in Section 4 below that Lemma 2 strongly restricts classical communication, and hence execution of local operations, classical communication (LOCC) protocols [7,34] by spacelike separated observers.
3.2. Reference and Pointer Measurements
The CCD in Figure 3 has a natural physical interpretation: it specifies the hierarchy of logical constraints that must be satisfied to identify the outcome values produced by the operators 
𝑀
𝑘
𝐴
…
𝑀
𝑛
𝐴
, and hence the components 
𝑘
…
𝑛
 of the pure state 
|
𝐵
〉
𝐴
, as the observed (effective or virtual) state 
𝜌
𝑆
 of an observed (effective or virtual) system S. The state 
𝜌
𝑆
 of any such S has by convention two components, a time-varying pointer state 
𝜌
𝑃
 that is of interest as a measurement outcome, and the remaining reference state 
𝜌
𝑅
 that by remaining fixed over the macroscopic time required for multiple cycles of pointer measurements enables the re-identification of the single, fixed system S with pointer state 
𝜌
𝑃
. The pointer state 
𝜌
𝑃
 here includes not just the traditional “pointer(s)” of S, but also any adjustable “settings” of S that may vary during a sequence of measurements. The reference state 
𝜌
𝑅
, in contrast, specifies the fixed properties of the system S that fix its identity and hence allow re-identification over time. If S is a macroscopic item of apparatus, for example, these include both the exterior size, shape, color, brand name, and location required to pick the apparatus out, e.g., by visual search, from the cluttered background of the laboratory as a whole, as well as the internal structural and functional properties that enable it to serve as a measurement device, i.e., as a QRF [10].
Following the notation of [6], we indicate by 
{
𝑀
𝑖
𝑃
}
 and 
{
𝑀
𝑗
𝑅
}
 the disjoint subsets of 
𝑀
𝑘
𝐴
…
𝑀
𝑛
𝐴
 that measure 
𝜌
𝑃
 and 
𝜌
𝑅
, respectively. Call the dimensions of these components 
𝑁
𝑃
+
𝑁
𝑅
=
𝑁
𝑆
. As 
𝜌
𝑅
 serves as a fixed reference, clearly 
∀
𝑖
,
𝑗
,
[
𝑀
𝑖
𝑃
,
𝑀
𝑗
𝑅
]
=
0
. Pointer state measurements, however, generically do not commute; adjustable apparatus settings, in particular, are useful only to the extent that they do not commute with pointer readings. Hence generically, 
∃
𝑖
,
𝑗
,
[
𝑀
𝑖
𝑃
,
𝑀
𝑗
𝑃
]
≠
0
. Call the set of mutually-commuting subsets of 
{
𝑀
𝑖
𝑃
}
, and hence of classifiers 
𝐴
𝑖
𝑃
, 
{
𝑃
𝑖
}
; in this case a state 
𝜌
𝑖
𝑆
 is computed by a CCD over 
𝑅
𝑃
𝑖
. This decomposition is shown in a simplified notation in Figure 4. Clearly under these conditions the joint state 
𝜌
𝑆
 must be separable as 
𝜌
𝑅
𝜌
𝑃
𝑖
, i.e., the system components R and P must be mutually decoherent.
Figure 4. A cocone diagram (CCD) computing an effective (or virtual) “system state” 
𝜌
𝑆
 comprises classifier channels computing an effective pointer state 
𝜌
𝑃
𝑖
 and an effective reference state 
𝜌
𝑅
 (cf. [6]). These channels define the effective “subsystems” R and 
𝑃
𝑖
 comprising S. The CCD acts on the pure physical state 
|
𝐵
〉
𝐴
 encoded by 
𝐻
𝐴
𝐵
 on the holographic screen 
𝐵
 (blue) separating A from B. The computation represented by the CCD is implemented by the internal dynamics 
𝐻
𝐴
.
3.3. Sequential Pointer Measurements Induce Decoherence
State transitions 
𝐺
𝑖
𝑗
:
𝜌
𝑖
𝑆
(
𝑡
)
→
𝜌
𝑗
𝑆
(
𝑡
+
Δ
𝑡
)
, although associative and invertible, in general do not commute, and have a set of multiple identities; hence they can be represented as elements of a groupoid [35,36] 
(
{
𝐺
𝑖
𝑗
}
,
∘
)
 such that 
𝐺
𝑖
𝑗
∘
𝐺
𝑗
𝑖
≠
𝐺
𝑗
𝑖
∘
𝐺
𝑖
𝑗
 if and only if 
[
𝑀
𝑖
𝑃
,
𝑀
𝑗
𝑃
]
≠
0
 [6]. The action of 
(
{
𝐺
𝑖
𝑗
}
,
∘
)
 on this set of system states, indexed by a macroscopic discrete time 
𝜏
, is illustrated in Figure 5 (for the formal definition of the action of a groupoid on a set, see, e.g., ([36], Section 10.4)).
Figure 5. A sequence of CCDs identifying R (blue triangles) and measuring pointer components 
𝑃
𝑖
,
𝑃
𝑗
,
𝑃
𝑘
…
𝑃
𝑙
. Transitions between CCDs are implemented by groupoid elements, e.g., 
𝐺
𝑖
𝑗
 and labeled by discrete macroscopic times 
𝜏
𝑖
. The operators 
𝑀
𝑖
𝑃
 can equally well be generalized to subsets 
{
𝑀
𝑃
}
𝑖
 of mutually-commuting pointer-state observables.
A reference state 
𝜌
𝑅
 computed by a CCD R from the outcome values of a set of operators 
𝑀
𝑗
𝑅
 is, effectively, a logical constraint on the identities of the qubits 
𝑞
𝑗
 that the 
𝑀
𝑗
𝑅
 measure. Hence we have:
Lemma 3. In any system 
𝐴
𝐵
 characterized by (2), fixing a reference state 
𝜌
𝑅
 over a macroscopic time interval τ locally breaks the 
𝑆
𝑁
 symmetry of the holographic screen 
𝐵
 encoding the eigenvalues of 
𝐻
𝐴
𝐵
.
Proof.  Suppose the 
𝑀
𝑗
𝑅
 measure the states of 
𝑁
𝑅
=
𝑛
−
𝑘
 qubits as shown in Figure 3; we can neglect P and assume that the other qubits constitute the environment and are swap-symmetric. Holding 
𝜌
𝑅
 fixed for 
𝜏
 is holding the 
𝑁
𝑅
 outcomes of the 
𝑀
𝑗
𝑅
 fixed for 
𝜏
; 
∀
𝑗
,
𝑀
𝑗
𝑅
|
𝑞
𝑗
(
𝑡
)
〉
=
|
1
〉
 or 
|
−
1
〉
 for t within 
𝜏
. This cannot be guaranteed if 
𝑞
𝑗
 is swapped for some environmental 
𝑞
𝑖
 with an unconstrained state; hence any such swap must be forbidden by a “selection rule.” This breaks the 
𝑆
𝑁
 symmetry on 
𝐵
. □
Lemma 3 is in fact obvious: the CCD R assigns each of the physical degrees of freedom 
𝑞
𝑗
 a specific role in the computation of 
𝜌
𝑅
, one that an arbitrary qubit in an arbitrary state cannot satisfy. The qubits 
𝑞
𝑗
 are classically phase-locked by R, while the phases of the environmental qubits can vary freely, preserving their swap symmetry. The CCD R effectively divides 
𝐵
 into two (not necessarily simply connected) regions, one in which the qubits are classically phase-correlated and the other in which they are not. Any such division induces decoherence between non-swap-symmetric and swap-symmetric qubits. These conditions equally hold for any CCD measuring a pointer state 
𝜌
𝑃
𝑖
.
Lemma 3 associates decoherence with system identification as a necessary prerequisite of pointer-state measurement. As Zurek emphasized ([22] p. 1794),
[T]he formulation of the measurement problem and its resolution through the appeal to decoherence require a universe split into systems. Yet, it is far from clear how one can define systems given an overall Hilbert space ‘of everything’ and the total Hamiltonian.
Subsequent work demonstrated that no preferred decomposition of an overall Hilbert space or its Hamiltonian could legitimately be assumed a priori [37,38], rendering all formulations of decoherence that assumed an a priori 
𝐵
=
𝑆
𝐸
 decomposition circular (see [24,25,26,27] for relevant discussion). By characterizing all observations as mediated by a holographic screen 
𝐵
, the GHP localizes the 
𝐵
=
𝑆
𝐸
 decomposition to the observer’s QRFs [5,6]. All systems S and states 
𝜌
𝑆
 are, therefore, virtual in the precise sense of computational processes implemented on underlying, observationally inaccessible hardware [39]: the observer A itself with its Hamiltonian 
𝐻
𝐴
.
3.4. Example: Mass and Hawking Radiation QRFs for a BH
Suppose A employs a local QRF 
𝑅
𝐵
𝐻
 (e.g., a local sample of the ambient photon field) to measure both the position x and the mass M of a distant BH B and a particle detector 
𝑅
𝑟
 to measure the momentum 
𝑝
→
 of one or more quanta of radiation. Her task, familiar from discussions of horizon complementarity [40] and the firewall paradox [15], is to determine whether her local quantum r is a Hawking quantum 
𝑟
𝐻
 from B (see [11,12,13,14] for relevant discussion). As illustrated in Figure 6, answering this question requires a QRF 
𝑅
𝐻
 that associates 
𝑝
→
 with an identified Hawking quantum.
Figure 6. Identifying a local quantum of radiation as a Hawking quantum 
𝑟
𝐻
 from a distant BH requires a local Hawking QRF 
𝑅
𝐻
. Lemma 3 rules this out.
Lemma 3 shows that the required 
𝑅
𝐻
 cannot be implemented, even in principle: determining that the BH has lost information requires observation over macroscopic time, inducing decoherence on 
𝐵
. Hence not only is A prevented by 
𝐵
 from obtaining information about the BH entanglement entropy 
𝑆
(
𝐵
)
 (Lemma 1); she cannot obtain entanglement information about identified systems if distinct QRFs are required for their identification. The entanglement entropy 
𝑆
(
|
𝐵
>
|
𝑟
>
)
 is, in particular, experimentally inaccessible even in principle. Horizon complementarity is, therefore, not required to prevent observations of no-cloning violations by Hawking quanta; such observations cannot be made because the QRF needed to identify an observable BH as the source of the quanta is unavailable. Thought experiments in which observers measure entanglement entropies before and after falling into a BH, as employed in, e.g., [15], are unrealizable even in principle.
The limitation imposed by Lemma 3 generalizes, via the ER = EPR hypothesis, to any system with a spatially-distributed purifier, e.g., an “octopus” BH topologically connected to its entangled Hawking quanta by ER bridges [11,41]. The complete system state is pure but unobservable in principle, as the QRFs required to localize the spatially-distributed components would induce decoherence separately on each component. This problem of QRF-induced decoherence in spatially-distributed purifiers is similarly relevant to treatments of potential entanglement effects surviving the inflationary epoch, e.g., [42,43,44,45]. Bell-type communication protocols, e.g., [46,47,48,49], circumvent this problem by employing classical communication, treated as an a priori preferred QRF, to provide localization information as discussed in Section 4 below.
3.5. Computation and Memory Costs Induce Coarse-Graining
Provided their intermediate states are not recorded to a persistent, classical memory, logically reversible computations can in principle be performed without net energetic cost; logically irreversible computations, in contrast, cost at least ln2 
𝑘
𝐵
𝑇
 per bit [50,51]. What is of interest in practice, however, is the incremental cost of computation, including the cost of writing intermediate states to a classical memory, even if the computational step is to be reversed later. An observation can only be considered to have been “made” if the result is written to a classical memory from which it can later reported, e.g., classically communicated to another observer [52]. A system S can, in particular, only be regarded as “observed” at a time t if its reference state 
𝜌
𝑅
(
𝑡
)
 is written to a classical memory from which it can be reported. System identification over macroscopic 
𝜏
 clearly requires writing to and reading from such a memory as discussed further in Section 6 below.
The free energy required to fund the incremental cost of computing and recording must be supplied by what Landauer [50] called the “non-information-bearing degrees of freedom” of the computer and/or its environment, even if this free energy is repaid in part later. Viewed on the output side, i.e., in terms of A’s action on B, these non-information-bearing degrees of freedom exhaust the waste heat generated by the computing and recording processes. This distinction between information-bearing and non-information-bearing degrees of freedom breaks 
𝑆
𝑁
 symmetry as discussed above. As shown in [6], this symmetry breaking can be expressed thermodynamically as the requirement that 
𝛽
𝑅
,
𝛽
𝑃
>
𝛽
𝐸
≥
 ln 2, where 
𝛽
𝐸
 is the efficiency of the operators 
𝑀
𝑘
𝐸
 acting on E. The environment E provides, in other words, the incremental free energy required to irreversibly identify R and measure P. The fuel value 
𝛽
𝐸
𝑘
𝐵
𝑇
 is independent of the bit value (+1 or −1) of the outcome; hence these outcomes are “non-information-bearing” for the computation implemented by 
𝐻
𝐴
. They therefore retain full permutation symmetry, justifying the trace over their joint state.
Any classical computation can be performed reversibly, e.g., with Toffoli gates, and any reversible computation can be performed with some unitary operator [46]. The only obligate classical steps in computing either 
𝜌
𝑅
 or 
𝜌
𝑃
𝑗
 are, therefore, the initial step of writing the “input” outcomes of the 
𝑀
𝑗
𝑅
 and the selected 
𝑀
𝑖
𝑃
 onto the relevant classifiers and the final step of writing the time-stamped joint state specification 
𝜌
𝑅
𝑃
𝑗
 on a classical memory. The criterion of classicality for the memory is operational: the time-stamped state specification must be reportable at any later time without disturbing other processes. For a perfectly efficient system, the free energy required to write each (Reference, Pointer) outcome 
𝜌
𝑅
𝑃
𝑗
 to memory is, therefore:
Δ
𝐻
𝑗
≥
ln
2
(
𝑁
𝑅
+
𝑁
𝑃
+
𝑁
𝜌
+
𝑁
𝜏
)
𝑘
𝐵
𝑇
,
	
(3)
where 
𝑁
𝑅
+
𝑁
𝑃
 is the number of bits required to record the inputs, 
𝑁
𝜌
 is the number of bits required to record 
𝜌
𝑅
𝑃
𝑗
, 
𝑁
𝜏
 is the number of bits required to record the timestamp. This incremental 
Δ
𝐻
𝑗
 must be supplied by E during each interval 
𝜏
𝑗
, so (3) places a lower limit 
𝑁
𝐸
≥
𝑁
𝑅
+
𝑁
𝑃
+
𝑁
𝜌
+
𝑁
𝜏
 on the number of qubits of E and therefore on the total area 
𝐴
(
𝐵
)
 of the holographic screen 
𝐵
. In any realistic system thermodynamic efficiency is less than ideal; hence 
𝛽
𝐸
>
 ln2 and 
𝐴
(
𝐵
)
 is correspondingly larger.
As 
𝑁
𝑅
+
𝑁
𝑃
 remains fixed, 
Δ
𝐻
𝑗
 is minimized as 
𝑁
𝜌
+
𝑁
𝜏
→
0
, i.e., as classical memory is coarse-grained relative to 
𝐵
. We can, therefore, generically expect QRFs to encode high redundancy over states 
|
𝐵
〉
𝐴
 mapped to the same 
𝜌
𝑅
𝑃
𝑗
, i.e., we can expect any CCD implementing a QRF to include logical OR gates and hence to lose information. Optimal coarse-graining jointly minimizes the cost of memory and the cost of redundancy. Furthermore, any QRF that is coarse-grained engenders redundancy and can be considered as a quantum error correcting code (QECC) [46]. This is relevant to the discussion in Section 3.4 above: a QEEC can be used to reconstruct local effective field theory observables, which as pointed out in [53], are applicable to BH states whose entanglement entropy falls short of saturating the Bekenstein-Hawking bound. Such local observables are designed to protect coherence in the Hilbert space of codes by correcting errors due to the emission of Hawking quanta, by entangling radiation within other regions of Hilbert space and inducing entanglement swaps that increase the entanglement entropy of the BH interior over time. As discussed in Section 3.4 above, such postulated entanglement swaps are unobservable even in principle.
4. Reference Frame Induced Entanglement
Communication protocols that employ shared entanglement depend on shared QRFs, e.g., shared z-axis QRFs for 
𝑠
𝑧
 measurements [8]. This suggests that the shared entanglement is in fact induced by the shared QRFs, a suggestion consistent with the general observer-dependence of entanglement [37,38].
Consider a Bell protocol described in the lab frame, as shown in Figure 7. An entangled state is distributed from a source to Alice and Bob, who remain spacelike-separated throughout the protocol’s operation. They are free to adjust their detector settings during the interval 
Δ
𝑡
𝑠
𝑒
𝑡
. Following data processing (the interval 
Δ
𝑡
𝑝
𝑟
𝑜
𝑐
), Bob sends his classically-encoded measurement outcomes to Alice via a classical channel. Alice can then compute the joint statistics, obtaining a Bell-inequality violation and hence an observation of entanglement at 
𝑡
𝑚
𝑒
𝑎
𝑠
. Alice’s ability to compute the joint statistics, and hence to observe entanglement, critically depends on two assumptions. First, Alice must know the code that Bob employs to encode his results; effectively, they must “speak the same language”. Second, the communication from Bob to Alice must be classical, i.e., must not involve a quantum measurement [7]. If the communication is not considered classical, i.e., if Bob sends Alice a QRF with which he is entangled, Alice must identify the transmitted QRF in order to measure its state, inducing decoherence as discussed in Section 3.4 above. These two assumptions are operationally equivalent: Alice scrambling Bob’s message by decohering a transmitted quantum state has the same effect as Alice scrambling Bob’s classical coding scheme by employing, e.g., an obsolete one-time pad. In either case, Alice does not “understand” Bob’s encoded results and her subsequent statistical analysis is meaningless [33].
Figure 7. A typical Bell protocol described in the lab frame. Sharing of measurements results via a classical channel is required to observe a Bell-inequality violation. If Alice’s interaction with Bob’s message is viewed as an ordinary quantum measurement, the entanglement disappears as in Section 3.4 above.
The assumption of classical communication is, effectively, the assumption of a preferred pointer measurement that returns the content of the communicated message without requiring prior identification, via a separate measurement, of the physical medium, i.e., the QRF, via which the message has been transmitted. Alice, in other words, does not have to identify Bob to receive his message, just as Wigner does not, in his famous thought experiment, have to identify his friend to receive his friend’s observational outcomes [54]. Hence assuming classical communication is assuming an a priori shared QRF [8]. This breaks the free-choice symmetry across 
𝐵
 as discussed in Section 2.3 above; if 
𝐵
 is considered a qubit array as in Figure 3, the assumption of classical communication is the assumption that A and B use identical z-axis QRFs on a subset of qubits as confirmed by Lemma 2 above. Call this subset of qubits 
𝑚
𝑒
𝑠
𝑠
𝑎
𝑔
𝑒
; the observed states 
|
𝑚
𝑒
𝑠
𝑠
𝑎
𝑔
𝑒
〉
𝐴
 and 
|
𝑚
𝑒
𝑠
𝑠
𝑎
𝑔
𝑒
〉
𝐵
 are superdeterministically correlated. Choosing a decomposition that identifies the shared QRFs shows that 
|
𝑚
𝑒
𝑠
𝑠
𝑎
𝑔
𝑒
〉
=
|
𝑚
𝑒
𝑠
𝑠
𝑎
𝑔
𝑒
〉
𝐴
𝐵
 is a single pure state. That such a pure state exists is the operational meaning of the requirement that Alice and Bob “share a language” for classical communication.
Redescribing the Bell protocol in the frame of the entangled state, as illustrated in Figure 8, makes both the shared QRF and its entangled state manifest. Hence we have:
Figure 8. (a) A Bell protocol in the frame of the entangled state (yellow circle). Alice and Bob collide at 
𝑡
𝑚
𝑒
𝑎
𝑠
, at which time they share, and together measure, the entangled state. (b) This is equivalent to Alice and Bob sharing an entangled QRF that reports consistent pointer outcomes to each observer.
Theorem 1. Sharing entanglement requires shared entanglement.
Proof.  Spacelike-separated Alice and Bob can observe entanglement only if they can compare their observational outcomes. By Lemma 2, this requires an a priori shared QRF. Classical transfer of a QRF also requires an a priori shared QRF [8]; hence the shared QRF can only be shared as an entangled state. □
Superdeterminist correlations, i.e., absence of free choice of QRFs, is a general feature of LOCC communication protocols. In the Bell protocol, Alice’s response to the bit string received from Bob is predetermined by the requirement of a shared QRF. Other protocols superdetermine the “choices” made during 
Δ
𝑡
𝑠
𝑒
𝑡
, and hence the outcomes observed. Entanglement-enabled secure communication protocols, for example, require Alice and Bob to deploy QRFs and execute measurement on an otherwise-uncharacterized qubit in the order specified by the protocol. These protocols avoid decoherence, and hence enable quantum communication, by rendering Alice’s and Bob’s QRFs effectively entangled for the duration of the protocol. Here again, avoiding decoherence is equivalent, operationally, to sharing a language in which, e.g., protocol instructions are classically communicated.
The special role played by classical communication in LOCC protocols has been investigated previously in extended Wigner’s friend experiments in which the outcomes of classical communication between pairs of “observers” and “friends” are contrasted with the outcomes of quantum measurements of “friends” by “observers” [55,56]. These experiments have been interpreted as showing, subject to an assumption of no superdeterminism, that observed events cannot be regarded as observer-independent. By treating all observed events as relative to observer-implemented QRFs, we show here that the assumption of classical communication between observers, widely regarded as physically inconsequential prior to [55,56], is to enforce local superdeterminism.
5. Reference Frame Induced Contextuality
Contextuality and entanglement are conceptually equivalent [57]. For a fixed P, switching between QRFs over 
𝑀
𝑗
𝑅
 and 
𝑀
𝑘
𝑇
, where 
[
𝑀
𝑗
𝑅
,
𝑀
𝑘
𝑇
]
≠
0
 for at least one pair 
𝑗
,
𝑘
 induces contextuality, i.e., no non-contextual probability distribution consistent with the Kolmogorov axioms and hence with Dutch-book coherence can be defined over the combined set of outcomes [6,33].
Consistent with the findings of [55,56], no Kolmogorov-consistent, non-contextual probability distribution can be defined over the combined outcomes of Alice’s and Bob’s observations unless it can be demonstrated, for the relevant P, and for R and T the QRFs deployed by Alice and Bob, respectively, that 
∀
𝑗
,
𝑘
,
[
𝑀
𝑗
𝑅
,
𝑀
𝑘
𝑇
]
=
0
. This cannot, however, be demonstrated by any observer of Alice and Bob, as no such observer has, by Lemma 1, access to the internal Hamiltonians 
𝐻
Alice
 or 
𝐻
Bob
.
This result can be stated in more formal terms of undecidability.
Theorem 2. Whether arbitrarily-chosen QRFs R and T compute the same function f is finite Turing undecidable.
Proof.  Let f designate the function computed by R, i.e., the function computed by the CCD representing R; we then ask whether T computes this same f. Whether an arbitrarily chosen computer computes any non-trivial function f cannot, however, be decided by any finite Turing machine [58]. Hence whether T computes f is finite Turing undecidable. □
As shown in [33], contextuality induced by non-commuting QRFs renders the Frame problem, the problem of circumscribing the degrees of freedom that do not change their values as the result of an action, e.g., a measurement [59] unsolvable even in domains with small numbers of degrees of freedom (cf. [60]).
Theorems 1 and 2 have as an obvious corollary:
Corollary 1. Whether two observers share an entangled state is finite Turing undecidable.
Hence whether Alice and Bob have successfully completed a quantum communication protocol is finite Turing undecidable.
6. Writing and Reading Classical Memories
As noted earlier, a sequence of observations made over macroscopic time is only reportable at some later time if the observations have been recorded on a classical memory. Distinguishing measurements made at different times requires, moreover, some method of distinguishing the memories. We therefore assume that the bit string composing each memory record includes a substring encoding a time stamp 
𝜏
𝑗
, which we take to be generated by the groupoid action of the 
𝐺
𝑖
𝑗
. Considering this classical memory to be implemented by 
𝐻
𝐴
 would prima facie require internal decoherence, i.e., disrupt the purity of 
|
𝐴
〉
. This can be avoided if A is regarded as writing all classical memories on 
𝐵
. As the result to be written to memory is coarse-grained compared to the input from which it was generated (Section 3.5), only a relatively small number of qubits on 
𝐵
 need be devoted to memory.
Reversing the arrows in a CCD yields the dual construction, a cone diagram (CD) with the single source classifier the limit over the bottom-level classifiers [32]. A CD can be constructed to encode any finite bit string on an underlying bit array, i.e., to write any finite bit string to memory. Regarding each memory bit as a preparation instruction for a corresponding qubit on 
𝐵
, we can represent a memory-write operation to 
𝐵
 as in Figure 9.
Figure 9. A CD 
𝑊
𝑗
𝑗
 (green triangle) specifies a memory-write operation of the time-stamped state 
(
𝜌
𝑅
𝑃
𝑗
,
𝜏
𝑗
)
 to 
𝐵
. The timestamp 
𝜏
𝑗
 is generated by the groupoid action 
𝐺
𝑖
𝑗
.
Reading the memory reverses the arrows on the memory-write CD 
𝑊
𝑗
𝑗
 to a CCD, i.e., a QRF for retrieving the time-stamped value 
(
𝜌
𝑅
𝑃
𝑗
,
𝜏
𝑗
)
. Writing readable memory records on 
𝐵
 imposes phase correlations across time on 
𝐵
; such correlations obviously further disrupt the 
𝑆
𝑁
 symmetry of 
𝐵
. Reading and rewriting memory records also imposes an energetic cost as in Section 3.
7. Conclusions
We have investigated, in this paper, a generalization of the HP in which interactions 
𝐻
𝐴
𝐵
 between finite quantum systems A and B that maintain a separable joint state are represented as exchanges of information across a holographic screen 
𝐵
. While the role of 
𝐵
 is ancillary to the action of 
𝐻
𝐴
𝐵
, the permutation symmetry of 
𝐵
 is broken when the internal Hamiltonian 
𝐻
𝐴
 is considered to implement QRFs that identify “systems” and measure their states. This symmetry breaking induces decoherence of identified systems by forcing the “environment” that remains to serve as both free energy source and waste heat sink. Observable entanglement, contextuality, and classical memory are, in this representation, logical and temporal relations between QRFs implemented by 
𝐻
𝐴
.
It is natural to interpret the holographic screen 
𝐵
 not merely as ancillary, but as a “physical” space, i.e., a stretched horizon, separating A from B. In this context, broken permutation symmetries on 
𝐵
 become broken exchange symmetries between points in the 
2
+
1
 spacetime defined by 
(
𝐵
,
𝜏
)
, 
𝜏
 a characteristic “macroscopic” time scale for 
𝐻
𝐴
𝐵
 as above (see [61] for details). From A’s observational perspective, exchange-inequivalent regions of 
(
𝐵
,
𝜏
)
 correspond to coarse-grained, decoherent “systems” while exchange-symmetric regions are “empty space” that supplies free energy and exhausts waste heat. That the GHP itself forces these virtual decoherent systems to obey gauge symmetries is shown in [61].
It is tempting to speculate that a third spatial dimension is induced when, but only when, A implements QRFs capable of identifying single systems across time while varying pointer observables such as size, shape, and color, and that inertial mass is a QRF representing the observable response of an identified system to actions by A. Whether the fundamental symmetries of space, time, and matter, or even all of physics can be completely reconstructed “within” an observer A, and hence viewed as a computation implemented by 
𝐻
𝐴
, remains to be determined. The fact that physics is done by physicists, systems that appear to interact with their environments via a Hamiltonian of the form (2), suggests that such a reconstruction is possible; possible routes forward are discussed in [62,63,64].
It is, finally, increasingly being suggested that the entropic structure of a BH may be more than a phenomenological correlate of its mass, possibly providing a route toward defining mass [65], specifying nontrivial internal topological structure [11], or even generating the phenomenology of dark energy or dark matter [66]. Theoretical investigation of the QRFs implemented by a BH may, therefore, offer exciting future developments.
Author Contributions
Conceptualization, C.F., J.F.G., A.M.; writing—original draft preparation, C.F.; writing—review and editing, C.F., J.F.G., A.M. All authors have read and agreed to the published version of the manuscript.
Funding
This research was funded by the Federico and Elvia Faggin Foundation (CF), the Shanghai Municipality, Grant No. KBH1512299 (AM), and by Fudan University, Grant No. JJH1512105 (AM).
Institutional Review Board Statement
Not applicable.
Informed Consent Statement
Not applicable.
Data Availability Statement
No data reported.
Acknowledgments
JFG wishes to thank J. McLennan for discussions on related topics.
Conflicts of Interest
The authors declare no conflict of interest.
Abbreviations
The following abbreviations are used in this manuscript:
BH	Black Hole
CCD	Cocone Diagram
CD	Cone Diagram
EPR	Einstein-Podolsky-Rosen
ER	Einstein-Rosen
GHP	Generalized Holographic Principle
LOCC	Local Operations, Classical Communication
QECC	Quantum Error-Correcting Code
QRF	Quantum Reference Frame
Appendix A. The Basics of Channel Theory Information Flow and Context Dependency
The Channel Theory of [28] introduces the idea of a “classifier” (or “classification”) as accommodating a “context” in terms of its constituent “tokens” in some language and the “types” to which they belong.
Definition A1. A classifier 
𝐴
 is a triple 
〈
𝑇
𝑜
𝑘
(
𝐴
)
,
𝑇
𝑦
𝑝
(
𝐴
)
,
⊨
𝐴
〉
 where 
𝑇
𝑜
𝑘
(
𝐴
)
 is a set of “tokens”, 
𝑇
𝑦
𝑝
(
𝐴
)
 is a set of “types”, and 
⊨
𝐴
 is a “classification” relation between tokens and types.
Note that this definition specifies a classifier/classification as an object in the category of Chu spaces [67,68,69] where ‘
⊨
𝐴
’ is realized by a satisfaction relation valued in some set 
𝐾
 (with no structure assumed). The arrows (morphisms) between classifiers are specified by the following:
Definition A2. Given two classifiers 
𝐴
=
〈
𝑇
𝑜
𝑘
(
𝐴
)
,
𝑇
𝑦
𝑝
(
𝐴
)
,
⊨
𝐴
〉
 and 
𝐵
=
〈
𝑇
𝑜
𝑘
(
𝐵
)
,
𝑇
𝑦
𝑝
(
𝐵
)
,
⊨
𝐵
〉
, an infomorphism 
𝑓
:
𝐴
→
𝐵
 is a pair of maps 
𝑓
→
:
𝑇
𝑜
𝑘
(
𝐵
)
→
𝑇
𝑜
𝑘
(
𝐴
)
 and 
𝑓
←
:
𝑇
𝑦
𝑝
(
𝐴
)
→
𝑇
𝑦
𝑝
(
𝐵
)
 such that 
∀
𝑏
∈
𝑇
𝑜
𝑘
(
𝐵
)
 and 
∀
𝑎
∈
𝑇
𝑦
𝑝
(
𝐴
)
, 
𝑓
→
(
𝑏
)
⊨
𝐴
𝑎
 if and only if 
𝑏
⊨
𝐵
𝑓
←
(
𝑎
)
.
Information is inherently a physical mode of distinctions and relationships between them, and not simply a reduction to a quantity of bits as it would be for Shannon information that passively neglects the substance of reasoning. Rather, it instead conforms to the set of logical constraints as imposed by Definition A2. An infomorphism as a mapping between classifiers provides the basic building blocks towards constructing multi-level, quasi-hierarchical classification systems. Such a framework of information transfer is indicative of causation, which itself may be viewed as a form of computation in view of the regular relations in a distributed system [70]. References [6,32,33] bring to the forefront many examples, and applications of the above concepts that include probability distributions, Bayesian belief networks, event space structures, formal concept analysis, and fuzzy relationships (as further relevant to this issue, let us point out that the Sorkin model of spacetime causal sets [71,72] has been interpreted in terms of classifiers (Chu spaces) in [73] (reviewed in [32])). In particular, Reference [33] focuses on orders of contextuality with ramifications to active inference and to the Frame Problem.
The specifics of transmitting information via classifiers and infomorphisms lead, in [28], to defining the idea of an information channel over classifiers, the most general of which leads to the categorical notion of a cocone with the core 
𝐶
 the colimit of all possible upward-going structure-preserving maps from the classifiers 
𝐴
𝑖
. Such a colimit core, when it exists, can be regarded as a classifier which embraces the totality of information that is common to the component classifiers 
𝐴
𝑖
. The resulting structure is a cocone diagram (CCD) as exemplified Figure 3. Within such a framework, the means by which channels encode sets of mutual constraints between classifiers is regulated by a local logic as presented formally in ([28], Ch. 12) (reviewed in [32,33]). Basically, the idea is that the types of a (regular) theory specify the logical structure of a given situation. A local logic is essentially a classifier having a (regular) theory along with a subset of tokens that satisfy all constraints of the theory as specified by a sequent (a sequent 
𝑀
⊨
𝐴
𝑁
 of a classifier 
𝐴
 is a pair of subsets 
𝑀
,
𝑁
 of 
𝑇
𝑦
𝑝
(
𝐴
)
 such that 
∀
𝑥
∈
𝑇
𝑜
𝑘
(
𝐴
)
, 
𝑥
⊨
𝐴
𝑀
⇒
𝑥
⊨
𝐴
𝑁
). An infomorphism preserving this additional logical structure is then promoted to a logic infomorphism. In short then, a local logic “identifies” the token(s) satisfying all of the types, the logic infomorphisms are those infomorphisms that transfer token-identification information between local logics, and the channels comprise sets of (logic) infomorphisms encoding mutual constraints that assemble multiple identified tokens. As demonstrated in [74], a sequent of a theory can be weakened to a conditional probability such that a CCD becomes a network of hierarchical Bayesian inference, as reviewed and formulated in [32,33] (cf. [75]), and whose structure is compatible with the variational free energy principle as the latter is fundamental to the precision of perceptual inference [76] (the sequent relation can be weakened by requiring only that if 
𝑥
⊨
𝐴
𝑀
, there is some probability 
𝑃
𝑟
𝑜
𝑏
(
𝑁
|
𝑀
)
 that 
𝑥
⊨
𝐴
𝑁
. Essentially it is how a conditional probability interprets the logical implication “⇒” [77]).
Appendix A.1. Example: Observables in Context
One fundamental example incorporating “context”, instrumental in [33] has the following Chu space ingredients: Consider the following countable (in practice, finite) sets:
(i)
A a set of “events” (in the general sense of the term, e.g., as observed value combinations or atomic), as related to
(ii)
a set B of conditions specifying “objects/contents” or “influences,” and
(iii)
a set R of contexts (or, in certain instances, a set of “detectors”, “measurements” or “methods”).
The set B can be decomposed as 
𝐵
=
𝐵
𝑀
∪
𝐵
𝐶
 (disjoint union), where 
𝐵
𝑀
 contains “objects/contents” or “degrees of freedom” that are observed or measured in some event 
𝑎
∈
𝐴
, and 
𝐵
𝐶
 contains what is not observed in the events in A. This leads to defining a ‘large’ space,
𝑋
:
=
𝐵
×
𝑅
=
(
𝐵
𝑀
∪
𝐵
𝐶
)
×
𝑅
,
	
(A1)
in assuming that 
𝐴
,
𝐵
 and R are subsets of the same (even larger) probability space 
𝑃
 (We do not make any assumptions about corresponding types of probability distributions (e.g., discrete versus continuous) in relationship to 
𝑃
. Neither do we specify the nature of random variables, nor the possible orders of “connectedness” (of distributions)). Thus, based on this data we consider the classifier,
𝐴
=
〈
𝐴
,
𝑋
,
⊨
𝐴
〉
,
	
(A2)
as comprising observables in context, where as in Section 3, the classification relation ‘
⊨
𝐴
’ is realized by the Chu space valuation in the set K
=
{
−
1
,
1
}
. Notably, in [33], ‘
⊨
𝐴
’ can be realized for an inferential process by the conditional probability 
𝑝
(
𝑎
|
𝑥
)
=
𝑝
(
𝑎
|
{
𝑏
,
𝑐
}
)
, whenever defined, for 
𝑎
∈
𝐴
,
𝑏
∈
𝐵
 and 
𝑐
∈
𝑅
, and which for suitable indexing, leads to an information flow of hierarchical Bayesian inference within a CCD [33]. The background to the results in Section 5 here can be found in ([33], Section 7). In particular, ([33], Th. 7.1) states the criteria for intrinsic contextuality (non-co-deployable observables) in terms of noncommutativity of a CCD. Note that the above classifier (Chu space) formulism of contextuality is very general. Special cases of the set 
𝑋
=
𝐵
×
𝑅
 are the sets of binary random variables labelled by a measurement (contents-context) system as basic to the theory of Contextuality-by-Default [78,79]. Much amounts to the question of determining the nature of an empirical model e relative to how a probability distribution can be obtained as the marginals of a global probability distribution on the outcomes to all measurements. For example, e is said to be contextual in [80] if the corresponding probability distribution cannot be obtained by such global means. This has a compatible interpretation in terms of the non-existence of a global section of a sheaf defined relative to a “measurement cover” in [81]. These methods of studying contextuality are also very general, and as for those of [33], can extend beyond quantum theory to such disciplines as linguistics and psychology. To see the explicit connections between these various approaches would indeed be a worthwhile undertaking.
References
Bousso, R. The holographic principle. Rev. Mod. Phys. 2002, 74, 825–874. [Google Scholar] [CrossRef]
Hooft, G. Dimensional reduction in quantum gravity. In Salamfestschrift; Ali, A., Ellis, J., Randjbar-Daemi, S., Eds.; World Scientific: Singapore, 1993; pp. 284–296. [Google Scholar]
Susskind, L. The world as a hologram. J. Math. Phys. 1995, 36, 6377–6396. [Google Scholar] [CrossRef]
Bekenstein, J.D. Black holes and information theory. Contemp. Phys. 2004, 45, 31–43. [Google Scholar] [CrossRef]
Fields, C.; Marcianò, A. Holographic screens are classical information channels. Quant. Rep. 2020, 2, 326–336. [Google Scholar] [CrossRef]
Fields, C.; Glazebrook, J.F. Representing measurement as a thermodynamic symmetry breaking. Symmetry 2020, 12, 810. [Google Scholar] [CrossRef]
Bartlett, S.D.; Rudolph, T.; Spekkens, R.W. Reference frames, superselection rules, and quantum information. Rev. Mod. Phys. 2007, 79, 555–609. [Google Scholar] [CrossRef]
Fields, C.; Marcianò, A. Sharing nonfungible information requires shared nonfungible information. Quant. Rep. 2019, 1, 252–259. [Google Scholar] [CrossRef]
Fuchs, C.A.; Schack, R. Quantum-bayesian coherence. Rev. Mod. Phys. 2013, 85, 1693–1715. [Google Scholar] [CrossRef]
Fields, C. Some consequences of the thermodynamic cost of system identification. Entropy 2018, 20, 797. [Google Scholar] [CrossRef] [PubMed]
Susskind, L. Entanglement is not enough. arXiv 2014, arXiv:1411.0690. [Google Scholar] [CrossRef]
Rovelli, C. Black holes have more states than those giving the Bekenstein-Hawking entropy: A simple argument. arXiv 2017, arXiv:1710:00218. [Google Scholar]
Rovelli, C. The subtle unphysical hypothesis of the firewall theorem. Entropy 2019, 21, 839. [Google Scholar] [CrossRef]
Almheiri, A.; Hartman, T.; Maldacena, J.; Shaghoulian, E.; Tajdini, A. The entropy of Hawking radiation. arXiv 2000, arXiv:2006.06872v1. [Google Scholar]
Almheiri, A.; Marolf, D.; Polchinski, J.; Sully, J. Black Holes: Complementarity or firewalls? J. High Energy Phys. 2013, 2013, 62. [Google Scholar] [CrossRef]
Tipler, F.J. Quantum nonlocality does not exist. Proc. Natl. Acad. Sci. USA 2014, 111, 11281–11286. [Google Scholar] [CrossRef]
Hooft, G.T. Deterministic quantum mechanics: The mathematical equations. Front. Phys. 2020, 8, 253. [Google Scholar] [CrossRef]
Zurek, W.H. Decoherence, einselection, and the quantum origins of the classical. Rev. Mod. Phys. 2003, 75, 715–775. [Google Scholar] [CrossRef]
Schlosshauer, M. Decoherence and the Quantum-To-Classical Transition; Springer: Berlin, Germany, 2007. [Google Scholar]
Schlosshauer, M. Quantum decoherence. Phys. Rep. 2019, 831. [Google Scholar] [CrossRef]
Landsman, N.P. Observation and superselection in quantum mechanics. Stud. Hist. Philos. Mod. Phys. 1995, 26, 45–73. [Google Scholar] [CrossRef]
Zurek, W.H. Decoherence, einselection and the existential interpretation (the rough guide). Philos. Trans. R. Soc. A 1998, 356, 1793–1821. [Google Scholar] [CrossRef]
Chiribella, G.; D’Ariano, G.M.; Perinotti, P. Quantum Theory: Informational Foundations and Foils; Chiribella, G.G., Spekkens, R.W., Eds.; Springer: Dordrecht, The Netherland, 2016; pp. 171–221. [Google Scholar]
Dugić, M.; Jeknixcx, J. What is “system”: Some decoherence-theory arguments. Int. J. Theor. Phys. 2006, 45, 2215–2225. [Google Scholar] [CrossRef]
Dugić, M.; Jeknixcx, J. What is “system”: The information-theoretic arguments. Int. J. Theor. Phys. 2008, 47, 805–813. [Google Scholar] [CrossRef]
Fields, C. Quantum Darwinism requires an extra-theoretical assumption of encoding redundancy. Int. J. Theor. Phys. 2010, 49, 2523–2527. [Google Scholar] [CrossRef]
Kastner, R.E. ‘Einselection’ of pointer observables: The new H-theorem? Stud. Hist. Philos. Mod. Phys. 2014, 48, 56–58. [Google Scholar] [CrossRef]
Barwise, J.; Seligman, J. Information Flow: The Logic of Distributed Systems; Cambridge Tracts in Theoretical Computer Science, 44; Cambridge University Press: Cambridge, UK, 1997. [Google Scholar]
Adámek, J.; Herrlich, H.; Strecker, G.E. Abstract and Concrete Categories: The Joy of Cats; Wiley: New York, NY, USA, 2004; Available online: http://katmat.math.uni-bremen.de/acc (accessed on 26 May 2019).
Awodey, S. Category Theory. In Oxford Logic Guides, 62; Oxford University Press: Oxford, UK, 2010. [Google Scholar]
Goguen, J.A. A categorical manifesto. Math. Struct. Comput. Sci. 1991, 1, 49–67. [Google Scholar] [CrossRef]
Fields, C.; Glazebrook, J.F. A mosaic of Chu spaces and Channel Theory I: Category-theoretic concepts and tools. J. Exp. Theor. Artif. Intell. 2019, 31, 177–213. [Google Scholar] [CrossRef]
Fields, C.; Glazebrook, J.F. Information flow in context-dependent hierarchical Bayesian inference. J. Expt. Theor. Artif. Intell. 2020, in press. [Google Scholar] [CrossRef]
Chitambar, E.; Leung, D.; Mančinska, L.; Ozols, M.; Winter, A. Everything you always wanted to know about LOCC (but were afraid to ask). Comms. Math. Phys. 2014, 328, 303–326. [Google Scholar] [CrossRef]
Weinstein, A. Groupoids: Unifying internal and external symmetry. Not. AMS 1996, 43, 744–752. [Google Scholar]
Brown, R. Topology and Groupoids; Ronald Brown: Deganwy, UK, 2006; Available online: www.groupoids.org.uk (accessed on 1 February 2021).
Zanardi, P. Virtual quantum subsystems. Phys. Rev. Lett. 2001, 87, 077901. [Google Scholar] [CrossRef] [PubMed]
Zanardi, P.; Lidar, D.A.; Lloyd, S. Quantum tensor product structures are observable-induced. Phys. Rev. Lett. 2004, 92, 060402. [Google Scholar] [CrossRef] [PubMed]
Smith, J.E.; Nair, R. The architecture of virtual machines. IEEE Comput. 2005, 38, 32–38. [Google Scholar] [CrossRef]
Susskind, L.; Thorlacius, L.; Uglum, J. The stretched horizon and black hole complementarity. Phys. Rev. D 1993, 48, 3743–3761. [Google Scholar] [CrossRef]
Maldacena, J.; Susskind, L. Cool horizons for entangled black holes. Fortschritte Der Phys. 2013, 61, 781–811. [Google Scholar] [CrossRef]
Maldecana, J.; Pimental, G.L. Entanglement entropy in de Sitter space. J. High Energy Phys. 2013, 2013, 38. [Google Scholar] [CrossRef]
Choudhury, S.; Panda, S.; Singh, R. Bell violation in the sky. Eur. Phys. J. C 2017, 77, 60. [Google Scholar] [CrossRef]
Kanno, S.; Soda, J. Infinite violation of Bell inequalities in inflation. Phys. Rev. D 2017, 96, 083501. [Google Scholar] [CrossRef]
Rangamani, M.; Takayanagi, T. Holographic entanglement entropy. In Holographic Entanglement Entropy; Lecture Notes in Physics; Springer: Cham, Switzerland, 2017; Volume 931, pp. 35–47. [Google Scholar]
Nielsen, M.A.; Chuang, I.L. Quantum Computation and Quantum Information; Cambridge Univeraity Press: Cambridge, UK, 2000. [Google Scholar]
Vazirani, U.; Vidick, T. Fully device-independent quantum key distribution. Phys. Rev. Lett. 2014, 113, 140501. [Google Scholar] [CrossRef] [PubMed]
Situ, H.; Qiu, D.W. Investigating the implementation of restricted sets of multiqubit operations on distant qubits: A communication complexity perspective. Quant. Inform. Process. 2011, 10, 609–618. [Google Scholar] [CrossRef]
Zou, X.; Qiu, D.W. Three-step semiquantum secure direct communication protocol. Sci. China G 2014, 57, 1696–1702. [Google Scholar] [CrossRef]
Landauer, R. Irreversibility and heat generation in the computing process. IBM J. Res. Dev. 1961, 5, 183–195. [Google Scholar] [CrossRef]
Bennett, C.H. The thermodynamics of computation. Int. J. Theor. Phys. 1982, 21, 905–940. [Google Scholar] [CrossRef]
Bohr, N. The quantum postulate and the recent development of atomic theory. Nature 1928, 121, 580–590. [Google Scholar] [CrossRef]
Verlinde, E.; Verlinde, H. Black hole entanglement and quantum error correction. J. High Energy Phys. 2013, 107. [Google Scholar] [CrossRef]
Wigner, E.P. Remarks on the mind-body question. In The Scientist Speculates; Good, I.J., Ed.; Heinemann: London, UK, 1961; pp. 284–302. [Google Scholar]
Brukner, C. A no-go theorem for observer-independent facts. Entropy 2018, 20, 350. [Google Scholar] [CrossRef] [PubMed]
Bong, K.-W.; Utreras-Alarcón, A.; Ghafari, F.; Liang, Y.-C.; Tischler, N.; Cavalcanti, E.G.; Pryde, G.J.; Wiseman, H.M. A strong no-go theorem on the Wigner’s friend paradox. Nat. Phys. 2020. [Google Scholar] [CrossRef]
Mermin, D. Hidden variables and the two theorems of John Bell. Rev. Mod. Phys. 1993, 65, 803–815. [Google Scholar] [CrossRef]
Rice, H.G. Classes of recursively enumerable sets and their decision problems. Trans. Am. Math. Soc. 1953, 74, 358–366. [Google Scholar] [CrossRef]
McCarthy, J.; Hayes, P.J. Some philosophical problems from the standpoint of artificial intelligence. In Machine Intelligence; Michie, D., Meltzer, B., Eds.; Edinburgh University Press: Edinburgh, UK, 1969; Volume 4, pp. 463–502. [Google Scholar]
Dietrich, E.; Fields, C. Equivalence of the Frame and Halting problems. Algorithms 2020, 13, 175. [Google Scholar] [CrossRef]
Addazi, A.; Chen, P.; Fabrocini, F.; Fields, C.; Greco, E.; Lutti, M.; Marcianò, A.; Pasechnik, R. Generalized holographic principle, gauge invariance and the emergence of gravity à la Wilczek. Front. Astron. Space Sci. in press. Available online: https://www.frontiersin.org/articles/10.3389/fspas.2021.563450/abstract (accessed on 1 February 2021).
Wheeler, J.A. Law without law. In Quantum Theory and Measurement; Wheeler, J.A., Zurek, W.H., Eds.; Princeton University Press: Princeton, NJ, USA, 1983; pp. 182–213. [Google Scholar]
Mermin, N.D. Making better sense of quantum mechanics. Rep. Prog. Phys. 2018, 82, 12002. [Google Scholar] [CrossRef] [PubMed]
Muller, M.P. Law without law: From observer states to physics via algorithmic information theory. Quantum 2020, 4, 301. [Google Scholar] [CrossRef]
Verlinde, E. On the origin of gravity and the laws of Newton. J. High Energy Phys. 2011, 2011, 29. [Google Scholar] [CrossRef]
Ng, Y.J. Entropy and gravitation. From black hole computers to dark energy and dark matter. Entropy 2019, 21, 1035. [Google Scholar] [CrossRef]
Barr, M. *-Autonomous Categories, with an Appendix by Po Hsiang Chu; Lecture Notes in Mathematics 752; Springer: Berlin, Germany, 1979. [Google Scholar]
Pratt, V. Chu spaces. In School on Category Theory and Applications (Coimbra 1999); Volume 21 of Textos Mat. Sér. B; University of Coimbra: Coimbra, Portugal, 1999; pp. 39–100. [Google Scholar]
Pratt, V. Chu spaces from the representational viewpoint. Ann. Pure Appl. Log. 1999, 96, 319–333. [Google Scholar] [CrossRef]
Collier, J. Information, causation and computation. In Information and Computation: Essays on Scientific and Philosophical Foundations of Information and Computation; World Scientific Series in Information Studies; Crnkovic, G.D., Burgin, M., Eds.; World Scientific Press: Hackensack, NJ, USA, 2011; Volume 2, pp. 89–105. [Google Scholar]
Sorkin, R.D. Finitary substitute for continuous topology. Int. J. Theoret. Phys. 1991, 30, 923–947. [Google Scholar] [CrossRef]
Sorkin, R.D. Spacetime and causal sets. In Relativity and Gravitation: Classical and Quantum; D’Olivo, J.C., Nahmad-Achar, E., Rosenbaum, M., Ryan, M.P., Jr., Urrutla, L.F., Zertuche, F., Eds.; World Scientific: Singapore, 1991; pp. 150–173. [Google Scholar]
Gratus, J.; Porter, T. A spatial view of information. Theor. Comp. Sci. 2006, 365, 206–215. [Google Scholar] [CrossRef]
Allwein, G.; Moskowitz, I.S.; Chang, L.-W. A New Framework for Shannon Information Theory; Technical Report A801024; Naval Research Laboratory: Washington, DC, USA, 2004; 17p. [Google Scholar]
Barwise, J. Information and Impossibilities. Notre Dame J. Form. Log. 1997, 38, 488–515. [Google Scholar] [CrossRef]
Friston, K.J.; Kiebel, S. Predictive coding under the free-energy principle. Philos. Trans. R. Soc. 2009, 364, 1211–1221. [Google Scholar] [CrossRef] [PubMed]
Adams, E.W. A Primer of Probabilistic Logic; University of Chicago Press: Chicago, IL, USA, 1998. [Google Scholar]
Dzhafarov, E.N.; Kujala, J.V.; Cervantes, V.H. Contextuality-by-default: A brief overview of concepts and terminology. In Lecture Notes in Computer Science 9525; Atmanspacher, H., Filik, T., Pothos, E., Eds.; Springer: Heidelberg, Germany, 2016; pp. 12–23. [Google Scholar]
Dzharfarov, E.N.; Kon, M. On universality of classical probability with contextually labeled random variables. J. Math. Psychol. 2018, 85, 17–24. [Google Scholar] [CrossRef]
Abramsky, S.; Barbosa, R.S.; Mansfield, S. Contextual fraction as a measure of contextuality. Phys. Rev. Lett. 2017, 119, 050504. [Google Scholar] [CrossRef]
Abramsky, S.; Brandenburger, A. The sheaf-theoretic structure of non-locality and contextuality. New J. Phys. 2011, 13, 113036. [Google Scholar] [CrossRef]
	
Publisher’s Note: MDPI stays neutral with regard to jurisdictional claims in published maps and institutional affiliations.

© 2021 by the authors. Licensee MDPI, Basel, Switzerland. This article is an open access article distributed under the terms and conditions of the Creative Commons Attribution (CC BY) license (http://creativecommons.org/licenses/by/4.0/).
Share and Cite
      
MDPI and ACS Style

Fields, C.; Glazebrook, J.F.; Marcianò, A. Reference Frame Induced Symmetry Breaking on Holographic Screens. Symmetry 2021, 13, 408. https://doi.org/10.3390/sym13030408

AMA Style


Fields C, Glazebrook JF, Marcianò A. Reference Frame Induced Symmetry Breaking on Holographic Screens. Symmetry. 2021; 13(3):408. https://doi.org/10.3390/sym13030408

Chicago/Turabian Style


Fields, Chris, James F. Glazebrook, and Antonino Marcianò. 2021. "Reference Frame Induced Symmetry Breaking on Holographic Screens" Symmetry 13, no. 3: 408. https://doi.org/10.3390/sym13030408

APA Style


Fields, C., Glazebrook, J. F., & Marcianò, A. (2021). Reference Frame Induced Symmetry Breaking on Holographic Screens. Symmetry, 13(3), 408. https://doi.org/10.3390/sym13030408

Note that from the first issue of 2016, this journal uses article numbers instead of page numbers. See further details here.
Article Metrics
Citations
Crossref
 
22
Scopus
 
25
Web of Science
 
23
ads
 
14
Google Scholar
 
[click to view]
Article Access Statistics
Article access statistics
Article Views
18. Mar
19. Mar
20. Mar
21. Mar
22. Mar
23. Mar
24. Mar
25. Mar
26. Mar
27. Mar
28. Mar
29. Mar
30. Mar
31. Mar
1. Apr
2. Apr
3. Apr
4. Apr
5. Apr
6. Apr
7. Apr
8. Apr
9. Apr
10. Apr
11. Apr
12. Apr
13. Apr
14. Apr
15. Apr
16. Apr
17. Apr
18. Apr
19. Apr
20. Apr
21. Apr
22. Apr
23. Apr
24. Apr
25. Apr
26. Apr
27. Apr
28. Apr
29. Apr
30. Apr
1. May
2. May
3. May
4. May
5. May
6. May
7. May
8. May
9. May
10. May
11. May
12. May
13. May
14. May
15. May
16. May
17. May
18. May
19. May
20. May
21. May
22. May
23. May
24. May
25. May
26. May
27. May
28. May
29. May
30. May
31. May
1. Jun
2. Jun
3. Jun
4. Jun
5. Jun
6. Jun
7. Jun
8. Jun
9. Jun
10. Jun
11. Jun
12. Jun
13. Jun
14. Jun
15. Jun
0k
1k
2k
3k
4k
5k
For more information on the journal statistics, click here.
Multiple requests from the same IP address are counted as one view.
Symmetry, EISSN 2073-8994, Published by MDPI
RSS Content Alert
Further Information
Article Processing Charges
Pay an Invoice
Open Access Policy
Contact MDPI
Jobs at MDPI
Guidelines
For Authors
For Reviewers
For Editors
For Librarians
For Publishers
For Societies
For Conference Organizers
MDPI Initiatives
Sciforum
MDPI Books
Preprints.org
Scilit
SciProfiles
Encyclopedia
JAMS
Proceedings Series
Follow MDPI
LinkedIn
Facebook
X
© 1996-2026 MDPI (Basel, Switzerland) unless otherwise stated
Disclaimer Legal Notice Terms and Conditions Privacy Policy Privacy Settings Accessibility