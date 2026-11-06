# Slides — 2nde hybridation (Learnable Galerkin methods)

Notes de travail issues de la discussion du 2026-11-02. À reprendre plus tard.
**État (2026-11-05) : structure alignée sur la 1ère hybridation** (voir « Organisation des fichiers »), une slide écrite, le reste en trous `\todo`. On avance petit à petit, dans l'ordre.

## Sources dans le manuscrit (`/home/flecourtier/Bureau/manuscrit_thesis_2026`)

- Dernier chapitre : `chapters/learnable_methods.tex` (chap. 5), avec ses sous-fichiers dans `chapters/learnable_methods/`
  - `dimensional_study.tex` : étude dimensionnelle (FEM/DG contre réseaux, d = 1..4)
  - `PINN_1D.tex` (+ `PINN_1D/fem_patchwise.tex`, `fem_cellwise.tex`, `dg_patchwise.tex`, `dg_cellwise.tex`) : cas non paramétrique 1D
  - `PINN_3D.tex` : cas non paramétrique 3D
  - `NO_1D.tex` (+ `NO_1D/linear_transport.tex`, `turning_point.tex`) : cas paramétrique
- Formalisme et bibliothèque : `chapters/sciml_library.tex` (chap. 4), en particulier §« Elements of Differentiable Programming », §« Solve framework » (Newton sans matrice, JVP) et §« Learnable formalization of the Galerkin schemes »
- Conclusion de la thèse : `chapters/conclusion.tex`
- Figures : dossier `figures/` du manuscrit (`figures/learnable_methods/...`)

## Budget de temps (total 45 min)

| Partie | Durée |
|---|---|
| Intro + méthodes classiques + conclusion | 10 min |
| 1ère hybridation (EFEM avec PINN) | 15 min |
| **2nde hybridation (Galerkin apprenable)** | **20 min** |

## Découpage de la 2nde hybridation (décidé le 2026-11-05, remplace A/B/C ci-dessous)

Plus de « Presentation / Numerical results » comme pour la 1ère hybridation : une section commune, puis **une section par cas**, chacune avec son principe et ses résultats, puis la conclusion (fichiers `sections/contrib2/{introduction,pinn,neural_operator,conclusion}.tex`).

1. **Introduction** (`introduction.tex`) : transition depuis la 1ère hybridation ; autodiff en bref (1 slide au plus, Scimba/JAX) ; slide « Two settings: PINN and neural operator » (PINN : μ fixé, perte PINN, un réseau par cas ; NO : famille de μ, perte supervisée, un réseau pour tous les μ).
2. **PINN** (`pinn.tex`, nav « PINN ») : « Neural-enriched Galerkin scheme » et « Learning θ: differentiating through the solver » **écrites pour ce cas : μ fixé, omis des formules et des schémas** ; puis algorithme et cas 1D, choix du multiplicateur, résultats 1D, cas 3D, coût 3D.
3. **Neural operator** (`neural_operator.tex`, nav « Neural operator ») : « Neural operator: principle » = **le même schéma avec μ** (flèche μ vers le solveur, perte supervisée sur un lot de μ, un solve adjoint par μ, trois bases : partagée, dépendant de μ, dépendant du champ ; la base partagée est un « opérateur constant » au sens du manuscrit, `NO_1D.tex`), puis transport linéaire et point de retournement.
4. **Conclusion** (`conclusion.tex`).

Conséquences : la slide de principe n'a plus à jongler avec μ (ligne « μ fixed » du bloc Galerkin, formules sans μ). `A_h(θ;μ)`, `β_h(θ;μ)`, `u_{h,θ}(x;μ)` ne reviennent que dans la section Neural operator.

## Plan proposé pour la 2nde hybridation (~10 slides de contenu, ~1,7 min/slide)

### A. Pourquoi et principe (~4 min, 3 slides)
1. **Transition depuis la 1ère hybridation** (voir « Décision 2 » plus bas) : du prior gelé à une base apprise à travers le solveur.
2. **Principe du schéma de Galerkin apprenable** : base de Lagrange × multiplicateur(réseau θ). Le solveur est une fonction de θ, on entraîne θ à travers le solveur. Schéma θ → base → schéma → U → loss.
3. **Différentier à travers le solveur** : autodiff, différentiation implicite et méthode adjointe, Newton sans matrice (JVP). Fusion des slides « Autodiff » et « Optimization problem » du squelette. Mentionner Scimba ici.

### B. Enrichissement non paramétrique, perte PINN (~8 min, 4 slides)
4. Algorithme et cas test 1D (advection-diffusion avec couche limite), choix du multiplicateur (positif, poids bornés).
5. Résultats 1D, FEM par patch : erreur L² en fonction de N_dofs, base classique contre base apprise (plusieurs ordres de grandeur). Une ligne sur les limites (échec pour k = 1, sensibilité à l'initialisation).
6. Cas 3D : cas test et gains par rapport à la FEM classique.
7. 3D, coût à temps égal : mêmes ordres d'erreur avec bien moins de dofs (message honnête sur le coût).

### C. Enrichissement paramétrique, opérateur neuronal (~7 min, 3 slides)
8. Principe : un seul réseau entraîné sur toute la famille de μ à partir de solutions de référence, sans réentraînement. Trois bases comparées : partagée, dépendant de μ, dépendant du champ de transport.
9. Transport linéaire : gains médians jusqu'à ~7000 selon (N_cells, k).
10. Point de retournement : la couche se déplace avec μ, la dépendance en μ devient indispensable (gain médian ~9).

### D. Bilan (~1 min, dans la conclusion générale `sections/common/conclusion.tex`)
- Conclusion, limites, perspectives de l'étude (exploratoire : peu de cas tests, 1D en paramétrique, entraînement sensible à l'initialisation, coût d'entraînement non étudié partout, pas d'estimation d'erreur).
- Perspective qui relie les deux contributions : **geler le multiplicateur appris** → on retombe sur l'approche multiplicative de la 1ère hybridation, dont l'estimation a priori (type Céa) s'applique.

## Annexe (pour les questions, pas dans le corps)
- Variantes cellwise et DG
- Étude de capacité (degré, nombre de cellules, largeur du réseau)
- Détail des paramétrisations du multiplicateur (affine, positif, positif borné)
- Modules de Scimba, φ-FEM et domaine à 3 lobes
- Problème inverse
- (Éventuellement) l'étude dimensionnelle complète

## Décisions prises

1. **FEM pour le cas non paramétrique, DG pour le cas paramétrique** (corrigé le 2026-11-02) : PINN 1D et 3D en FEM par patch ; le cas paramétrique (`NO_1D`, transport linéaire et point de retournement) n'existe qu'en **DG SIPG patchwise** dans le manuscrit (solutions de référence comprises), donc le DG est présenté dans le corps. Cellwise, upwind et NIPG vont en annexe.
   - Background : on part du **DG** (cadre général, comme au chapitre 4 : `extension_fem.tex`), puis le **CG** s'obtient avec une table de connectivité `conn` qui identifie les dofs partagés et supprime les termes d'interface. Les anciennes slides FEM sont reprises mais adaptées à cet enchaînement.

## Méthode de travail décidée

On commence par le **cœur du sujet** (slides 2 à 10 du plan). La slide de transition depuis la 1ère hybridation (slide 1) et le sort de l'étude dimensionnelle sont **mis de côté, avec un trou laissé dans le squelette** (`\todo`). On y reviendra quand les autres parties (intro, 1ère hybridation, conclusion) seront en place, car ces choix en dépendent.

## Organisation des fichiers (faite)

Un fichier `.tex` par section dans chaque hybridation, `main.tex` ne fait que des `\input` :
- `sections/contrib1/` : `presentation.tex`, `numerical_results.tex`, `conclusion.tex`
- `sections/contrib2/` : même schéma que contrib1 (décidé le 2026-11-05) :
  - (ancien découpage, remplacé le 2026-11-05 : voir « Découpage de la 2nde hybridation » en haut) `presentation.tex` : transition, principe, gradient ;
  - `numerical_results.tex` : `\subsection{Non-parametric}` (algorithme + cas 1D, choix du multiplicateur, résultats 1D, 3D, coût 3D) puis `\subsection{Parametric}` (principe, transport linéaire, point de retournement). Chaque sous-section commence par sa slide d'algorithme/perte, car la perte diffère selon le cas ;
  - `conclusion.tex` : limites de l'étude et perspective « geler le multiplicateur » (conclusion propre à la 2nde hybridation ; la conclusion générale ne reprend que le lien entre les deux).
- Préambule de `presentation.tex` : ajout de `bm`, de `\argmin` et des bibliothèques TikZ `positioning`, `arrows.meta`.
- Pour compiler sans toucher à `presentation.pdf` du dépôt : `lualatex -interaction=nonstopmode -output-directory=<dossier temporaire> presentation.tex`. Ne pas utiliser `latexmk` sans biber : il s'arrête sur une erreur bibtex et ne reconstruit plus le PDF (le rendu reste alors celui d'avant la modification).

## Avancement des slides (2nde hybridation)

- [x] « Neural-enriched Galerkin scheme » (ex « Learnable Galerkin scheme: principle ») (maintenant dans `pinn.tex`) : réécrite le 2026-11-05 (version « A′ »). Schéma en 4 overlays (`images/contrib2/learnable_scheme/` : **3 boîtes vertes** neural network → enriched space → Galerkin solver, puis sortie `β_h(θ)` (overlay 3) puis `u_{h,θ}` (overlay 4) ; même style que l'intro, via `images/hybscheme.sty`) + un bloc de texte par étape. Décisions : **cadre DG** (note « CG : same with conn » sur la slide) ; **(modifié le 2026-11-05 : μ fixé et omis dans la section PINN : `A_h(θ)`, `β_h(θ)`, `b_h(θ)`, `u_{h,θ}(x)` ; la version avec μ vient dans la section Neural operator)** ; `m_θ(x)` sans μ dans le principe (μ n'entre dans le multiplicateur que dans les variantes paramétriques) ; **corps = un seul réseau sur Ω partagé par toutes les cellules** (base « patchwise » du manuscrit, le mot n'est pas sur la slide), annexe = cellwise (principe et résultats) ; **pas de figure de bases enrichies ici** (c'est le contenu des résultats numériques) ; le **problème bi-niveau** passe sur la slide suivante. Corrections de l'auteure conservées : `\phi` (pas `\varphi`), `m_θ` sans « positif », `\mathcal{J}` perte générique.
- [x] « Learning θ: differentiating through the solver » : écrite le 2026-11-05 (option A, une slide, 3 overlays). Figure `images/contrib2/learnable_scheme/` pages 5 et 6 (regroupée avec le schéma de la slide précédente le 2026-11-05 ; ancien dossier `gradient_scheme/` devenu inutile) (même schéma que la page 4 de `learnable_scheme` : flèche de sortie avec `β_h(θ)` au-dessus et `u_{h,θ}` au bout ; la perte est en dessous, branchée sur `u_{h,θ}`, puis retour en pointillés vers le réseau) ; blocs : problème bi-niveau, gradient (termes direct / à travers les dofs, le second en rouge), adjoint (un solve linéaire au lieu de `N_θ`). `μ` fixé, formules sans `μ` comme le manuscrit. Le tableau des 4 étapes n'est pas sur la slide (répète le bloc adjoint) : à mettre en annexe si besoin. Scimba / JAX : à l'oral.
- [x] ex `numerical_results.tex` : réparti dans `pinn.tex` (5 slides en trous `\todo`) et `neural_operator.tex` (3 slides en trous `\todo`) ; `introduction.tex` : transition, autodiff, deux cas (trous `\todo`)
- [x] `pinn.tex` : slide « 1D test case and PINN loss » écrite (cas test, perte PINN, évaluation contre la référence FEM) ; l'algorithme d'entraînement va en annexe
- [x] `pinn.tex` : slide « Learned basis: affine multiplier » (solutions + erreurs ℚ3, puis multiplicateur, base classique, base enrichie ; figures PNG du manuscrit) ; suivante : « Choice of the multiplier » (option A : affine → positif → positif à poids bornés)
- [x] `pinn.tex` : « Choice of the multiplier » (profils + tableau forme/signe de l'affine) et « Retained multiplier: positive, bounded weights » (histogramme des gains puis heatmap des 20 configurations) ; figures PNG du manuscrit sous `images/contrib2/multiplier/`. La slide « 1D results » est supprimée (déjà couverte par « Learned basis: affine multiplier » et « Results at fixed budget »). Suivante : « 3D test case »
- [x] `pinn.tex` : cas 3D en 3 slides (écrites 2026-11-05, à relire) : « 3D test case » (PDE, perte résiduelle seule, évaluation sur solution manufacturée), « 3D: learned multiplier » (3 overlays : réglages, animation `animate` de u exacte et m_θ — lue par Acrobat, frames sous `images/contrib2/pinn3d/anim_diagonal/` —, puis solution et multiplicateur sur la diagonale), « 3D: gain and cost » (tableau à temps égal du manuscrit lu depuis le CSV, puis 3 messages : erreur ×6 au mieux (×1 pour la seed 3), 534 à 729 fois moins de dofs, temps appris dominé par la compilation ; courbe erreur/N_dofs et ligne « N_dofs fixé » retirées, gardées pour les questions). Une slide de plus que prévu (3 au lieu de 2). Suivante : `neural_operator.tex`.
- **Section Neural operator : 4 slides (décidé le 2026-11-05, la section PINN en ayant déjà 9)** : (1) « Neural operator: principle » ; (2) « Three enriched bases » (schéma TikZ, une ligne par base : partagée `m_θ(x)`, dépendant de μ `m_θ(x,μ)`, dépendant du champ `m_θ(x,z)`, `z=ℰ_θ(C(·,μ))`, le terme qui change en rouge ; 321 / 353 / 1381 poids) ; (3) « Linear transport » (cartes de gain (8,2) des trois bases, puis multiplicateurs) ; (4) « Turning point » (multiplicateurs : bosse centrale du partagé contre pic à la couche du dépendant de μ ; gains (8,2) : 0.5–2.1 contre 85–366). Tableaux complets et cartes d'erreur en annexe. Limites (1D, 32 μ d'entraînement, meilleure graine, coût d'entraînement) en une ligne sur la slide 4 ou dans `conclusion.tex`.
- [x] `neural_operator.tex` : « Neural operator: principle » écrite le 2026-11-05 (option A, une slide, 3 overlays). Figure `images/contrib2/operator_scheme/` (3 pages : μ vers le solveur et sorties ; perte `𝒥_data` ; gradient). Blocs : schéma paramétrique (`A_h(θ;μ)β_h(θ;μ)=b_h(θ;μ)`), perte supervisée, entraînement (lot de 16, un adjoint par μ) et validité (μ jamais vus, pas de réentraînement). Les trois bases viennent sur la slide suivante.
- [x] `neural_operator.tex` : les trois autres slides écrites le 2026-11-05, **à relire** : « Three enriched bases » (figure `images/contrib2/enriched_bases/`, 3 pages, une ligne par base), « Linear transport » (3 overlays : solutions de référence, cartes de gain (8,2) des trois bases, multiplicateurs ; PNG du manuscrit sous `images/contrib2/operator/lt_*`), « Turning point » (4 overlays : solutions de référence en 5 μ, multiplicateurs, gains (8,2), ligne « Limits »). Chiffres relus dans les CSV et le texte du manuscrit (gains médians transport linéaire : partagée 14–167, μ-dep 116–7143). Le champ-dépendant n'est présenté que sur le transport linéaire. Note : après ajout d'une slide, **compiler deux fois** (le numéro de page, TikZ `current page`, n'est correct qu'à la 2e passe).
- [x] `conclusion.tex` : écrite le 2026-11-05 en l'absence de l'auteure, **à relire** (1 slide, 3 blocs : résultats 1D / 3D / opérateur, limites, perspective « geler le multiplicateur »). Chiffres relus dans `chapters/learnable_methods/` et `chapters/conclusion.tex`. Conclusion générale (`common/conclusion.tex`) et `contrib1/conclusion.tex` : toujours des trous (pas demandés).
- [x] `introduction.tex` : les trois slides écrites le 2026-11-05 en l'absence de l'auteure, **à relire** : « From frozen prior to learned basis » (transition, figure `images/contrib2/transition/`, 2 pages, 4 overlays ; à garder ou mettre de côté), « ScimBa: differentiable programming » (ex « Automatic differentiation », refaite le 2026-11-06 (**version actuelle, 3e (QR code « See ScimBa » en bas à droite au dernier overlay, repris de 2026_07_03 : `images/qrcode_scimba.png`) : besoin → pourquoi l'autodiff (pastilles symbolique / différences finies / autodiff) → ScimBa en JAX avec `vjp`, `vmap`, `jit`**, validée par l'auteure ; 2e version : besoin, puis « outil JAX → usage » `vjp/jvp/vmap/jit`, puis une ligne d'autodiff ; une seule note de bas de frame `\myfootcite{Bradbury2018,Baydin2018}` avec `\myfootcite[1]{Baydin2018}` dans le tableau et `\myfootcite[2]{Bradbury2018}` sur JAX (numéro optionnel ajouté à la macro de `presentation.tex`)) ; le bloc déroulé/implicite est retiré ; citations `Baydin2018`, `Bradbury2018` recopiées du manuscrit dans `biblio.bib`), (slide supprimée le 2026-11-06, option B : l'annonce des deux cas se fait à l'oral et par les `\SectionNote` des pages de section) ex-« Two settings: fixed μ or family of μ » (deux cartes vertes *loss / network / new μ* + bandeau ; refaite le 2026-11-06 : l'axe est μ, pas PINN contre opérateur). Alternatives non implémentées : `PROPOSITIONS_hybridation2.md`. Notes d'oral dans `NOTES_ORAL.md`.

## Points à ne pas oublier pour la suite

- **Choix du multiplicateur** : il ne doit pas être présenté comme « positif » dans le principe, c'est un résultat de l'étude 1D. À raconter sur une slide du cas 1D : multiplicateur affine → échecs (changement de signe = pire que la FEM, échelon = entraînement qui explose) → positif `m_θ = exp(2 tanh(v_θ))` → poids bornés par softclip (facteur 4). Chiffres de la comparaison à 400 entraînements : explosions 45 / 45 / 0 ; battent la FEM 281 / 287 / 376 (affine / positif / positif à poids bornés). Cela pourrait faire une slide de plus dans la partie B.
- La perte 𝒥 est définie plus tard dans chaque cas : perte PINN (non paramétrique), perte sur solutions de référence (paramétrique).

## Décisions en suspens

2. **Étude dimensionnelle** : elle justifie l'hybridation en général (donc aussi la 1ère), elle n'a pas sa place comme ouverture de la 2nde. Il faut une suite logique avec la 1ère hybridation. **Proposition (à valider)** :
   - Déplacer le message « limite d'expressivité des bases classiques » dans l'intro commune (Context and motivation), avec un seul chiffre ou une seule figure : à d = 4, FEM/DG ~5·10⁻² avec 38 000 à 65 000 dofs, contre un réseau de 641 poids sous 10⁻⁴. Sinon, backup.
   - Ouvrir la 2nde hybridation par une slide de transition, construite à partir de la conclusion de la thèse : la 1ère hybridation donne des garanties (convergence conservée, gain explicite) mais le gain est borné par la qualité du prior, qui est entraîné puis gelé, indépendamment du schéma (cf. Navier–Stokes à Rayleigh élevé). Question qui lance la 2nde : et si la base était apprise *à travers le solveur*, donc adaptée à la solution du schéma ? Tableau comparatif prior gelé contre base apprise : garanties a priori contre expressivité et adaptation.
3. **Poids du cas paramétrique** : 7 min proposés (3 slides), c'est la réponse à la limite « coût des résolutions répétées » mais c'est exploratoire (1D seulement). Variante : 4 + 3 + 2 slides pour donner plus de place au 3D. À débattre ensuite.

## Remarque sur l'état du squelette

`sections/contrib2/main.tex` ne fait plus que quatre `\input` (introduction, pinn, neural_operator, conclusion). `sections/contrib2/appendix.tex` est vide (deux frames `\todo`).

## Cohérence des schémas d'hybridation (décidée le 2026-11-05)

Une seule ossature pour tous les schémas : **réseau → espace enrichi → solveur de Galerkin**, plus un repère de perte. Macros communes : `images/hybscheme.sty` (`\hybstyle{one|two}`, `\hscompactsize` / `\hslargesize`, `\hsbox`, `\hsloss`, `\hsout`, styles `hsarrow` / `hsdash`).
- **1ère (bleu)** : perte PINN **à gauche**, hors ligne, réseau « neural prior » figé ensuite, pas de retour.
- **2nde (vert)** : perte **à droite**, gradient qui remonte au réseau (pointillés), réseau « neural network ».
- Vocabulaire fixe : « enriched space » et « Galerkin solver » dans les deux (les blocs de texte parlent de base enrichie). L'intro n'est plus la référence : c'est `hybscheme.sty` ; `outline.tex` (intro) et `learnable_scheme.tex` l'utilisent déjà.
- **À faire** : slide « Pipeline of the hybrid method » de la 1ère hybridation (mêmes boîtes, taille large), slide du gradient de la 2nde, slide de transition (les deux variantes côte à côte). Lien avec la figure encodeur/approximation/décodeur : à garder pour la slide du gradient, pas dans l'ossature.

## Portée de la slide de principe (décidée le 2026-11-05)

La slide « Neural-enriched Galerkin scheme » est **générique**, rédigée pour le cas PINN (μ fixé, `m_θ(x)`). Le cas opérateur (NO) est le même schéma (réseau partagé par toutes les cellules, base `m_θ φ_{c,i}`, solveur fixe) : seules changent l'entrée du réseau (`x` → `(x,μ)`, ou champ via l'encodeur) et la perte (PINN → supervisée). Cela se dit sur la slide « Neural operator: principle » de la sous-section Parametric, pas sur la slide de principe. Le multiplicateur n'est introduit qu'au bloc « Enriched basis » (le bloc « Network » ne parle que du réseau) ; sa forme `g(v_θ)` est un résultat de l'étude 1D.

- 2026-11-06 : slide « Neural operator: principle » refaite (option A) : `operator_scheme` montre **un réseau, un espace enrichi, puis deux solveurs empilés μ^(1) … μ^(N)** (μ en rouge dans chaque boîte, « ⋮ » entre les deux) ; la perte réunit les sorties de tous les μ. Texte : « One network, one Galerkin solve per μ » (μ = paramètre de l'EDP, pas une entrée du solveur). Slide dense en bas : « Validity » touche presque le numéro de page.
- 2026-11-06 : **règle NO : pas de chiffres sur les slides d'explication** (principe, trois bases) ; ils sont sur « Linear transport » (ligne « Setup » + « Weights N_θ »). Figure `operator_scheme` : indices `i_1 … i_B` (batch aléatoire), plus de valeur.
- 2026-11-06 : « Neural operator: principle » **restructurée en 4 overlays** (échantillonnage 𝒟_train ⊂ 𝓜 puis batch par epoch → un solve par μ du batch → perte → entraînement) ; bloc « Validity » déplacé à l'oral. Équation sans indice (`A_h(θ;μ) β_h(θ;μ) = b_h(θ;μ)`).
- 2026-11-06 : slide « Three enriched bases » renommée **« Constructing the enriched space (three ways) »** (transition du « how ? » de la slide principe) ; figure refaite : réseau → `v_θ(…)` → enriched space → équation `φ^θ_{c,i}(x;·) = m_θ(x,·) φ_{c,i}(x)` pour les trois bases.

