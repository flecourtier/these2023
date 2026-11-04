# Notes pour l'oral

Points à dire qui ne sont **pas** sur la slide (la slide = mots-clés). Pas un texte rédigé : des rappels, slide par slide, dans l'ordre de la présentation. Ajouté au fil des échanges ; `(→ source)` renvoie au manuscrit.

---

## Background — Discontinuous Galerkin method

*Public : connaît déjà ces méthodes. But de la slide : aperçu rapide + préparer le lien DG ↔ CG, pas une démonstration.*

- Pas de relation entre les dofs de cellules voisines : chaque `φ_{c,i}` est une base locale de ℙ_k(K_c) **prolongée par zéro** (support = une seule cellule), d'où une base de `V_h^disc`.
- Pas de relation entre les dofs de cellules voisines : chaque cellule a sa propre base locale.
- `dim V_h^disc = N_cells · N_b` (`N_b` = nombre de fonctions de base par cellule).
- Le degré `k` est volontairement non précisé (ℙ_k / ℚ_k). Dans les chapitres 4–5 de la thèse : ℚ_k sur hypercubes ; en 1D c'est identique à ℙ_k. Si on me pose la question : `N_b = C(k+d, d)` pour ℙ_k, `(k+1)^d` pour ℚ_k.

**Overlay 2 : couplage des cellules**
- Pourquoi une trace ambiguë : `u_h` et `D∇u_h` prennent deux valeurs sur chaque interface (une par cellule voisine). Le saut de `u_h` en rouge sur la figure.
- D'où vient le flux (dérivation **non** montrée) : on teste l'EDP sur chaque cellule, on intègre par parties le terme de diffusion, il reste un terme de bord `∫_{∂K} D∇u_h · n_K v_K` qui est ambigu → on le remplace par des **flux numériques** `û_h`, `ς̂_h^D`. (→ `chapters/numerical_methods/dg.tex`)
- Le manuscrit passe par une variable auxiliaire `ς_h` (le gradient) ; la slide l'omet.
- « Le flux définit la méthode » : c'est lui qui fixe les propriétés du DG (consistance, stabilité). Il existe plusieurs flux (SIPG, NIPG, LDG…), on en prend un.
- **SIPG** = *Symmetric Interior Penalty Galerkin* : `û_h = {u_h}`, `ς̂_h^D = {D∇_h u_h} − κ[[u_h]]`.

**Overlay 3 : formulation variationnelle** — notation : `l` sans indice (ne dépend pas de `h`, même forme que le CG) ; `a_h^disc` = forme unifiée du manuscrit, `a_h^sipg` = cas du flux SIPG
- Moyenne et saut sur une face entre `K_L` et `K_R` : `{v} = ½(v_L + v_R)`, `[[v]] = v_L n_L + v_R n_R` (pas écrits sur la slide).
- `κ = η_F / h_F`, `η_F` assez grand pour la stabilité. Dans les expériences : `η_F = 10 k(k+1)` (1D, chap. 5) ; `η_F = k(k+1)d` (étude dimensionnelle).
- `∇_h` = gradient « brisé », calculé cellule par cellule (ignore les sauts).
- Les *cell terms* sont exactement ceux du CG (au gradient brisé près) → c'est le lien avec la slide suivante.

**Overlay 4 : ce que le DG ajoute au CG** (termes volumiques en gris = le CG, « cell terms » ; termes d'interface encadrés en rouge, étiquetés « consistency » et « penalty »)
- *consistency* : termes symétriques en `u_h` et `v_h` aux interfaces (d'où le « Symmetric »).
- *penalty* : contrôle les sauts, donne la stabilité.

**Ce qui reviendra dans la suite (à relier à l'oral)**
- Les sauts de `u_h` : dans la 2nde hybridation (DG cellwise, 1D, `μ^(1)`, `N_cells=4`), le saut en `x=0.75` passe d'environ `7·10⁻³` (k=3) / `2·10⁻³` (k=4) pour la solution classique à `1.4·10⁻⁷` / `3·10⁻⁹` avec la base apprise. (→ `chapters/learnable_methods/PINN_1D/dg_cellwise.tex`)
- On n'écrira plus `𝒯_h` ni `V_h^disc` : seulement `K_c`, `N_cells`, `φ_{c,i}`, `N_dofs`. Le système `A_h β_h = b_h` est ce que la 2nde hybridation rend dépendant de `θ`.

**Conditions aux limites et régime (ajouté)**
- Page de chapitre « Numerical methods » : note « homogeneous Dirichlet ». Valable pour FEM et DG ; autres CL (Neumann, Robin, `g ≠ 0`) = même cadre, flux de bord adaptés (→ `dg.tex:89`). En DG le Dirichlet est faible (Nitsche : les intégrales sur `𝓕_h` incluent les faces du bord) ; en CG il est fort (`V_h^0`). Les CL mixtes réapparaissent en 1ère hybridation (cas anneau, `lap2DMixRing`).
- Overlay 2 du DG (« Focus on diffusion-dominated regimes : flux for the diffusion term only ») : on se place en diffusion dominante (Pe petit), donc seul le terme de diffusion a un flux (SIPG) ; convection et réaction sont de simples termes volumiques. Ce n'est pas le flux SIPG qui limite, c'est l'absence de flux pour la convection : en régime convectif, on ajoute le flux amont `c_h^up` : `a_h^sipg + c_h^up` (→ `dg.tex:46`, `:185-226`). Le taux de convergence dépend de Pe et de R hors de ce régime (→ `dg.tex:183`).

## Background — Continuous Galerkin method

*Message : le CG = le DG où la continuité est imposée sur les dofs (table de connectivité), plus par des flux. Même figure que le DG, mais plus de sauts.*

- Mêmes cellules `K_c`, mêmes bases locales `φ_{c,i}` que le DG ; seule change la relation `β_{c,i} ↔ vecteur global`. DG : `β_{c,i}` indépendants (`N_dofs = N_cells · N_b`). CG : `β_{c,i} = β_I`, `I = conn(c,i)` (`N_dofs = ∏_ℓ (N_ℓ k + 1)` sur grille logique ; sur la figure `k=1`, 3 cellules → 4 dofs au lieu de 6). (→ `chapters/sciml_library/extension_fem.tex`, §« Continuous FE variables »)
- Base de `V_h` = fonctions de forme nodales `φ_I = Σ_{conn(c,i)=I} φ_{c,i}` (« global » = l'indice `I` ; le support est le patch des cellules qui contiennent le nœud `I`, pas une seule cellule comme `φ_{c,i}` en DG) (chapeau en rouge sur la figure : somme des deux fonctions locales de l'interface ; au bord, `φ_I = φ_{c,i}`). `u_h = Σ_I β_I φ_I = Σ_{c,i} β_{conn(c,i)} φ_{c,i}` (même fonction, `β_{c,i} = β_I`, `I = conn(c,i)` ; non écrit). Les `φ_{c,i}` locales sont celles de `V_h^disc` ; les `φ_I` sont celles du chap. 2 et de la 1ère hybridation (`φ^θ_I = m_θ φ_I`). La 2nde hybridation travaille avec les `φ_{c,i}` + `conn`.
- `conn` est construite une fois pour toutes à partir de la topologie du maillage ; un nœud d'une face commune reçoit le même indice global dans les deux cellules. Pas de coût de calcul : pas de terme à assembler (≠ pénalité).
- Pourquoi `u_h` continue : sur une face `F`, les deux cellules voisines ont les mêmes dofs *et* les mêmes fonctions de forme (interpolants de Lagrange de `F`) → traces égales. Vrai si les bases elles-mêmes sont continues (automatique pour une base analytique ou patchwise ; à imposer par construction pour une base *cellwise*, cf. 2nde hybridation).
- `V_h = V_h^disc ∩ C⁰(Ω)` : le CG est un sous-espace du DG (même degré, même maillage). `V_h^0` = `V_h` + CL de Dirichlet homogène (`V_h^0 ⊂ V_h`, nul sur `Γ`).
- Formulation : `a(u_h,v_h)=l(v_h)` **sans indice `h`** (ne dépend pas de `h`, contrairement à `a_h^disc` à cause de `κ = η_F/h_F`). `∇` ordinaire au lieu de `∇_h` (plus de saut à réconcilier). Ce sont exactement les *cell terms* de la slide DG (encadré gris) ; consistance et pénalité disparaissent, ainsi que tout le passage sur `𝓕_h^I` dans l'assemblage.
- Hypercubes dans la thèse : base `ℚ_k` (pas `ℙ_k`) pour le CG de ScimBa, pour partager maillage / quadrature / mapping avec le DG. En FEM classique du ch. 2 : simplexes, `ℙ_k`.
- Dirichlet (pas sur la slide) : **fort** en CG (`R_I = β_I − g(x^{(I)})` pour les nœuds du bord), **faible** en DG (état fantôme dans le flux). À reprendre dans la comparaison DG/CG si on en fait une slide.
- Notation : la slide DG disait « on n'écrira plus `V_h^disc` » ; ici `V_h^disc` est réutilisé une seule fois, pour définir `V_h` comme son sous-espace continu.
- Résultat de convergence (pas sur la slide) : `|u−u_h|_{H¹} ≤ C h^s |u|_{H^{s+1}}`, `‖u−u_h‖_{L²} ≤ C h^{s+1}|u|_{H^{s+1}}`, `1 ≤ s ≤ k` (Céa + Aubin–Nitsche). (→ `chapters/numerical_methods/fem.tex`)
