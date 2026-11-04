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
