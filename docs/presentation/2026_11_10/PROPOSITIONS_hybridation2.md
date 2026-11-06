# Propositions de slides (2nde hybridation) — à choisir

Rédigées le 2026-11-05 en ton absence. Pour chaque slide : la version **implémentée** (recommandée) et des alternatives, en mots-clés, non implémentées. Dis laquelle tu gardes et on ajuste.

## 1. Transition — « From frozen prior to learned basis » (`introduction.tex`)

**Implémentée (A)** : deux schémas en rangée (`images/contrib2/transition/`, bleu puis vert), avec entre les deux : 1ère = prior figé / estimations a priori → **Limit** : gain borné par la qualité du prior → 2nde = base apprise à travers le solveur → **Price** : pas d'estimateur d'erreur, un solve par itération. 4 overlays.
- **B** : une seule figure à deux colonnes (cartes `contribbox` bleue et verte, comme l'intro « Two hybridizations »), lignes *Network / Guarantees / Gain / Limit*, puis une question « What if the basis were learned through the solver? ». Plus compact, mais redondant avec l'intro (mémoire : « on se répète »).
- **C** : une seule phrase-question en grand (« Gain bounded by the prior… what if the basis were learned *through the solver*? ») et le schéma vert seul. Très aéré, 1 min de parole, mais perd le rappel de la 1ère.
- **D** : supprimer la slide et dire la transition à l'oral (« Limit of the 1st hybridization » en dernière ligne de la conclusion de la 1ère).
- Étude dimensionnelle : toujours non placée (voir plan, décision 2) ; option : un chiffre dans la ligne « Limit » ou un seul point d'intro.

## 2. Autodiff — « Automatic differentiation » (`introduction.tex`)

**Implémentée (A)** : 3 blocs : principe (graphe + règle de la chaîne ; JAX, ScimBa) ; modes direct (JVP) / inverse (VJP), coût indépendant du nombre d'entrées en rouge ; à travers un solveur : déroulé contre implicite.
- **B** : un schéma standalone d'un graphe de calcul avec les deux sens de parcours (flèches JVP vers l'avant, VJP vers l'arrière), un bloc de texte en dessous ; plus visuel, mais un TikZ à dessiner.
- **C** : supprimer cette slide d'intro, dire le mode inverse à l'oral au moment de l'adjoint (« Learning θ »). Gagne ~1,5 min ; l'adjoint reste moins ancré.
- **D** : version « JAX » : `jax.grad`, `jax.vjp`, `jax.jvp`, `vmap`, `jit` en blocs de code ; plus concret pour ScimBa, mais moins de maths.

## 3. Deux cas — « Two settings: PINN and neural operator » (`introduction.tex`)

**Implémentée (A)** : deux cartes neutres (comme l'intro) en pastilles *parameter / loss / reuse*, puis bandeau « Same learnable Galerkin scheme, different input and loss ».
- **B** : un schéma standalone : le même schéma vert, avec en haut une flèche μ fixe (PINN) / une famille de μ (NO) qui entre dans le solveur, et la perte (résidu / données) à droite ; pas de cartes.
- **C** : tableau 2 colonnes × 4 lignes (parameter, loss, data, reuse) ; déconseillé : redondant avec les slides suivantes.
- La slide est un aperçu : elle peut aussi se supprimer, la page de section (`\SectionNote`) annonçant déjà le cas.

## 4. Conclusion — « Conclusion, limitations and perspectives » (`conclusion.tex`)

**Implémentée (A)** : une slide, 3 blocs en `itemize` (résultats : 1D, 3D, opérateur ; limites ; perspective « geler le multiplicateur »).
- **B** : deux slides : « Conclusion » (cadre vert avec les trois résultats) puis « Limitations and perspectives » avec un schéma : multiplicateur appris → gelé → approche multiplicative (bleu) → estimation a priori.
- **C** : garder limites + perspective sur une slide avec le schéma bleu/vert côte à côte (cohérent avec la transition), et laisser les résultats chiffrés à la conclusion générale.
- La conclusion générale (`sections/common/conclusion.tex`) et celle de la 1ère hybridation sont **toujours des trous** (tu as demandé uniquement celle-ci).
