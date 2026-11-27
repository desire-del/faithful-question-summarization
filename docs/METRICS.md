# Métriques, récompenses et statistiques

> Référence unique pour tout ce que le projet mesure. Le code correspondant est indiqué pour chaque bloc.
> Documents liés : [PROTOCOL.md](PROTOCOL.md) (règles de décision) · [EXTRACTOR.md](EXTRACTOR.md) · [FINDINGS.md](FINDINGS.md)

## Notation

| Symbole | Sens |
|---|---|
| **x** | la question du patient (la source) |
| **y** | le résumé écrit par l'expert (la référence) |
| **ŷ** | le résumé produit par le modèle (la prédiction) |
| \|t\| | la longueur d'un texte en mots |
| **J(t)** | l'ensemble des termes de jargon trouvés dans le texte t (voir §4) |
| **J\*** = J(y) | le jargon essentiel, celui que l'expert a gardé |

## Vue d'ensemble

| Métrique | Compare ŷ à… | Mesure | Dans la récompense R ? | Rôle |
|---|---|---|---|---|
| ROUGE-1/2/L | y | recouvrement de mots | non | comparaison avec la littérature (H4) |
| BERTScore | y | similarité de sens | non | comparaison avec la littérature |
| SummaC-ZS | x | implication NLI | non | **descriptif uniquement** (récompense la recopie) |
| JR, LT | y | jargon essentiel gardé / perdu | **oui** | fidélité (signalée « optimisée ») |
| HR | x et y | termes inventés | **oui** | fidélité (signalée « optimisée ») |
| CR | x | recopie de la source | non | détection du hacking |
| Longueur, format_ok | — | forme | partiellement | contrôle |
| H (humain) | x et y | jugement d'expert | non | **preuve principale de fidélité** |

Règle : une métrique présente dans la récompense ne peut jamais être la seule preuve. Les métriques tenues à
l'écart de l'entraînement (ROUGE, BERTScore, l'annotation humaine) font foi.

---

## 1. ROUGE (`metrics.rouge`, paquet `rouge-score`, stemmer de Porter)

**ROUGE-1 / ROUGE-2** : n-grammes (mots isolés pour N = 1, paires consécutives pour N = 2) communs à ŷ et y,
comptés avec leur multiplicité minimale. Les mots sont ramenés à leur racine (« treatments » → « treatment »).
```
R_N = nb de n-grammes communs / nb de n-grammes de y
P_N = nb de n-grammes communs / nb de n-grammes de ŷ
ROUGE-N = 2·P_N·R_N / (P_N + R_N)
```
**ROUGE-L** : LCS = la plus longue sous-suite commune (les mêmes mots dans le même ordre, pas forcément collés).
```
R = LCS(ŷ, y) / |y|      P = LCS(ŷ, y) / |ŷ|      ROUGE-L = 2·P·R / (P + R)
```
**Piège**, avec y = « Is levothyroxine safe during pregnancy? » :

| ŷ | LCS | ROUGE-L | Erreur |
|---|---|---|---|
| « Is levothyroxine safe? » | 3 | 0,75 | oublie « pregnancy » |
| « Is Synthroid safe during pregnancy? » | 4 | 0,80 | invente un médicament |

ROUGE préfère ici l'hallucination à l'oubli. C'est ce que teste H1.

## 2. BERTScore (`metrics.bertscore`, RoBERTa-large, couche 17, configuration par défaut)

Chaque mot est représenté par un vecteur contextuel e. Chaque mot cherche son meilleur partenaire dans l'autre texte :
```
P = (1/|ŷ|) · Σ_{t ∈ ŷ} max_{s ∈ y} cos(e_t, e_s)
R = (1/|y|) · Σ_{s ∈ y} max_{t ∈ ŷ} cos(e_s, e_t)
BERTScore = 2·P·R / (P + R)
```
- **Avantage** : tolère les synonymes.
- **Limite** : « levothyroxine » et « Synthroid » sont proches dans l'espace des vecteurs, donc l'échange est peu puni.
- **À savoir** : les valeurs sont tassées entre 0,80 et 0,95, donc de petits écarts comptent.

## 3. SummaC-ZS (`metrics.SummaCZS`, NLI ALBERT-xlarge « vitc », granularité phrase)

**La NLI** (inférence en langue naturelle) : pour une prémisse P et une hypothèse H, le modèle donne trois probabilités
qui somment à 1 : implication p_E, contradiction p_C, neutre p_N.

**SummaC** :
1. On découpe x en phrases x₁…xₘ et ŷ en phrases ŷ₁…ŷₙ (phrases NLTK de plus de 10 caractères).
2. On calcule E[i,j] = p_E(xᵢ ⇒ ŷⱼ) et C[i,j] = p_C(xᵢ, ŷⱼ).
3. On agrège :
```
score_j = max_i E[i,j] − max_i C[i,j]        meilleur soutien − pire contradiction
SummaC  = (1/n) · Σ_j score_j                 ∈ [−1, +1]
```
SummaC ne regarde **jamais** y : c'est une fidélité à la source. Notre implémentation est identique au paquet
`summac==0.0.4` (écart ≤ 1e-6, `reports/summac/equivalence.json`).

**Pourquoi il est seulement descriptif** (FINDINGS du 2026-11-27, dev) :

| Système | SummaC | ROUGE-L |
|---|---|---|
| Résumés d'experts | −0,02 | 1,00 |
| B0 : recopier la dernière phrase-question du patient | **+0,74** | 0,14 |

Une phrase recopiée est impliquée par elle-même (E ≈ 1). Une question n'affirme rien, ce que les modèles NLI jugent mal.
Et les experts reformulent. SummaC récompense donc l'extraction. Il est toujours affiché avec CR et B0, et il ne sert
plus de critère de succès (PROTOCOL §8, H4 amendé).

## 4. Métriques de jargon (`jargon.py`, `reward.py`)

**Extraction J(t)** : un ensemble NER choisi sur des annotations de référence (CHQA, F1 = 0,73 ; voir EXTRACTOR.md),
qui couvre maladies, médicaments, symptômes, anatomie, examens et actes. S'y ajoutent des règles pour le **contexte
critique** : doses (« 50 mcg »), âges, durées, grossesse, température.

**Correspondance** match(a, b) sur les formes normalisées (minuscules, ponctuation retirée, pluriel simple replié).
Elle est vraie dans trois cas :
1. a = b ;
2. l'un contient l'autre **en mots entiers**, le plus court ayant au moins 4 caractères (« type 2 diabetes » ⊃ « diabetes ») ;
3. la similarité de caractères est d'au moins 85 % (fautes : « diabeties » ≈ « diabetes »), **seulement si** les deux
   termes ont au moins 5 caractères, **exactement les mêmes nombres** (« 50 mcg » ≠ « 500 mcg », « type 1 » ≠ « type 2 »)
   et **aucun préfixe opposé** (hypo/hyper, brady/tachy, micro/macro).

**Définitions**
```
JR (jargon retenu)  = |{ t ∈ J* : ∃ u ∈ J(ŷ), match(t, u) }| / |J*|            (= 1 si J* est vide)
LT (termes perdus)  = 1 − JR
HR (hallucinations) = |{ u ∈ J(ŷ) : ∄ t ∈ J(x) ∪ J*, match(u, t) }| / max(1, |J(ŷ)|)
```
Exemple, avec J\* = {levothyroxine, hypothyroidism, pregnancy} :

| ŷ | JR | HR |
|---|---|---|
| « Is levothyroxine safe during pregnancy for hypothyroidism? » | 1 | 0 |
| « Is Synthroid safe during pregnancy? » | 1/3 | 1/2 |

**Limite** : l'extracteur a un rappel de 0,63. Un terme non détecté ne peut pas être compté comme perdu, donc les
métriques de jargon sont *indulgentes*, jamais injustes.

## 5. Métriques de contrôle

```
CR (recopie) = |4-grammes(ŷ) ∩ 4-grammes(x)| / |4-grammes(ŷ)|      1 = recopie pure ; experts ≈ 0,05
Longueur     = |ŷ| en mots                                          experts ≈ 9-10 mots
format_ok    = 1 si ŷ est une seule question finissant par « ? », sans préambule ni saut de ligne
```
100 % des résumés d'experts passent `format_ok`.

## 6. Score humain H

Pour chaque (question, résumé) annoté à l'aveugle, on compte un point par critère respecté, soit H ∈ {0, …, 4} :
jargon conservé · pas de terme paraphrasé · pas de contexte critique perdu · pas d'hallucination.
La grille complète se trouve dans `docs/annotation_guidelines.md` (phase 5).

---

## 7. Récompenses (des signaux d'entraînement, pas des métriques)

**R, avec référence** (`reward.compute`) : utilisée **seulement à l'entraînement** (paires DPO, GRPO), où y est connu.
```
R = 1.0·JR − 0.25·(nb de termes de J* perdus) − 1.0·HR + 0.5·format_ok − 1.0·max(0, |ŷ|/|y| − 1.5)
```

**R_free, sans référence** (`reward.compute_reference_free`) : utilisée pour le **best-of-N**, y compris sur le test.
```
R_free = 1.0·JR_src − 1.0·HR_src + 0.5·format_ok − 1.0·max(0, |ŷ|/L − 1.5)
  JR_src = part des termes de J(x) retrouvés dans ŷ
  HR_src = part des termes de J(ŷ) absents de J(x)
  L      = longueur médiane des références d'entraînement (une constante)
```
**Pourquoi deux versions** : R lit le corrigé y. Trier des candidats avec R sur le test reviendrait à laisser le
corrigé choisir la meilleure copie. R_free n'utilise que la question et des statistiques du train (PROTOCOL §8).
Le best-of-N trié par R n'est autorisé que sur le dev, comme borne supérieure « oracle ».

## 8. Outils statistiques (`stats.py`)

- **IC bootstrap à 95 %** : on tire 10 000 fois, avec remise, autant d'exemples que l'échantillon, on recalcule la moyenne,
  et on garde les percentiles 2,5 % et 97,5 %. **Apparié** : les mêmes exemples sont tirés pour deux systèmes, et on calcule
  l'IC de leur différence. Une différence est retenue si son IC exclut 0.
- **Kendall τ_b** : sur toutes les paires d'exemples, C = paires classées dans le même ordre par la métrique et par l'humain,
  D = paires en ordre inverse, Tₓ et T_y = égalités d'un seul côté.
  ```
  τ_b = (C − D) / √((C + D + Tₓ)·(C + D + T_y))      ∈ [−1, 1]
  ```
  Il sert à H1 : quelle métrique s'accorde le mieux avec le jugement humain ?
- **McNemar exact** : on compare deux systèmes jugés oui/non sur les mêmes questions. b = cas où seul A est bon,
  c = cas où seul B est bon. La p-valeur bilatérale est 2·P(X ≤ min(b, c)) avec X ~ Binomiale(b + c, ½).
- **Seeds** : chaque configuration est entraînée 3 fois (seeds 0, 1, 2), et on rapporte moyenne ± écart-type.
