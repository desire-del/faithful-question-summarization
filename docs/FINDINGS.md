# Journal des constats

> Les résultats notables, au fil de l'eau, destinés au README final. Chaque entrée renvoie à sa preuve.

## 2026-11-27 : 2 % du test MEDIQA 2021 sont dans le jeu d'entraînement officiel
- 2 des 100 questions du test officiel MEDIQA 2021, tâche 1, apparaissent **mot pour mot** dans
  MeQSum, qui est le jeu d'entraînement recommandé par la tâche (Jaccard = 1,0 sur les 3-grammes de mots) :
  - `mediqa2021_test:276` = `meqsum:1-123706665.xml.txt` (seconde dose de ROR administrée par erreur) ;
  - `mediqa2021_test:318` = `meqsum:17713.txt` (perte du goût et de l'odorat après une chirurgie).
- Conséquence : tout système entraîné sur MeQSum complet (le cas de la littérature, FocusMed compris)
  a vu ces 2 questions et leurs références. L'effet sur la moyenne est faible, mais c'est une fuite.
  Nous les retirons de l'entraînement.
- Preuve : `data/manifests/splits_v1.json`, section `audit` ; règle dans DESIGN §3 ; écart noté dans PROTOCOL §8.
- Pour le rapport : on pourra évaluer aussi « test sans ces 2 questions » pour montrer que la conclusion ne change pas.

## 2026-11-27 : observations du go/no-go (sondes synthétiques, zero-shot)
- Qwen3.5-0.8B perd du jargon (« hypothyroidism ») et recopie certaines questions.
- Qwen3.5-4B garde le jargon mais produit des résumés 2 à 3 fois plus longs que les références.
- Preuve : `docs/GO_NO_GO.md`, `reports/go_no_go/`.

## 2026-11-27 : SummaC sous transformers 5 donne silencieusement de faux scores
- Le paquet de référence (`summac==0.0.4`) ne s'installe pas avec transformers 5 (il exige huggingface-hub ≤ 0.17).
  Notre réimplémentation donnait d'abord un écart moyen de **0,34** avec la référence (et jusqu'à 1,12), pour deux raisons :
  1. le tokenizer ALBERT ne renvoie plus les `token_type_ids` par défaut, et le modèle NLI les utilise ;
  2. transformers 5 convertit `spiece.model` autrement que la 4.35 (« -marie », « (cmt) »...).
- Correctif : on demande explicitement les segments, et on embarque le `tokenizer.json` exporté par transformers 4.35.2
  (hash vérifié au chargement). Résultat : un écart maximal de **1e-6** sur 60 paires, comparé à la référence installée dans un
  environnement isolé (`reports/summac/equivalence.json`, essais ratés conservés).
- Pour le rapport : toute comparaison de SummaC avec la littérature dépend de ce détail. Nos chiffres sont alignés sur le paquet d'origine.

## 2026-11-27 : extracteur de jargon choisi sur des annotations de référence
- Un ensemble OpenMed ×3 + procédures DistilBERT donne F1 = 0,73 sur CHQA (P 0,87, R 0,63), après correction de l'agrégation (v3). Détails dans `docs/EXTRACTOR.md`.

## 2026-11-27 : question ouverte sur SummaC et l'objectif H4
- Sur le dev, les **résumés d'experts eux-mêmes** obtiennent un SummaC-ZS moyen de **−0,016**
  (`runs/smoke_oracle`, implémentation alignée sur `summac==0.0.4`). FocusMed annonce 0,603 pour son système
  et 0,448 pour BART sur le test MEDIQA.
- SummaC (NLI source ⇒ résumé) récompense mécaniquement les phrases **recopiées** de la source : un objectif
  « SummaC > 0,603 » peut entrer en tension avec notre pénalité anti-recopie.
- Vérification prévue : la reproduction de BART (phase 1) donnera notre SummaC pour BART, à comparer à son 0,448.
  Si l'écart est grand, la configuration de FocusMed diffère de celle par défaut. On le documentera, et la partie
  SummaC de H4 sera rapportée avec cette réserve. *(Mise à jour : la baseline de recopie ci-dessous a tranché, le critère est retiré de H4.)*

## 2026-11-27 : une baseline qui recopie la question bat l'« état de l'art » en SummaC
- Sur le dev (150 questions), avec notre SummaC-ZS aligné sur `summac==0.0.4` :

  | Système | SummaC-ZS | ROUGE-L | Mots |
  |---|---|---|---|
  | Résumés d'experts (références) | **−0,016** | 1,000 | 9,6 |
  | « Recopier la dernière phrase-question du patient » | **+0,738** | 0,137 | 15,1 |

  Par exemple, « Any advice? » obtient +0,93 alors que la référence d'expert (« Where can I find information on hair loss
  on the lower extremities in men, including causes and treatment? ») obtient −0,79.
- Interprétation : SummaC mesure si chaque phrase du résumé est **impliquée par une phrase de la source** (NLI). Une phrase
  recopiée est trivialement impliquée par elle-même. Les reformulations d'experts, elles, ne le sont pas, d'autant qu'un modèle
  NLI juge mal une *question* (qui n'affirme rien).
- Conséquence : sur cette tâche, SummaC n'est **pas** une preuve de fidélité. Il récompense l'extraction. Le 0,603 de FocusMed
  n'est pas un objectif pertinent en soi. On rapportera toujours SummaC **avec** le taux de recopie et la baseline de recopie
  comme repère. **Décision du 2026-11-27 : le critère SummaC est retiré de H4** (PROTOCOL §8), avant tout résultat de nos systèmes.

