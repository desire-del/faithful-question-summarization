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
- Un ensemble OpenMed ×3 + procédures DistilBERT donne F1 = 0,73 sur CHQA (P 0,84, R 0,65). Détails dans `docs/EXTRACTOR.md`.
