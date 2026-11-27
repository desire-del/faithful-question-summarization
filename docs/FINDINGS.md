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
