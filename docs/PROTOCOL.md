# Protocole expérimental (pré-enregistrement léger)

> Figé par le commit qui introduit ce fichier, **avant toute expérience**.
> Toute modification ultérieure est ajoutée dans la section « Écarts au protocole »,
> avec sa date et sa raison. On ne réécrit jamais le texte d'origine.

## 1. Question
Une récompense qui pénalise la perte du jargon médical, utilisée en post-training
(best-of-N, DPO, GRPO), fait-elle conserver aux LLM les termes médicaux essentiels,
sans halluciner ni recopier, et dépasser l'état de l'art sur le test MEDIQA 2021 ?

## 2. Données
- **Entraînement** : MeQSum, 900 exemples. **Dev** : les 100 exemples restants de MeQSum, plus les 50 de la validation MEDIQA 2021.
- **Test** : les 100 questions du test officiel MEDIQA 2021, tâche 1.
- Seed du split : `42`. Les quasi-doublons (Jaccard ≥ 0.8 sur les 3-grammes de mots) sont groupés dans un même split. Le manifeste (IDs et SHA-256) est versionné dans `data/manifests/`.
- **Accès au test**, deux au total, tous journalisés dans `reports/test_ledger.jsonl` :
  1. la reproduction de BART-large (phase 1), qui n'influence aucun réglage ;
  2. l'évaluation finale de tous les systèmes retenus, en une seule fois (phase 5).

## 3. Systèmes
B1, P0–P5, P5-abl, G0, G1, G3–G5, définis dans DESIGN §4. Seeds d'entraînement : `0, 1, 2`.
Décodage : greedy, `max_new_tokens = 64`, même prompt versionné pour tous les décodeurs.
Checkpoints et hyperparamètres choisis **sur le dev uniquement**. La liste des systèmes
envoyés au test est écrite dans ce fichier (section 7) **avant** l'accès final.

## 4. Métriques
| Abréviation | Définition |
|---|---|
| JR | Rappel du jargon essentiel : \|J(ŷ) ∩ J\*\| / \|J\*\|, avec J\* le jargon de la référence (extracteur de DESIGN §5.1) |
| LT | Taux de termes perdus : 1 − JR |
| HR | Taux d'hallucination : \|J(ŷ) \ (J(x) ∪ J\*)\| / max(1, \|J(ŷ)\|) |
| R-1/2/L | `rouge-score`, F1, avec stemmer |
| BS | BERTScore F1, configuration par défaut de `bert-score` pour l'anglais (modèle et version rapportés) |
| SummaC | SummaC_ZS, configuration par défaut du paquet `summac` (rapportée) |
| CR | Ratio de recopie : part des 4-grammes de ŷ présents dans x |
| H | Score humain par exemple, de 0 à 4 : un point par critère satisfait (jargon conservé, pas de paraphrase de terme, pas de contexte critique perdu, pas d'hallucination) |

Chaque métrique est moyennée sur les exemples, puis on rapporte moyenne ± écart-type
sur les 3 seeds. IC à 95 % par **bootstrap apparié** sur les exemples (10 000
rééchantillonnages, seed `0`).

## 5. Hypothèses et règles de décision
| | Hypothèse | Soutenue si… |
|---|---|---|
| **H1** | ROUGE s'accorde mal avec le jugement humain ; nos métriques s'accordent mieux | Sur les 150 jugements : τ_b de Kendall (JR − HR, H) > τ_b (R-L, H), et l'IC à 95 % de la différence (bootstrap apparié) exclut 0 |
| **H2** | L'encodeur-décodeur hallucine moins à ROUGE comparable | HR(B1) < HR(meilleur décodeur SFT) avec un IC de la différence qui exclut 0, **et** \|ΔR-L\| ≤ 0.01 ou un IC de ΔR-L qui contient 0 |
| **H3** | GRPO réduit la perte de jargon par rapport au SFT, à coût ROUGE limité | Pour chaque taille : LT(GRPO) ≤ 0.7 × LT(SFT) **et** R-L(GRPO) ≥ R-L(SFT) − 0.01, les deux évalués avec leurs IC. Testé séparément pour le 4B (P5 vs meilleur P1/P2) et le 27B (G5 vs G1) |
| **H3b** | Sans pénalité de longueur, la politique recopie | CR(P5-abl) > CR(P5) et longueur moyenne(P5-abl) > 1.5 × longueur moyenne des références |
| **H4** | État de l'art sur MEDIQA 2021 | Meilleur système G : SummaC > 0.603 **et** R-L ≥ 0.386. Meilleur système P : R-L > 0.339. Chiffres de référence : FocusMed, tableaux II et V ; on signale les différences d'implémentation des métriques |

Un test humain apparié (McNemar) accompagne H2 et H3 sur les 50 questions annotées.

## 6. Bake-off du modèle phare (phase 1, inférence seulement)
- Candidats : Qwen3.8-27B, Gemma 4 31B-it, MedGemma 27B text-it. Few-shot k = 5 (exemples tirés de l'entraînement, identiques pour tous), sur le dev.
- **Règle** : on retient le meilleur JR − HR. Si les IC à 95 % du meilleur et du deuxième se recouvrent, on garde Qwen3.8-27B. ROUGE-L sert uniquement à départager.
- Conséquence sur le petit modèle : Gemma 4 → Gemma 4 E4B ; MedGemma → MedGemma 1.5 4B.

## 7. Systèmes envoyés au test final
_À compléter à la fin de la phase 4, avant l'accès final._

## 8. Écarts au protocole
- **2026-11-27, entraînement à 898 exemples au lieu de 900.** La construction du split a détecté
  que 2 questions du **test officiel MEDIQA 2021** (`mediqa2021_test:276` et `:318`) sont des copies
  exactes (Jaccard = 1,0) de questions MeQSum (`1-123706665.xml.txt`, `17713.txt`). Comme le prévoit
  la règle de recouvrement (DESIGN §3), elles sont retirées de l'entraînement, ce qui donne
  898 / 100 / 150 / 100. Le dev et le test sont inchangés. Voir `data/manifests/splits_v1.json` (audit).
- **2026-11-27, best-of-N trié par R_free (sans référence) et non par R.** R lit la référence
  (le jargon J\* du résumé d'expert). C'est légitime pour l'entraînement (paires DPO, GRPO), mais pas pour
  trier des candidats au moment du test, où aucun système n'a la référence. P3 / G3 utilisent donc
  R_free (`reward.compute_reference_free`) : J\* est remplacé par le jargon de la source, |y| par la longueur
  médiane des références d'entraînement, et `lost` vaut 0. Un best-of-N « oracle » trié par R peut être
  rapporté **sur le dev uniquement**, comme borne supérieure diagnostique, jamais sur le test.
  Corrigé avant toute expérience de best-of-N.

