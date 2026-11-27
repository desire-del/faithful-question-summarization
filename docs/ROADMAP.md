# Roadmap

> Objectif : **finir**. 7 phases, environ 9 à 10 semaines au maximum à temps partiel.
> GPU : ~20 h de L4 sur Colab (4B, BART ; recalé avec la mesure de la phase 0) et ~30 h sur le cluster (27B, à recaler).

## Règles anti-dérive (à relire au début de chaque session)

1. **Le périmètre est figé** : MeQSum + MEDIQA 2021, BART-large, Qwen3.5-4B, Qwen3.8-27B et
   les systèmes de DESIGN §4. Toute nouvelle idée, nouveau modèle ou nouveau dataset va
   dans [PARKING_LOT.md](PARKING_LOT.md), jamais dans le code avant la v1.0.
2. **Chaque phase a une durée limite.** Si une phase dépasse de plus de 50 %, on applique l'ordre de coupe.
3. **Une phase ne commence que lorsque la précédente a atteint son DoD.** Chaque phase se termine par un tag git.
4. **Les phases ne grossissent pas** : on peut seulement retirer des tâches.
5. **On met au point sur le petit modèle, on transfère sur le gros.** On ne réajuste jamais la recette sur le 27B (seul le taux d'apprentissage change).

## Le minimum viable

> B1 BART (reproduit), P0 / P1 / P5 (4B : zero-shot, SFT, GRPO), G0 / G1 / G3 (27B :
> zero-shot, SFT, best-of-N), les métriques avec IC, le tableau face à l'état de l'art et l'annotation humaine.

## L'ordre de coupe (décidé maintenant, appliqué sans débat)

1. La démo Gradio
2. G4, DPO sur le 27B
3. P2, full fine-tuning du 4B
4. P5-abl, l'ablation de reward hacking
5. G5, GRPO sur le 27B (on garde le best-of-N G3)
6. P4, DPO sur le 4B

---

## Phase 0 : cadrage et go/no-go (≤ 1 semaine) → `v0.0`
- [ ] Squelette du dépôt : uv, `src/faithqs`, ruff, pytest, CI, Makefile, `configs/env/{local,colab,cluster}`.
- [ ] Go/no-go (DESIGN §9) : un pas de GRPO du 4B (Colab) et du 27B (cluster), tokenizers, corpus CHQs, accès et quotas du cluster.
- [ ] `docs/PROTOCOL.md` : H1–H4 et la règle du bake-off, figées par commit.
- [ ] v1 : bandeau, archivage, correction de la model card HF.

**DoD** : CI verte ; chaque ligne du go/no-go est tranchée ; budget GPU recalé.

## Phase 1 : données, jargon, évaluation, ancre et bake-off (≤ 2 semaines) → `v0.1`
- [ ] `data build` : MeQSum (900 / 100), validation et test MEDIQA, quasi-doublons, recouvrement, manifeste.
- [ ] `jargon.py` et `reward.py`, avec tests adverses ; validation de l'extracteur.
- [ ] `generate.py` (HF et vLLM, best-of-N), `metrics.py` (avec SummaC_ZS), `stats.py`, `report.py`, test lock.
- [ ] **Ancre** : BART-large SFT × 3 seeds, puis **un** accès journalisé au test pour la reproduction.
- [ ] **Bake-off** (≤ 1 jour) : Qwen3.8-27B vs Gemma 4 31B vs MedGemma 27B en few-shot sur le dev → décision selon la règle d'ADR-001 §5.

**DoD** : ROUGE-L de BART sur MEDIQA à ±0.02 de 0.293 (sinon, écart expliqué avant d'aller plus loin) ; phare confirmé ; test de fumée en CI.

## Phase 2 : SFT (≤ 1,5 semaine) → `v0.2`
- [ ] P0 / G0 zero-shot sur le dev.
- [ ] 4B : recherche d'hyperparamètres (3 taux d'apprentissage) pour LoRA et full, puis P1 et P2 × 3 seeds (Colab).
- [ ] 27B : recherche d'hyperparamètres LoRA (3 taux d'apprentissage), puis G1 × 3 seeds (cluster).

**DoD** : tableau du dev pour B1, P0–P2, G0–G1 (qualité, jargon, SummaC, coût) avec IC → première réponse à H2.

## Phase 3 : post-training du 4B, mise au point de la recette (≤ 1,5 semaine) → `v0.3`
- [ ] Réglage des poids de R sur le dev, puis gel et hachage.
- [ ] P3 best-of-N ; P4 DPO × 3 ; P5 GRPO × 3 ; P5-abl × 1.
- [ ] **Gel de la recette** (R, β, groupes, époques) dans `configs/recipe_v1.yaml`.

**DoD** : tableau du dev pour P0–P5 ; courbes W&B ; figure de hacking ; recette figée.

## Phase 4 : post-training du 27B, transfert de la recette (≤ 1,5 semaine) → `v0.4`
- [ ] G3 best-of-N ; G4 DPO × 3 ; G5 GRPO × 3 (cluster), avec la recette figée.

**DoD** : tableau du dev complet ; choix des systèmes envoyés à l'évaluation finale (écrit **avant** l'accès au test).

## Phase 5 : évaluation finale et humaine (≤ 1 semaine) → `v0.5`
- [ ] Un seul accès `--final` au test MEDIQA (journalisé) pour tous les systèmes retenus.
- [ ] Tableau face à l'état de l'art (FocusMed, LLaMA3.1-8B SFT, GPT-4o, FaMeSumm, BART).
- [ ] 150 jugements humains à l'aveugle ; 20 refaits une semaine plus tard ; réponses chiffrées à H1–H4 ; typologie des erreurs.

**DoD** : `results/` versionné (prédictions et annotations).

## Phase 6 : livraison (≤ 1 semaine) → `v1.0`
- [ ] README en anglais (= le rapport) avec figures générées.
- [ ] Modèles et model cards sur le Hub HF.
- [ ] Démo Gradio avec le 4B (limitée à 1 jour).
- [ ] Post LinkedIn.

**DoD (fin du projet)** : un inconnu regénère le tableau principal depuis `results/` en moins de 5 minutes, sans GPU.

---

## Budget GPU estimé
| Bloc | Où | Temps |
|---|---|---|
| Bake-off (3 modèles de ~30B, inférence vLLM) | cluster | ~3 h |
| BART × 3 | Colab | ~0,5 h |
| 4B : recherche d'hyperparamètres et SFT (LoRA et full) | Colab | ~4 h |
| 4B : best-of-N, DPO, GRPO (3 × ~3,1 h mesurés), ablation | Colab (L4) | ~14 h |
| 27B : recherche d'hyperparamètres et SFT LoRA | cluster | ~6 h |
| 27B : best-of-N, DPO, GRPO | cluster | ~20 h |
| Génération finale | les deux | ~1,5 h |
| **Total** | | **~20 h Colab (L4) + ~30 h cluster** |

Le poste le plus incertain est GRPO sur le 27B : il est mesuré en phase 0, et on recale ce tableau à ce moment-là.
