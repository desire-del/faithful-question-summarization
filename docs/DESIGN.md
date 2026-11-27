# Préserver le jargon médical par post-training : résumé fidèle de questions de patients

> Statut : v0.4, 2026-11-27. Modèles choisis dans [MODEL_SELECTION.md](MODEL_SELECTION.md) (ADR-001).
> Documents liés : [ROADMAP.md](ROADMAP.md) · [PARKING_LOT.md](PARKING_LOT.md) · [V1_POSTMORTEM.md](V1_POSTMORTEM.md)

## 1. Le problème

Un patient écrit : *« J'ai 34 ans, je suis enceinte, on m'a diagnostiqué une
hypothyroïdie et je prends du Levothyrox. Est-ce que je peux continuer pendant la
grossesse ? »* (exemple traduit ; les données sont en anglais)

| Résumé | ROUGE | Problème |
|---|---|---|
| Le Levothyrox est-il sûr pendant la grossesse en cas d'hypothyroïdie ? | élevé | ✅ |
| Puis-je continuer mon traitement pour la thyroïde enceinte ? | **élevé** | ❌ jargon paraphrasé |
| Le Levothyrox est-il sûr ? | **élevé** | ❌ contexte critique perdu |
| Le Synthroid est-il sûr pendant la grossesse ? | **élevé** | ❌ terme halluciné |

Les termes médicaux (maladies, médicaments, examens, doses) sont ce dont un moteur
de recherche ou un médecin a besoin. ROUGE ne voit presque pas qu'ils ont été perdus.

## 2. La question centrale et les hypothèses

> **Peut-on, par post-training avec une récompense qui pénalise la perte du jargon
> médical, apprendre à un LLM à conserver les termes médicaux essentiels sans
> halluciner ni recopier, et dépasser l'état de l'art sur le benchmark MEDIQA 2021 ?**

Les hypothèses sont figées par commit avant les expériences, dans `docs/PROTOCOL.md`.

| | Hypothèse |
|---|---|
| **H1 Mesure** | ROUGE s'accorde mal avec le jugement humain sur la préservation du jargon ; nos métriques de jargon s'accordent mieux. |
| **H2 Architecture** | Après SFT, l'encodeur-décodeur de référence (BART-large) hallucine moins de termes que des décodeurs bien plus gros, à ROUGE comparable. |
| **H3 Post-training** | GRPO réduit d'au moins 30 % (en relatif) la perte de jargon par rapport au SFT, pour une perte ≤ 1 point de ROUGE-L. **H3b** : sans pénalité de longueur, le modèle triche en recopiant la question. |
| **H4 État de l'art** | Sur le test MEDIQA 2021, le phare après post-training dépasse FocusMed en SummaC_ZS (0.603) et égale ou dépasse son ROUGE-L (0.386). Le petit modèle après post-training dépasse le SFT LLaMA3.1-8B (ROUGE-L 0.339). |

Les résultats négatifs sont publiés tels quels.

## 3. Données : le protocole MEDIQA 2021

| Jeu | Rôle | Taille |
|---|---|---|
| MeQSum (HF `albertvillanova/meqsum`) | entraînement | 900 (sur 1 000) |
| MeQSum, 100 exemples mis de côté, plus la validation MEDIQA 2021 | **dev** (sélection, réglage de R) | 100 + 50 |
| **Test officiel MEDIQA 2021, tâche 1** (GitHub `abachaa/MEDIQA2021`, CC BY 4.0) | **test**, comparable à la littérature | 100 |

- On entraîne sur 900 exemples au lieu de 1 000 comme les papiers : c'est un handicap **conservateur**, et on le déclare.
- Normalisation des espaces uniquement ; les quasi-doublons sont groupés dans le même split ; le recouvrement MeQSum ↔ MEDIQA est vérifié ; le manifeste (IDs et hash) est versionné.
- Pas de « test MeQSum » : faute de split officiel, ses chiffres ne seraient pas comparables.

## 4. Modèles et systèmes

Le détail de l'analyse est dans [MODEL_SELECTION.md](MODEL_SELECTION.md).

| Rôle | Modèle |
|---|---|
| Ancre, encodeur-décodeur | **BART-large** |
| Phare, vise l'état de l'art | **Qwen3.8-27B** (à confirmer par le bake-off contre Gemma 4 31B et MedGemma 27B) |
| Petit modèle, où la recette est mise au point | **Qwen3.5-4B** |
| Développement local uniquement | Qwen3.5-0.8B (absent des résultats) |

| ID | Système | Seeds |
|---|---|---|
| B1 | BART-large, SFT | 3 |
| P0 / G0 | Qwen3.5-4B / Qwen3.8-27B, zero-shot | — |
| P1 | 4B, SFT LoRA | 3 |
| P2 | 4B, SFT full fine-tuning | 3 |
| G1 | 27B, SFT LoRA | 3 |
| P3 / G3 | Best-of-N (N = 8), trié par R, au-dessus du SFT | 3 |
| P4 / G4 | **DPO** à partir du SFT, paires construites par R | 3 |
| P5 / G5 | **GRPO** à partir du SFT, récompense R | 3 |
| P5-abl | GRPO 4B sans pénalité de longueur | 1 |

- **Équité** : même format d'entrée, même décodage (greedy, `max_new_tokens = 64`), même budget de recherche d'hyperparamètres, sélection sur le dev uniquement.
- **Règle de transfert** : la recette de post-training est réglée sur le 4B, puis transférée au 27B. Seul le taux d'apprentissage est réajusté.
- **Outils** : TRL (`SFTTrainer`, `DPOTrainer`, `GRPOTrainer`) et Unsloth si la VRAM manque ; `Seq2SeqTrainer` pour BART ; vLLM pour la génération des gros modèles.

## 5. La récompense de jargon R

### 5.1 Extraire le jargon
- scispaCy (`en_core_sci_md`) détecte les termes biomédicaux. Des règles ajoutent les doses, nombres, âges, durées et la grossesse.
- Normalisation : minuscules et lemmes, plus une correspondance floue pour les fautes des patients.
- La correspondance se fait **sur la forme des termes** : le but est de garder le jargon, pas seulement le concept.
- L'extracteur est validé sur les annotations d'entités du **corpus CHQA** (Kilicoglu et al., 2018) : 436 questions MeQSum y sont annotées (vérifié en phase 0). On complète par 30 résumés annotés à la main, car CHQA n'annote que les questions.

### 5.2 Formule
J\* = le jargon de la référence (le jargon essentiel), J(ŷ) = celui de la prédiction, J(x) = celui de la source.
```
R(x, ŷ, y) =  1.0 · |J(ŷ) ∩ J*| / |J*|                  # jargon essentiel conservé
            − γ   · |J* \ J(ŷ)|                          # pénalité par terme essentiel perdu
            − α   · |J(ŷ) \ (J(x) ∪ J*)| / max(1,|J(ŷ)|) # termes hallucinés
            + β   · Format(ŷ)                            # une seule question, « ? », sans préambule
            − λ   · max(0, |ŷ| / |y| − 1.5)              # pénalité de longueur (anti-recopie)
```
- Les poids sont réglés sur le dev avec le 4B, puis **figés et hachés**.
- R est une fonction pure, testée sur des cas adverses : recopier la source, sortie vide, paraphrase d'un terme, terme inventé, préambule.

### 5.3 Utilisation
- **Best-of-N** : 8 candidats, on garde le meilleur selon **R_free**, la variante sans référence (jargon de la source, longueur médiane des références d'entraînement). R lit la référence, donc il ne peut pas servir au test (voir PROTOCOL §8). Aucun entraînement : c'est la baseline qui dit si DPO et GRPO apportent vraiment quelque chose.
- **DPO** : K = 8 candidats par prompt d'entraînement, générés par le SFT (T = 1.0). Le meilleur selon R devient « chosen », le pire « rejected », et on écarte les paires dont la marge est < δ.
- **GRPO** : groupes de 8, LoRA, KL vers le SFT.
- **Monitoring** (W&B) : chaque terme de R, la longueur et le taux de recopie, pour repérer le reward hacking.

## 6. Évaluation

Règle : **le jargon est dans R, donc il ne peut pas être la seule preuve**.

| Métrique | Dans R ? | Rôle |
|---|---|---|
| Rappel du jargon essentiel, termes perdus, termes hallucinés | **oui** | Rapportés, mais signalés « optimisés » |
| ROUGE-1/2/L | non | Comparaison avec l'état de l'art |
| BERTScore | non | Comparaison avec l'état de l'art |
| **SummaC_ZS** | non | Fidélité par NLI, **métrique du SOTA** (FocusMed) |
| Longueur, ratio de recopie | non | Détection du hacking |
| **Annotation humaine** | non | Preuve principale de fidélité |
| Paramètres, VRAM, temps GPU | — | Coût |

**Annotation humaine** : 50 questions du test × 3 systèmes (B1, le meilleur SFT du
phare, le meilleur post-training du phare) = 150 jugements, à l'aveugle et mélangés.
Grille écrite avant de commencer : jargon conservé, terme paraphrasé, contexte critique
perdu, hallucination, question bien formée, type d'erreur. L'annotation se fait dans
un tableur, et 20 jugements sont refaits une semaine plus tard (κ).

**Rigueur**
- 3 seeds, bootstrap apparié (10 000 rééchantillonnages), IC à 95 %.
- Tableau de l'état de l'art : nos chiffres face aux chiffres publiés, avec un avertissement sur les implémentations des métriques.
- **Test lock** : `--split test` exige `--final`, et chaque accès est journalisé dans `reports/test_ledger.jsonl`. Il y a deux accès prévus et déclarés : (1) la reproduction de BART en phase 1, qui n'influence aucun réglage ; (2) l'évaluation finale.

## 7. Architecture logicielle

### Principes
1. **Les prédictions sont la source de vérité** : chaque génération écrit un `predictions.jsonl`, et toutes les métriques se recalculent hors ligne, sans GPU.
2. **Un seul module de génération**, avec des prompts versionnés : HF pour BART et le 4B, vLLM pour les gros.
3. **La récompense et les métriques de jargon partagent le même code** (`jargon.py`).
4. **Chaque run garde sa provenance** (config, commit, hash des données et de R, versions, GPU, seed). On n'écrase jamais un run existant.
5. **Trois environnements, une seule CLI** : le PC (dev, 0.8B), Colab (4B, BART) et le cluster (27B). Le code est identique partout, seule la config change (`configs/env/`).

### Flux
```
MeQSum + MEDIQA ─► data build ─► splits + manifest
                                     │
          ┌──────────────────────────┼──────────────────────────┐
          ▼                          ▼                          ▼
  BART SFT (Seq2Seq)        Qwen3.5-4B SFT (TRL)       Qwen3.8-27B SFT (TRL)
                                     │                          │
                          BoN / DPO / GRPO ──recette figée──►  BoN / DPO / GRPO
          └──────────────────────────┴────────────┬─────────────┘
                                                  ▼
                generate ─► predictions.jsonl ─► score ─► report (tableaux, état de l'art, figures)
```

### Arborescence
```
faithful-question-summarization/
├── pyproject.toml / Makefile
├── configs/                  # env/ (local, colab, cluster), model/, sft/, dpo/, grpo/, reward/
├── src/faithqs/
│   ├── data.py               # MeQSum, MEDIQA, quasi-doublons, splits, manifeste
│   ├── prompts.py            # formats versionnés
│   ├── jargon.py             # extraction et correspondance : partagé par R et les métriques
│   ├── reward.py
│   ├── train_sft.py          # TRL SFTTrainer (décodeurs) et Seq2SeqTrainer (BART)
│   ├── build_pairs.py / train_dpo.py / train_grpo.py
│   ├── generate.py           # HF et vLLM, best-of-N
│   ├── metrics.py            # ROUGE, BERTScore, SummaC, recopie
│   ├── stats.py / report.py  # bootstrap, McNemar, tableaux, test lock
│   └── cli.py
├── notebooks/colab/          # lanceurs minces
├── scripts/cluster/          # scripts SLURM minces
├── results/ reports/ docs/
└── tests/                    # unitaires (R adverse, splits, jargon) et un test de fumée CPU
```

## 8. Livrables
1. **README en anglais = le rapport** : la figure phare (jargon conservé vs ROUGE par système), le tableau face à l'état de l'art, des exemples avant / après GRPO, la figure de reward hacking, les limites.
2. Les modèles (SFT, DPO, GRPO du 4B et du 27B) sur le Hub HF, avec des model cards honnêtes.
3. Une démo Gradio avec le 4B : SFT vs GRPO côte à côte, jargon surligné.
4. Le post-mortem de la v1.

**Pitch d'entretien** : « Les modèles de résumé paraphrasent ou suppriment le jargon
médical, et ROUGE ne le voit pas. J'ai conçu une récompense qui pénalise la perte de
termes médicaux, et j'ai post-entraîné Qwen 4B et 27B (best-of-N, DPO, GRPO) contre
une ancre BART reproduite. Sur le benchmark officiel MEDIQA 2021, face à l'état de l'art : … »

## 9. Go/no-go (phase 0)
| Point | Si oui | Si non |
|---|---|---|
| Un pas de GRPO du 4B sur Colab (TRL, enveloppe Qwen3.5) | TRL | Unsloth ; sinon charger le modèle de langage seul |
| Un pas de GRPO du 27B sur le cluster | Plan tel quel | vLLM sur un 2ᵉ GPU ou en FP8 ; sinon le 27B s'arrête à SFT + BoN + DPO |
| Tokenizers 4B et 27B identiques (fichiers comparés) | Distillation v1.1 possible | Noter la limite |
| Corpus CHQs accessible | Validation de l'extracteur sur des annotations de référence | 30 exemples annotés à la main |
| Accès au cluster (quota, file d'attente) | Plan tel quel | Le 27B passe en QLoRA sur un A100 Colab |
