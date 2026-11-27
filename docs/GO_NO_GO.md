# Phase 0 : état du go/no-go

> Mis à jour le 2026-11-27. Les rapports JSON détaillés (avec provenance) sont dans `reports/go_no_go/`.

| Point (DESIGN §9) | Statut | Preuve | Suite |
|---|---|---|---|
| Tokenizers 4B et 27B identiques | ✅ **GO** | `tokenizers.json` : même vocabulaire (SHA-256), mêmes encodages des sondes, mêmes tokens spéciaux, même rendu du gabarit de chat | Distillation v1.1 possible |
| Chargement texte seul et génération (Qwen3.5) | ✅ **GO** (0.8B, local) | `generate_qwen35_0_8b_local.json` : `Qwen3_5ForCausalLM` sans la partie vision, thinking bien désactivé, pic de 1,7 Go | À refaire sur 4B (Colab) et 27B (cluster) |
| GRPO TRL + LoRA sur Qwen3.5 | ✅ **GO** (0.8B, local) | `grpo_qwen35_0_8b_local.json` : 4 pas, 5,2 s par pas (8 complétions, génération HF), pic de 2,0 Go | La plomberie TRL et Qwen3.5 fonctionne |
| GRPO du 4B sur Colab | ⏳ **à lancer** | `notebooks/colab/00_go_no_go.ipynb` | Ouvrir le notebook dans Colab (runtime GPU) et tout exécuter |
| GRPO du 27B sur le cluster | ⏳ **à lancer** | `scripts/cluster/go_no_go_27b.sbatch` (bf16 ; repli 4 bits commenté) | Adapter les lignes `#SBATCH` au cluster |
| Accès au cluster (quota, file d'attente) | ⏳ **à vérifier** | — | — |
| Corpus CHQs (annotations d'entités de référence) | ✅ **GO** | Téléchargeable (CHQA-Corpus-1.0, 12,5 Mo, format brat). **436 des 1 000 questions MeQSum** y sont annotées (CHQA-email, correspondance des textes et des IDs) | Validation de l'extracteur de jargon côté source. Les annotations ne sont pas adjudiquées : choisir une méthode (MACE). Licence à vérifier |
| Test MEDIQA 2021 | ✅ **GO** | GitHub `abachaa/MEDIQA2021` : validation (50) et test (100) avec questions et références, CC BY 4.0 | Chargement en phase 1 |

## Premières observations (0.8B en zero-shot, sondes synthétiques)
Elles illustrent déjà le problème que vise le projet :
- **Jargon perdu** : « hypothyroidism » disparaît du résumé de la sonde 1 (« Can I continue taking levothyroxine 50 mcg every morning during pregnancy? »).
- **Recopie** : les sondes 6 et 8 sont recopiées presque mot pour mot.
- **Sens déformé** : la sonde 3 devient une question sur la prescription au lieu des effets secondaires.

## Budget GPU
Il sera recalé quand les mesures du 4B (Colab) et du 27B (cluster) seront disponibles.
Repère local : le GRPO complet sur 0.8B (14 400 complétions) est estimé à ~2,6 h sur la RTX 4050, avec la génération HF.
