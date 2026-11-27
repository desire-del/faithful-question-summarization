# Parking lot

> Les idées qui **ne rentrent pas** dans la v1.0. On les note ici pour libérer
> l'esprit, sans les coder. On ne les relit qu'après la v1.0.

## v1.1 : la première chose à faire après la v1.0
- **Distillation on-policy** : le 27B après GRPO comme enseignant, le 4B (ou 2B / 0.8B) comme élève, avec le `DistillationTrainer` de TRL et la génération on-policy par vLLM. Le tokenizer partagé est vérifié en phase 0. Voir ADR-001 §7.

## Sortis du périmètre (2026-11-27)

| Idée | Pourquoi sortie |
|---|---|
| T5Gemma 2 (encodeur-décodeur moderne) | Version pt seulement, aucune référence dans la littérature, pas de post-training TRL. BART joue le rôle d'encodeur-décodeur |
| Gemma 4, MedGemma en fine-tuning | Seulement concurrents du bake-off (inférence). L'un d'eux remplace Qwen s'il gagne, sans s'ajouter |
| LLaMA3.1-8B (reproduire le SFT de FocusMed) | L'ancre est déjà BART ; on cite leurs chiffres |
| Qwen3.5-9B, PEGASUS, modèles MoE | Rôles redondants ; MoE et GRPO instables |
| Split MeQSum « à la FocusMed » (400/100/500) | Split non publié, donc non reproductible ; le test officiel MEDIQA suffit |
| CHQ-Summ, HealthCareMagic, iCliniq | Élargit les données ; CHQ-Summ demande en plus l'accès Yahoo L6 |
| QLoRA vs LoRA, balayage du rang | Question secondaire (QLoRA reste une **solution de repli** technique pour le 27B) |
| LLM-as-judge | L'annotation humaine joue ce rôle ; **à ajouter si on est en avance après la phase 5** |
| Récompense sans référence (jargon de la source seulement) | Un 2ᵉ axe de récompense |
| Correspondance par concept UMLS (synonymes) | On garde volontairement la forme des termes |
| Audit de contamination | Secondaire |
| Données synthétiques | Une autre question (lien avec GroundedDistill) |
| Questions de patients en français | Projet à part |
| DPO / GRPO from scratch | Projet suivant ; ici on utilise TRL et Unsloth |

## Nouvelles idées pendant le projet
<!-- Une ligne par idée, avec la date. Ne pas coder. -->
