# ADR-001 : choix des modèles

> Statut : proposé, 2026-11-27. Disponibilité, licences et tokenizers vérifiés sur le Hub
> Hugging Face le même jour. La taille des modèles **n'est pas** un critère :
> on dispose de Colab Pro et du cluster de l'école.

## 1. Ce que le problème exige

| Propriété de la tâche | Valeur | Conséquence pour le choix |
|---|---|---|
| Entrée courte, sortie très courte | ~60 mots en entrée, ~10–11 mots en sortie (MEDIQA / MeQSum) | La longueur de contexte ne compte pas. La génération coûte peu, donc GRPO reste abordable même sur un gros modèle. |
| Très peu de données | 1 000 paires d'entraînement | Il faut un **prior fort** (modèle instruit, grand). Le full fine-tuning d'un gros modèle risquerait le surapprentissage, d'où LoRA pour les gros. |
| Jargon médical **et** langage de patient | fautes, mots courants à transformer en termes médicaux (« sugar » → « diabetes ») | La **connaissance médicale** du modèle compte : c'est un argument pour un grand modèle ou un modèle médical. |
| Fidélité avant le style | ne pas paraphraser ni inventer de termes | Argument pour l'encodeur-décodeur (cross-attention vers la source), à tester. |
| Post-training DPO / GRPO avec TRL ou Unsloth, pas from scratch | — | Ces étapes exigent un **décodeur seul** : TRL 1.19 ne gère pas les encodeurs-décodeurs (vérifié dans le code source). |
| Comparaison avec l'état de l'art | — | Il faut (a) le **même benchmark** que la littérature et (b) un modèle de la littérature **reproduit** pour valider notre pipeline (leçon de la v1). |
| Distillation on-policy plus tard | — | L'enseignant et l'élève doivent partager le **même tokenizer** (la distillation sur les logits l'exige). |
| Reproductibilité | — | Poids ouverts, licence permissive, idéalement sans accès restreint (gated). |

## 2. L'état de l'art à battre

**MeQSum n'a pas de split officiel** : chaque papier refait le sien (par exemple
400 / 100 / 500 pour FocusMed), donc les chiffres ne sont pas comparables entre papiers.
**Le seul benchmark vraiment comparable est le test officiel de MEDIQA 2021, tâche 1**
(100 questions, références d'experts, CC BY 4.0). L'entraînement s'y fait sur MeQSum.

Test officiel MEDIQA 2021 :

| Système | Type | ROUGE-L | BERTScore | SummaC_ZS |
|---|---|---|---|---|
| BART (relancé par FocusMed) | encodeur-décodeur, SFT | 0.293 | 0.747 | 0.448 |
| FaMeSumm (EMNLP 2023) | BART + apprentissage contrastif sur les termes médicaux | 0.301 | 0.753 | 0.472 |
| MEDAL | — | 0.302 | 0.754 | 0.496 |
| GPT-4o, zero-shot | API | 0.309 | — | — |
| Damo (MEDIQA 2021) | — | 0.313 | — | — |
| Qwen2.5-7B, SFT QLoRA | décodeur | 0.334 | — | — |
| LLaMA3.1-8B, SFT QLoRA | décodeur | 0.339 | — | — |
| **FocusMed (2025), état de l'art** | LLaMA3.1-8B QLoRA, plus extraction du focus (Qwen) et sélection par un juge (DeepSeek-R1) | **0.386** | **0.782** | **0.603** |

Source : tableaux II et V de FocusMed (arXiv 2510.04671). Les chiffres marqués comme
cités dans ce papier viennent des articles d'origine. Les implémentations de ROUGE
et de BERTScore diffèrent selon les papiers, ce qu'on signale dans le rapport.

**Positionnement** : FaMeSumm est le travail le plus proche de notre idée (forcer les
termes médicaux), mais il le fait par apprentissage contrastif sur un encodeur-décodeur.
Nous le faisons par une **récompense** (best-of-N, DPO, GRPO) sur des LLM, avec des
métriques de jargon explicites. Aucun papier ne rapporte de métriques de préservation
du jargon sur MEDIQA : sur cet aspect, on établit la référence.

## 3. Candidats examinés

| Modèle | Type, taille | Licence, date | Pour | Contre | Verdict |
|---|---|---|---|---|---|
| **BART-large** | enc-déc, 406M | Apache-2.0, 2019 | Base de tous les SOTA MeQSum / MEDIQA ; chiffre publié à reproduire (0.293) | Ancien ; pas de post-training possible avec TRL | ✅ **ancre** |
| PEGASUS-large | enc-déc, 568M | 2020 | 2ᵉ de MEDIQA 2021 | Fait doublon avec BART | ❌ |
| T5Gemma 2 (1B-1B, 4B-4B) | enc-déc moderne | Gemma, oct. 2025 | Encodeur-décodeur récent | Version pt seulement (pas d'IT) ; aucune référence dans la littérature ; pas de post-training TRL | ❌ parking lot |
| **Qwen3.8-27B** | décodeur dense, 27.8B | **Apache-2.0**, août 2026, non gated | Le plus récent ; même famille que les modèles du SOTA (Qwen2.5) ; **même tokenizer que Qwen3.5-4B** (vocabulaire de 248 320, même classe) ; dense, pas de MoE | Enveloppe multimodale ; mode « thinking » à désactiver ; GRPO 27B demande un GPU de 80 Go (cluster) | ✅ **phare** (à confirmer par le bake-off) |
| Gemma 4 31B | décodeur dense, 31.3B | Apache-2.0, mars 2026 | Très bon ; petits frères E2B / E4B (même vocabulaire) | Même rôle que Qwen | 🔶 concurrent du bake-off |
| Gemma 4 26B-A4B | MoE | Apache-2.0 | Inférence rapide | MoE + GRPO = instabilité du routeur | ❌ |
| MedGemma 27B text-it | décodeur, 27B, **médical** | licence HAI-DEF, gated, mai 2025 | Prior médical, directement utile pour le jargon | Génération Gemma 3 (plus ancienne) ; licence restrictive ; l'adaptation médicale ne bat pas toujours le modèle général (Jeong et al., EMNLP 2024) | 🔶 concurrent du bake-off |
| **Qwen3.5-4B** | décodeur dense, 4.7B | **Apache-2.0**, fév. 2026 | Itération rapide ; même tokenizer que le phare (élève idéal) ; démo déployable | Prior plus faible | ✅ **petit modèle** |
| Qwen3.5-9B | décodeur, 9.7B | Apache-2.0 | Proche de la taille du SOTA (8B) | Rôle trop proche du phare ; pas assez petit pour la démo | ❌ |
| LLaMA3.1-8B | décodeur, 8B | Llama 3.1, 2024 | Base de FocusMed | Ancien ; le rôle d'ancre est déjà tenu par BART | ❌ |
| GPT-4o et autres API | — | — | — | Ni fine-tunable ni reproductible | Chiffres cités uniquement |

## 4. Décision : 3 modèles, chacun avec un rôle unique

| Rôle | Modèle | Pourquoi lui | Ce qu'on en fait |
|---|---|---|---|
| **Ancre, encodeur-décodeur** | BART-large | Reproduire 0.293 sur MEDIQA **valide notre pipeline** avant tout résultat. Il répond aussi à « l'encodeur-décodeur garde-t-il mieux le jargon ? » | SFT (full), 3 seeds |
| **Phare, vise l'état de l'art** | Qwen3.8-27B | Le meilleur prior ouvert sous licence permissive ; deux générations après le Qwen du SOTA ; enseignant pour la distillation future | zero-shot, SFT LoRA, best-of-N, DPO, GRPO |
| **Petit modèle** | Qwen3.5-4B | C'est là qu'on met au point la recette de post-training (rapide et peu coûteux). Teste « un 4B avec GRPO de jargon bat-il le SFT 8B du SOTA ? ». Sert pour la démo et comme futur élève | zero-shot, SFT LoRA **et** full, best-of-N, DPO, GRPO, ablation |

Pour le développement local sur la RTX 4050 : **Qwen3.5-0.8B** (même famille), uniquement
pour faire tourner la pipeline. Il n'apparaît pas dans les résultats.

- **Pourquoi pas 2 modèles ?** Sans BART, pas d'ancre ni d'encodeur-décodeur. Sans le petit modèle, pas d'itération rapide, de démo ni d'élève. Sans le gros, aucune chance d'atteindre l'état de l'art.
- **Pourquoi pas 4 ?** Chaque modèle de plus multiplie les runs, et les questions sont déjà couvertes.
- **Règle de transfert, qui limite le périmètre** : la recette de post-training (poids de R, β, taille des groupes, époques) est réglée **sur le 4B**, puis transférée telle quelle au 27B. Seul le taux d'apprentissage est réajusté.

## 5. Bake-off avant de figer (≤ 1 jour, inférence seulement)

Avant tout fine-tuning, on évalue les 3 candidats phares en few-shot (k = 5) sur le dev :
Qwen3.8-27B, Gemma 4 31B et MedGemma 27B text.

**Règle, écrite avant de lancer** : on choisit le meilleur score de jargon (rappel du
jargon essentiel moins le taux d'hallucination) ; ROUGE-L départage les ex æquo.
Si les IC se recouvrent, on garde Qwen (licence, famille, distillation).

- Si Gemma 4 gagne : le petit modèle devient Gemma 4 E4B (même vocabulaire).
- Si MedGemma gagne : le petit modèle devient MedGemma 1.5 4B (tokenizer Gemma 3), et on accepte la licence HAI-DEF.

C'est aussi un **mini-résultat** en soi : *un modèle médical spécialisé garde-t-il mieux le jargon, à taille égale ?*

## 6. Risques liés aux modèles

| Risque | Parade |
|---|---|
| GRPO 27B : poids bf16 (~55 Go) et vLLM ne tiennent pas ensemble sur un GPU | vLLM en mode serveur sur un 2ᵉ GPU du cluster, ou checkpoint FP8 (`Qwen3.8-27B-FP8`) pour la génération. Sinon, le phare s'arrête à SFT + best-of-N + DPO (DPO est hors ligne) |
| Enveloppe multimodale (`Qwen3_5ForConditionalGeneration`) mal gérée par TRL | Test d'un pas de GRPO en phase 0 ; sinon chargement du seul modèle de langage |
| Mode « thinking » de Qwen | Désactivé ; on entraîne sur un format sans thinking |
| Surapprentissage (900 exemples) | LoRA pour les gros ; arrêt anticipé sur le dev |

## 7. Plus tard (v1.1, hors v1.0) : distillation on-policy

Enseignant : Qwen3.8-27B après GRPO. Élève : Qwen3.5-4B (ou 2B / 0.8B). On utiliserait
le `DistillationTrainer` de TRL 1.19, qui gère la génération on-policy avec vLLM
(vérifié dans le code source). Le tokenizer partagé rend la distillation sur les logits directe.

## Sources
- FocusMed, arXiv 2510.04671 : https://arxiv.org/html/2510.04671
- FaMeSumm, EMNLP 2023 : https://aclanthology.org/2023.emnlp-main.673
- Données MEDIQA 2021 tâche 1 (CC BY 4.0) : https://github.com/abachaa/MEDIQA2021
- Entity-driven contrastive learning (ECL) : https://arxiv.org/html/2304.07437
- MeQSum, Ben Abacha & Demner-Fushman, ACL 2019 : https://lhncbc.nlm.nih.gov/LHC-publications/PDF/pub9963.pdf
- Fiches des modèles sur le Hub HF (consultées le 2026-11-27) : google/gemma-4-*, Qwen/Qwen3.8-27B, Qwen/Qwen3.5-*, google/medgemma-*, google/t5gemma-2-*
