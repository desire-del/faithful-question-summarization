# Post-mortem de la v1 (`clinical_text_sum`)

> Pourquoi ce document : la v1 annonçait ROUGE-1 = 0.736 sur MeQSum. C'est environ
> 20 points au-dessus de ce que rapporte la littérature. Un résultat aussi bon doit
> être audité avant d'être affiché. Ce post-mortem fait partie du portfolio : savoir
> détecter un résultat trop beau compte plus qu'un score.

## 1. Le signal d'alerte

| Source | ROUGE-1 sur MeQSum |
|---|---|
| Pointer-generator, papier original (Ben Abacha & Demner-Fushman, ACL 2019) | ~44 |
| BART et méthodes récentes (2021–2023) | ~50–55 |
| `NouRed/medqsum-bart-large-xsum-meqsum`, sur **sa propre** validation (model card) | 54.3 |
| Le même modèle BART, évalué sur **notre** split de test v1 | **75.7** |
| Notre Flan-T5-Large + LoRA (v1) | **73.6** |

Le même modèle BART passe de 54 à 76 selon le split. Il a très probablement été
entraîné sur des exemples de notre test, car son split n'est pas documenté et le
dataset fait 1 000 exemples. Notre propre score est du même ordre, ce qui pointe
vers le même problème.

## 2. Constats

| # | Constat | Où | Impact |
|---|---|---|---|
| 1 | Split aléatoire **sans seed** (`train_test_split(test_size=0.1)`), recalculé à chaque session | notebook, cellule 17 | Le split change à chaque exécution |
| 2 | Entraînement **repris depuis un checkpoint Drive** (`resume_from_checkpoint=True`, `checkpoint-2025`), puis évaluation dans une session avec un nouveau split | cellules 33, 35, 49 | Les exemples de test de la session d'évaluation ont presque sûrement été vus à l'entraînement : **fuite train → test** |
| 3 | Comparaison avec un BART tiers entraîné sur MeQSum, split inconnu | cellule 44 | Comparaison invalide (contamination) |
| 4 | Prompts incohérents : entraînement avec `"summarize medical question: "`, inférence LoRA **sans** préfixe, baseline avec `"summarize: "`, double préfixe dans la cellule 48 | cellules 11, 42, 48, 49 | Les systèmes ne sont pas comparés à conditions égales |
| 5 | La « baseline T5 » est en réalité Flan-T5-Large en zero-shot, et elle est évaluée sur 50 exemples dans une cellule et sur le test complet dans une autre | cellules 46, 49 | Mauvais étiquetage, chiffres non reproductibles |
| 6 | Test de **50 exemples**, une seule seed, aucun intervalle de confiance | cellule 17 | Différences non significatives présentées comme des gains |
| 7 | ROUGE seul | partout | Ne mesure pas la fidélité, alors que c'est le vrai risque en médical |
| 8 | Hygiène : `data/opi.csv` (comptes rendus de radio IU X-ray, sans rapport avec la tâche, jamais utilisé), `tensorflow` dans les dépendances, faute de frappe `pert` pour `peft`, tout le projet dans un seul notebook | dépôt | Crédibilité du dépôt |

## 3. Ce que la v2 change

| Constat | Mesure dans la v2 |
|---|---|
| 1, 2 | Split figé une seule fois, avec manifeste (IDs et hash) versionné ; groupement des quasi-doublons avant le split ; vérification automatique de l'absence de recouvrement |
| 3 | Aucun modèle tiers fine-tuné sur MeQSum dans les comparaisons ; tous les systèmes sont entraînés par nous, sur le même split |
| 4, 5 | Prompts versionnés dans le code ; un seul module de génération pour tous les systèmes ; décodage identique et déclaré dans la config |
| 6 | 3 seeds, bootstrap apparié, IC à 95 % ; évaluation sur le **test officiel MEDIQA 2021** (fixe, comparable à la littérature) |
| 7 | Métriques de fidélité (hallucination et omission d'entités), validées par annotation humaine |
| 8 | Package Python testé, CI, configs, notebooks Colab réduits à des lanceurs |
| Usage répété du test | « Test lock » : le test ne s'évalue qu'avec `--final`, et chaque accès est journalisé |

## 4. Actions sur l'existant

Textes prêts à coller (rien n'est encore publié) :

**Bandeau en tête du README de `clinical_text_sum`**
```markdown
> ⚠️ **Archived. The scores below are not valid.** The random train/test split was
> regenerated across sessions, so test questions were most likely seen during training,
> and the BART baseline was trained on MeQSum with an unknown split. See the post-mortem
> and the rebuilt project: <link to the v2 repository>/blob/main/docs/V1_POSTMORTEM.md
```

**Note en tête de la model card `autodidacte228/meqsum-lora-T5-finetuned`**
```markdown
> ⚠️ The ROUGE scores previously reported for this model are invalid (train/test leakage,
> see <link to the v2 post-mortem>). Kept for reference only; not for clinical use.
```


- [ ] Ajouter un bandeau au README de `clinical_text_sum` : « Archivé : les scores sont invalides (fuite de données), voir la v2 et ce post-mortem », puis archiver le dépôt.
- [ ] Corriger la model card Hugging Face `autodidacte228/meqsum-lora-T5-finetuned` et le Space (retirer les scores ou les marquer comme invalides).
- [ ] Facultatif : reproduire la fuite en une cellule (même BART : split propre vs split v1) et mettre le chiffre dans ce document.
