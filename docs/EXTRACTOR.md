# Choix de l'extracteur de jargon

> Décidé le 2026-11-27 selon une règle fixée à l'avance : le meilleur F1 (recouvrement de spans) sur
> la moitié « sélection », rapporté sur l'autre moitié. Rapport complet : `reports/extractor/validation_v2.json`.

**Données de référence** : corpus CHQA-email (Kilicoglu et al., 2018), 1 720 vrais e-mails de patients
adressés à la NLM (la même source que MeQSum), entités annotées par des experts, fautes comprises.
Adjudication MACE. Jargon de référence = Problem, DrugSupplement, Substance, Anatomy,
DiagnosticProcedure, ProcedureDevice, GeneProtein, CellularEntity.

| Candidat | P (rapport) | R (rapport) | F1 sélection | **F1 rapport** | Contexte vs Measurement (R) | s / 100 e-mails |
|---|---|---|---|---|---|---|
| **OpenMed ×3 + procédures DistilBERT** ✅ | 0,837 | **0,648** | **0,721** | **0,731** | 0,636 | 15,5 |
| OpenMed ×3 (maladie, médicament, anatomie) | 0,906 | 0,587 | 0,711 | 0,712 | 0,582 | 9,9 |
| DistilBERT MACCROBAT | 0,773 | 0,548 | 0,641 | 0,641 | 0,636 | 1,0 |
| DeBERTa MACCROBAT | 0,936 | 0,388 | 0,549 | 0,549 | 0,636 | 2,8 |

**Pourquoi l'ensemble** : OpenMed est précis sur les maladies, les médicaments et l'anatomie, mais rate les
examens et les actes (rappel de 0,25 et 0,19). DistilBERT y est bon. En ajoutant ses seuls labels « procédure »
et « contexte », le rappel passe de 0,59 à 0,65 (examens 0,70, actes 0,50), pour une précision de 0,84.

**Limites, à rapporter** :
- Rappel de 0,65 : environ un terme de référence sur trois n'est pas détecté. La récompense est donc *indulgente*
  (un terme non détecté ne peut pas être compté comme perdu), jamais injuste.
- CHQA n'annote que des questions de patients, pas des résumés. L'extracteur sera aussi vérifié sur 30 résumés
  de référence annotés à la main (phase 1).
- Coût : 4 modèles NER, mais les résumés sont courts (~11 mots). Dans GRPO, la récompense porte sur des
  complétions courtes, et les termes des sources et des références sont mis en cache.
