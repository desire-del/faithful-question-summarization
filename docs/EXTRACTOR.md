# Choix de l'extracteur de jargon

> Décidé le 2026-11-27 selon une règle fixée à l'avance : le meilleur F1 (recouvrement de spans) sur
> la moitié « sélection », rapporté sur l'autre moitié. Rapport complet : `reports/extractor/validation_v3.json` (v3 = agrégation au niveau du mot ; la v2, qui coupait des mots en morceaux, est conservée pour trace).

**Données de référence** : corpus CHQA-email (Kilicoglu et al., 2018), 1 720 vrais e-mails de patients
adressés à la NLM (la même source que MeQSum), entités annotées par des experts, fautes comprises.
Adjudication MACE. Jargon de référence = Problem, DrugSupplement, Substance, Anatomy,
DiagnosticProcedure, ProcedureDevice, GeneProtein, CellularEntity.

| Candidat | P (rapport) | R (rapport) | F1 sélection | **F1 rapport** | Contexte vs Measurement (R) | s / 100 e-mails |
|---|---|---|---|---|---|---|
| **OpenMed ×3 + procédures DistilBERT** ✅ | 0,867 | **0,629** | **0,730** | **0,729** | 0,618 | 17,8 |
| OpenMed ×3 (maladie, médicament, anatomie) | 0,952 | 0,573 | 0,727 | 0,715 | 0,582 | 10,8 |
| DistilBERT MACCROBAT | 0,771 | 0,511 | 0,613 | 0,615 | 0,618 | 1,3 |
| DeBERTa MACCROBAT | 0,941 | 0,369 | 0,534 | 0,530 | 0,673 | 3,0 |

L'écart de F1 en sélection est faible (0,730 contre 0,727), mais l'ensemble a un **rappel** nettement meilleur
(0,63 contre 0,57). C'est ce qui compte pour la récompense : un terme de référence non détecté ne peut pas être
compté comme perdu.

**Pourquoi l'ensemble** : OpenMed est précis sur les maladies, les médicaments et l'anatomie, mais rate les
examens et les actes (rappel de 0,25 et 0,19). DistilBERT y est bon. En ajoutant ses seuls labels « procédure »
et « contexte », le rappel passe de 0,57 à 0,63, pour une précision de 0,87.

**Limites, à rapporter** :
- Rappel de 0,63 : environ un terme de référence sur trois n'est pas détecté. La récompense est donc *indulgente*
  (un terme non détecté ne peut pas être compté comme perdu), jamais injuste.
- CHQA n'annote que des questions de patients, pas des résumés. L'extracteur sera aussi vérifié sur 30 résumés
  de référence annotés à la main (phase 1).
- Coût : 4 modèles NER, mais les résumés sont courts (~11 mots). Dans GRPO, la récompense porte sur des
  complétions courtes, et les termes des sources et des références sont mis en cache.
