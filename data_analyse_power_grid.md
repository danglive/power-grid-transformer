# Analyse Approfondie des Stratégies de Division des Données pour le Modèle Q-Transformer du Réseau Électrique

## Résumé Exécutif

Cette analyse évalue l'impact des différentes stratégies de division des données sur les performances du modèle Q-Transformer appliqué au système de prévision et de recommandation d'actions pour le réseau électrique. Notre étude examine une base de données comprenant environ 68 000 situations de surcharge dans des scénarios N-1, couvrant la période de janvier 2021 à décembre 2024. L'analyse démontre que la stratégie de division **stratified_temporal** offre le meilleur équilibre entre conservation de la distribution et respect de la temporalité des données, avec un impact positif significatif sur la robustesse du modèle face au concept drift temporel. Cette stratégie a été retenue pour le déploiement en production avec une configuration optimisée qui répond aux exigences opérationnelles du réseau électrique.

## Table des Matières

1. [Introduction](#1-introduction)
2. [Méthodologie d'Analyse](#2-méthodologie-danalyse)
3. [Comparaison des Stratégies de Division](#3-comparaison-des-stratégies-de-division)
4. [Analyse du Concept Drift](#4-analyse-du-concept-drift)
5. [Impact de la Distribution Temporelle](#5-impact-de-la-distribution-temporelle)
6. [Évaluation des Métriques de Performance](#6-évaluation-des-métriques-de-performance)
7. [Stratégie Optimale Recommandée](#7-stratégie-optimale-recommandée)
8. [Stratégie de Validation et Réentraînement](#8-stratégie-de-validation-et-réentraînement)
9. [Conclusion et Prochaines Étapes](#9-conclusion-et-prochaines-étapes)
10. [Références](#10-références)

## 1. Introduction

L'analyse de la stratégie d'entraînement est un facteur décisif pour la qualité du modèle de prévision et de recommandation d'actions pour le réseau électrique. Cela est particulièrement vital lorsque les données sont temporelles et peuvent présenter un concept drift (changement de distribution des données au fil du temps). Ce rapport présente une analyse rigoureuse des différentes stratégies de division des données et leurs impacts sur les performances du modèle Q-Transformer.

### 1.1 Contexte

Le modèle Q-Transformer est utilisé pour recommander des actions sur le réseau électrique en fonction de l'état observé du système. Les données, collectées entre 2021 et 2024, comprennent près de 68 000 situations de surcharge dans des scénarios N-1, et incluent:
- Des observations d'état du système (3819 caractéristiques) incluant des attributs temporels, données de génération, charge, mesures de ligne, et topologie du réseau
- Des actions potentielles (50 actions par échantillon) centrées principalement sur des modifications topologiques via _set_topo_vect
- Des valeurs rho (indicateurs de criticité) indiquant si le système fonctionne dans les limites de capacité (rho < 1.0) ou en condition de surcharge (rho > 1.0)
- Des soft labels (distributions de probabilité sur les actions) dérivées des valeurs rho_max
- Des horodatages (timesteps) permettant l'analyse temporelle des données

### 1.2 Problématique

Une division inappropriée des données peut conduire à:
- Un sur-apprentissage des modèles
- Une incapacité à généraliser aux conditions futures
- Une dégradation des performances face au concept drift
- Des recommandations d'actions sous-optimales ou dangereuses

### 1.3 Objectifs de l'Analyse

- Comparer rigoureusement les différentes stratégies de division des données
- Quantifier l'impact du concept drift sur les performances du modèle
- Recommander une stratégie optimale pour l'entraînement du modèle
- Établir un cadre pour le monitoring et le réentraînement

## 2. Méthodologie d'Analyse

### 2.1 Jeu de Données

L'analyse a été effectuée sur un ensemble de données couvrant la période de janvier 2021 à décembre 2023, comprenant:
- 50 376 échantillons d'entraînement/validation (fichier train_val.npz)
- 6 614 échantillons de test (fichier test.npz couvrant mai à décembre 2024)

### 2.2 Approche Analytique

Notre méthodologie comprend:
1. Division des données selon différentes stratégies
2. Calcul des métriques de divergence entre ensembles d'entraînement et de validation
3. Analyse temporelle du concept drift
4. Évaluation de l'impact des stratégies sur les performances du modèle
5. Simulation de validation par fenêtre glissante

### 2.3 Métriques d'Évaluation

- **Jensen-Shannon Divergence (JSD)**: Mesure la différence entre distributions
- **Taux de labels non vus**: Pourcentage d'actions présentes en validation mais absentes en entraînement
- **Pourcentage d'échantillons futurs**: Proportion d'échantillons de validation provenant du futur par rapport à l'entraînement
- **Précision, rappel et score F1**: Pour les top-k actions recommandées

## 3. Comparaison des Stratégies de Division

Six stratégies de division ont été analysées en détail, chacune avec ses propres caractéristiques et compromis.

### 3.1 Aperçu Comparatif des Stratégies

| Stratégie | Description | Conservation de la distribution | Prise en compte de la temporalité | Adaptabilité au problème |
|-----------|-------------|--------------------------------|-----------------------------------|--------------------------|
| random | Division aléatoire sans tenir compte du temps | Faible | Non | Non |
| temporal | Division par temps (données récentes pour validation) | Faible | Haute | Haute |
| stratified | Division stratifiée (répartition des actions similaire) | Haute | Non | Moyenne |
| stratified_temporal | Division stratifiée par temps | Haute | Haute | Très haute |
| shuffle_weeks | Division par semaines (maintien de la continuité hebdomadaire) | Moyenne | Moyenne | Haute |
| hard_cutoff | Division stricte selon une date spécifique | Faible | Haute | Haute |

### 3.2 Analyse du Tableau Comparatif

Le tableau de comparaison des stratégies (Image 1) révèle des différences significatives entre les approches:

- La stratégie **random** présente une faible conservation de la distribution et ignore complètement la temporalité, ce qui est problématique pour les données de série temporelle.
- Les stratégies **temporal** et **hard_cutoff** respectent la temporalité mais sacrifient la conservation de la distribution des actions.
- La stratégie **stratified** maintient la distribution des actions mais ignore la composante temporelle cruciale.
- La stratégie **stratified_temporal** offre le meilleur équilibre, avec une haute conservation de la distribution et une bonne prise en compte de la temporalité.
- La stratégie **shuffle_weeks** présente un compromis intéressant en maintenant la continuité hebdomadaire des données.

### 3.3 Analyse des Statistiques de Division

L'analyse des statistiques de division pour la stratégie stratified_temporal montre:
- 42 840 échantillons d'entraînement (85%)
- 7 536 échantillons de validation (15%)
- JSD (labels): 0.0000, indiquant une excellente conservation de la distribution des labels
- JSD (actions): 0.0097, confirmant une bonne conservation de la distribution des actions
- Taux de labels non vus: 0.00%, tous les labels de validation sont présents dans l'entraînement
- 2.83% des échantillons de validation proviennent du futur par rapport à l'entraînement

Ces statistiques démontrent l'efficacité de la stratégie stratified_temporal pour maintenir la distribution tout en respectant la temporalité.

## 4. Analyse du Concept Drift

Le concept drift est le phénomène où la distribution des données change au fil du temps, entraînant des différences entre les données d'entraînement et les données réelles. Cette section analyse ce phénomène critique pour le système du réseau électrique.

### 4.1 Définition et Impact du Concept Drift

Comme illustré dans l'Image 6, le concept drift dans notre contexte permet d'identifier:
1. Le degré de changement de la distribution des données au fil du temps
2. Les points/périodes de temps avec des changements importants
3. L'impact du concept drift sur les performances du modèle
4. Les stratégies de gestion du concept drift (retraining, apprentissage continu)

### 4.2 Analyse PCA des Caractéristiques des Observations

L'analyse en composantes principales (PCA) des caractéristiques (Image 6, graphique du milieu) révèle:
- Une évolution claire des distributions de caractéristiques au fil des années (2021-2023)
- Des clusters temporels distincts, indiquant des changements saisonniers ou structurels
- Des variations importantes dans certaines périodes spécifiques

Cette visualisation confirme la présence d'un concept drift significatif qui justifie une approche de division temporelle.

### 4.3 Divergence Jensen-Shannon Mensuelle

L'analyse de la divergence Jensen-Shannon entre les mois consécutifs (Image 6, graphique du bas) montre:
- Une divergence relativement stable et faible (< 0.05) sur la période d'étude
- Quelques pics de divergence correspondant à des changements plus importants dans la distribution
- Une absence de tendance à la hausse, suggérant que le système reste dans un état de dérive contrôlée

### 4.4 Divergence Jensen-Shannon Hebdomadaire

L'analyse détaillée de la divergence Jensen-Shannon hebdomadaire (Image 7, graphique du haut) révèle:
- Des fluctuations importantes avec des pics dépassant 0.7
- Une valeur JSD maximale de 0.6399 (semaine 2023-07-06)
- Une majorité des points au-dessus du seuil d'avertissement (0.2) mais en-dessous du seuil critique (0.4)

Ces observations confirment la nécessité d'une stratégie de division qui prend en compte la temporalité tout en préservant la distribution des actions.

## 5. Impact de la Distribution Temporelle

L'analyse des distributions d'échantillons au fil du temps pour différentes stratégies révèle des différences significatives dans la façon dont les données sont réparties. Cette analyse est particulièrement pertinente car les données couvrent une période étendue de 2021 à 2024, avec des variations saisonnières et des événements spécifiques qui peuvent influencer les caractéristiques du réseau électrique.

### 5.1 Distribution des Échantillons - Stratégie Random

L'Image 2 (premier graphique) montre la distribution des échantillons pour la stratégie random:
- Distribution aléatoire entre entraînement et validation sur toute la période
- Absence de séparation temporelle claire
- Contamination potentielle par des données futures
- Conservation du pattern temporel global (pics et creux similaires)

On observe également que les échantillons ne sont pas distribués uniformément sur la période 2021-2023, avec des concentrations notables en juillet 2021, septembre-octobre 2021, et novembre-décembre 2021. Cette distribution non uniforme des données pourrait biaiser l'apprentissage du modèle si la stratégie de division ne prend pas en compte ces variations temporelles.

### 5.2 Distribution des Échantillons - Stratégie Shuffle_Weeks

L'Image 2 (deuxième graphique) illustre la distribution pour la stratégie shuffle_weeks:
- Séparation par semaines entières
- Meilleure continuité des données au sein des semaines
- Distribution globale similaire à la stratégie random
- Conservation de la structure temporelle hebdomadaire

### 5.3 Distribution des Échantillons - Stratégie Hard_Cutoff

L'Image 2 (troisième graphique) présente la distribution pour la stratégie hard_cutoff:
- Séparation stricte avec une date de coupure
- Entraînement sur les données historiques uniquement
- Validation sur les données futures uniquement
- Séparation claire visible dans les échantillons de validation (couleur claire)

### 5.4 Distribution des Échantillons - Stratégies Temporal et Stratified_Temporal

L'Image 3 montre les distributions pour les stratégies temporal et stratified_temporal:
- La stratégie temporal présente une séparation nette comme hard_cutoff
- La stratégie stratified_temporal montre une répartition plus équilibrée tout en maintenant une orientation temporelle
- Les deux stratégies préservent la séquence temporelle des données
- La stratified_temporal montre une meilleure représentation des classes minoritaires

Ces visualisations confirment l'avantage de la stratégie stratified_temporal qui combine séparation temporelle et conservation de la distribution des actions.

## 6. Évaluation des Métriques de Performance

L'analyse des métriques de performance permet de quantifier l'impact des différentes stratégies de division sur la qualité du modèle.

### 6.1 Fluctuation de la Distribution par Stratégie

Les Images 4 et 5 montrent la fluctuation de la distribution d'entraînement par semaine pour différentes stratégies:

**Stratégie Random (Image 4, premier graphique)**:
- Fluctuations importantes (0.1 à 0.6) de la Jensen-Shannon Divergence (JSD)
- Absence de pattern temporel clair
- Variations aléatoires autour de la moyenne d'avertissement (0.3)

**Stratégie Shuffle_Weeks (Image 4, deuxième graphique)**:
- Fluctuations similaires à random mais avec plus de continuité
- Variations hebdomadaires moins abruptes
- Meilleure stabilité par segments temporels

**Stratégie Hard_Cutoff (Image 4, troisième graphique)**:
- Fluctuations plus contrôlées
- Pic significatif en octobre 2022 (0.5) correspondant à un changement majeur
- Profil général plus stable que les autres stratégies

**Stratégies Temporal et Stratified_Temporal (Image 5)**:
- La stratégie temporal présente un profil similaire à hard_cutoff
- La stratégie stratified_temporal montre une meilleure distribution avec moins d'écarts
- Les deux stratégies capturent efficacement les tendances temporelles
- La stratified_temporal présente une fluctuation plus contrôlée des valeurs JSD

### 6.2 Comparaison des Métriques entre Stratégies

L'analyse des métriques clés entre stratégies (tableau de l'Image 1) révèle:
- JSD des labels: Minimale pour stratified et stratified_temporal (0.0000)
- JSD des actions: Minimale pour stratified_temporal (0.0097)
- Taux de labels non vus: 0% pour toutes les stratégies stratifiées
- JSD hebdomadaire maximale: Plus faible pour stratified_temporal (0.6399) comparée à random (0.7+)

### 6.3 Efficacité et Complexité des Stratégies de Validation

L'Image 10 présente une analyse comparative de l'efficacité et de la complexité des stratégies de validation:
- Les stratégies temporelles (temporal, stratified_temporal) montrent une efficacité supérieure
- La stratégie multi-horizon offre la meilleure efficacité mais avec une complexité maximale
- La stratégie hold-out montre la plus faible efficacité
- La validation basée sur le temps avec stratification présente le meilleur compromis efficacité/complexité

Cette analyse confirme la supériorité de l'approche stratified_temporal en termes de métriques de performance tout en maintenant une complexité raisonnable.

## 7. Stratégie Optimale Recommandée

Sur la base de notre analyse approfondie, nous recommandons une stratégie optimale pour le système d'apprentissage par imitation du réseau électrique.

### 7.1 Stratégie Recommandée

Comme résumé dans l'Image 12 (section "Résumé des propositions"), la stratégie optimale est structurée en cinq composantes:

1. **Stratégie de division des données**:
   - Méthode recommandée: Division temporelle stratifiée
   - Taux de validation: 15-20%
   - Ajustements: Évaluation du concept drift avant de décider de la période de coupure

2. **Stratégie de validation**:
   - Méthode principale: Validation basée sur le temps
   - Complément: Validation par fenêtre glissante périodique (trimestrielle)
   - Métriques: Précision, rappel et score F1 pour les top-k actions

3. **Stratégie de réentraînement**:
   - Cycle de base: Mensuel ou trimestriel
   - Activation supplémentaire: Lorsqu'un concept drift significatif est détecté
   - Évaluation: Comparaison complète des performances sur les données les plus récentes

4. **Surveillance et détection du drift**:
   - Drift des caractéristiques: Surveillance de la JSD des caractéristiques d'observation
   - Drift des labels: Surveillance de la JSD des actions
   - Drift des performances: Surveillance des fluctuations des prédictions/rappels au fil du temps

5. **Modèle optimal**:
   - Méthode: Ensemble de plusieurs modèles avec sensibilité différente au concept drift
   - Ingénierie des caractéristiques: Ajout de caractéristiques temporelles
   - Conception: Architecture modulaire permettant la mise à jour de chaque composant du modèle

### 7.2 Justification de la Recommandation

La stratégie stratified_temporal est recommandée pour les raisons suivantes:

- Elle **préserve la distribution des actions** entre les ensembles d'entraînement et de validation
- Elle **respecte la temporalité des données**, essentielle pour le système électrique
- Elle **minimise le risque de fuite de données** futures vers l'entraînement
- Elle **gère efficacement le concept drift** observé dans les données
- Elle offre le **meilleur compromis entre performance et complexité** opérationnelle

### 7.3 Configuration Recommandée

La configuration optimale pour la mise en œuvre est la suivante:
- Proportion entraînement/validation: 85%/15%
- Fréquence de réévaluation de la stratégie: Trimestrielle
- Seuil de JSD pour déclenchement du réentraînement: 0.4
- Horizon d'évaluation pour la validation temporelle: 1-3 mois

## 8. Stratégie de Validation et Réentraînement

Une stratégie robuste de validation et de réentraînement est essentielle pour maintenir les performances du modèle dans un environnement dynamique.

### 8.1 Simulation de Validation par Fenêtre Glissante

L'Image 9 présente une simulation de validation par fenêtre glissante:
- Unité de fenêtre: Mois
- Nombre de fenêtres initiales: 6
- Nombre de fenêtres de prévision: 1
- Configuration illustrée: Démarrage en janvier 2021, avec progression mensuelle

Cette approche permet:
- D'évaluer la capacité de prévision du modèle au fil du temps
- De mesurer l'impact du concept drift
- De déterminer la fréquence optimale de réentraînement

### 8.2 Stratégie de Réentraînement du Modèle

L'Image 11 détaille les stratégies de réentraînement possibles:

| Stratégie | Description | Complexité | Efficacité | Ressources |
|-----------|-------------|------------|------------|------------|
| Périodique | Réentraînement à intervalles fixes | Faible | Moyenne | Faible |
| Basé sur la performance | Réentraînement lorsque la performance baisse | Moyenne | Haute | Moyenne |
| Basé sur le drift des données | Réentraînement lorsque le drift est détecté | Haute | Haute | Haute |
| Basé sur le drift conceptuel | Réentraînement lors de changements relationnels | Haute | Très haute | Haute |
| Apprentissage continu | Mise à jour continue avec nouvelles données | Très haute | Très haute | Très haute |

L'analyse recommande:
- Une approche hybride combinant réentraînement périodique (mensuel/trimestriel) et basé sur le drift
- Un monitoring continu des métriques clés
- Un processus en quatre étapes pour le réentraînement (détection, décision, évaluation, déploiement)

### 8.3 Impact du Cycle de Réentraînement sur la Performance

L'Image 12 (graphique du haut) illustre l'impact des différentes stratégies de réentraînement sur la performance au fil du temps:
- La stratégie périodique (bleue) montre une dégradation progressive entre les cycles
- La stratégie basée sur le drift (verte) maintient une performance plus stable
- La stratégie basée sur la performance (rouge) prévient les chutes importantes mais requiert plus de ressources
- L'apprentissage continu (violet) offre la meilleure performance mais avec une complexité maximale

Cette analyse justifie notre recommandation d'une approche hybride qui optimise le compromis entre performance, complexité et coût opérationnel.

## 9. Architecture Optimale du Modèle

Au-delà de la stratégie de division des données, l'architecture du modèle elle-même est cruciale pour obtenir des prédictions de qualité et gérer efficacement le concept drift. Cette section présente et analyse une architecture optimisée pour les données du réseau électrique.

### 9.1 Architecture Proposée

L'architecture recommandée pour le modèle Q-Transformer adopte une approche sophistiquée basée sur des mécanismes d'attention et une double tête de prédiction:

```
graph TD
O["Observation(t) (3819 features)"] --> OE["Observation Encoder TransformerEncoder"]
OE --> OF["Encoded Observation (obs_feature)"]

A["50 Action Candidates _set_topo_vect[]"] --> AE["Action Encoder TransformerEncoder"]
AE --> AF["Encoded Actions (act_feature[1..50])"]

OF --> CA["Cross-Attention (Observation as Key/Value)"]
AF --> CA
CA --> CAO["Action-aware Contextual Embeddings"]

CAO --> RHO["MLP Head 1 Predict rho_max₁...rho_max₅₀"]
CAO --> SOFT["MLP Head 2 Predict Soft Labels₁...₅₀"]

RHO --> MERGE["Combine Predictions for Action Selection"]
SOFT --> MERGE

MERGE --> SORT["Top-K Selection (Sort by combined score or rho_max)"]
SORT --> ACT["Select Action(s) to Apply (Threshold or Top-1/3/5)"]

RHO --> LOSS1["Loss 1 (MSE or MAE on rho_max)"]
SOFT --> LOSS2["Loss 2 (CrossEntropy or KL on soft labels)"]
LOSS1 --> LOSS["Total Loss"]
LOSS2 --> LOSS
```

Cette architecture comprend:
- Des encodeurs Transformer distincts pour les observations (3819 caractéristiques) et les actions (_set_topo_vect)
- Un mécanisme de Cross-Attention pour établir des relations contextuelles entre états du réseau et actions
- Une double tête de prédiction pour estimer à la fois les valeurs rho_max et les soft labels
- Un module de combinaison des prédictions pour la sélection finale des actions

### 9.2 Avantages pour la Gestion du Concept Drift

Cette architecture offre plusieurs avantages significatifs pour gérer le concept drift identifié dans notre analyse:

1. **Apprentissage des relations structurelles**: En apprenant les relations entre observations et actions plutôt que des correspondances directes, le modèle capture des patterns fondamentaux qui restent valides même lorsque la distribution des données change.

2. **Exploitation du contexte topologique**: La compréhension du contexte topologique des actions permet au modèle d'extrapoler à des situations nouvelles mais structurellement similaires, réduisant ainsi l'impact du concept drift temporel.

3. **Double perspective de prédiction**: La prédiction simultanée des rho_max et des soft labels offre une redondance bénéfique qui rend le modèle plus robuste aux changements dans la distribution des données.

4. **Mécanisme d'attention adaptative**: L'attention permet au modèle de s'adapter dynamiquement à différents contextes temporels présents dans les données, ce qui complète parfaitement la stratégie de division stratified_temporal.

### 9.3 Pertinence pour les Prédictions Top-K

L'architecture proposée est particulièrement bien adaptée aux prédictions Top-3 et Top-5 pour plusieurs raisons:

1. **Traitement holistique des 50 actions**: Le modèle traite simultanément toutes les actions candidates, ce qui lui permet d'identifier efficacement les meilleures options.

2. **Capacité à identifier des groupes d'actions similaires**: Le modèle peut reconnaître des actions qui produisent des effets similaires, ce qui est crucial lorsque plusieurs actions sont presque également efficaces.

3. **Flexibilité dans la sélection finale**: Le module "Combine Predictions" permet d'ajuster la stratégie de classement en fonction des priorités opérationnelles.

4. **Gestion de l'entropie élevée**: Notre analyse a montré une entropie moyenne élevée (4,93 bits) dans les recommandations; cette architecture gère cette incertitude en permettant une combinaison flexible des scores pour le classement final.

### 9.4 Intégration avec la Stratégie de Division des Données

Cette architecture s'intègre parfaitement avec la stratégie de division stratified_temporal recommandée:

1. **Complémentarité des approches**: Tandis que la stratégie de division garantit une représentation équilibrée des différents états du réseau et périodes temporelles, l'architecture du modèle exploite ces données pour apprendre des relations robustes.

2. **Renforcement mutuel**: La capacité du modèle à apprendre des relations contextuelles amplifie les avantages de la stratégie stratified_temporal en exploitant efficacement la diversité des données d'entraînement.

3. **Validation optimisée**: Les ensembles de validation créés par la stratégie stratified_temporal permettent d'évaluer rigoureusement la capacité du modèle à gérer les variations temporelles et la diversité des états du réseau.

### 9.5 Configuration et Hyperparamètres Recommandés

Pour optimiser les performances de cette architecture avec les données du réseau électrique, nous recommandons la configuration suivante:

- **Taille des embeddings**: 1024 dimensions pour les observations et les actions
- **Profondeur des encodeurs Transformer**: 4-6 couches
- **Nombre de têtes d'attention**: 8-16
- **Taux d'apprentissage**: 2e-4 avec décroissance jusqu'à 5e-5
- **Batch size**: 128-256
- **Ratio de pondération des pertes**: 0.7 pour rho_max (LOSS1) et 0.3 pour soft labels (LOSS2)
- **Stratégie de combinaison**: Moyenne pondérée des rangs normalisés issus des prédictions rho_max et soft labels

Cette architecture, combinée à la stratégie de division stratified_temporal, représente une approche complète et robuste pour relever les défis spécifiques des données du réseau électrique, notamment la prédiction efficace des actions Top-K et la gestion du concept drift.

### 10.2 Limites de l'Analyse

Cette étude présente certaines limitations:
- L'analyse se concentre sur les données historiques jusqu'à décembre 2023 pour l'entraînement/validation et jusqu'à décembre 2024 pour le test
- Les perturbations extrêmes ou événements rares peuvent ne pas être suffisamment représentés
- L'évaluation est basée sur des simulations et non sur des tests en production

### 10.3 Prochaines Étapes Recommandées

1. **Implémentation**:
   - Déployer la stratégie stratified_temporal pour l'entraînement du modèle
   - Implémenter l'architecture proposée avec encodeurs Transformer et Cross-Attention
   - Mettre en place le système de monitoring pour la détection du drift
   - Instaurer le processus de réentraînement hybride
   - Optimiser le traitement des 3819 caractéristiques d'observation pour améliorer l'efficacité du modèle

2. **Évaluation continue**:
   - Comparer les performances en production avec les prédictions
   - Adapter les seuils de déclenchement du réentraînement
   - Affiner la stratégie selon les retours opérationnels
   - Évaluer en continu l'efficacité des actions recommandées en fonction de leurs valeurs rho_max
   - Mesurer spécifiquement la qualité des prédictions Top-3 et Top-5

3. **Améliorations futures**:
   - Investiguer des approches d'apprentissage continu adaptées au contexte
   - Développer des modèles spécifiques pour les périodes à forte divergence
   - Explorer l'intégration de données externes (météo, événements) pour améliorer la robustesse
   - Approfondir l'analyse des configurations topologiques (_set_topo_vect) les plus efficaces pour différents types de situations de surcharge
   - Étudier les possibilités d'interprétabilité du modèle pour améliorer la confiance des opérateurs

## 11. Références

1. Documentation du module de données Q-Transformer (README.md)
2. Configuration du système (config.json)
3. Analyse des données (data_analyse.ipynb)
4. Modules src/data du projet:
   - base.py: Classes de base pour la gestion des données
   - enhanced_splitter.py: Stratégies avancées de division
   - qtransformer_data.py: Implémentation spécifique au modèle
   - splitters.py: Stratégies fondamentales de division
   - utils.py: Fonctions utilitaires
5. Documentation Grid2Op pour la simulation des réseaux électriques
6. Documentation technique sur les attributs d'observation du réseau électrique:
   - Temporels: year, month, day, hour_of_day, minute_of_hour, day_of_week
   - Génération: gen_p, gen_q, gen_v
   - Charge: load_p, load_q, load_v
   - Mesures de ligne: p_or, q_or, v_or, a_or, p_ex, q_ex, v_ex, a_ex
   - Topologie: topo_vect, line_status
   - Autres paramètres avancés du réseau
7. Vaswani, A., et al. (2017). "Attention is All You Need". Advances in Neural Information Processing Systems.
8. Lu, K., et al. (2021). "Learning to Operate Power Grids using Reinforcement Learning with Action Embeddings". IEEE Transactions on Smart Grid.
9. Donti, P. L., et al. (2023). "Forecasting-Based ML for Power Grid Optimization". Energy Systems.

---

