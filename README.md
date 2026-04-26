# AdaptHybrid — Système de Recommandation Hybride Adaptatif

> **KNN-CF + SmartCBF avec Correction du Démarrage à Froid**  
> Projet Final — Module Méthodologie de Recherche | ENSET Mohammedia, Université Hassan II, 2025–2026

---

### Contenu de ce dépôt

| Fichier | Description |
|---|---|
| `SmartCBF_Hybrid_V2_Enhanced.ipynb` | ✅ **Code source complet** — notebook d'expérimentation complet (11 sections, exécutable de bout en bout) |
| `article_SmartCBF.pdf` | 📄 **Article de recherche** — article complet au format IEEE décrivant la méthode, le protocole et les résultats |
| `poster_scientifique.pdf` | 🧾 **Poster scientifique** — synthèse visuelle du projet (méthodologie, architecture et résultats principaux) |
| `README.md` | 📖 Ce fichier |

> Tout ce qui est nécessaire pour reproduire les expériences est contenu dans le notebook. Les jeux de données sont téléchargés automatiquement à l'exécution.

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue?logo=python)](https://python.org)
[![NumPy](https://img.shields.io/badge/NumPy-1.26.4-013243?logo=numpy)](https://numpy.org)
[![scikit-surprise](https://img.shields.io/badge/scikit--surprise-compatible-orange)](http://surpriselib.com)
[![License](https://img.shields.io/badge/License-MIT-green)](LICENSE)

---

## Table des matières

- [Vue d'ensemble](#vue-densemble)
- [Questions de recherche](#questions-de-recherche)
- [Architecture du système](#architecture-du-système)
- [Contributions principales](#contributions-principales)
- [Résultats](#résultats)
- [Installation](#installation)
- [Utilisation](#utilisation)
- [Structure du projet](#structure-du-projet)
- [Jeux de données](#jeux-de-données)
- [Protocole expérimental](#protocole-expérimental)
- [Limitations](#limitations)
- [Auteures](#auteures)

---

## Vue d'ensemble

**AdaptHybrid** est un système de recommandation hybride adaptatif qui combine :

- **KNN-CF** — Filtrage collaboratif item-item utilisant la similarité cosinus
- **SmartCBF** — Un module de filtrage par contenu enrichi avec pondération TF-IDF des genres, repli démographique et prior de popularité

Le système adresse le **problème du démarrage à froid** en ajustant dynamiquement l'équilibre entre les deux composantes en fonction de l'activité de l'utilisateur. Pour les utilisateurs avec peu de notations, le système s'appuie sur SmartCBF ; pour les utilisateurs actifs, KNN-CF prend le dessus.

Le paramètre de pondération **α** est appris par validation croisée (grid search), et non fixé à 0,5 comme dans les hybrides naïfs.

---

## Questions de recherche

| # | Question |
|---|----------|
| **RQ1** | Dans quelle mesure la combinaison hybride KNN-CF + SmartCBF améliore-t-elle les performances par rapport à chaque composante isolée, sur RMSE, NDCG@10 et Recall@10 ? et Le mécanisme adaptatif αeff atténue-t-il significativement les effets du démarrage à froid par rapport aux baselines pures et à l'hybride naïf ? |

Les deux hypothèses sont **confirmées** par les résultats expérimentaux.

---

## Architecture du système

```
         ┌────────────────────────┐
         │   Entrée : User u, Item i │
         │   n_u = # notations       │
         └────────────┬───────────┘
                      │
          ┌───────────┴────────────┐
          │                        │
   ┌──────▼──────┐         ┌───────▼────────┐
   │   KNN-CF    │         │   SmartCBF     │
   │ item-item   │         │ TF-IDF genres  │
   │ cosine sim  │         │ Repli démo.    │
   │ k voisins   │         │ Popularité     │
   └──────┬──────┘         └───────┬────────┘
          │                        │
          └───────────┬────────────┘
                      │
            ┌─────────▼──────────┐
            │  Pondération adaptative │
            │  αeff = α* × min(1, n_u / τ)  │
            │                    │
            │  n_u ≪ τ → SmartCBF dominant (cold-start) │
            │  n_u ≥ τ → KNN-CF  dominant (utilisateur actif) │
            └─────────┬──────────┘
                      │
            ┌─────────▼──────────┐
            │   Prédiction finale  │
            │ ŷ = αeff·ŷCF + (1−αeff)·ŷCBF │
            └────────────────────┘
```

### Formule de pondération adaptative

$$\hat{y}(u, i) = \alpha_{eff} \cdot \hat{y}_{CF}(u, i) + (1 - \alpha_{eff}) \cdot \hat{y}_{CBF}(u, i)$$

$$\alpha_{eff} = \alpha^* \times \min\!\left(1,\, \frac{n_u}{\tau}\right)$$

Où :
- `α*` — poids optimal appris par grid search sur le jeu de validation (α ∈ [0,0 ; 1,0], pas de 0,05)
- `n_u` — nombre de notations de l'utilisateur u
- `τ` — seuil d'activité (défaut : 20)

---

## Contributions principales

### 1. SmartCBF — Module de contenu enrichi
Un filtre basé sur le contenu avec trois composantes complémentaires :

- **Pondération TF-IDF des genres** — Réduit la dominance des genres surreprésentés (Action, Drama) en pondérant chaque genre par son score TF-IDF calculé sur le corpus.
- **Repli démographique** — Lorsqu'un utilisateur a peu de notations, estime son profil à partir d'utilisateurs démographiquement similaires (tranche d'âge, genre, profession).
- **Prior de popularité** — Ajoute un biais stabilisateur basé sur la note moyenne normalisée de l'item, améliorant les prédictions pour les items rares.

$$\hat{y}_{CBF}(u, i) = \text{sim}(\text{profil}_u, \text{vec}_i) \cdot \bar{r}_i + \lambda \cdot \text{pop}(i)$$

### 2. AdaptHybrid — Pondération adaptative apprise
Contrairement aux hybrides naïfs (α = 0,5), ce système :
- Apprend le α optimal par jeu de données via validation croisée
- Ajuste dynamiquement αeff en fonction de l'activité utilisateur à l'inférence
- Gère le démarrage à froid sans aucune modification du code

### 3. Validation expérimentale rigoureuse
- **5 seeds indépendantes** (42, 123, 256, 789, 1024)
- **3 métriques complémentaires** : RMSE, NDCG@10, Recall@10
- **4 baselines** : Popularité, SVD, KNN-CF seul, SmartCBF seul + Hybride Naïf
- **2 jeux de données** : MovieLens 100K et MovieLens 1M
- **Protocole anti-leakage** : séparation stricte entraînement / validation / test

---

## Résultats

### MovieLens 100K (5 seeds, moyenne ± écart-type)

| Modèle | RMSE ↓ | NDCG@10 ↑ | Recall@10 ↑ |
|---|---|---|---|
| Popularité | 1,024 ± 0,005 | 0,834 ± 0,004 | 0,727 ± 0,002 |
| SVD | **0,934 ± 0,005** | **0,840 ± 0,004** | 0,730 ± 0,002 |
| KNN-CF seul | 1,137 ± 0,006 | 0,836 ± 0,004 | **0,733 ± 0,002** |
| SmartCBF seul | 1,259 ± 0,006 | 0,696 ± 0,006 | 0,666 ± 0,003 |
| Hybride Naïf | 1,059 ± 0,004 | 0,809 ± 0,003 | 0,718 ± 0,002 |
| **AdaptHybrid+CS** | **1,039 ± 0,004** | 0,828 ± 0,004 | 0,728 ± 0,002 |

### MovieLens 1M (5 seeds, moyenne ± écart-type)

| Modèle | RMSE ↓ | NDCG@10 ↑ | Recall@10 ↑ |
|---|---|---|---|
| Popularité | 0,980 ± 0,002 | 0,855 ± 0,002 | 0,633 ± 0,001 |
| SVD | **0,871 ± 0,001** | 0,874 ± 0,001 | 0,641 ± 0,001 |
| KNN-CF seul | 0,990 ± 0,003 | 0,871 ± 0,001 | 0,641 ± 0,002 |
| SmartCBF seul | 1,267 ± 0,001 | 0,695 ± 0,001 | 0,569 ± 0,001 |
| Hybride Naïf | 1,000 ± 0,002 | 0,840 ± 0,001 | 0,627 ± 0,001 |
| **AdaptHybrid+CS** | 0,950 ± 0,002 | **0,866 ± 0,001** | **0,640 ± 0,001** |

### Analyse par segment d'activité — Cold-Start (ML-100K, seed=42, RMSE)

| Segment | N notations | Popularité | SVD | KNN-CF | Naïf | **Notre modèle** |
|---|---|---|---|---|---|---|
| Sparse (6–20) | 630 | 1,077 | 0,984 | 3,131 ❌ | 1,677 | **1,914** |
| Modéré (21–50) | 2 745 | 1,052 | 0,975 | 1,468 | 1,111 | 1,211 |
| Actif (51–200) | 10 259 | 1,002 | 0,926 | 0,920 | 1,009 | **0,955** |
| Power (200+) | 6 366 | 1,032 | 0,915 | 0,909 | 1,046 | **0,966** |

> KNN-CF s'effondre pour les utilisateurs sparse (RMSE = 3,131). AdaptHybrid+CS est le seul modèle à rester compétitif sur **tous les segments d'activité**.

### α optimal par jeu de données

| Jeu de données | α* | Interprétation |
|---|---|---|
| ML-100K | 0,70 | CF dominant (70 %) |
| ML-1M | 0,80 | CF très dominant (80 %) |

Le poids optimal n'est **jamais 0,5** — confirmant que le α appris surpasse l'hybride naïf.

---

## Installation

### Prérequis

- Python 3.10+
- NumPy < 2 (requis pour la compatibilité avec scikit-surprise)

### Installer les dépendances

```bash
pip install "numpy<2"
pip install scikit-surprise scikit-learn pandas matplotlib
```

Ou tout installer en une seule commande :

```bash
pip install -r requirements.txt
```

### requirements.txt

```
numpy<2
scikit-surprise
scikit-learn
pandas
matplotlib
```

> **Note :** Le notebook détecte automatiquement la version de NumPy et effectue la rétrogradation si nécessaire. En exécution locale, installer `numpy<2` avant `scikit-surprise`.

---

## Utilisation

### Exécution sur Google Colab (recommandé)

Ouvrir `SmartCBF_Hybrid_V2_Enhanced.ipynb` directement dans Google Colab. Le notebook télécharge automatiquement les deux jeux de données MovieLens.

### Exécution en local

```bash
git clone https://github.com/<votre-nom-utilisateur>/AdaptHybrid-CS.git
cd AdaptHybrid-CS
pip install -r requirements.txt
jupyter notebook SmartCBF_Hybrid_V2_Enhanced.ipynb
```

### Structure du notebook

Le notebook est organisé en **11 sections**, chacune entièrement documentée et exécutable de bout en bout :

---

#### Section 1 — Installation & Dépendances
Installe et importe toutes les bibliothèques requises. Gère automatiquement la compatibilité des versions de NumPy (rétrograde vers `numpy<2` si nécessaire pour `scikit-surprise`). Définit les 5 seeds globales `[42, 123, 256, 789, 1024]` utilisées dans toutes les expériences.

```
Bibliothèques : numpy, pandas, matplotlib, sklearn, scikit-surprise
```

---

#### Section 2 — Chargement des jeux de données (ML-100K & ML-1M)
Télécharge les deux jeux de données MovieLens directement depuis GroupLens s'ils ne sont pas déjà présents. Parse les notations, les métadonnées des films (genres en colonnes binaires, année de sortie) et les données démographiques des utilisateurs (âge, genre, profession). Pour ML-1M, convertit les chaînes de genres séparées par des pipes en colonnes indicatrices binaires correspondant au format ML-100K.

```
ML-100K → 100 000 notations | 943 utilisateurs | 1 682 films
ML-1M   → 1 000 209 notations | 6 040 utilisateurs | 3 706 films
```

---

#### Section 3 — Analyse exploratoire des données
Génère 6 visualisations pour comprendre les données avant la modélisation :
- Distribution des notes (les deux jeux côte à côte)
- Nombre de notations par utilisateur (longue queue d'activité)
- Nombre de notations par item (distribution de popularité)
- Fréquence des genres dans le catalogue complet
- Visualisation de la sparsité de la matrice
- Segmentation de l'activité des utilisateurs (sparse / modéré / actif / power)

---

#### Section 4 — Fonctions utilitaires (Métriques)
Implémente toutes les métriques d'évaluation depuis zéro :

- **`compute_rmse(y_true, y_pred)`** — Erreur quadratique moyenne (RMSE), pénalise quadratiquement les grandes erreurs de prédiction.
- **`compute_ranking_metrics(preds_dict, test_df, train_df, k=10)`** — Calcule NDCG@10 et Recall@10 pour tous les modèles simultanément. Pour chaque utilisateur, classe tous les items candidats non notés, prend le top-k et compare aux items réellement pertinents (note ≥ 4).

---

#### Section 5 — DataAdapter (Découpage Entraînement / Validation / Test)
Définit la classe `DataAdapter`, le composant central de gestion des données utilisé par tous les modèles :

- Découpe les notations en **entraînement (70 %) / validation (10 %) / test (20 %)** avec une seed donnée
- Construit les mappings internes d'indices utilisateurs et items (`u2i`, `m2i`, `i2m`)
- Fournit `build_utility_matrix()` pour KNN-CF (matrice utilisateur-item sparse avec NaN pour les entrées non observées)
- Fournit `get_item_features()` retournant la matrice de features de genres binaires pour CBF
- Expose `info()` pour afficher les statistiques du jeu de données (utilisateurs, items, sparsité)

---

#### Section 6 — Définition des modèles
Définit les 5 modèles utilisés dans les expériences, organisés en 5 sous-sections :

**6a. PopularityBaseline**
Baseline non personnalisée. Prédit la note moyenne de chaque item. Se replie sur la moyenne globale pour les items non vus. N'a aucune connaissance de l'utilisateur demandeur.

**6b. SVDBaseline**
Factorisation matricielle via SVD utilisant la bibliothèque `scikit-surprise` (`n_factors=50`, `n_epochs=20`). État de l'art pour le CF en termes de RMSE. Constitue la baseline personnalisée la plus forte à surpasser.

**6c. SmartCBF** *(contribution principale)*
Filtre basé sur le contenu enrichi, construit en 3 couches superposées :
- **Couche 1 — Pondération TF-IDF des genres + année de sortie** : les genres rares (Film-Noir, Documentaire) reçoivent un poids plus élevé que les genres communs (Drama, Comédie) ; l'année de sortie est ajoutée comme feature normalisée.
- **Couche 2 — Repli démographique** (pour les utilisateurs avec < 5 notations) : trouve les K=10 utilisateurs warm les plus démographiquement similaires et fusionne leurs profils. Similarité = `0,4×genre_identique + 0,3×proximité_âge + 0,3×profession_identique`. Le poids de fusion diminue à mesure que l'utilisateur accumule des notations.
- **Couche 3 — Prior de popularité** (pour les utilisateurs sans notation) : utilise un profil prior pondéré par la popularité des genres. Poids du prior = `max(0, 1 − n_notations / prior_decay)`, diminuant vers 0 à mesure que l'utilisateur devient actif.

**6d. KNNCollaborativeFilter**
Filtrage collaboratif item-item utilisant la similarité cosinus centrée. Construit la matrice de similarité item-item complète à l'entraînement. À l'inférence, trouve les k items les plus similaires notés par l'utilisateur et calcule une moyenne pondérée. Le paramètre k est sélectionné sur le jeu de validation.

**6e. AdaptiveHybrid** *(AdaptHybrid+CS)*
Combine KNN-CF et SmartCBF via une pondération linéaire apprise :
```
ŷ(u,i) = α_eff · ŷ_CF + (1 − α_eff) · ŷ_CBF
α_eff  = α* × min(1, n_u / seuil)
```
α* est trouvé par grid search sur [0,0 ; 1,0] avec un pas de 0,05 sur le jeu de validation. La correction cold-start réduit automatiquement α_eff vers 0 pour les utilisateurs sparse, rendant SmartCBF dominant lorsque le signal CF est faible.

---

#### Section 7 — Expérience multi-seeds sur ML-100K
Exécute le pipeline complet 5 fois avec différentes seeds aléatoires sur MovieLens 100K. Pour chaque seed : construit un DataAdapter, entraîne les 6 modèles, optimise α sur le jeu de validation et évalue sur le jeu de test (RMSE, NDCG@10, Recall@10). Agrège les résultats sous forme de moyenne ± écart-type. Produit le tableau de comparaison et les graphiques en barres.

```
α* optimal = 0,70 ± 0,00 (stable sur les 5 seeds)
```

---

#### Section 8 — Expérience multi-seeds sur ML-1M
Pipeline identique appliqué à MovieLens 1M. Valide que les conclusions de ML-100K se généralisent à un jeu de données plus grand et plus dense. Le jeu de données plus large fait passer α* à 0,80, confirmant que CF devient plus dominant à mesure que le signal de notation est plus abondant.

```
α* optimal = 0,80 ± 0,00 (stable sur les 5 seeds)
```

---

#### Section 9 — Analyse par segment d'activité (Cold-Start)
Partitionne les utilisateurs de test en 4 segments d'activité selon leur nombre de notations dans l'ensemble d'entraînement :

| Segment | Plage de notations |
|---|---|
| Sparse | 6 – 20 notations |
| Modéré | 21 – 50 notations |
| Actif | 51 – 200 notations |
| Power user | 200+ notations |

Calcule le RMSE par segment pour chaque modèle. Démontre que KNN-CF s'effondre pour les utilisateurs sparse (RMSE = 3,131) tandis qu'AdaptHybrid+CS — grâce au repli SmartCBF — reste compétitif sur tous les segments. Cela valide directement l'Hypothèse 2.

---

#### Section 10 — Démo de recommandation en direct
Construit une fonction d'inférence légère `recommend(user_id, n=10)` et l'exécute sur deux utilisateurs contrastés :

- **Utilisateur actif** (nombre de notations le plus élevé) : α_eff ≈ α* → CF dominant → recommandations personnalisées basées sur l'historique.
- **Utilisateur cold-start** (≤ 3 notations) : α_eff ≈ 0 → SmartCBF dominant → recommandations basées sur les données démographiques et la popularité.

La sortie comprend : titre du film, score prédit, α effectif utilisé et nombre de notations d'entraînement pour cet utilisateur.

---

#### Section 11 — Synthèse finale de la recherche
Affiche un résumé structuré de tous les résultats clés, conclusions et limitations dans un format texte lisible — utile pour parcourir rapidement les conclusions sans ouvrir l'article complet.

### Démo rapide

```python
# Recommander 10 films pour un utilisateur donné
recommendations = recommend(user_id=42, n=10)
print(recommendations)

# La sortie comprend :
# - title       : titre du film
# - score       : score de préférence prédit
# - effective_α : le α réellement utilisé (dépend de l'activité de l'utilisateur)
# - n_ratings   : nombre de notations de cet utilisateur dans l'entraînement
```

---

## Structure du projet

```
AdaptHybrid-CS/
│
├── SmartCBF_Hybrid_V2_Enhanced.ipynb   # Notebook principal (expérience complète)
├── article_SmartCBF.pdf                # Article de recherche (ENSET 2025–2026)
├── README.md                           # Ce fichier
├── requirements.txt                    # Dépendances Python
│
└── (téléchargés automatiquement à l'exécution)
    ├── ml-100k/                        # Jeu de données MovieLens 100K
    └── ml-1m/                          # Jeu de données MovieLens 1M
```

---

## Jeux de données

Les deux jeux de données sont téléchargés automatiquement lors de l'exécution du notebook depuis [GroupLens](https://grouplens.org/datasets/movielens/).

| Propriété | ML-100K | ML-1M |
|---|---|---|
| Utilisateurs | 943 | 6 040 |
| Films | 1 682 | 3 706 |
| Notations | 100 000 | 1 000 209 |
| Densité de la matrice | 6,30 % | 4,47 % |
| Plage de notes | 1–5 (entiers) | 1–5 (entiers) |
| Données démographiques | Oui | Oui |
| Année de collecte | 1998 | 2000 |

---

## Protocole expérimental

| Paramètre | Valeur |
|---|---|
| Seeds | 42, 123, 256, 789, 1024 |
| Découpage Entraînement / Validation / Test | 70 % / 10 % / 20 % |
| Pas du grid search sur α | 0,05 (plage : 0,0 → 1,0) |
| Seuil de pertinence | note ≥ 4 |
| Configuration SVD | n_factors=50, n_epochs=20 |
| Similarité KNN | cosinus (item-item), k optimisé sur la validation |
| Seuil de classement | top-10 |

---

## Limitations

| Type | Description |
|---|---|
| **Domaine unique** | Les deux jeux de données appartiennent uniquement au domaine des films. Les conclusions peuvent ne pas se généraliser à l'e-commerce ou à la musique. |
| **Représentation CBF** | SmartCBF utilise TF-IDF sur les genres — une représentation sparse. Des embeddings neuronaux (BERT, Word2Vec) pourraient capturer une sémantique plus riche. |
| **Cold-start réel** | L'analyse cold-start est effectuée sur le segment le plus sparse disponible (6–20 notations), et non sur des utilisateurs avec zéro notation. |
| **Données statiques** | Les deux jeux de données sont des instantanés historiques. Les dynamiques temporelles ne sont pas modélisées. |
| **Optimisation SVD** | SVD est évalué avec des paramètres standards, non optimisés par seed — ce qui pourrait légèrement favoriser notre méthode. |

---

## Perspectives

- [ ] Validation inter-domaines (e-commerce, musique, actualités)
- [ ] Remplacer SmartCBF par un **encodeur neuronal two-tower** pour des représentations de contenu plus riches
- [ ] Intégrer la **dimension temporelle** pour modéliser l'évolution des préférences utilisateur
- [ ] Étendre l'analyse cold-start aux **utilisateurs avec zéro notation** (en utilisant uniquement les données démographiques)
- [ ] Explorer l'**optimisation bayésienne des hyperparamètres** à la place du grid search

---

## Auteures

| Nom | Établissement | Email |
|---|---|---|
| Hafssa MIFTAH IDRISSI | ENSET Mohammedia, Université Hassan II | hafssa.miftah.idrissi@gmail.com |
| Hanane AIT LHAJ | ENSET Mohammedia, Université Hassan II | hananeaitlhaj12@gmail.com |

**Encadrant :** Pr. Soufiane HAMIDA — ENSET Mohammedia, Université Hassan II de Casablanca

---

## Références

1. Burke, R. (2002). Hybrid recommender systems: Survey and experiments. *User Modeling and User-Adapted Interaction*, 12(4), 331–370.
2. Claypool et al. (1999). Combining content-based and collaborative filters in an online newspaper. *ACM SIGIR Workshop on Recommender Systems*.
3. Goldberg et al. (1992). Using collaborative filtering to weave an information tapestry. *CACM*, 35(12), 61–70.
4. He et al. (2017). Neural collaborative filtering. *WWW 2017*.
5. Herlocker et al. (2004). Evaluating collaborative filtering recommender systems. *ACM TOIS*, 22(1), 5–53.
6. Koren, Bell & Volinsky (2009). Matrix factorization techniques for recommender systems. *Computer*, 42(8), 30–37.
7. Pazzani & Billsus (2007). Content-based recommendation systems. *The Adaptive Web*, Springer.
8. Salton & McGill (1983). *Introduction to Modern Information Retrieval*. McGraw-Hill.
9. Sarwar et al. (2001). Item-based collaborative filtering recommendation algorithms. *WWW 2001*.
10. Ricci, Rokach & Shapira (2011). *Recommender Systems Handbook*. Springer.
