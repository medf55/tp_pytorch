# Activité Pratique : Fondements de PyTorch et Deep Learning

Ce dépôt présente le travail pratique portant sur la prise en main de la bibliothèque PyTorch, la manipulation de tenseurs, ainsi que la conception, l'entraînement et l'évaluation de modèles d'apprentissage automatique et d'apprentissage profond.

---

## 1. Objectifs pédagogiques

* Maîtrise des concepts fondamentaux de PyTorch : manipulation de tenseurs, différenciation automatique (`autograd`), boucles d'entraînement canoniques et gestion des modes d'évaluation (`model.eval()`, `torch.no_grad()`).
* Prétraitement rigoureux des données : découpage stratifié, standardisation statistique (`StandardScaler`, `transforms.Normalize`) et chargement efficace par mini-lots avec `DataLoader`.
* Implémentation de plusieurs familles d'architectures :
  * Régression linéaire multiple avec descente de gradient stochastique.
  * Perceptron multicouche (MLP) pour la classification d'images et de données tabulaires.
  * Réseau de neurones convolutif (CNN) intégrant couches de convolution, sous-échantillonnage (*pooling*) et régularisation par *Dropout*.
* Évaluation multi-critères et analyse d'erreurs : matrices de confusion, métriques de précision, rappel, F1-score et inspection visuelle des prédictions erronées.
* Persistance des modèles : sérialisation et chargement des états de paramètres (`state_dict`).

---

## 2. Organisation du projet

```text
lab_pytorch_v1/
├── Exercices_Activite_Pratique_PyTorch.ipynb   # Notebook principal contenant la résolution complète des 4 exercices
├── Activité pratique Bases de Pytorch.ipynb    # Support de cours et énoncé des exercices
├── mnist_mlp.pth                               # Poids entraînés du modèle MLP (Exercice 2)
├── mnist_cnn.pth                               # Poids entraînés du modèle CNN (Exercice 4)
├── pyproject.toml                              # Définition des dépendances et du projet Python
├── uv.lock                                     # Fichier de verrouillage des versions exactes des dépendances
├── .python-version                             # Version de Python ciblée (Python 3.12)
├── .gitignore                                  # Règles d'exclusion des données brutes et caches
└── README.md                                   # Documentation technique et compte-rendu du projet
```

---

## 3. Détail des exercices et méthodologie

### Exercice 1 : Régression linéaire multiple (Prédiction de salaires)
* **Formulation du problème :** Modélisation du salaire mensuel (en Dirhams) à partir de deux variables explicatives : l'ancienneté (en années) et le niveau d'expertise (échelle de 1 à 5).
  $$\text{Salaire} = 4\,000 + 600 \times \text{Ancienneté} + 1\,500 \times \text{Niveau} + \epsilon, \quad \epsilon \sim \mathcal{N}(0, 500^2)$$
* **Méthodologie :**
  * Génération de 1 000 observations synthétiques et découpage 80% entraînement / 20% test.
  * Standardisation des variables d'entrée et de sortie afin de stabiliser la convergence du gradient.
  * Optimisation par descente de gradient stochastique (`optim.SGD`, $\text{lr}=0.01$) avec critère d'erreur quadratique moyenne (`nn.MSELoss`).
* **Résultats obtenus :**
  * Erreur absolue moyenne sur le jeu de test : $\text{MAE} \approx 381{,}6\text{ DH}$, ce qui est en accord direct avec le bruit stochastique injecté ($\sigma = 500\text{ DH}$).
  * Dénormalisation analytique des paramètres : les coefficients estimés ($w_1 \approx 592{,}7$, $w_2 \approx 1\,490{,}7$, $b \approx 4\,090{,}3$) convergent fidèlement vers les paramètres théoriques ($600$, $1\,500$, $4\,000$).
  * Prédiction pour un nouveau profil (7 ans d'ancienneté, niveau 4) : $\approx 14\,202\text{ DH}$ pour une valeur théorique de $14\,200\text{ DH}$.

---

### Exercice 2 : Classification de chiffres manuscrits (MNIST) via un MLP
* **Jeu de données :** Base MNIST composée de 60 000 images d'entraînement et 10 000 images de test (chiffres de 0 à 9 en niveaux de gris, $28 \times 28$ pixels).
* **Architecture :**
  $$\text{Entrée (784)} \longrightarrow \text{Linéaire (128)} \longrightarrow \text{ReLU} \longrightarrow \text{Linéaire (10 logits)}$$
* **Entraînement :** 5 époques, optimiseur `Adam` ($\text{lr} = 0.001$), taille de lot de 64, critère `nn.CrossEntropyLoss`.
* **Résultats obtenus :**
  * Précision globale (*Accuracy*) sur le jeu de test : **97,58 %**.
  * Analyse par classe : rapport complet incluant Précision, Rappel et F1-Score par chiffre.
  * Matrice de confusion et extraction des cas d'échecs (ambiguïtés calligraphiques courantes entre 4 et 9, 7 et 2).
  * Sauvegarde et validation du rechargement des poids dans le fichier `mnist_mlp.pth`.

---

### Exercice 3 : Classification multiclasse sur le jeu de données Iris
* **Jeu de données :** 150 spécimens d'Iris caractérisés par 4 variables biométriques et répartis en 3 espèces (*Setosa*, *Versicolor*, *Virginica*).
* **Particularités techniques :**
  * Découpage stratifié pour conserver la proportion égale des classes (10 individus par classe dans le jeu de test).
  * Justification formelle de l'encodage des labels sous type `torch.long` (indices entiers discrets requis par `CrossEntropyLoss`).
* **Comparaison architecturale :**
  * **Modèle linéaire direct (`nn.Linear(4, 3)`) :** Exactitude de **90,00 %** (limité par la séparabilité non linéaire entre *Versicolor* et *Virginica*).
  * **Réseau avec couche cachée (`Linear(4, 16) -> ReLU -> Linear(16, 3)`) :** Exactitude de **100,00 %** sur l'ensemble de test.
* **Inférence :** Prédiction sur un nouvel échantillon $[5.1, 3.5, 1.4, 0.2]$ classé sans ambiguïté comme *Iris-setosa* ($p > 99{,}9\,\%$ via Softmax).

---

### Exercice 4 : Classification MNIST via Réseau Convolutif (CNN)
* **Architecture du réseau :**
  1. Couche convolutionnelle 1 : 1 canal entrant, 16 filtres de taille $3 \times 3$, padding 0, stride 1 $\to$ activation ReLU.
  2. Couche de sous-échantillonnage : Max-Pooling $2 \times 2$, stride 2.
  3. Couche convolutionnelle 2 : 16 canaux entrants, 32 filtres de taille $3 \times 3$, padding 0, stride 1 $\to$ activation ReLU.
  4. Couche de sous-échantillonnage : Max-Pooling $2 \times 2$, stride 2.
  5. Couche dense de classification : 800 neurones entrants $\to$ 128 neurones avec activation ReLU.
  6. Régularisation par *Dropout* ($p = 0.5$).
  7. Couche de sortie : 128 neurones $\to$ 10 logits.
* **Démonstration des dimensions spatiales :**
  $$28 \times 28 \xrightarrow{\text{Conv1}} 26 \times 26 \xrightarrow{\text{Pool1}} 13 \times 13 \xrightarrow{\text{Conv2}} 11 \times 11 \xrightarrow{\text{Pool2}} 5 \times 5$$
  Le vecteur aplati en entrée de la couche dense a donc une taille de $32 \times 5 \times 5 = 800$ caractéristiques (ou $32 \times 7 \times 7 = 1\,568$ si un padding de 1 est utilisé).
* **Résultats obtenus :**
  * Précision sur le jeu de test : **98,92 %** (dépassement net du seuil attendu de 98 %).
  * Visualisation comparative de 10 prédictions erronées mettant en évidence des chiffres manuscrits fortement déformés.
  * Sauvegarde des poids dans `mnist_cnn.pth` et vérification de l'inférence après rechargement.

---

## 4. Tableau récapitulatif des résultats

| Exercice | Modèle | Type de tâche | Critère d'optimisation | Performance obtenue |
| :--- | :--- | :--- | :--- | :--- |
| **Exercice 1** | Régression linéaire multiple | Régression | Erreur quadratique moyenne (MSE) | $\text{MAE} \approx 381{,}6\text{ DH}$ |
| **Exercice 2** | MLP (784-128-10) | Classification (10 classes) | Entropie croisée (*Cross-Entropy*) | $\text{Accuracy} = 97{,}58\,\%$ |
| **Exercice 3** | MLP (4-16-3) | Classification (3 classes) | Entropie croisée (*Cross-Entropy*) | $\text{Accuracy} = 100{,}00\,\%$ |
| **Exercice 4** | CNN (2 Conv + MaxPool + Dropout) | Classification (10 classes) | Entropie croisée (*Cross-Entropy*) | $\text{Accuracy} = \mathbf{98{,}92\,\%}$ |

---

## 5. Environnement d'exécution et reproductibilité

Le projet est configuré avec le gestionnaire d'environnement Python `uv`.

### Prérequis
* Python 3.12 ou supérieur
* Outil `uv` installé sur le système

### Installation et exécution

1. Synchroniser l'environnement virtuel et installer l'ensemble des bibliothèques (`torch`, `torchvision`, `scikit-learn`, `matplotlib`, `numpy`) :
   ```bash
   uv sync
   ```

2. Lancer l'environnement Jupyter pour visualiser ou réexécuter les cellules :
   ```bash
   uv run jupyter lab
   ```
   ou
   ```bash
   uv run jupyter notebook
   ```

3. Exécution sélective d'un script ou vérification de l'installation :
   ```bash
   uv run python -c "import torch; print('PyTorch version:', torch.__version__)"
   ```
