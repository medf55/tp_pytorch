# TP : Bases de PyTorch

Ce projet contient la résolution complète et détaillée des 4 exercices de l'**Activité Pratique sur les bases de PyTorch**.

## 📁 Contenu du dépôt

* **`Exercices_Activite_Pratique_PyTorch.ipynb`** : Le notebook principal contenant les 4 exercices entièrement résolus avec code, visualisations (courbes de perte, matrices de confusion, chiffres manuscrits) et explications théoriques.
* **`Activité pratique Bases de Pytorch.ipynb`** : Le support original de cours et de TP.
* **`mnist_mlp.pth`** : Poids sauvegardés du modèle MLP (Exercice 2).
* **`mnist_cnn.pth`** : Poids sauvegardés du modèle CNN (Exercice 4).
* **`pyproject.toml` / `uv.lock`** : Gestion des dépendances de l'environnement Python.

---

## 🚀 Les 4 Exercices résolus

1. **Exercice 1 : Régression linéaire multiple (Salaires)**
   * Génération de données synthétiques (ancienneté, niveau d'expertise, bruit gaussien).
   * Modèle linéaire `nn.Linear(2, 1)`, perte MSE, optimiseur SGD.
   * Dénormalisation analytique des poids et prédiction sur de nouveaux profils.
2. **Exercice 2 : Classification de chiffres manuscrits MNIST avec un MLP**
   * Architecture dense `Flatten -> Linear(784, 128) -> ReLU -> Linear(128, 10)`.
   * Entraînement sur mini-batches (`DataLoader`), rapport complet (Précision, Rappel, F1-Score, matrice de confusion).
   * Visualisation d'images mal classées et persistance du modèle (`state_dict`).
3. **Exercice 3 : Classification des fleurs Iris**
   * Réseau `Linear(4, 16) -> ReLU -> Linear(16, 3)` avec `CrossEntropyLoss`.
   * Stratification 80/20, normalisation `StandardScaler`, comparaison avec un modèle linéaire sans couche cachée.
   * Prédiction d'une nouvelle observation avec probabilités Softmax.
4. **Exercice 4 : Classification MNIST avec un CNN**
   * Architecture convolutive `Conv2d -> MaxPool -> Conv2d -> MaxPool -> Linear + Dropout`.
   * Démonstration mathématique des dimensions spatiales des tenseurs d'activation.
   * Évaluation (Accuracy > 98%), analyse des confusions et persistance des poids.

---

## 🛠️ Installation et Exécution

Ce projet utilise [uv](https://github.com/astral-sh/uv) pour la gestion rapide et reproductible de l'environnement :

```bash
# Installer les dépendances
uv sync

# Lancer Jupyter Lab ou Notebook
uv run jupyter lab
```
