# Projet de Master : Amélioration de la Calibration des Réseaux Neuronaux Profonds

## Contexte
Ce projet a été réalisé dans le cadre de notre programme de Master 1 en *Statistique et Science des Données* à Montpellier. L'objectif principal était de contribuer à l'amélioration de la calibration des réseaux neuronaux profonds afin d'obtenir une évaluation plus précise de l'incertitude dans le contexte de l'apprentissage profond.

## Méthode Proposée : Cohérence des Attentes (EC)
Le projet présente une nouvelle méthode de calibrage appelée *cohérence des attentes* (EC). Cette approche implique un redimensionnement des poids de la dernière couche après l'entraînement. L'objectif de cette manipulation est d'aligner la confiance moyenne de validation avec la proportion moyenne d'étiquettes correctes.

## Objectif
L'objectif principal de ce travail est d'empiriquement démontrer que la méthode EC offre des performances de calibrage comparables à celles de la mise à l'échelle de la température (TS) pour diverses architectures de réseaux neuronaux. De plus, cette démonstration devrait montrer que la méthode EC atteint ces performances tout en nécessitant des ressources similaires en termes d'échantillons de validation et de puissance informatique.

## Avantages de la Méthode EC
- Performances de calibrage comparables à la mise à l'échelle de la température (TS).
- Nécessite des ressources similaires en termes d'échantillons de validation et de puissance informatique.

Le code pour obtenir les résulats qui se trouve dans les tableaux du rapport est dans le fichier `comparison2.py` et le code pour obtenir les résulats et les diagrammes de fiabilités se trouve dans `diagrams.py` .

Les référentiels suivants ont été partiellement utilisés :
  - https://github.com/SPOC-group/expectation-consistency
  - https://github.com/chenyaofo/pytorch-cifar-models
  - https://github.com/hollance/reliability-diagrams
