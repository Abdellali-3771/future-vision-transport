## Rapport d'Évaluation - Modèle de Segmentation Sémantique

## 🏷️ Identification de l'Expérience
- **Nom de l'expérience**: exp_003_finetune
- **Description**: Fine-tuning MobileNetV2 avec encoder dÃ©gelÃ© (exp_003)
- **Date d'évaluation**: 2025-10-25 19:22:38

## ⚙️ Configuration du Modèle
- **Architecture**: MobileNetV2-UNet
- **Backbone**: MobileNetV2 (ImageNet pré-entraîné)
- **Encoder trainable**: True
- **Classes de segmentation**: 8
- **Groupes de classes**: N/A

## 📋 Paramètres d'Entraînement
- **Taille d'image**: (224, 224)
- **Batch size**: 8
- **Époques d'entraînement**: 5
- **Taux d'apprentissage**: 1e-05
- **Division validation**: 0.2
- **Augmentation de données**: False

## 📊 Performances Finales sur Dataset de Test
- **Perte finale**: 0.39918768405914307
- **Précision finale**: N/A
- **MeanIoU finale**: N/A

## 🎯 Dataset et Évaluation
- **Dataset**: Cityscapes (segmentation urbaine)
- **Split utilisé pour test**: Dataset 'val' original (évite data leakage)
- **Nombre d'images de test**: ~500 images
- **Méthode d'évaluation**: IoU par classe, précision globale, matrice de confusion

## 📈 Mapping des Classes
Les classes originales de Cityscapes ont été regroupées en catégories pertinentes pour la navigation autonome:

1. **N/A** (0)

## 🚗 Applications et Déploiement
- **Cas d'usage**: Système de vision pour véhicules autonomes
- **Format de modèle**: Keras (.keras) - compatible TensorFlow Lite
- **Optimisations**: MobileNetV2 conçu pour l'efficacité mobile
- **Intégration**: API FastAPI + Frontend Next.js (en développement)

## 📂 Artefacts Générés
- **Modèle entraîné**: `experiments/exp_003_finetune/models/cityscapes_segmentation_model.keras`
- **Configuration**: `experiments/exp_003_finetune/results/experiment_config.json`
- **Historique d'entraînement**: `experiments/exp_003_finetune/results/training_history.json`
- **Métriques détaillées**: `experiments/exp_003_finetune/results/`
- **Visualisations**: Matrice de confusion, exemples de prédictions

## 🔄 Suivi et Reproductibilité
- **Structure d'expériences**: Organisation modulaire par dossiers
- **Intégration MLflow**: Suivi des métriques et artefacts
- **Graine aléatoire**: 42 (reproductibilité garantie)
- **Environnement**: TensorFlow 2.18.0, Keras 3.8.0

## 📝 Notes Techniques
- Le modèle utilise `SparseCategoricalCrossentropy` comme fonction de perte.
- L'encoder MobileNetV2 peut être gelé ou entraînable selon la configuration.
- La segmentation est réalisée à 224x224 puis peut être redimensionnée.
- L'augmentation de données inclut flip horizontal et variation de luminosité.

## 🎯 Recommandations
1. **Améliorer les performances** : Entraîner avec encoder dégelé ou plus d'epochs.
2. **Pour le déploiement** : Conversion en TensorFlow Lite pour usage mobile embarqué.
3. **Pour l'évaluation** : Comparer Mean IoU entre expériences pour analyse.
4. **Pour la production** : Étendre l'entraînement à d'autres environnements urbains.

---
*Rapport généré automatiquement le 2025-10-25 19:30:04*
