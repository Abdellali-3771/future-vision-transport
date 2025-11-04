
# Rapport d'Évaluation - Modèle de Segmentation Sémantique

## 🏷️ Identification de l'Expérience
- **Nom de l'expérience**: exp_002_no_augmentation
- **Description**: MobileNetV2-UNet avec frozen encoder, sans data augmentation
- **Date d'évaluation**: 2025-10-25 18:18:58

## ⚙️ Configuration du Modèle
- **Architecture**: MobileNetV2-UNet
- **Backbone**: MobileNetV2 (ImageNet pré-entraîné)
- **Encoder trainable**: False
- **Classes de segmentation**: 8 groupes
- **Groupes de classes**: flat, human, vehicle, construction, object, nature, sky, void

## 📋 Paramètres d'Entraînement
- **Taille d'image**: (224, 224)
- **Batch size**: 8
- **Époques d'entraînement**: 19
- **Taux d'apprentissage**: 0.0001
- **Division validation**: 0.2
- **Augmentation de données**: False

## 📊 Performances Finales sur Dataset de Test
- **Perte finale**: 0.3947
- **Précision finale**: 0.8789
- **MeanIoU finale**: 0.6316

## 🎯 Dataset et Évaluation
- **Dataset**: Cityscapes (segmentation urbaine)
- **Split utilisé pour test**: Dataset 'val' original
- **Nombre d'images de test**: ~500 images
- **Méthode d'évaluation**: IoU par classe, précision globale, matrice de confusion

## 📈 Mapping des Classes
1. **flat** (0)
2. **human** (1)
3. **vehicle** (2)
4. **construction** (3)
5. **object** (4)
6. **nature** (5)
7. **sky** (6)
8. **void** (7)

## 🚗 Applications et Déploiement
- **Cas d'usage**: Système de vision pour véhicules autonomes
- **Format du modèle**: `.keras` (TensorFlow)
- **Optimisation mobile**: MobileNetV2 pour efficacité embarquée
- **Intégration prévue**: API FastAPI + Frontend Next.js

## 🔄 Suivi et Reproductibilité
- **Graine aléatoire**: 42
- **Environnement**: TensorFlow 2.20.0, Keras 3.11.3

---
*Rapport généré automatiquement le 2025-10-25 à 18:18:58*
