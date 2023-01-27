## Application de prédictions de maladie de la rétine 

1) Télécharger le depo git 

2) Crée un environnement python (3.9 de préference)

3) L'arborescence doit ressembler à ça : (sans le model, car il est trop volumineux pour github)

![arbo](https://user-images.githubusercontent.com/95628428/215041934-aa0d5338-d68b-4fa3-8cfe-3b227ac98116.PNG)

4) Renommer votre model Best-retinal.h5

5) Dans le terminal installer les bibliothèques avec pip install pip install -r requirements.txt (À exécuter dans le répertoire où se trouve le main.py)


## Fonctionnement de l'application.

1) Cliquer sur le bouton central.

![screen1](https://user-images.githubusercontent.com/95628428/215043141-09aa0207-2ec3-41a2-a5a6-5398679a3c21.PNG)

2) Nous voila sur l'écran principal, ici, on a deux choix.

La prédiction sur une seul image avec une fonctionnalité de rognage ou la prédiction sur plusieurs image (limité à 4 car il y a 4 classes)

![screen2](https://user-images.githubusercontent.com/95628428/215043500-664cd4b5-4135-471c-8c18-708f6a77bb2f.PNG)

3) Prédictions sur plusieurs images.

Pour utiliser la prédiction sur plusieurs images il suffit de sélectionner quelque photo et d'appuyer le bouton Multi prédiction.

![screen3](https://user-images.githubusercontent.com/95628428/215045898-2c3e0d82-ceab-4f24-aca1-2cc580189aef.PNG)

3) Prédictions une image + rognage.

Pour utiliser la prédiction, il suffit juste de sélectionner une image et d'appuyer sur le bouton prédictions sur image.

Il est aussi possible d'effectuer un rognage comme sur l'image ci-dessus, avec le rognage, on peut être plus précis sur la zone de recherche de la maladie.

![screen4](https://user-images.githubusercontent.com/95628428/215046509-e2725aed-6a7e-44de-b4e7-1c5ee985d15a.PNG)


