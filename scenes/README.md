# Bibliothèque de décors

Des photographies de **pièces vides**, calibrées une fois, réutilisées indéfiniment.
Le pipeline y incruste les tapis. **Coût par image : zéro.**

## Pourquoi c'est mieux que générer le décor

| | Générateur d'images | Bibliothèque de décors |
|---|---|---|
| Coût par image | ~0,13 $ | **0 $** |
| Photoréalisme | à espérer | **c'est une vraie photo** |
| Plan du sol | deviné par un modèle | **calibré une fois, exact** |
| Cohérence de marque | aléatoire | **la vôtre** |
| Variété | infinie | limitée à votre bibliothèque |

Les deux premiers risques du projet — « est-ce que ça passera pour une photo ? » et
« le modèle situe-t-il bien le sol ? » — disparaissent purement et simplement.

## Arborescence

```
scenes/
├── salon_parisien_01.jpg      la photo
├── salon_parisien_01.json     sa fiche (plan du sol + métadonnées)
├── salle_a_manger_sud_02.jpg
└── salle_a_manger_sud_02.json
```

Même nom de base pour la photo et sa fiche.

## Ajouter un décor

**1. Photographier la pièce VIDE.** Aucun tapis au sol, et la zone où le tapis ira
doit être entièrement dégagée : pas de table basse, pas de pied de fauteuil qui
mordrait dessus. Le pipeline incruste le tapis *par-dessus* — il ne sait pas
faire passer un pied de meuble devant.

**2. Calibrer le plan du sol.** Ouvrez `tools/calibrer-decor.html` dans un
navigateur (double-clic, aucun serveur ni installation), déposez la photo,
cliquez les 4 coins de la zone de sol, renseignez ses dimensions réelles, puis
téléchargez le `.json`.

**3. Déposer les deux fichiers ici** et vérifier :

```bash
python -m babarug.cli scenes
```

## Ce qui fait un bon décor

- **Définition** : 2400 px minimum sur le grand côté. En-dessous, le tapis
  incrusté sera plus net que la pièce et l'œil le verra.
- **Sol dégagé et généreux** : plus la zone libre est grande, plus le moteur a de
  latitude pour poser des tapis de tailles différentes.
- **Lumière naturelle, franche et directionnelle.** Le relighting reprend le
  dégradé lumineux du sol pour l'appliquer au tapis ; un sol uniformément éclairé
  donne un tapis plat.
- **Angle et hauteur** : caméra entre 1,2 et 1,7 m, légère plongée. Un angle trop
  rasant écrase le tapis et le rend illisible.
- **Sobriété.** Un décor chargé vole la vedette au produit.

## Combien en faut-il

| Nombre | Ce que ça donne |
|---|---|
| 4–6 | utilisable, mais les tapis se répètent vite |
| **15–25** | **bon équilibre pour un catalogue** |
| 40+ | confortable pour plusieurs milliers de tapis |

Deux images d'un même tapis ne tombent jamais dans le même décor. La variété vient
aussi du recadrage, de l'étalonnage lumineux et de la position du tapis au sol :
un même décor ne redonne pas deux images identiques.

Prévoyez plusieurs styles (`parisien_contemporain`, `mediterraneen`, `japandi`…)
et plusieurs pièces (`salon`, `salle_a_manger`, `chambre`) : le moteur choisit
automatiquement deux combinaisons franchement différentes par tapis.

## Où trouver les photos

- **Les vôtres.** Une journée de shooting dans quelques intérieurs vous donne une
  bibliothèque exclusive, cohérente avec la marque, et réglée définitivement.
- **Banques d'images.** Vérifiez la licence : l'usage commercial et la modification
  doivent être autorisés. Attention aux visages et aux œuvres d'art dans le champ.

## Utilisation

```bash
python -m babarug.cli run BABA-RUG-0001 photos/BABA-RUG-0001/ \
    --size 170x133 --generation library
```
