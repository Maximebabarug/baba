# Photos sources

Un sous-dossier par tapis, nomme avec son identifiant produit.

```
photos/
├── BABA-RUG-0001/
│   ├── 01_face.jpg          <- LA PLUS IMPORTANTE : tapis entier, a plat, frontal
│   ├── 02_angle.jpg
│   ├── 03_champ.jpg
│   ├── 04_medaillon.jpg
│   ├── 05_bordure.jpg
│   ├── 06_champ_bordure_oblique.jpg
│   ├── 07_pli_lisiere.jpg
│   └── 08_tranche_lisiere.jpg
└── BABA-RUG-0002/
    └── ...
```

Formats acceptes : `.jpg` `.jpeg` `.png` `.webp`. L'ordre alphabetique n'a pas
d'importance : c'est l'analyse qui choisit la photo servant de plate.

Puis :

```bash
python -m babarug.cli run BABA-RUG-0001 photos/BABA-RUG-0001/ --size 200x150
python -m babarug.cli batch photos/          # tout le shooting
```

## La photo qui decide de tout

`01_face.jpg` fournit les pixels reels du rendu final. Les sept autres servent a
l'analyse. Pour celle-la :

- tapis **entier**, **franges comprises**, aucun bord coupe ;
- **a plat** (au sol de preference), appareil **au-dessus du centre**, pas en
  contre-plongee ;
- **fond franchement contrastant** : gris moyen ou sombre. Un tapis a champ
  ivoire sur un mur creme rend le detourage des franges impossible ;
- lumiere homogene, pas d'ombre portee en travers du tapis.

## Et les dimensions

Mesurez le tapis au metre ruban et passez `--size <longueur>x<largeur>` en cm.
C'est l'information qui apporte le plus de precision pour le moins d'effort :
sans elle, les proportions dans la scene restent une estimation.
