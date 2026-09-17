# Baba Rug Visual Studio

Moteur de production d'images lifestyle pour tapis d'Orient, kilims et tapis faits main.

**N photos d'un tapis réel → 2 photographies d'intérieur photoréalistes montrant *ce*
tapis.** Conçu pour le traitement en masse après un shooting.

## Principe

Le tapis n'est **jamais** redessiné par un modèle génératif. Il est détouré, redressé, puis
**incrusté géométriquement** dans un décor généré vide. Les pixels du tapis dans l'image
finale sont les pixels de la photo d'origine.

> Une belle image montrant un autre tapis est un échec.
> La fidélité produit est ici une propriété de construction, pas une espérance statistique.

Voir [`ARCHITECTURE.md`](ARCHITECTURE.md) pour le raisonnement complet et les limites connues.

## Installation

```bash
pip install -r requirements.txt
python -m babarug.cli doctor        # vérifie l'installation et les clés
```

## Utilisation

```bash
# Un tapis. --size est fortement recommandé : c'est l'info la plus rentable.
python -m babarug.cli run BABA-RUG-0001 photos/BABA-RUG-0001/ --size 200x150

# Tout un shooting : un sous-dossier par tapis
python -m babarug.cli batch shooting_2026_09/

# Analyse seule (RUG DNA), sans générer d'image
python -m babarug.cli dna BABA-RUG-0001 photos/BABA-RUG-0001/ --size 200x150

# Sans aucune clé API : la chaîne tourne, le décor est schématique
python -m babarug.cli run BABA-RUG-0001 photos/ --generation offline --no-review
```

### Options utiles

| Option | Effet |
|---|---|
| `--size 200x150` | dimensions réelles en cm — **fiabilise les proportions** |
| `--style parisien_contemporain --room salon` | impose les scènes (sinon « surprends-moi ») |
| `--generation gemini\|flux\|offline` | choix du générateur de décor |
| `--max-kb 400` | budget de poids du JPEG exporté |
| `--filename "quelle-taille-tapis-{room}-baba-rug"` | nommage SEO |
| `--no-review` | désactive l'avis de scène (moins cher, moins sûr) |

## Configuration

```bash
export ANTHROPIC_API_KEY=...            # RUG DNA + contrôle qualité
export GEMINI_API_KEY=...               # génération de décor
export BFL_API_KEY=...                  # alternative FLUX.1 Kontext
export BABARUG_MATTING_ENDPOINT=...     # détourage hébergé (SAM 3 / BiRefNet)
```

## Sortie

```
data/BABA-RUG-0001/
├── rug_dna.json      identité du tapis
├── plate.png         tapis redressé, RGBA, franges comprises
├── lifestyle_A.png   rendu pleine résolution
├── lifestyle_B.png
├── export/*.jpg      1600×900, prêts pour Shopify
└── report.json       verdicts, mesures, coûts
```

## Contrôle qualité

Deux niveaux **strictement séparés** :

1. **Fidélité produit** — mesurée sans IA. Le tapis est ré-extrait de l'image finale par
   homographie inverse et comparé à la plate : ΔE2000, SSIM, appariement ORB, histogramme,
   rétention de texture, silhouette, occultation, proportions. **Seul ce niveau peut rejeter.**
2. **Réalisme de scène** — avis d'un modèle de vision. Consultatif : il peut déclencher une
   correction, il ne peut **jamais** repêcher une image ayant échoué au niveau 1.

Verdicts : `APPROVED` · `REVIEW_REQUIRED` · `REJECTED`.

## Tests

```bash
python -m pytest tests/ -q     # 32 tests, sans clé API ni coût
```

Dont dix tests adversariaux vérifiant que le contrôle qualité **rejette** : un autre tapis,
un motif central modifié, des couleurs embellies, une texture lissée, des franges
supprimées, des proportions déformées, un tapis hors cadre — et qu'un avis de scène
élogieux ne peut pas repêcher un échec de fidélité mesuré.

## Avant un shooting — protocole recommandé

Ces trois points valent plus que n'importe quelle amélioration de modèle :

1. **Une photo à plat, frontale, du tapis entier**, franges comprises, sans perspective.
   C'est elle qui fournit les pixels du rendu final. Tout le reste est secondaire.
2. **Un fond franchement contrastant** (gris moyen ou sombre). Un tapis à champ ivoire sur
   un mur crème rend le détourage des franges impossible — constaté sur BABA-RUG-0001.
3. **Les dimensions au mètre ruban.** Elles suppriment toute incertitude sur les proportions.
