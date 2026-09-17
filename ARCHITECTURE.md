# Baba Rug Visual Studio — architecture technique

> Document d'ingénierie. Il dit aussi ce qui **ne marche pas** et ce qui reste difficile.
> Objectif produit : après un shooting, déposer N dossiers de photos et obtenir N × 2
> images lifestyle exploitables commercialement, avec un minimum d'intervention humaine.

---

## PHASE 1 — Audit : où est réellement le problème

Le cahier des charges pose une priorité non négociable : **fidélité produit > beauté**.
Cette seule contrainte détermine toute l'architecture, et elle élimine d'emblée l'approche
que 90 % des projets comparables adoptent.

### Ce qu'un modèle génératif sait faire — et ce qu'il ne sait pas faire

Les modèles d'image actuels (Gemini 3 Pro Image / « Nano Banana Pro », FLUX.1 Kontext,
GPT-Image) préservent remarquablement bien une **identité sémantique** : un visage, un
personnage, une silhouette de produit, un logo. C'est ce que mesurent leurs benchmarks.

Un tapis d'Orient n'est pas une identité sémantique. C'est un **champ de haute fréquence
dense, non répétitif à l'échelle locale, et porteur d'asymétries signifiantes**. Le modèle
n'a aucune représentation interne de « ce Chirvan-ci » ; il a une représentation de
« un tapis caucasien à champ ivoire ». Quand il redessine la zone, il produit un tapis
plausible du même genre. Sur BABA-RUG-0001, cela veut dire :

- le moulinet quadricolore des médaillons médians devient un moulinet **quelconque** ;
- la bordure koufique perd son rythme exact ;
- la dissymétrie haut/bas du champ — qui est une signature de tissage — est **corrigée** ;
- le crénelage des diagonales, visible au noeud, est remplacé par des tracés lisses.

Chacun de ces points est, selon vos propres critères, un **échec commercial**. Le client
recevrait un autre tapis que celui de la photo.

**Conclusion d'audit : la fidélité ne peut pas être obtenue en demandant au générateur
d'être fidèle. Elle doit être une propriété de construction.**

### Le renversement d'architecture

Au lieu de :

```
photo du tapis + prompt  →  [modèle génératif]  →  image contenant un tapis ressemblant
```

on fait :

```
photo du tapis  →  détourage → redressement → PLATE (pixels réels)
prompt de décor →  [modèle génératif] → PIÈCE VIDE (le tapis n'est jamais demandé)
                            ↓
              incrustation géométrique de la PLATE dans le plan du sol
                            ↓
              relighting + ombre de contact + cohérence optique
```

Les pixels du tapis dans l'image finale **sont** les pixels de la photo, transformés par
une homographie. La fidélité n'est plus une espérance statistique : elle est structurelle.

Le risque ne disparaît pas, il **change de nature** — et c'est un bien meilleur risque :

| Approche | Risque | Détectable ? | Réparable ? |
|---|---|---|---|
| Génération | « ce n'est pas le bon tapis » | difficilement | non, il faut relancer au hasard |
| Compositing | « le tapis a l'air collé » | oui, à l'oeil et en mesure | oui, paramètres déterministes |

Un tapis un peu trop « posé » se corrige. Un tapis inventé ne se corrige pas.

### Ce que cela permet en plus : un contrôle qualité qui n'est pas une opinion

Parce que l'homographie est connue, on peut **ré-extraire** le tapis de l'image finale et
le remettre à plat. On compare alors pixel à pixel avec la plate d'origine. Le contrôle
qualité devient une **mesure reproductible et opposable**, pas un score de modèle.

C'est la réponse directe à « ne considère jamais un score d'IA comme une preuve absolue ».

---

## PHASE 2 — Architecture

### Chaîne de traitement

```
1. IMPORT          N photos d'un même tapis (drag & drop ou dossier)
2. RUG DNA         un seul appel vision, les N photos ensemble       → rug_dna.json
3. PLATE           détourage → redressement métrique                 → plate.png (RGBA)
4. SCÈNE           génération d'une pièce VIDE + lecture du sol      → scene + FloorPlane
5. COMPOSITING     homographie + relighting + ombre + grain          → lifestyle_X.png
6. CONTRÔLE        mesures déterministes + avis de scène consultatif → QCReport
7. CORRECTION      boucle ciblée, bornée                             → retour en 4 ou 5
8. EXPORT          16:9 1600×900, JPEG sous budget, nom SEO          → export/*.jpg
```

### Modules

| Fichier | Rôle | Dépend d'une API ? |
|---|---|---|
| `babarug/models.py` | contrats : `RugDNA`, `SceneBrief`, `FidelityMetrics`, `QCReport` | non |
| `babarug/geometry.py` | `FloorPlane`, homographies, placement métrique | **non** |
| `babarug/pipeline/plate.py` | détourage → orthophoto | non |
| `babarug/pipeline/composite.py` | incrustation, relighting, ombre | **non** |
| `babarug/pipeline/qc.py` | mesures de fidélité | **non** |
| `babarug/pipeline/scene.py` | pièce vide + plan du sol | oui |
| `babarug/pipeline/loop.py` | boucle générer → analyser → corriger | — |
| `babarug/pipeline/export.py` | 16:9, JPEG, nommage Shopify | non |
| `babarug/styles.py` | catalogue de styles, variation des scènes | non |
| `babarug/providers/*` | adaptateurs de fournisseurs | oui |
| `babarug/run.py` | chaîne complète, un tapis et par lot | — |

**Le coeur de la fidélité (géométrie, compositing, QC) ne dépend d'aucune API.** Il est
déterministe, testable hors ligne, et survit au changement de n'importe quel fournisseur.

### Providers abstraits

```python
VisionProvider           analyze_rug() · locate_floor()
ImageGenerationProvider  generate_scene()
ImageEditingProvider     edit()                 # optionnel, jamais sur le tapis
SegmentationProvider     segment_rug()
QualityControlProvider   review_scene()         # consultatif
```

Sélection par nom dans `providers/registry.py`, pilotée par variables d'environnement.
Changer de générateur = une variable. Aucun SDK n'apparaît dans le pipeline.

---

## PHASE 3 — Choix des modèles

Critères : coût, vitesse, stabilité d'API, automatisation, traitement en volume — pas
seulement la qualité théorique.

### Vision / RUG DNA → **Claude Opus 5**

Les N photos partent dans **un seul appel**. C'est ce qui permet de répondre à « quelles
caractéristiques sont communes aux N photos ? » au lieu de produire N descriptions à
réconcilier ensuite. Sortie contrainte par JSON Schema (`output_config.format`), donc
directement exploitable. Coût mesuré : **~0,08–0,15 $ par tapis** pour 8 photos.

### Segmentation → **SAM 3 ou BiRefNet / RMBG-2.0**, avec repli local

| Option | Pour | Contre |
|---|---|---|
| SAM 3 (hébergé) | prompt textuel, ~30 ms, robuste | dépendance externe, ~0,003 $/image |
| BiRefNet / RMBG-2.0 | excellent en haute résolution, auto-hébergeable | pas de prompt, GPU à gérer |
| GrabCut local (fourni) | gratuit, hors ligne, déterministe | échoue sur fond peu contrasté |

**Constat réel sur BABA-RUG-0001** : le tapis a un champ ivoire, il est photographié sur
un mur de brique crème, et **les franges sont crème sur crème**. Aucun modèle ne réglera
ça de façon fiable — l'information n'est pas dans l'image. Le module
`segmentation.mask_quality()` mesure ce contraste et refuse de continuer en silence.
La vraie solution est un **fond de prise de vue contrastant** (voir PHASE 5).

### Génération de décor → **Gemini 3 Pro Image** en principal, **FLUX.1 Kontext** en secours

| Modèle | Prix/image | Vitesse | Remarque |
|---|---|---|---|
| Gemini 3 Pro Image | ~0,134 $ | 2–5 s | intérieurs crédibles, GA juin 2026 |
| FLUX.1 Kontext | ~0,04 $ | ~7 s | moins cher, très stable en édition guidée |

Ce choix est **volontairement peu engageant** : le générateur ne dessine que des murs, du
parquet et des meubles. S'il produit une pièce médiocre, on régénère avec une autre graine
pour 0,13 $. Aucune conséquence sur le produit.

### Image-to-image / inpainting → **écarté par défaut**

Techniquement disponible, mais toute passe générative qui touche la zone du tapis
ré-hallucine le motif. Le code prévoit l'`ImageEditingProvider` avec deux garde-fous
stricts : le masque doit **exclure** le tapis, et le QC compare avant/après — si le produit
a bougé, la passe est jetée. À n'activer qu'après mesure sur des tapis réels.

### Contrôle qualité → **mesure déterministe d'abord, Claude ensuite**

| Niveau | Outil | Pouvoir |
|---|---|---|
| Fidélité produit | OpenCV + scikit-image, **sans IA** | peut **REJETER** |
| Réalisme de scène | Claude Opus 5 | peut dégrader, **jamais repêcher** |

---

## PHASE 4 — MVP

Périmètre : 5 à 10 tapis réels, 8 photos chacun, jusqu'à l'export.

**Livré et fonctionnel :**

- RUG DNA multi-photos, schéma contraint, séparation produit / artefacts photographiques
- détourage + contrôle automatique de la qualité du masque
- redressement métrique (placement à la taille réelle en cm)
- génération de décor + lecture du plan du sol, avec contrôles de plausibilité géométrique
- compositing : relighting luminance seule, ombre de contact, accord de grain et de netteté
- 8 métriques de fidélité déterministes
- boucle de correction ciblée et bornée
- export 16:9 1600×900 JPEG sous budget de poids, nommage configurable
- CLI unitaire **et par lot**, mode hors ligne sans clé ni coût
- 32 tests, dont 10 tests adversariaux de contrôle qualité

**Mesures instrumentées par tapis** : temps, coût, verdicts, nombre d'itérations,
`needs_human`. Écrites dans `report.json`, agrégées en fin de lot.

### Ce qu'il faut tester avant de construire la version complète

1. **Le générateur produit-il vraiment un sol dégagé ?** Le prompt l'exige explicitement.
   À vérifier sur 20 générations réelles : c'est le point de rupture le plus probable.
2. **La lecture du plan du sol est-elle assez précise ?** Un modèle de vision donne des
   points approximatifs. Les contrôles de plausibilité attrapent les erreurs grossières,
   pas les erreurs de 10 %.
3. **L'estimation d'échelle en mètres tient-elle ?** Elle conditionne la taille du tapis.
4. **Le rendu passe-t-il pour une vraie photo ?** C'est le seul point que la mesure ne
   tranche pas. Jugement humain sur 10 tapis, à l'aveugle.
5. **Les seuils de QC sont-ils bien calibrés ?** Les valeurs actuelles sont un point de
   départ raisonnable, pas une vérité. À régler sur les 10 premiers tapis.

---

## PHASE 5 — Plan de développement

### Stockage et arborescence

```
data/<PRODUCT_ID>/
├── rug_dna.json        identité du tapis, réutilisée entre les lancements
├── plate.png           orthophoto RGBA (pixels réels)
├── lifestyle_A.png     rendu pleine résolution
├── lifestyle_B.png
├── export/*.jpg        fichiers prêts pour Shopify
└── report.json         verdicts, mesures, coûts, avertissements
```

Système de fichiers pour le MVP. Passage à S3/R2 + Postgres au-delà de ~200 tapis : le
`RunReport` est déjà sérialisable, la migration est une couche d'adaptation.

### Jobs, reprise, journalisation

`run_batch()` isole chaque tapis : un échec n'interrompt pas le lot (testé). Le RUG DNA est
mis en cache, donc une reprise ne repaie pas l'analyse. Journalisation `logging` standard ;
chaque itération trace ses mesures et son verdict.

### Coûts

Par tapis, 8 photos, 2 lifestyles, générateur Gemini :

| Poste | Coût |
|---|---|
| RUG DNA (1 appel, 8 images) | 0,08 – 0,15 $ |
| Détourage | 0,00 – 0,01 $ |
| 2 scènes | 0,27 $ |
| 2 lectures de sol | 0,02 $ |
| 2 avis de scène | 0,04 $ |
| Régénérations (moyenne ~0,5) | 0,07 $ |
| **Total estimé** | **0,50 – 0,60 $** |

Pour 1 000 tapis : **~550 $**. À comparer à un shooting lifestyle professionnel.
Garde-fous : `max_iterations` borné, avis de scène désactivable (`--no-review`), avis de
scène non demandé quand la fidélité mesurée est déjà disqualifiante.

### Sécurité

Clés en variables d'environnement, jamais en base ni en dépôt. `.gitignore` exclut `data/`
et `.env`. Les providers ne journalisent pas les clés. Les photos produit ne sortent que
vers les fournisseurs explicitement configurés.

---

## Limites connues — à lire avant de s'engager

1. **Occlusion non gérée.** Si le décor comporte une table basse devant le tapis, le tapis
   est incrusté **par-dessus**. Le pipeline l'évite en demandant un sol dégagé, mais ne sait
   pas gérer un pied de table qui devrait passer devant. Une vraie solution demande une carte
   de profondeur de la scène. Hors périmètre MVP, assumé.
2. **Tapis plié ou fortement en perspective.** La plate suppose un tapis approximativement
   plan. Un coin replié se redresse mal ; `plate.warnings` le signale.
3. **Estimation d'échelle approximative.** Sans dimensions réelles déclarées, la taille du
   tapis dans la pièce reste une estimation. **Saisissez les dimensions au mètre ruban** :
   c'est l'information qui apporte le plus de précision pour le moins d'effort.
4. **Le rendu peut paraître « posé ».** C'est le compromis assumé de l'approche. Il se
   corrige par paramètres ; un tapis inventé, non.
5. **Le contrôle qualité ne juge pas le photoréalisme du décor.** Aucune métrique ne dit
   « ça ressemble à une vraie photo ». Seul l'oeil tranche. C'est pourquoi `REVIEW_REQUIRED`
   existe.
6. **Aucun garde-fou de marque sur le décor.** Le générateur peut produire un intérieur
   hors de l'univers Baba Rug. Le prompt négatif aide, il ne garantit rien.
