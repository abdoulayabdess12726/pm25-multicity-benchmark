# AirPhyNet — étude de faisabilité (portage comme baseline externe)

**Candidat** : AirPhyNet, Hettige et al., ICLR 2024, arXiv:2402.03784,
`github.com/kethmih/AirPhyNet` (commit `e77576c`).
**Statut** : étude de faisabilité **uniquement**. Aucun des 9 runs (3 villes × 3
seeds) n'a été lancé. Ce document est la base de la décision « on porte / on ne
porte pas », et, si on porte, du paragraphe de méthode à écrire dans le
manuscrit.

Matériel : Apple M1, 8 cœurs, 8 Go, **CPU uniquement** (le code force
`cuda if available else cpu`, pas de branche MPS).
Environnement : Python 3.13.0, torch 2.11.0, torchdiffeq 0.2.5, numpy 2.4.4.

---

## 1. Le pipeline d'origine tourne — avec deux correctifs d'installation

### 1.1 Installation

`pip install -r requirements.txt` **échoue en l'état** : le fichier déclare
`pytorch>=1.7.1`, qui n'est pas le nom PyPI du paquet (`torch`). Trois
dépendances réellement importées ne sont pas déclarées du tout :

| Paquet | Importé par | Déclaré dans requirements.txt |
|---|---|---|
| `torch` | partout | non (déclaré comme `pytorch`) |
| `haversine` | `utils.py:16` | non |
| `torch_geometric` | `utils.py:17` (`dense_to_sparse`) | non |
| `arrow` | `airphynet_supervisor.py:4` | non |

Installées à la main : `torchdiffeq 0.2.5`, `haversine`, `arrow` (`torch` et
`torch_geometric` étaient déjà dans notre venv). Aucune modification du code
n'a été nécessaire : le dépôt tourne tel quel sous Python 3.13 / torch 2.11.

### 1.2 Données d'exemple

Récupérées depuis le Google Drive indiqué dans le README (`external/AirPhyNet/data/`) :
`station.csv` (35 stations Beijing, lat/lon), `train.npz`, `val.npz`, `test.npz`.

```
train  x,y : (2180, 24, 35, 6)
val    x,y : ( 311, 24, 35, 6)
test   x,y : ( 623, 24, 35, 6)
x_offsets = -23..0   y_offsets = 1..24   (stride 1)
features   = [PM2.5, temperature, pressure, humidity, ws, wd]
```

Soit 3114 fenêtres, découpe 70/10/20 chronologique — conforme au « 7:1:2 »
annoncé dans le papier.

> **Résolution temporelle : 3 heures, pas 1 heure.** Le papier le dit
> explicitement (« 3 hours as a time step », 24 pas = 72 h) et le code le
> confirme : `evaluate_more` évalue aux pas `[2, 4, 8, 16, 24]`, c'est-à-dire
> 6 h / 12 h / 24 h / 48 h / 72 h. C'est le point de départ de tout le §3.

*Réserve mineure* : l'échantillon Drive contient 3161 pas de temps ≈ 395 jours
à 3 h, alors que le papier annonce Beijing 2017/01/01 → 2018/05/30 (515 jours).
L'écart reste inexpliqué, mais l'accord numérique obtenu (§1.4, < 1 % sur les
six valeurs) indique que c'est bien le jeu de la Table 2.

### 1.3 Exécution

`python main.py --config_filename config.yaml`, `config.yaml` **inchangé**
(`filter_type: diff`, seq_len 24, horizon 24, 100 époques, patience 20,
lr 5e-4, batch 32).

- Modèle construit : **37 869 paramètres entraînables**.
- Entraînement stable, décroissance normale :
  `train_mae 53.80 → 43.58`, `val_mae 29.12 → 27.22` (meilleure époque : 8),
  `test_mae 32.01` à l'époque 9.
- Une seule alerte, bénigne (`sparse invariant checks disabled`).

### 1.4 Reproduction des chiffres publiés

Run mené à terme : arrêt anticipé à l'**époque 23** (meilleure val_mae 27.2082
à l'époque 3, patience 20).

| Horizon | MAE reproduite | MAE publiée | écart | RMSE reproduite | RMSE publiée | écart |
|---|---|---|---|---|---|---|
| 24 h (pas 8) | 29.30 | 29.11 | **+0,7 %** | 42.24 | 42.16 | **+0,2 %** |
| 48 h (pas 16) | 36.44 | 36.69 | **−0,7 %** | 48.23 | 48.66 | **−0,9 %** |
| 72 h (pas 24) | 41.93 | 42.23 | **−0,7 %** | 52.58 | 53.07 | **−0,9 %** |

Les six valeurs tombent à moins de 1 % des valeurs publiées, sans aucun réglage.

*Observation de code, à conserver en tête pour le portage* : `main.py` appelle
`supervisor.train()` puis `supervisor.evaluate_more('test')` **sans recharger le
meilleur checkpoint**. Les métriques de test sont donc celles du modèle à la
**dernière** époque (23), pas à la meilleure (3) — les checkpoints sauvegardés
ne sont jamais relus. C'est le comportement qui produit les chiffres publiés :
il est reproductible, mais ce n'est pas de l'early stopping au sens usuel
(cf. M14).

**Conclusion §1 : le pipeline d'origine fonctionne sans modification, et
reproduit les résultats publiés à moins de 1 % près.**

---

## 2. Coût mesuré, et ce que coûteraient 9 runs

### 2.1 Mesures

Sur l'exemple d'origine : **65–80 s/époque** (35 nœuds, 69 batches, `diff`).

Coût par batch mesuré directement sur les classes AirPhyNet non modifiées, avec
nos dimensions (batch 32, forward+backward, 8 threads, machine au repos) :

| nœuds | horizon | filtre | s/batch | NFE |
|---|---|---|---|---|
| 35 | 24 | `diff` | 1.042 | 14 |
| 12 | 24 | `diff` | 0.296 | 14 |
| 7 | 24 | `diff` | 0.240 | 14 |
| 35 | 24 | `diff_adv` | **52.66** | 20 |
| 12 | 24 | `diff_adv` | 15.62 | 20 |
| 7 | 24 | `diff_adv` | 7.98 | 20 |
| 12 | 1 | `diff` (grille d'origine) | 0.097 | **2** |
| 7 | 1 | `diff` (grille d'origine) | 0.076 | **2** |
| 12 | 1 | `diff` (**grille corrigée**, cf. M3) | 0.647 | 14 |
| 8 | 1 | `diff` (grille corrigée) | 0.512 | 14 |
| 7 | 1 | `diff` (grille corrigée) | 0.655 | 14 |
| 12 | 1 | `diff_adv` (grille corrigée) | 11.02 | 20 |
| 7 | 1 | `diff_adv` (grille corrigée) | 6.88 | 20 |

Les lignes « grille d'origine » à horizon 1 sont **trompeuses de bon marché** :
NFE = 2 parce qu'aucune intégration n'a lieu (cf. §3.2). Elles ne doivent pas
servir de base d'estimation.

### 2.2 Extrapolation à nos jeux de données

Nos trois villes font **T = 35 064 pas horaires** chacune (Beijing 12 stations,
London 8, Madrid 7). Avec seq_len 24, horizon 1, stride 1 : 35 040 fenêtres →
70/15/15 → 24 528 train (766 batches de 32) + 5 256 val (164 batches, forward
seul ≈ 0,35×) ≈ **825 batch-équivalents par époque**, soit **11× l'exemple
Beijing du dépôt**.

| Configuration | min/époque | 50 époques (notre protocole) | 100 époques (sa config) |
|---|---|---|---|
| `diff`, horizon 1 corrigé | 7 – 9 | 6 – 7,5 h / run | 12 – 15 h / run |
| `diff_adv`, horizon 1 corrigé | 95 – 150 | 80 – 125 h / run | 160 – 250 h / run |

**Pour 3 villes × 3 seeds = 9 runs :**

- **`diff` seul** : **55 – 68 h** (≈ 2,5–3 jours) à 50 époques ;
  **110 – 135 h** (≈ 5 jours) à 100 époques.
- **`diff_adv`** (le modèle complet du papier, diffusion **+** advection) :
  **700 – 1 100 h**, soit **1 à 1,5 mois**. Infaisable en l'état.

### 2.3 Pourquoi `diff_adv` explose, et ce qu'on peut y faire

`ODEFunc.__init__` (`ode_func.py:141-162`) reconstruit, **à chaque forward**,
une matrice d'adjacence d'advection par échantillon du batch via une double
boucle Python `for batch_index in range(batch_size): for edge_id in
range(num_edges)`, puis appelle `calculate_scaled_laplacian` (scipy) une fois
par échantillon. À 12 nœuds et batch 32, cela fait 4 224 itérations Python et
32 factorisations scipy **par batch**, pour un état ODE de dimension 48.

C'est vectorisable (un `scatter` dense + un Laplacien batché) **sans changer
une seule ligne de mathématiques**. Gain attendu : un ordre de grandeur. C'est
la modification M11 ci-dessous, et elle conditionne la possibilité même
d'évaluer le modèle complet.

**Augmenter le batch ne sauve pas** : mesuré 32 → 64, gain ~10 % seulement (le
coût des convolutions de graphe croît en batch × nœuds ; ce n'est pas de la
surcharge Python fixe).

---

## 3. Analyse d'écart avec notre protocole

### 3.1 Features attendues vs nos 5

Le papier dit consommer « PM2.5, wind speed and wind direction as auxiliary
variables ». **Le code le confirme et va plus loin** : dans
`Encoder_z0_RNN.forward` (`airphynet_model.py:151-152`),

```python
pm25      = inputs[:, :, 0].unsqueeze(-1)   # seule entrée du GRU
wind_vars = inputs[:, :, -2:]               # (ws, wd), passées à l'ODE
```

Les colonnes 1, 2, 3 (`temperature`, `pressure`, `humidity`) sont chargées puis
**jetées**. Le modèle n'a donc jamais vu la météo au-delà du vent — ce n'est
pas nous qui le bridons, c'est son design.

Conséquences pour nous :

| | notre GCN-Transformer | AirPhyNet porté |
|---|---|---|
| PM2.5 | ✅ | ✅ |
| TEMP, PRES, DEWP | ✅ (3 features) | ignorées par l'encodeur |
| WSPM (vitesse vent) | ✅ | ✅ (advection seulement) |
| **direction du vent** | absente de nos données | **requise** |

**Disponibilité de la direction du vent dans nos 3 jeux :**

- **Beijing** : disponible. La colonne `wd` existe dans les CSV bruts PRSA
  (rose des vents 16 points, `"NNW"`…) ; il faut la décoder en degrés. Aucun
  re-téléchargement.
- **London / Madrid** : disponible **après re-téléchargement**. La météo vient
  d'Open-Meteo (`01d_`, `01f_`), qui n'a demandé que `wind_speed_10m` ;
  `wind_direction_10m` s'obtient sur la même API, par la même requête, avec une
  variable de plus.

**Mais** — et c'est le point qui doit figurer dans le manuscrit — la météo
Open-Meteo est une réanalyse sur grille : les stations proches tombent dans la
même maille. Mesuré sur nos parquets :

| ville | stations | séries WSPM distinctes | paires strictement identiques | σ inter-stations / σ total |
|---|---|---|---|---|
| London | 8 | **5** | 4 / 28 | 0,134 |
| Madrid | 7 | **3** | 7 / 21 | 0,142 |

Or le poids d'arête d'advection vaut `flow_net(vent_source) −
flow_net(vent_cible)` (`ode_func.py:136-139`) : pour toute paire de stations
partageant une maille, il est **exactement nul**. Le terme d'advection est donc
structurellement dégénéré sur une fraction importante des arêtes de London et
Madrid. C'est une limite de nos **données**, pas du modèle — mais elle doit
être écrite, sinon un relecteur y verra un bridage.

### 3.2 Horizon — le point bloquant

Le modèle est conçu pour 24 pas de 3 h (72 h). Nous prédisons **1 pas de 1 h**.
Deux problèmes distincts, à ne pas confondre.

**(a) `horizon: 1` "marche", mais désactive entièrement la physique.**

`airphynet_model.py:93-94` construit la grille temporelle de l'ODE :

```python
time_steps_to_predict = torch.arange(start=0, end=self.horizon, step=1).float()
time_steps_to_predict = time_steps_to_predict / len(time_steps_to_predict)
```

- `horizon = 24` → `[0, 1/24, …, 23/24]`
- `horizon = 1` → **`[0.0]`**, un point unique.

Vérifié directement sur torchdiffeq : `odeint(f, y0, [0.])` **renvoie `y0`
inchangé** — aucune intégration n'est effectuée (NFE = 2 correspond à
l'initialisation du solveur, pas à un pas). Le forward ne plante pas, il rend
simplement `Decoder(z0)`.

Et le décodeur (`airphynet_model.py:171-196`) **n'a aucun paramètre** : il
moyenne les 4 dimensions latentes. Donc à `horizon = 1`, AirPhyNet se réduit
exactement à :

> un GRU(64) appliqué **par station** au seul historique PM2.5 de cette
> station, → MLP → moyenne de 4 scalaires.

**Pas de graphe. Pas de diffusion. Pas d'advection. Pas de vent.** Publier ça
sous le nom « AirPhyNet » dans notre Table 3 serait indéfendable : nous
comparerions un GRU sans graphe à nos modèles à graphe, dans un papier dont la
thèse porte précisément sur l'apport du graphe.

*Note annexe sur le code d'origine* : le même mécanisme fait que, même à
`horizon = 24`, `output[0]` — la prédiction à t+1 — est `Decoder(z0)`, non
intégrée. Le papier n'évaluant qu'à partir du pas 2, ses chiffres publiés à
24/48/72 h ne sont pas concernés.

**Correctif (M3)** : grille `arange(0, horizon+1)/horizon`, puis ne garder que
`solution[1:]`. Trois lignes, aucune couche ajoutée, chaque pas prédit devient
le résultat d'une intégration réelle. C'est une modification qui **avantage**
le modèle. Coût : ×7 sur le temps de calcul (0,097 → 0,647 s/batch), déjà
intégré au §2.

**(b) 1 h est en dessous du pas natif du modèle.** Même corrigé, un pas de 1 h
n'est pas un pas de 3 h. La grille ODE étant normalisée sur `[0,1]`, l'échelle
physique est absorbée par les poids appris, mais `latent_dim = 4`,
`rnn_units = 64`, `coeff = 0.1` (coefficient de diffusion **codé en dur**,
`ode_func.py:202`) et le lr ont été calibrés pour 24 pas de 3 h. On ne peut pas
prouver a priori que ce transfert est neutre. → voir la recommandation R1.

### 3.3 Construction du graphe

`utils.get_adjacency_matrix` :

- **graphe complet**, poids `1 / distance_haversine(km)`. Le commentaire
  « Apply threshold to adjacency matrix » (`utils.py:39-40`) ne fait rien :
  c'est une copie sans seuil.
- **pas de self-loops** (diagonale nulle).
- symétrique.
- `calculate_scaled_laplacian(λ_max = 2)` symétrise l'entrée puis produit
  `L̃ = −D^{-1/2} A D^{-1/2}` ; Chebyshev ordre 2.

**Entrées dont dépend le graphe :**

| Composante | Entrées | Disponible chez nous ? |
|---|---|---|
| Diffusion | lat/lon des stations uniquement | **oui**, 3 villes |
| Advection | (ws, wd) par station, **au dernier pas d'entrée seulement** (`wind_vars[-1]`) | ws oui ; wd après re-téléchargement (§3.1), mais dégénérée sur London/Madrid |

Deux remarques sur le code, pour éviter les fausses pistes :

- `edge_attr` (distance, cap) est construit dans `utils.py`, converti en
  tenseur dans `ODEFunc` (`ode_func.py:99, 132`)… puis **jamais lu**. C'est une
  variable morte.
- Il y a un vrai bug ligne 61 de `utils.py` — `dest_lat` est initialisé avec la
  **longitude** de la station cible — mais comme il n'affecte que le cap stocké
  dans `edge_attr`, il est **sans effet numérique**. Ne pas le « corriger » :
  ce serait s'écarter du code publié pour rien.

**Écart avec nos topologies** : nous utilisons des kNN k=5, asymétriques
(Beijing 59 arêtes sur 132 possibles, London 39/56, Madrid 34/42) et deux
variantes (distance, corrélation). AirPhyNet n'a pas de paramètre k. Injecter
nos matrices est techniquement trivial (`calculate_scaled_laplacian` symétrise
déjà), mais ce n'est pas neutre — voir M6.

### 3.4 Découpe train/val/test et évaluation

| | AirPhyNet | Nous | Verdict |
|---|---|---|---|
| découpe | 7:1:2 chronologique | 70/15/15 chronologique | alignement trivial, nous générons les `.npz` |
| normalisation | StandardScaler (z-score), ajusté sur le PM2.5 du train | MinMax train | interne au modèle, métriques recalculées en unités d'origine → **garder le sien** |
| loss | MAE **masquée** : `y_true < 1e-4 → exclu` | MSE sur tous les points | son design, à conserver (M9) |
| métriques | MAE/MAPE/RMSE **masquées**, **moyennées par batch** | RMSE/MAE/R² sur prédictions poolées | **incompatible**, à recalculer (M8) |
| R² | jamais calculé | métrique centrale de nos tables | à ajouter (M8) |
| par station | jamais rapporté | requis (Table 6, `raw_results.csv`) | récupérable : la sortie est `(horizon, batch, N)` |
| seeds | **aucun seeding nulle part** dans le dépôt | 42 / 123 / 777 | à ajouter (M7) |

Trois pièges d'évaluation à neutraliser explicitement :

1. **Masquage.** `metrics.py` met à 0 puis exclut toute cible `< 1e-4`.
   Mesuré : **London a 3,05 % de PM2.5 exactement nuls** (8 561 points),
   Madrid 0,02 %, Beijing 0. Ces points disparaîtraient de son MAE mais pas du
   nôtre.
2. **Moyenne par batch.** `np.mean(rmse_losses)` moyenne des RMSE de batch —
   ce n'est pas le RMSE global.
3. **Padding.** `DataLoader(pad_with_last_sample=True)` duplique le dernier
   échantillon pour compléter le dernier batch ; ces doublons entrent dans les
   métriques de test.

La parade est la même pour les trois : sauver les prédictions brutes et
calculer **nos** métriques nous-mêmes, padding retiré.

### 3.5 MENDEZ ALVARO

Pas de blocage. Le StandardScaler est global (pas par station), donc la
variance nulle sur le train ne provoque aucune division par zéro, et la loss
masquée conserve la station (valeur 6.0 > 1e-4). Conformément à
`REVISION_BRIEF.md`, elle entre dans les agrégats sans traitement particulier.

---

## 4. Liste des modifications requises, et lesquelles brident le modèle

Légende de la colonne **Risque** : ⬛ neutre · 🟩 avantage le modèle · 🟥 risque
de le désavantager.

| # | Modification | Où | Risque |
|---|---|---|---|
| **M1** | Générer nos `.npz` (24 pas in, 1 pas out, 70/15/15 chronologique, stride 1) | nouveau script | ⬛ |
| **M2** | Passer d'un pas de **3 h à 1 h** | données | 🟥 **oui, non quantifiable a priori.** Hyperparamètres (latent_dim 4, rnn_units 64, `coeff=0.1` codé en dur) calibrés pour 24 pas de 3 h. → mitigation R1 |
| **M3** | Grille ODE `arange(0, H+1)/H`, garder `solution[1:]` | `airphynet_model.py:93` | 🟩 **obligatoire.** Sans elle, l'ODE est inerte à H=1 et « AirPhyNet » = GRU sans graphe (§3.2) |
| **M4** | Ajouter la direction du vent : décoder `wd` (Beijing) ; re-télécharger `wind_direction_10m` (London, Madrid) | `01a_`, `01d_`, `01f_` | 🟩 sans elle le mode advection est inutilisable. **Mais** dégénérescence ERA5 sur London/Madrid à documenter (§3.1) |
| **M5** | Réordonner nos features en `[PM2.5, TEMP, PRES, DEWP, WSPM, WD]` (il lit index 0 et les 2 derniers) | loader | ⬛ |
| **M6** | Injecter nos topologies kNN k=5 (distance, corrélation) au lieu du graphe complet 1/d | `utils.get_adjacency_matrix` | 🟥 **oui.** Son opérateur de diffusion est conçu sur un graphe complet pondéré. → mitigation R2 |
| **M7** | Seeding explicite 42 / 123 / 777 (absent du dépôt) | `main.py` | ⬛ |
| **M8** | Sauver les prédictions ; calculer RMSE/MAE/**R²** non masqués, par station et agrégés, padding retiré | nouveau script | ⬛ correction d'évaluation, aucun effet sur l'entraînement |
| **M9** | Conserver sa loss masquée à l'entraînement | — (non-modification) | ⬛ léger coût sur London (3 % de cibles ignorées), mais c'est **son** design ; y toucher serait plus discutable que ne pas y toucher |
| **M10** | Budget d'époques : utiliser **sa** config (100 époques, patience 20), pas la nôtre (50 / 8) | `config.yaml` | 🟥 si on impose 50/8. Recommandation : 100/20 et le dire explicitement |
| **M11** | Vectoriser la construction du Laplacien d'advection (perf pure, maths inchangées) | `ode_func.py:141-162` | ⬛ en soi — mais **sans elle, `diff_adv` est hors budget** et on n'évalue que la moitié du modèle, ce qui **est** un bridage (§2.3) |
| **M12** | Écriture au schéma `results/raw_results.csv` | nouveau script | ⬛ |
| **M13** | Pas de branche MPS (`cuda`/`cpu` en dur) : AirPhyNet tournera CPU, nos modèles tournent MPS | — | ⬛ écart de temps d'horloge, pas d'équité |
| **M14** | Recharger le meilleur checkpoint avant l'évaluation test (le code évalue la **dernière** époque, cf. §1.4) | `main.py` | 🟩 l'avantagerait — mais **s'écarterait du comportement qui produit ses chiffres publiés**. Recommandation : ne pas le faire, et le mentionner |

**Bilan honnête** : trois modifications présentent un risque réel de
désavantager le modèle — **M2** (résolution), **M6** (topologie), **M10**
(budget d'entraînement, si on impose le nôtre) — plus **M11** par omission. Les
trois premières sont imposées par notre protocole ; aucune ne peut être évitée
si l'on veut une comparaison « protocole identique ». La réponse défendable
n'est donc pas de les éviter, mais de **mesurer ce qu'elles coûtent** (R1, R2).

---

## 5. Recommandations avant lancement

**R0 — Ne pas lancer les 9 runs en l'état.** Sans M3, le résultat serait un GRU
sans graphe étiqueté AirPhyNet.

**R1 — Ajouter un run de contrôle au régime natif du modèle.** En plus des runs
sous notre protocole (1 h, horizon 1 pas), faire tourner AirPhyNet à **3 h /
24 pas** sur nos 3 villes, seed 42 seulement, et rapporter les deux en annexe.
C'est ce qui permet de répondre à « vous l'avez bridé » par une mesure plutôt
que par un argument. Coût : 3 runs supplémentaires.

**R2 — Rapporter les deux graphes.** Run principal avec nos topologies k=5
(cohérence avec STGCN et Graph WaveNet dans `14_sota_baselines.py`, qui
utilisent déjà `build_correlation_graph(k=K_NEIGHBORS)`), plus un run de
contrôle avec son graphe natif complet 1/d, seed 42. Coût : 3 runs
supplémentaires.

**R3 — Faire M11 avant de décider du filtre.** Tant que la boucle Python n'est
pas vectorisée, seul `filter_type: diff` est atteignable, or diffusion **+**
advection est la contribution centrale du papier. Chiffrer le gain réel de M11
est un travail d'une demi-journée qui décide si `diff_adv` est jouable.

**R4 — Budget.** Avec M11 réussi et `diff_adv`, compter grossièrement **3 à
5 jours-machine** pour 9 runs + 6 runs de contrôle. Sans M11, `diff` seul :
**2,5 à 5 jours** selon le budget d'époques. Dans les deux cas la machine est
mobilisée : à lancer en arrière-plan avec log horodaté, jamais en session.

### Trois décisions qui t'appartiennent

1. **Résolution** : 1 h sous notre protocole seul, ou 1 h + contrôle 3 h natif (R1) ?
2. **Topologie** : nos k=5 seuls, ou nos k=5 + contrôle graphe natif (R2) ?
3. **Filtre** : `diff` seul (moins cher, mais ampute la contribution du papier),
   ou investir M11 pour évaluer `diff_adv` ?

---

## Annexe — fichiers produits

- `external/AirPhyNet/` — clone `e77576c`, **non modifié**
- `external/AirPhyNet/data/` — échantillon Beijing (Google Drive), non versionné
- `external/AirPhyNet/logs/run_asis_diff.log` — log du run d'origine
- Dépendances ajoutées au venv : `torchdiffeq`, `haversine`, `arrow`, `gdown`
  (à reporter dans un `requirements-external.txt` si le portage est validé)
