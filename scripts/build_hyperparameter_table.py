#!/usr/bin/env python3
"""
build_hyperparameter_table.py — Table R2.3 : hyperparamètres par modèle
==========================================================================
Réponse à la demande R2.3 du reviewer (manuscrit §4) : un tableau unique
couvrant les 10 modèles de la Table 3, avec pour chacun l'architecture, le
schedule d'entraînement, la construction du graphe, le nombre total de
paramètres et la PROVENANCE de chaque jeu d'hyperparamètres.

CE SCRIPT NE CALCULE RIEN : il ne fait que mettre en forme des valeurs
relevées directement dans le code source (fichier + ligne cités en
commentaire pour chaque valeur non triviale). Il ne lit pas
results/raw_results.csv et ne modifie aucune logique de calcul existante.

Sources primaires (source de vérité = le .py, pas les YAML) :
  - 06_train_multistation.py        Linear-Transformer, GCN-Transformer
  - 09_controls_oversmoothing.py    GAT (2L), GCN 1L         (E13 / Table 8)
  - 10_external_baselines.py        Persistence, ARIMA, XGBoost, LSTM  (E1)
  - 14_sota_baselines.py            STGCN, Graph WaveNet     (E11 / Table 3)
Recoupement (non normatif) : configs/experiments/{canonical_benchmark,
e13_oversmoothing_gat_control,external_baselines,sota_stgcn_graphwavenet}.yaml

NOMBRES DE PARAMÈTRES — aucun compte n'est journalisé nulle part dans le
dépôt (grep "numel|n_params|param_count" sur *.py, logs/*.log, results/ :
aucun résultat). Ils ont donc été obtenus par INSTANTIATION SEULE (aucun
entraînement) des classes de modèles avec leurs hyperparamètres établis,
puis sum(p.numel() for p in model.parameters()) — exemple de référence
Beijing (N=12 stations, F=5 features). Vérifié : seul Graph WaveNet a un
compte dépendant de N (embeddings adaptatifs E1, E2 de taille N x 10) ;
tous les autres encodeurs sont invariants en N (les poids GCN/GAT/Cheb sont
partagés entre nœuds, l'adjacence est un buffer et non un paramètre).

Usage :
    python3 scripts/build_hyperparameter_table.py
Sortie :
    manuscript/tables/table_R2.3_hyperparameters.md + .docx
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from scripts.regenerate_tables import write_table  # noqa: E402 — même writer .md/.docx que les Tables 1-9

# --------------------------------------------------------------------------- #
# Blocs partagés — évitent de dupliquer (et de désynchroniser) les valeurs
# communes à plusieurs lignes.
# --------------------------------------------------------------------------- #

# Schedule canonique de 06_train_multistation.py:49-59 + train_model():644-690.
# Réutilisé tel quel par 14_sota_baselines.py (appelle b.train_model, 14:410-411)
# et RÉIMPLÉMENTÉ à l'identique en valeurs par 10 (LSTM, 10:275-299) et par 09
# (09:288-307, avec deux écarts documentés ligne par ligne plus bas).
_ADAM = "Adam (weight_decay 1e-5)"          # 06:648
_LR = "1e-3"                                # 06:56 (LR)
_NO_SCHED = "aucun"                         # grep lr_scheduler/StepLR/ReduceLROnPlateau/Cosine/OneCycle sur tout le dépôt : 0 résultat
_EPOCHS = "50"                              # 06:59 (MAX_EPOCHS)
_ES_06 = ("MSE de validation (échelle MinMax) ; patience 8 ; "
          "restauration du meilleur checkpoint ; clip_grad_norm 1.0")   # 06:58, 663, 679-689
_ES_09 = ("MSE de validation (échelle MinMax) ; patience 8 ; min_delta 1e-5 ; "
          "restauration du meilleur checkpoint ; PAS de clipping")      # 09:302-308
_BATCH = "64"                               # 06:51 (BATCH_SIZE)
_SEL_ES = "Early stopping sur la perte de validation (split chronologique 15 % médian)"

# Transformer temporel de 06 (TemporalTransformer, 06:573-599) — FFN = 4 x d_model.
_TR_06 = ("Transformer temporel 2 couches, d_model 64, 4 têtes, FFN 256 (=4 x d_model), "
          "dropout 0.1, encodage positionnel sinusoïdal ; tête Linear(64 -> 1)")  # 06:52-55, 576-581, 622
# Transformer temporel de 09 (09:167-179) — ARGUMENTS POSITIONNELS :
# nn.TransformerEncoderLayer(d_model, heads, d_model*2, dropout) => FFN = 2 x d_model,
# soit la MOITIÉ de 06. Écart réel entre les deux implémentations, rapporté tel quel.
_TR_09 = ("Transformer temporel 2 couches, d_model 64, 4 têtes, FFN 128 (=2 x d_model), "
          "dropout 0.1, encodage positionnel sinusoïdal ; tête Linear(64 -> 1)")

# Graphe k-NN de 06 (build_graph 06:98-120 / build_correlation_graph 06:123-176).
_GRAPH_06 = ("k-NN dirigé, k=5 (k_eff = min(k, N-1) ; jamais plafonné ici car N >= 7) ; "
             "topologie distance (poids 1/haversine) OU corrélation (Pearson PM2.5 sur le "
             "train seul) ; poids min-max normalisés ; self-loops ajoutés dans "
             "Â = D^-1/2 (A+I) D^-1/2")
# Graphe k-NN de 09 (build_edges 09:135-153) — réimplémentation locale, K=5 (09:58).
_GRAPH_09 = ("k-NN dirigé, k=5 (constante K locale, 09:58) ; topologie distance "
             "uniquement pour les runs de la Table 8 (wrapper e13_run_oversmoothing.sh) ; "
             "poids min-max normalisés ; self-loops ajoutés par la couche PyG "
             "(add_self_loops=True par défaut)")

_PROV_SCHED = ("schedule fixé par nous pour cohérence de protocole (Adam 1e-3 / wd 1e-5 / "
               "batch 64 / 50 époques max / patience 8 / SEQ_LEN 24 h / horizon 1 h "
               "identiques pour tous les modèles entraînés)")

# --------------------------------------------------------------------------- #
# Une entrée par modèle de la Table 3 (ordre du manuscrit).
# --------------------------------------------------------------------------- #
MODELS = [
    dict(  # -------------------------------------------------- Persistence --
        model="Persistence (t-1)",
        arch="n/a — règle déterministe P̂[t] = PM2.5[t-1], aucun paramètre appris",  # 10:144-148
        optimizer="n/a",
        lr="n/a",
        schedule="n/a",
        epochs="n/a",
        early_stop="n/a",
        batch="n/a",
        graph="n/a",
        params="0",
        selection="Aucune — modèle sans ajustement",
        provenance="n/a — le modèle n'a aucun hyperparamètre",
    ),
    dict(  # -------------------------------------------------------- ARIMA --
        model="ARIMA",
        # ARIMA_ORDER = (2, 1, 2), 10:76 ; un modèle univarié PM2.5 par station, 10:154-177
        arch="n/a — ARIMA(p,d,q) = (2,1,2), univarié PM2.5, un modèle indépendant par station ; "
             "prévision 1 pas en avant sans refit (statsmodels .append(refit=False), 10:166)",
        optimizer="Maximum de vraisemblance (statsmodels, ARIMA.fit() sans argument -> lbfgs par défaut)",
        lr="n/a",
        schedule="n/a",
        epochs="n/a (itérations MLE, maxiter par défaut de statsmodels)",
        early_stop="n/a — critère de convergence MLE",
        batch="n/a (ajustement sur la série d'entraînement complète)",
        graph="n/a",
        # 5 coefficients estimés vérifiés empiriquement (param_names de statsmodels 0.14.6) :
        # ar.L1, ar.L2, ma.L1, ma.L2, sigma2
        params="5 coefficients estimés par station (ar.L1, ar.L2, ma.L1, ma.L2, σ²) ; "
               "soit 60 au total pour Beijing (N=12), les modèles étant indépendants",
        # Cascade de repli 10:163-176 : ordre suivant essayé si le fit lève, puis persistance.
        selection="Ordre FIXE, aucune sélection AIC/BIC ; cascade de repli en cas d'échec du fit : "
                  "(2,1,2) -> (1,1,1) -> (1,1,0) -> (0,1,1) -> persistance",
        provenance="Fixé par nous (ordre a priori, aucune recherche sur validation dans le dépôt) ; "
                   "justification du choix (2,1,2) : UNKNOWN",
    ),
    dict(  # ------------------------------------------------------ XGBoost --
        # XGBRegressor(...), 10:209-211 ; design matrix _xgb_design, 10:183-196
        model="XGBoost",
        arch="n/a (ensemble d'arbres) — 400 arbres, max_depth 6, subsample 0.8, "
             "colsample_bytree 0.8, objectif reg:squarederror ; un modèle par station ; "
             "28 features = lags PM2.5 t-1..t-24 + 4 covariables météo à t-1",
        optimizer="Gradient boosting (XGBoost 3.2.0), n_jobs=4, random_state=42",
        lr="0.05 (shrinkage)",
        schedule="aucun",
        epochs="n/a — 400 rounds de boosting fixes",
        early_stop="aucun (aucun eval_set passé à .fit() ; le split de validation n'est pas utilisé)",
        batch="n/a (ajustement plein lot)",
        graph="n/a",
        params="Pas de compte fixe — la taille des arbres dépend des données ; "
               "borne supérieure 400 x (2^6 - 1) = 25 200 splits",
        selection="Hyperparamètres fixes, aucune recherche ; split de validation inutilisé",
        provenance="Fixé par nous (valeurs a priori, aucune recherche sur validation dans le dépôt) ; "
                   "justification des valeurs (400 / 6 / 0.05 / 0.8 / 0.8) : UNKNOWN",
    ),
    dict(  # --------------------------------------------------------- LSTM --
        # LSTMReg, 10:220-235 ; boucle d'entraînement 10:272-300
        model="LSTM",
        arch="nn.LSTM 2 couches, hidden 64, dropout 0.1 ; tête Linear(64 -> 1) ; "
             "skip de persistance résiduel (pred = PM2.5[t-1] + LSTM(...)) ; "
             "nœuds poolés (un modèle partagé, séquences empilées sur la dimension station)",
        optimizer=_ADAM,          # 10:275
        lr=_LR,                   # 10:275
        schedule=_NO_SCHED,
        epochs=_EPOCHS,           # 10:279 (range(1, 51))
        early_stop=_ES_06,        # 10:288 (clip 1.0), 10:293-299 (patience 8)
        batch=_BATCH,             # 10:282 (pas de 64)
        graph="n/a — modèle purement temporel",
        params="51 521 (Beijing N=12 ; invariant en N : les nœuds sont poolés)",
        selection=_SEL_ES,
        provenance="Architecture : fixée par nous pour cohérence de protocole (hidden 64 = d_model "
                   "des modèles Transformer, 2 couches = N_LAYERS) ; " + _PROV_SCHED +
                   " ; le skip de persistance résiduel est un choix de conception explicite, "
                   "motivé empiriquement (sans lui, R² 0.17 vs 0.80 sur Madrid — docstring 10:220-225), "
                   "pas issu d'une recherche d'hyperparamètres",
    ),
    dict(  # -------------------------------------------- Linear-Transformer --
        # SpatioTemporalModel(encoder_type='linear'), 06:602-630 ; LinearEncoder 06:559-570
        model="Linear-Transformer",
        arch="Encodeur Linear(5 -> 64) + LayerNorm, AUCUNE communication inter-stations ; " + _TR_06,
        optimizer=_ADAM,
        lr=_LR,
        schedule=_NO_SCHED,
        epochs=_EPOCHS,
        early_stop=_ES_06,
        batch=_BATCH,
        graph="n/a — topologie-indépendant (LinearEncoder ignore edge_index/edge_weight, 06:566)",
        params="100 545 (Beijing N=12 ; invariant en N)",
        selection=_SEL_ES,
        provenance="Architecture : Transformer encoder standard (nn.TransformerEncoder de PyTorch) ; "
                   "valeurs (d_model 64, 4 têtes, 2 couches, dropout 0.1) fixées par nous ; " + _PROV_SCHED,
    ),
    dict(  # ----------------------------------------------- GCN-Transformer --
        # SpatioTemporalModel(encoder_type='gcn2') -> GCNEncoder(n_gcn_layers=2), 06:442-483
        model="GCN-Transformer (2 couches, canonique)",
        arch="Encodeur GCN 2 couches, Â = D^-1/2 (A+I) D^-1/2, 5 -> 64 -> 64, ReLU intermédiaire, "
             "LayerNorm en sortie, sans biais ; " + _TR_06,
        optimizer=_ADAM,
        lr=_LR,
        schedule=_NO_SCHED,
        epochs=_EPOCHS,
        early_stop=_ES_06,
        batch=_BATCH,
        graph=_GRAPH_06,
        params="104 577 (Beijing N=12 ; invariant en N — Â est un buffer, pas un paramètre)",
        selection=_SEL_ES,
        provenance="Architecture : propagation GCN normalisée symétrique avec self-loops, formulation "
                   "standard — AUCUNE référence bibliographique citée dans 06_train_multistation.py, "
                   "l'attribution n'est donc pas établissable depuis le dépôt ; hyperparamètres "
                   "(d_model 64, 4 têtes, 2 couches GCN, k=5, dropout 0.1) fixés par nous ; " + _PROV_SCHED,
    ),
    dict(  # ---------------------------------------------------------- GAT --
        # SpatialEncoder(kind='gat'), 09:192-194 ; Model, 09:227-250 ; run(), 09:267-319
        model="GAT (GAT-Transformer, 2 couches)",
        arch="2 x GATConv (PyTorch Geometric) : 4 têtes x 16 canaux concaténés -> 64 ; "
             "LeakyReLU 0.2 et dropout d'attention 0.0 (défauts PyG, non surchargés dans 09) ; "
             "ReLU entre les couches ; " + _TR_09,
        optimizer=_ADAM,          # 09:288
        lr=_LR,                   # 09:288
        schedule=_NO_SCHED,
        epochs=_EPOCHS,           # 09:291 (range(50))
        early_stop=_ES_09,
        batch=_BATCH,             # 09:294 (découpage manuel par pas de 64)
        graph=_GRAPH_09,
        params="71 809 (Beijing N=12 ; invariant en N)",
        selection=_SEL_ES,
        provenance="Architecture : couche GATConv de PyTorch Geometric (implémentation de référence de "
                   "GAT ; aucune citation explicite dans 09_controls_oversmoothing.py) ; nombre de "
                   "têtes d'attention du GAT = 4, fixé par nous pour s'aligner sur N_HEADS du "
                   "Transformer ; " + _PROV_SCHED + ". ATTENTION — 09 est une RÉIMPLÉMENTATION "
                   "autonome du protocole (docstring 09:21-23) et non un import de 06 : deux écarts "
                   "réels subsistent, FFN = 2 x d_model (128) au lieu de 4 x d_model (256), et "
                   "absence de clip_grad_norm ; l'early stopping y utilise en outre un min_delta 1e-5",
    ),
    dict(  # ------------------------------------------------------- GCN 1L --
        # SpatialEncoder(kind='gcn', n_layers=1), 09:189-191 ; variants 09:337
        model="GCN 1L (contrôle 1 couche)",
        arch="1 x GCNConv (PyTorch Geometric) 5 -> 64 + ReLU (une seule agrégation de voisinage, "
             "l'over-smoothing exigeant l'empilement) ; " + _TR_09,
        optimizer=_ADAM,
        lr=_LR,
        schedule=_NO_SCHED,
        epochs=_EPOCHS,
        early_stop=_ES_09,
        batch=_BATCH,
        graph=_GRAPH_09,
        params="67 393 (Beijing N=12 ; invariant en N)",
        selection=_SEL_ES,
        provenance="Architecture : couche GCNConv de PyTorch Geometric ; la profondeur 1 n'est PAS un "
                   "réglage mais la variable manipulée du contrôle anti-over-smoothing (E13, Table 8) — "
                   "seule la profondeur change vs le GCN 2 couches de la même implémentation ; "
                   + _PROV_SCHED + ". Mêmes écarts d'implémentation vs 06 que la ligne GAT "
                   "(FFN 2 x d_model, pas de clipping, min_delta 1e-5)",
    ),
    dict(  # -------------------------------------------------------- STGCN --
        # STGCNModel, 14:186-212 ; STConvBlock 14:158-183 ; ChebGraphConv 14:107-142
        model="STGCN",
        arch="2 blocs ST-Conv sandwich (TemporalGatedConv kt=3 -> ChebGraphConv K=3 -> "
             "TemporalGatedConv kt=3), canaux bloc 1 : 5 -> 32 -> 64, bloc 2 : 64 -> 32 -> 64, "
             "LayerNorm + dropout 0.1 par bloc ; conv. temporelle de sortie Conv2d(64, 64, "
             "kernel (16,1)) ramenant T=24 à T=1 ; FC 64 -> 1",
        optimizer=_ADAM,          # via b.train_model, 14:410-411 -> 06:648
        lr=_LR,
        schedule=_NO_SCHED,
        epochs=_EPOCHS,           # b.MAX_EPOCHS, 14:411
        early_stop=_ES_06,        # b.PATIENCE + clipping de 06 (train_model réutilisé tel quel)
        batch=_BATCH,             # b.BATCH_SIZE, 14:405-407
        graph="Topologie CORRÉLATION uniquement (b.build_correlation_graph, k = b.K_NEIGHBORS = 5, "
              "14:399) ; adjacence SYMÉTRISÉE (A + Aᵀ)/2 avant normalisation — le Laplacien de "
              "Chebyshev exige un graphe non dirigé ; Laplacien normalisé rééchelonné par λ_max",
        params="110 337 (Beijing N=12 ; invariant en N — θ de Chebyshev partagé entre nœuds)",
        selection=_SEL_ES,
        provenance="Architecture : héritée de l'article d'origine — Yu, Yin & Zhu, IJCAI 2018 "
                   "(« Spatio-Temporal Graph Convolutional Networks ») ; convolution spectrale de "
                   "Chebyshev d'après Defferrard, Bresson & Vandergheynst, NeurIPS 2016 — les deux "
                   "explicitement cités dans la docstring 14:31-40. Seule adaptation DÉCLARÉE dans le "
                   "code : sortie à horizon 1 pas au lieu du seq2seq multi-pas natif ; "
                   + _PROV_SCHED,
    ),
    dict(  # ------------------------------------------------- Graph WaveNet --
        # GraphWaveNetModel, 14:256-318 ; GatedDilatedConv 14:244-253 ; DiffusionGraphConv 14:220-241
        model="Graph WaveNet",
        arch="start_conv Conv2d(5 -> 32) ; 4 blocs de dilatations (1, 2, 1, 2) : TCN gated "
             "kernel 2 (tanh x sigmoid) -> conv. de diffusion d'ordre 2 sur 3 supports "
             "(avant / arrière / adaptatif), dropout 0.1 -> skip Conv2d(32 -> 64) + résiduel "
             "Conv2d(32 -> 32) ; end_conv1 64 -> 64, end_conv2 64 -> 1 ; embeddings adaptatifs "
             "appris E1, E2 de dimension N x 10, A_adp = softmax(ReLU(E1 E2ᵀ))",
        optimizer=_ADAM,
        lr=_LR,
        schedule=_NO_SCHED,
        epochs=_EPOCHS,
        early_stop=_ES_06,
        batch=_BATCH,
        graph="Topologie CORRÉLATION uniquement (b.build_correlation_graph, k = b.K_NEIGHBORS = 5, "
              "14:399) ; adjacence NON symétrisée (la conv. de diffusion gère nativement les graphes "
              "dirigés) ; supports = normalisations en ligne de A (avant) et Aᵀ (arrière), PLUS une "
              "adjacence adaptative apprise",
        params="62 769 (Beijing N=12) — SEUL modèle dont le compte dépend de N, via E1 et E2 "
               "(2 x N x 10) : 62 669 pour Madrid (N=7), 62 689 pour London (N=8), "
               "62 929 pour CZT (N=20)",
        selection=_SEL_ES,
        provenance="Architecture : héritée de l'article d'origine — Wu, Pan, Long, Jiang & Zhang, "
                   "IJCAI 2019 (« Graph WaveNet for Deep Spatial-Temporal Graph Modeling ») ; "
                   "convolution de diffusion d'après Li, Yu, Shahabi & Liu, ICLR 2018 (DCRNN) — les "
                   "deux explicitement cités dans la docstring 14:41-47. Seule adaptation DÉCLARÉE "
                   "dans le code : sortie à horizon 1 pas au lieu du seq2seq multi-pas natif ; "
                   + _PROV_SCHED,
    ),
]

HEADERS = [
    "Modèle",
    "Taille d'architecture",
    "Optimiseur",
    "Taux d'apprentissage",
    "Schedule LR",
    "Époques (max)",
    "Early stopping (critère + patience)",
    "Batch",
    "Construction du graphe (k, topologie)",
    "Paramètres (total)",
    "Sélection de modèle",
    "Provenance des hyperparamètres",
]

KEYS = ["model", "arch", "optimizer", "lr", "schedule", "epochs", "early_stop",
        "batch", "graph", "params", "selection", "provenance"]

NOTE = (
    "Table R2.3 — hyperparamètres relevés directement dans le code source, sans re-run : "
    "06_train_multistation.py (Linear-Transformer, GCN-Transformer), "
    "09_controls_oversmoothing.py (GAT, GCN 1L — E13/Table 8), "
    "10_external_baselines.py (Persistence, ARIMA, XGBoost, LSTM — E1), "
    "14_sota_baselines.py (STGCN, Graph WaveNet — E11). "
    "Protocole commun à tous les modèles entraînés : SEQ_LEN 24 h, horizon 1 h, 5 features "
    "(PM2.5, TEMP, PRES, DEWP, WSPM), splits chronologiques 70/15/15, MinMax ajusté sur le train "
    "seul, seeds 42/123/777 (sauf XGBoost, seed 42 unique, et ARIMA/Persistence, déterministes). "
    "Le mode --quick de 06_train_multistation.py réduit SEEDS/MAX_EPOCHS/D_MODEL/SEQ_LEN et n'est "
    "JAMAIS utilisé pour les résultats publiés. "
    "PROVENANCE — trois catégories : (i) « hérité de l'article d'origine » = architecture reprise "
    "d'une publication identifiable, citée dans la docstring du script (STGCN, Graph WaveNet) ; "
    "(ii) « réglé sur validation » = valeur issue d'une recherche d'hyperparamètres — AUCUNE ligne "
    "de cette table n'entre dans cette catégorie : le dépôt ne contient aucune recherche "
    "d'hyperparamètres (grep grid/GridSearch/tune sur *.py : aucun résultat), le split de "
    "validation servant exclusivement à l'early stopping ; (iii) « fixé par nous pour cohérence de "
    "protocole » = valeur choisie pour que tous les modèles partagent le même budget "
    "d'entraînement et la même largeur (Adam 1e-3, wd 1e-5, batch 64, 50 époques max, patience 8, "
    "SEQ_LEN 24 h, d_model/hidden 64, k=5). "
    "AUCUN SCHEDULE DE TAUX D'APPRENTISSAGE n'existe dans le dépôt (« aucun » est ici une absence "
    "vérifiée, pas une valeur manquante). "
    "CELLULES UNKNOWN — deux, toutes deux dans la colonne Provenance : la justification de l'ordre "
    "ARIMA (2,1,2) et celle des valeurs XGBoost (400 arbres / max_depth 6 / lr 0.05 / subsample 0.8 "
    "/ colsample 0.8). Les valeurs elles-mêmes sont établies (10_external_baselines.py:76 et "
    "10_external_baselines.py:209-211), mais aucun document du dépôt (code, commentaires, "
    "configs/experiments/external_baselines.yaml, README.md, CHANGELOG_TABLES.md, REVISION_BRIEF.md, "
    "AUDIT.md) n'explique pourquoi ces valeurs-là ont été retenues ; elles sont rapportées comme "
    "fixées a priori, sans justification documentée. "
    "NOMBRES DE PARAMÈTRES — aucun compte n'est journalisé dans le dépôt ; ils ont été obtenus par "
    "instantiation seule des classes de modèles (aucun entraînement) puis "
    "sum(p.numel() for p in model.parameters()), avec Beijing (N=12 stations, F=5) comme exemple de "
    "référence. Seul Graph WaveNet a un compte dépendant de N (embeddings adaptatifs) ; tous les "
    "autres encodeurs spatiaux partagent leurs poids entre nœuds et sont donc invariants en N. "
    "ÉCART D'IMPLÉMENTATION À SIGNALER — 09_controls_oversmoothing.py est une réimplémentation "
    "autonome du protocole (et non un import de 06) : son Transformer temporel utilise "
    "dim_feedforward = 2 x d_model (128) au lieu de 4 x d_model (256), n'applique pas de "
    "clip_grad_norm, et son early stopping impose un min_delta de 1e-5. Les lignes GAT et GCN 1L "
    "ne sont donc pas strictement iso-capacité avec le GCN-Transformer canonique de la Table 2/3 "
    "(71 809 / 67 393 vs 104 577 paramètres) — rapporté tel quel."
)


def main():
    rows = [[m[k] for k in KEYS] for m in MODELS]
    assert len(rows) == 10, f"10 modèles attendus (Table 3), {len(rows)} trouvés"
    for r in rows:
        assert len(r) == len(HEADERS), f"largeur de ligne incohérente : {r[0]}"
        assert not any("|" in str(c) for c in r), f"pipe interdit dans une cellule : {r[0]}"
    write_table(
        "table_R2.3_hyperparameters",
        "Table [T? proposée] — Hyperparamètres par modèle (R2.3)",
        HEADERS,
        rows,
        note=NOTE,
    )


if __name__ == "__main__":
    main()
