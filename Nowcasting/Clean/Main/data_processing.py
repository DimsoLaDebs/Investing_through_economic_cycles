######### OS ######## A changer c'est pas propre
import sys
import os

# Ajouter le dossier Clean au path pour que utils/ soit trouvé
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import numpy as np
import pandas as pd
import warnings

from sklearn.preprocessing import StandardScaler, MinMaxScaler
from arch.bootstrap import optimal_block_length, StationaryBootstrap, CircularBlockBootstrap


######### Modelling ########
# Cross validation
from sklearn.model_selection import cross_val_score, GridSearchCV
# CLassification
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, AdaBoostClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier

from utils.helpers import *
from Main.data_loader import *
from Main.Deep_models import train_lstm_model

import logging
from tqdm import tqdm

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')



'''
normalize_data :
block_bootstrap :
pca était chelou (on avait aucune garantie ni de garder 99% de la variance ni meme de garder les n composantes les plus importantes)
'''

#100 sufficient for RF & XGBoost, facile pour le moment mais à changer



class Data_Processing:
    """
    A class for preprocessing and model training.
    """


    def __init__(self, db: Data) -> None:
        """
        Initialize the "process" object and the variables necessary for processing.

        """

        self.db = db
        self.bb_iteration=None
        self.lstm_hyperparams = None



    def block_bootstrap(self, block_size=None, n_samples=10, method = 'Circular', random_state=42, Verbose = False):

        '''
        A effectuer après le train test split et avant normalization 
        '''

        # Sécurisation du random_state
        if random_state is not None and not isinstance(random_state, np.random.RandomState):
            random_state = np.random.RandomState(random_state)

        
        X_train, y_train, X_test, y_test = self.sets

        df = pd.concat([X_train, y_train], axis=1)

        if block_size is None:
            block_lengths  = optimal_block_length(df)  # Calculer la taille de bloc optimale)
            b_sb = int(block_lengths ['stationary'].iloc[0])  # taille moyenne pour StationaryBootstrap
            b_cb = int(block_lengths ['circular'].iloc[0])    # taille fixe pour CircularBlockBootstrap

            if Verbose:
                print(f"Taille de bloc optimale pour StationaryBootstrap : {b_sb.iloc[0]:.2f}")
                print(f"Taille de bloc optimale pour CircularBlockBootstrap : {b_cb.iloc[0]:.2f}")
        else:
            b_sb = b_cb = int(block_size)
            if Verbose:
                print(f"Taille de bloc fixe : {b_sb}")

        # Create a stationary bootstrap object
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=FutureWarning)
            stationary_bootstrap = StationaryBootstrap(b_sb, df, random_state=random_state)
            circular_bootstrap = CircularBlockBootstrap(b_cb, df, random_state=random_state)


        final_sets = {}

        for i in range(n_samples): #on aura plus la première version entière

            if method == 'Circular' :
                df_bootstraped = pd.DataFrame(list(circular_bootstrap.bootstrap(1))[0][0][0], columns=df.columns)
            elif method == 'Stationary' :
                df_bootstraped = pd.DataFrame(list(stationary_bootstrap.bootstrap(1))[0][0][0], columns=df.columns)

            X_bootstraped = df_bootstraped.iloc[:, :-1]
            y_bootstraped = df_bootstraped.iloc[:, -1].round().astype(int)
            final_sets[i] = (X_bootstraped, y_bootstraped, X_test, y_test)
        
        self.bb_sets = final_sets #on aura plus la première version entière, c'est un choix
        #print(f'block bootstrap done {n_samples} times')

        

    def aggregate_predictions(self, models: list, vintage_date, method='soft', drop_old=False):
        """
        Aggrège les prédictions bootstrapées (self.y_test_label / self.y_test_probs) par modèle :
        - moyenne forte : moyenne des labels → 0 ou 1
        - moyenne faible : moyenne pondérée par proba
        - en cas d'égalité (0.5), on retient la prédiction faible
        """
        for model_name in models:
            # Récupérer toutes les colonnes de ce modèle
            label_cols = [col for col in self.y_test_label.columns if col.startswith(f"{model_name}_") and col.endswith("_label") and col != f"{model_name}_label" and col != f"{model_name}_bb_label"]
            prob_cols  = [col for col in self.y_test_probs.columns if col.startswith(f"{model_name}_") and col.endswith("_probs") and col != f"{model_name}_probs" and col != f"{model_name}_bb_probs"]


            # Vérification
            if not label_cols or not prob_cols:
                continue

            # Extraire les prédictions pour la date donnée
            labels = self.y_test_label.loc[vintage_date, label_cols].astype(float)
            probs  = self.y_test_probs.loc[vintage_date, prob_cols].astype(float)
            
            # Moyenne forte
            hard_vote = labels.mean()

            # Moyenne faible (pondérée) - avec vérification de la somme des poids
            if probs.sum() > 0:
                soft_vote = np.average(labels, weights=probs)
            else:
                # Si les poids sont tous zéros, utiliser une moyenne simple
                soft_vote = hard_vote

            if method =='hard':
                # Décision finale
                if hard_vote > 0.5:
                    final_label = 1
                elif hard_vote < 0.5:
                    final_label = 0
                else:
                    final_label = round(soft_vote)

            elif method == 'soft':
                final_label = round(soft_vote)

            final_prob = probs.mean()

            # Stockage
            self.y_test_label.loc[vintage_date, f"{model_name}_bb_label"] = final_label
            self.y_test_probs.loc[vintage_date, f"{model_name}_bb_probs"] = final_prob

                    # Nettoyage si demandé
            if drop_old:
                self.y_test_label.drop(columns=label_cols, inplace=True, errors='ignore')
                self.y_test_probs.drop(columns=prob_cols, inplace=True, errors='ignore')



    def normalize_data(self, method='standard'):
        
        '''
        After train test split and block bootstrap
        Normalisation des features d'entrainement et de test
        '''

        if isinstance(self.sets, dict):  # block bootstrap case
            for i, (X_train, y_train, X_test, y_test) in self.sets.items():
                
                # Forcer le format DataFrame (cas d'une seule ligne)
                if isinstance(X_train, pd.Series):
                    X_train = X_train.to_frame().T
                if isinstance(X_test, pd.Series):
                    X_test = X_test.to_frame().T

                # Choix du scaler
                if method == 'standard':
                    scaler = StandardScaler()
                elif method == 'minmax':
                    scaler = MinMaxScaler()
                else:
                    raise ValueError("Méthode de normalisation inconnue. Utilisez 'standard' ou 'minmax'.")

                # Fit sur X_train uniquement (bon usage du train-test split)
                scaler.fit(X_train)
                X_train_scaled = pd.DataFrame(scaler.transform(X_train), columns=X_train.columns, index=X_train.index)
                X_test_scaled = pd.DataFrame(scaler.transform(X_test), columns=X_test.columns, index=X_test.index)

                # Mise à jour du dictionnaire
                self.sets[i] = (X_train_scaled, y_train, X_test_scaled, y_test)

            #print(f'scaling done, mean={X_train_scaled.mean().mean()} std={X_train_scaled.std().mean()} ')

        else:
            X_train, y_train, X_test, y_test = self.sets

            # Forcer 2D si Series (cas d'une seule ligne)
            if isinstance(X_train, pd.Series):
                X_train = X_train.to_frame().T
            if isinstance(X_test, pd.Series):
                X_test = X_test.to_frame().T
                
            if method == 'standard':
                scaler = StandardScaler()
            elif method == 'minmax':
                scaler = MinMaxScaler()
            else:
                raise ValueError("Méthode de normalisation inconnue. Utilisez 'standard' ou 'minmax'.")

            scaler.fit(X_train)
            X_train_scaled = pd.DataFrame(scaler.transform(X_train), columns=X_train.columns, index=X_train.index)
            X_test_scaled = pd.DataFrame(scaler.transform(X_test), columns=X_test.columns, index=X_test.index)

            #print(f'scaling done, mean={X_train_scaled.mean().mean()} std={X_train_scaled.std().mean()} ')

            self.sets = (X_train_scaled, y_train, X_test_scaled, y_test)



    def training_shot(self, models: list = None, optimize=False, vintage_date=None, threshold_tuning=None, cv=None, epochs=5): #unité d'entrainement par vintage

        X_train_US, Y_train_US, X_test_US, _ = self.sets

        for model_name in models:
            if model_name == 'RF':
                if optimize:
                    grid = {
                        'n_estimators': [ 300, 500],
                        'max_depth': [None, 10, 30, 50],
                        #'min_samples_split': [2, 5, 10],
                        #'min_samples_leaf': [2, 5],
                        #'max_features': ['sqrt', 'log2', None],
                        'bootstrap': [True]
                        }           

                    gs = GridSearchCV(RandomForestClassifier(random_state=42), grid, cv=cv)
                    gs.fit(X_train_US, Y_train_US)
                    model = gs.best_estimator_
                    self.gridsearch_reports[(vintage_date, model_name)] = pd.DataFrame(gs.cv_results_)
                else:
                    model = RandomForestClassifier(n_estimators=10, random_state=42).fit(X_train_US, Y_train_US) #good with 100+
                self.var_importances['RF'].loc[vintage_date] = model.feature_importances_ * 100

            elif model_name == 'GB':
                if optimize:
                    grid = {'n_estimators': [10, 30, 50, 100, 200, 300], 'learning_rate': [0.05, 0.1]}
                    gs = GridSearchCV(GradientBoostingClassifier(random_state=42), grid, cv=cv)
                    gs.fit(X_train_US, Y_train_US)
                    model = gs.best_estimator_
                    self.gridsearch_reports[(vintage_date, model_name)] = pd.DataFrame(gs.cv_results_)
                else:
                    model = GradientBoostingClassifier(n_estimators=100, learning_rate=0.1, random_state=42).fit(X_train_US, Y_train_US)
                self.var_importances['GB'].loc[vintage_date] = model.feature_importances_ * 100

            elif model_name == 'ADA':
                if optimize:
                    grid = {'n_estimators': [10, 50, 100, 300],
                            'learning_rate': [0.1, 0.5, 1.0],
                            'estimator': [DecisionTreeClassifier(max_depth=1), DecisionTreeClassifier(max_depth=2), DecisionTreeClassifier(max_depth=3)]
                            }
                    gs = GridSearchCV(AdaBoostClassifier(random_state=42), grid, cv=cv)
                    gs.fit(X_train_US, Y_train_US)
                    model = gs.best_estimator_
                    self.gridsearch_reports[(vintage_date, model_name)] = pd.DataFrame(gs.cv_results_)
                else:
                    model = AdaBoostClassifier(n_estimators=150, random_state=42).fit(X_train_US, Y_train_US)
                self.var_importances['ADA'].loc[vintage_date] = model.feature_importances_ * 100


            elif model_name == 'ENET':
                if optimize:
                    grid = {'C': [0.1, 1.0, 10.0], 'l1_ratio': [0.2, 0.5, 0.8]}
                    base = LogisticRegression(penalty='elasticnet', solver='saga', max_iter=10000, random_state=42)
                    gs = GridSearchCV(base, grid, cv=cv)
                    gs.fit(X_train_US, Y_train_US)
                    model = gs.best_estimator_
                    self.gridsearch_reports[(vintage_date, model_name)] = pd.DataFrame(gs.cv_results_)
                else:
                    model = LogisticRegression(penalty='elasticnet', solver='saga', l1_ratio=0.8, C=0.1, max_iter=10000, random_state=42).fit(X_train_US, Y_train_US)
                self.var_importances['ENET'].loc[vintage_date] = np.abs(model.coef_[0]) * 100

            elif model_name == 'SVC':
                if optimize:
                    grid = {'C': [0.1, 1.0, 10.0], 'gamma': ['scale', 'auto']}
                    gs = GridSearchCV(SVC(probability=True, random_state=42), grid, cv=cv)
                    gs.fit(X_train_US, Y_train_US)
                    model = gs.best_estimator_
                    self.gridsearch_reports[(vintage_date, model_name)] = pd.DataFrame(gs.cv_results_)
                else:
                    model = SVC(kernel='rbf', C=0.1, probability=True, random_state=42).fit(X_train_US, Y_train_US)

            elif model_name == 'KNN':
                if optimize:
                    grid = {'n_neighbors': [5, 10, 25, 50, 100, 200]}
                    gs = GridSearchCV(KNeighborsClassifier(), grid, cv=cv)
                    gs.fit(X_train_US, Y_train_US)
                    model = gs.best_estimator_
                    self.gridsearch_reports[(vintage_date, model_name)] = pd.DataFrame(gs.cv_results_)
                else:
                    model = KNeighborsClassifier(n_neighbors=5).fit(X_train_US, Y_train_US)

            elif model_name == 'MLP':
                if optimize:
                    grid = {
                        'hidden_layer_sizes': [(32,), (128, 32), (516, 128, 32)],
                        'alpha': [0.0001, 0.001, 0.01],
                        'activation': ['relu', 'tanh']
                    }
                    base = MLPClassifier(max_iter=5000, random_state=42)
                    gs = GridSearchCV(base, grid, cv=cv)
                    gs.fit(X_train_US, Y_train_US)
                    model = gs.best_estimator_
                    self.gridsearch_reports[(vintage_date, model_name)] = pd.DataFrame(gs.cv_results_)
                else:
                    model = MLPClassifier(hidden_layer_sizes=(32, 8, 2), activation='relu', solver='adam', max_iter=10000, random_state=42).fit(X_train_US, Y_train_US)

            elif model_name == 'LSTM':

                lstm_params = {
                    'lookback': 6,
                    'hidden_size': 6,
                    'dropout': 0,
                    'epochs': 15,
                    'batch_size': 16,
                    'patience': 3,
                    'num_layers': 1
                }

                model, proba, train_losses, val_losses = train_lstm_model(X_train_US, Y_train_US, X_test_US, **lstm_params)

                # Enregistrement pour plus tard dans save_results_and_plots
                self.lstm_hyperparams = lstm_params

                y_pred = int(proba >= 0.5)

                # Stockage des prédictions
                if self.bb_iteration is not None:
                    suffix = f"_{self.bb_iteration}"
                    self.y_test_label.loc[vintage_date, f"{model_name}{suffix}_label"] = y_pred
                    self.y_test_probs.loc[vintage_date, f"{model_name}{suffix}_probs"] = proba
                else:
                    self.y_test_label.loc[vintage_date, f"{model_name}_label"] = y_pred
                    self.y_test_probs.loc[vintage_date, f"{model_name}_probs"] = proba
                    
                # Enregistrement des courbes de loss
                self.train_loss.loc[vintage_date, :len(train_losses)-1] = train_losses
                self.val_loss.loc[vintage_date, :len(val_losses)-1] = val_losses


                # Enregistrement du modèle
                self.models[model_name] = model
 

            # Enregistrement des prédictions
            if model_name != 'LSTM':
                if self.bb_iteration is not None:
                    self.y_test_label.loc[vintage_date, f'{model_name}_{self.bb_iteration}_label'] = model.predict(X_test_US)[0]
                    self.y_test_probs.loc[vintage_date, f'{model_name}_{self.bb_iteration}_probs'] = model.predict_proba(X_test_US)[0][1]
                else:
                    self.y_test_label.loc[vintage_date, f'{model_name}_label'] = model.predict(X_test_US)[0]
                    self.y_test_probs.loc[vintage_date, f'{model_name}_probs'] = model.predict_proba(X_test_US)[0][1]

                self.models[model_name] = model


        # --- Threshold tuning --- 
        if threshold_tuning:
            threshold_dict = get_optimal_thresholds(self.models, X_train_US, Y_train_US)
            for model_name, thresholds in threshold_dict.items():
                self.opt_threshold[model_name].loc[vintage_date] = thresholds



    def fit(self, threshold_tuning = None, models: list = None, optimize: bool = False, cv: int = 3, split_expanding=False, normalize=True, bootstrap_blocks=0, epochs=15):

        """
        Train the models with predefined hyperparameters.   
        """

        #None par défaut : on entraine tous les modèles
        if models is None:
            models = ['RF', 'GB', 'ADA', 'ENET', 'SVC', 'KNN', 'MLP']

        #Initialisation de variales utiles au split expanding
        X_train_old, y_train_old = pd.DataFrame(), pd.DataFrame()
        assert X_train_old.shape == (0, 0)

        index, columns = self.db.data.index, list(self.db.data.columns) + [f'y_lag_{lag}' for lag in range(1, self.db.y_lags + 1)]

        # --- Initialisation des dataframes stockant les résultats ---
        self.y_test_label = pd.DataFrame(np.nan, index=index, columns=['target'])
        self.y_test_probs = pd.DataFrame(np.nan, index=index, columns=['target'])
        self.train_loss = pd.DataFrame(np.nan, index=index, columns=list(range(epochs)))
        self.val_loss = pd.DataFrame(np.nan, index=index, columns=list(range(epochs)))
        
        # --- Var Imp --- Pour le moment on ne s'en préoccupe pas mais après à mettre sous self. et enregistrer
        models_with_var_imp = ['RF', 'GB', 'ADA', 'ENET']
        self.var_importances = {model: pd.DataFrame(np.nan, index=index, columns=columns) for model in models if model in models_with_var_imp}

        # --- Threshold tuning ---
        if threshold_tuning:
            metrics = ['log_score', 'g_means', 'J', 'f1_score']
            self.opt_threshold = {model: pd.DataFrame(np.nan, index=index, columns=metrics) for model in models}  

        # clé = (date, model_name), valeur = DataFrame
        self.gridsearch_reports = {}
        
        # --- Training models ---
        for vintage_date in tqdm(list(self.db.tp.columns)[:-1], desc="Training models"):

            if split_expanding: 
                self.sets, X_train_old, y_train_old = self.db.split_expanding(date = vintage_date, X_train_old=X_train_old, y_train_old=y_train_old)
            else:
                self.sets = self.db.nowcast_split(vintage_date)

            if isinstance(self.sets[-2], pd.Series): #Met X_test en forme 2D pour compatibilité
               X_train, y_train, X_test, y_test = self.sets
               X_test = X_test.to_frame().T
               self.sets = X_train, y_train, X_test, y_test
                
            #On stocke la target pour le vintage en cours
            self.y_test_label.loc[vintage_date, 'target'], self.y_test_probs.loc[vintage_date, 'target'] = self.sets[-1], self.sets[-1] 

            # --- Store models --- 
            self.models = {}

            if bootstrap_blocks>0:
                self.block_bootstrap(n_samples=bootstrap_blocks)

            if normalize:
                self.normalize_data()
                #print('data_normalized')

            if bootstrap_blocks>0: #block bootstrap avec un méta modèle de plusieurs modèles (forest de modèles) 
                for i, sets in self.bb_sets.items():
                    self.bb_iteration = i #gérer les doublons de colonne et faire ensuite une moyenne sur le dataset_probs
                    self.sets = sets
                    self.training_shot(models=models, vintage_date=vintage_date, optimize=optimize, cv=cv, epochs=epochs)
                    
                self.bb_iteration=None
                self.aggregate_predictions(models=models, vintage_date=vintage_date, method='hard', drop_old=False)      
            else:
                self.training_shot(models=models, vintage_date=vintage_date, optimize=optimize, cv=cv, epochs=epochs)


    

        # --- Store results ---
        self.y_test_probs.dropna(how='all', inplace=True) #On drop les lignes vides
        self.y_test_label.dropna(how='all', inplace=True)
        self.train_loss.dropna(how='all', inplace=True)
        self.val_loss.dropna(how='all', inplace=True)

    














