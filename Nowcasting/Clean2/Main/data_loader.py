import os
import numpy as np
import pandas as pd
from statsmodels.tsa.stattools import adfuller
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.decomposition import PCA
import statsmodels.api as sm
import logging
from IPython.display import display
from tqdm import tqdm

logging.basicConfig(level=logging.INFO)



'''
Ajouter preprocessing du notebbok modelling (filtre tp et data sur la même plage de données)
Implémenter l'utilisation du cli plutot que des tp pour une stratégie de trading en update de poids continus (plus tard)

nowcast split : 
split_per_revision :
split_expanding :
'''

def in_notebook() -> bool:

    '''
    Détecte l'environnement d'éxecution (notebook ou fichier python classique py)
    '''

    try:
        from IPython import get_ipython
        shell = get_ipython().__class__.__name__
        if shell == 'ZMQInteractiveShell':
            return True   # Jupyter notebook ou JupyterLab
        else:
            return False  # Autre environnement
    except (NameError, ImportError):
        return False      # Pas IPython donc script classique
    


#On pourra après ajouter tout le preprocessing qui a permis de passer de data à data_croped 
#pour avoir une pipeline plus flexible mais ce n'est pas le plus urgent actuellement

class Data:
    def __init__(self, 
                 BDD_path: str = '../Data/data_croped.csv', 
                 BDD_sheet: str = None, 
                 turning_points: str = '../Data/cli_diff_bin_vintage.csv', # '../Data/tp_croped.csv' ou '../Data/cli_diff_bin_vintage.csv'
                 TP_sheet: str = None,
                 y_lags = 0,
                 cli = True,
                 Verbose = True):  #Verbose à développer
        
        """
        Initialise l'objet Data en chargeant les fichiers de features et de turning points.

        Args:
            BDD_path (str): Chemin vers le fichier de données principales (.csv ou .xlsx).
            BDD_sheet (str, optional): Feuille Excel pour les features si besoin.
            turning_points (str): Chemin vers le fichier de turning points (.csv ou .xlsx).
            TP_sheet (str, optional): Feuille Excel pour les turning points si besoin.
        Raises:
            FileNotFoundError: Si un fichier n'existe pas.
            ValueError: Si un format non supporté est rencontré.
        """

        base_dir = os.path.dirname(os.path.abspath(__file__))

        # ------- Gestion robuste des chemins relatifs -------
        if not os.path.isabs(BDD_path):
            BDD_path = os.path.normpath(os.path.join(base_dir, BDD_path))
        if not os.path.isabs(turning_points):
            turning_points = os.path.normpath(os.path.join(base_dir, turning_points))

        if not os.path.exists(BDD_path):
            raise FileNotFoundError(f"Fichier introuvable : {BDD_path}")
        if not os.path.exists(turning_points):
            raise FileNotFoundError(f"Fichier introuvable : {turning_points}")


        # ------- Lecture des données de features -------
        extension = os.path.splitext(BDD_path)[1].lower()
        if extension == '.csv':
            self.data = pd.read_csv(BDD_path, index_col=0)
        elif extension in ['.xls', '.xlsx']:
            if BDD_sheet is None:
                raise ValueError("BDD_sheet doit être spécifié pour un fichier Excel de features.")
            self.data = pd.read_excel(BDD_path, sheet_name=BDD_sheet, index_col=0)
        else:
            raise ValueError(f"Extension {extension} non supportée pour les features.")

        self.data.index = pd.to_datetime(self.data.index, errors='coerce').map(lambda x: x.replace(day=15))
        if self.data.index.isnull().any():
            raise ValueError("Index de 'data' contient des dates invalides après conversion.")


        # -------  Lecture des turning points -------
        extension_tp = os.path.splitext(turning_points)[1].lower()
        if extension_tp == '.csv':
            self.tp = pd.read_csv(turning_points, index_col=0)
        elif extension_tp in ['.xls', '.xlsx']:
            if TP_sheet is None:
                raise ValueError("TP_sheet doit être spécifié pour un fichier Excel de turning points.")
            self.tp = pd.read_excel(turning_points, sheet_name=TP_sheet, index_col=0)
        else:
            raise ValueError(f"Extension {extension_tp} non supportée pour les turning points.")

        self.tp.index = pd.to_datetime(self.tp.index, errors='coerce').map(lambda x: x.replace(day=15))
        if self.tp.index.isnull().any():
            raise ValueError("Index de 'tp' contient des dates invalides après conversion.")
        
        self.tp.columns = pd.to_datetime(self.tp.columns, errors='coerce').map(lambda x: x.replace(day=15))
        if self.tp.columns.isnull().any():
            raise ValueError("Certaines colonnes de 'tp' contiennent des dates invalides après conversion.")


        # Vérifie et corrige si nécessaire les colonnes mal alignées
        self.align_tp_columns_index(Verbose=Verbose)
        #Shaping and croping the two dataframes
        self.cropping(Verbose=Verbose)
        # Vérification de la continuité mensuelle des données
        self.ensure_monthly_continuity(Verbose=Verbose)


        # Copie de self.data pour éviter les modifications non désirées
        self.features = self.data.copy()
        self.y_lags=y_lags

    
    def cropping(self, Verbose=False):
        """
        Algorithme de synchronisation temporelle entre deux datasets
        """

        # 3. Détermination de la période de synchronisation
        # Calcul de la date de début commune (maximum des minimums)
        min_date = max(self.tp.index.min(), self.data.index.min())

        # Calcul de la date de fin commune (minimum des maximums)
        max_date = min(self.tp.index.max(), self.data.index.max())

        # 4. Synchronisation des données
        # Filtrage des données après la date de début commune
        tp_sync = self.tp[self.tp.index >= min_date]
        data_sync = self.data[self.data.index >= min_date]

        # Filtrage de tp sur la période définie
        # Filtrage des lignes (index) jusqu'à la date de fin
        tp_sync = tp_sync[tp_sync.index <= max_date]

        # Filtrage des colonnes jusqu'à la date de fin
        tp_sync = tp_sync.loc[:, tp_sync.columns <= max_date]

        self.data = data_sync
        self.tp = tp_sync

        if Verbose:
            print("[INFO] Période de synchronisation calculée :")
            print(f" - de {min_date} à {max_date}")

        
    def align_tp_columns_index(self, Verbose=False):

        """
        Aligne les dates des colonnes de self.tp avec les last_valid_index de chaque colonne.
        Si un décalage constant en mois est détecté, les colonnes sont corrigées automatiquement
        avec pandas.DateOffset.
        
        Args:
            Verbose (bool): Affiche les informations de correction si True.
        Raises:
            ValueError: Si les décalages sont incohérents ou multiples.
        """

        offsets = []
        
        for col in self.tp.columns:
            last_valid = self.tp[col].last_valid_index()
            if last_valid is None:
                continue  # Colonne vide, on ignore
            delta = (col.year - last_valid.year) * 12 + (col.month - last_valid.month)
            offsets.append(delta)

        unique_offsets = set(offsets)
        
        if len(unique_offsets) == 1 and list(unique_offsets)[0] != 0:
            offset_months = list(unique_offsets)[0]
            
            if Verbose:
                print(f"[INFO] Décalage détecté de {offset_months} mois entre colonnes et leur last_valid_index.")
                print("[INFO] Correction automatique des dates de colonnes en cours...")

            self.tp.columns = self.tp.columns + pd.DateOffset(months=offset_months)

            if Verbose:
                print("[INFO] Correction des colonnes effectuée.")
        elif len(unique_offsets) > 1:
            raise ValueError("Décalages multiples détectés entre colonnes et last_valid_index dans 'tp'. Vérifiez les données.")

    def ensure_monthly_continuity(self, Verbose=True):
        """
        Vérifie que les index et colonnes de `self.data` et `self.tp` ont une fréquence mensuelle continue.
        Insère les dates manquantes avec les valeurs de la période précédente (forward-fill manuelle).

        Args:
            Verbose (bool): Si True, imprime les dates corrigées et les actions entreprises.
        """

        def fill_missing_months(df, axis_name):
            """Utilitaire pour remplir les mois manquants sur un index donné (ligne ou colonne)."""
            full_range = pd.date_range(start=df.index.min(), end=df.index.max(), freq='MS') + pd.DateOffset(days=14)
            missing = full_range.difference(df.index)
            if not missing.empty and Verbose:
                print(f"[INFO] Dates manquantes dans l'index de {axis_name} : {[d.strftime('%Y-%m-%d') for d in missing]}")
            
            for date in missing:
                prev_date = date - pd.DateOffset(months=1)
                if prev_date in df.index:
                    new_row = df.loc[prev_date].copy()
                    df.loc[date] = new_row
                    if Verbose:
                        print(f"[FIX] Ligne ajoutée à {axis_name} pour {date.date()}, copiée depuis {prev_date.date()}")
                else:
                    df.loc[date] = np.nan
                    if Verbose:
                        print(f"[WARN] Ligne ajoutée vide à {axis_name} pour {date.date()}, car {prev_date.date()} manquant")
            return df.sort_index()

        # Correction index de self.data et self.tp
        self.data = fill_missing_months(self.data, 'data')
        self.tp = fill_missing_months(self.tp, 'tp')

        # Correction colonnes de self.tp
        col_range = pd.date_range(start=self.tp.columns.min(), end=self.tp.columns.max(), freq='MS') + pd.DateOffset(days=14)
        missing_cols = col_range.difference(self.tp.columns)

        if not missing_cols.empty and Verbose:
            print(f"[INFO] Colonnes manquantes dans `tp` : {[d.strftime('%Y-%m-%d') for d in missing_cols]}")

        for date in missing_cols:
            prev_col = date - pd.DateOffset(months=1)
            next_col = date + pd.DateOffset(months=1)
            if prev_col in self.tp.columns:
                new_col = self.tp[prev_col].copy()
                self.tp[date] = new_col

                # Remplacement croisé (diagonale) : chercher la prochaine colonne non vide à droite
                if date in self.tp.index:
                    # Colonnes futures après `date`, triées
                    future_cols = [col for col in self.tp.columns if col > date] #pas vraiment besoin de boucle for par construction de cette liste, mais bon ça rend plus robuste au cas où
                    replaced = False

                    for col_to_check in future_cols:
                        val = self.tp.loc[date, col_to_check]
                        #print(f"Checking ({date}, {col_to_check}): {val}")
                        if pd.notna(val):
                            self.tp.loc[date, date] = val
                            if Verbose:
                                print(f"[FIX] Cellule diagonale ({date.date()}, {date.date()}) remplacée par celle de ({date.date()}, {col_to_check.date()})")
                            replaced = True
                            break

                    if not replaced and Verbose:
                        print(f"[WARN] Impossible de remplacer la cellule diagonale ({date.date()}, {date.date()}) : aucune colonne suivante non vide trouvée")

                elif Verbose:
                    print(f"[WARN] Impossible de remplacer la cellule diagonale ({date.date()}, {date.date()}) : index date absente")
            else:
                if Verbose:
                    print(f"[WARN] Colonne {date.date()} impossible à remplacer car {prev_col.date()} manquant")


        # Vérification 1 : fréquence mensuelle continue
        def is_monthly_continuous(dates, name=""):
            dates = pd.DatetimeIndex(dates).sort_values()

            # Recalage correct de la fenêtre pour '15 du mois'
            start = (dates.min() - pd.DateOffset(days=14)).replace(day=1)
            end = (dates.max() - pd.DateOffset(days=14)).replace(day=1)

            expected = pd.date_range(start=start, end=end, freq='MS') + pd.DateOffset(days=14)
            expected = pd.DatetimeIndex(expected)

            missing = expected.difference(dates)
            extra = dates.difference(expected)

            if missing.empty and extra.empty:
                if Verbose:
                    print(f"[✅] {name} : succession mensuelle continue")
                return True
            else:
                if Verbose:
                    print(f"[❌] {name} :")
                    if not missing.empty:
                        print(f"   - Dates manquantes : {[d.strftime('%Y-%m-%d') for d in missing]}")
                    if not extra.empty:
                        print(f"   - Dates en trop : {[d.strftime('%Y-%m-%d') for d in extra]}")
                return False

        # Vérification 2 : diagonale de tp bien remplie
        def diagonal_valid(tp):
            invalid_dates = []
            for date in tp.columns:
                if date in tp.index:
                    val = tp.loc[date, date]
                    if pd.isna(val):
                        invalid_dates.append(date)

            if not invalid_dates:
                if Verbose:
                    print("[✅] Diagonale de `tp` : toutes les cellules (date, date) sont remplies")
                return True
            else:
                if Verbose:
                    print("[❌] Diagonale de `tp` : cellules vides pour les dates :")
                    for d in invalid_dates:
                        print(f"   - {d.strftime('%Y-%m-%d')}")
                return False

        # === Lancer les vérifications ===
        index_ok = is_monthly_continuous(self.tp.index, name='tp_index')
        columns_ok = is_monthly_continuous(self.tp.columns, name='tp_columns')
        data_ok = is_monthly_continuous(self.data.index, name='data_index')
        diagonal_ok = diagonal_valid(self.tp)

        # === Résumé final ===
        if index_ok and columns_ok and data_ok and diagonal_ok:
            if Verbose:
                print("[✅] Succession mensuelle des dates assurée pour `data` et `tp` (index, colonnes, diagonale)")
        else:
            if Verbose:
                print("[❌] Problèmes détectés dans la structure temporelle :")
                if not index_ok:
                    print("   - ❌ Index de `tp` non mensuel ou incomplet")
                if not columns_ok:
                    print("   - ❌ Colonnes de `tp` non mensuelles ou incomplètes")
                if not data_ok:
                    print("   - ❌ Index de `data` non mensuel ou incomplet")
                if not diagonal_ok:
                    print("   - ❌ Certaines cellules diagonales (date, date) dans `tp` sont manquantes ou vides")


    

    def data_type_and_sample(self, resample: bool = False) -> None: #pas le truc le plus utile mais bon
        """
        Transforme les données en float (features) et int (turning points),
        rééchantillonne à fréquence journalière si demandé.

        Args:
            resample (bool): Si True, rééchantillonne à fréquence journalière ('D').
        """

        try:
            self.data = self.data.astype(float)
            self.tp = self.tp.astype('Int64')
        except Exception as e:
            logging.error(f"Erreur lors de la conversion des données: {e}")
            raise

        if resample:
            try:
                self.data = self.data.resample('D', axis=0).interpolate('linear')
                self.tp = self.tp.resample('D', axis=0).ffill()
            except Exception as e:
                logging.error(f"Erreur lors du resampling: {e}")
                raise

        logging.info("Data processed successfully.")



    def covariates(self, returns: bool = False, log: bool = False, ts_analysis: bool = False, diff: bool = False, dates_columns: bool = True) -> pd.DataFrame:

        """
        Génère les covariables (features) selon différentes transformations.

        Args:
            returns (bool): Si True, ajoute les rendements (ou log-rendements).
            log (bool): Si True et returns=True, utilise log-rendements.
            ts_analysis (bool): Si True, ajoute moyennes mobiles et décomposition saisonnière.
            diff (bool): Si True, ajoute les différences successives (lag).

        Returns:
            pd.DataFrame: DataFrame enrichi en nouvelles covariables.
        """
        print('data shape before covariates', self.data.shape)


        
        self.data = self.features

        if returns:
            aux = self.features.loc[:, (self.features > 0).all()]
            if log:
                df_returns = np.log(aux) - np.log(aux.shift(1))
                df_returns.columns = [col + '_log_change' for col in df_returns.columns]
            else:
                df_returns = aux.pct_change()
                df_returns.columns = [col + '_change' for col in df_returns.columns]

            # On remplace la première ligne qui est NaN
            df_returns.iloc[0] = df_returns.iloc[1]
            # Merge features + returns
            self.data = pd.concat([self.data, df_returns], axis=1)
            

        if ts_analysis:
            rolling = self.features.rolling(window=3).mean()
            rolling.columns = [col + '_rolling_avg' for col in self.features.columns]
            self.data = pd.concat([self.data, rolling], axis=1)

            # Dictionnaire pour stocker toutes les colonnes trend/seasonal/residual
            decomposition_dict = {}

            for col in self.features.columns:
                try:
                    decomposition = sm.tsa.seasonal_decompose(self.features[col], model='additive', period=3)
                    decomposition_dict[f"{col}_trend"] = decomposition.trend
                    decomposition_dict[f"{col}_seasonal"] = decomposition.seasonal
                    decomposition_dict[f"{col}_residual"] = decomposition.resid
                except Exception:
                    continue

            # Une seule concat pour éviter fragmentation
            df_decomp = pd.DataFrame(decomposition_dict, index=self.features.index)
            self.data = pd.concat([self.data, df_decomp], axis=1)

            self.data.bfill(inplace=True)
            self.data.ffill(inplace=True)



        if diff:
            diff_features = {}

            for col in self.features.columns:
                for lag in range(1, 51):
                    diff_features[f"{col}_diff_{lag}"] = self.features[col].diff(lag)

            # Création d'un DataFrame unique, puis concaténation une fois
            df_diff = pd.DataFrame(diff_features, index=self.features.index)
            self.data = pd.concat([self.data, df_diff], axis=1)

            self.data.bfill(inplace=True)

        if dates_columns:
            #A mettre dans le dataframe de base
            # Ajout des covariables temporelles : mois et année
            self.data['mois'] = self.data.index.month
            self.data['annee'] = self.data.index.year

            # Encodage cyclique du mois (sinus/cosinus)
            self.data['mois_sin'] = np.sin(2 * np.pi * self.data['mois'] / 12)
            self.data['mois_cos'] = np.cos(2 * np.pi * self.data['mois'] / 12) 

        print('data shape after covariates', self.data.shape)
        print('total number of covariates with y_lags = ', self.data.shape[-1] + self.y_lags)

    
    
    def nowcast_split(self, date=None):

        if date is None:
            date = self.tp.columns[-2]

        date = pd.to_datetime(date)
        next_date = date + pd.DateOffset(months=1)

        y_train = self.tp[date].dropna(how='all')

        X_train = self.data.loc[:date].copy()

        # === Ajout des lags de y (1 à 12 mois)
        for lag in range(1, self.y_lags+1):
            lagged_series = self.tp.shift(lag)[date]
            X_train[f'y_lag_{lag}'] = lagged_series.bfill()

        X_test = self.data.loc[next_date].copy().to_frame().T
        for lag in range(1, self.y_lags+1):
            try:
                lagged_value = self.tp.shift(lag).loc[next_date, date]
            except KeyError:
                lagged_value = np.nan

            if pd.isna(lagged_value):
                lagged_value = self.tp[date].ffill().bfill().iloc[-1]

            X_test[f'y_lag_{lag}'] = lagged_value

        # === y_test
        y_test = self.tp.loc[next_date, next_date]

        # === Vérification : quelles structures contiennent des NaN ?
        problèmes = []

        if X_train.isna().any().any():
            problèmes.append("X_train")
        if X_test.isna().any().any():
            problèmes.append("X_test")
        if y_train.isna().any():
            problèmes.append("y_train")
        if pd.isna(y_test):
            problèmes.append("y_test")

        # Détection fine des NaN dans X_train
        nan_mask = X_train.isna()

        # Lignes et colonnes concernées
        nan_positions = nan_mask[nan_mask].stack().index.tolist()

        if nan_positions:
            print(f"🚨 {len(nan_positions)} NaN détectés dans X_train aux positions suivantes (ligne, colonne) :")
            for row, col in nan_positions:
                print(f" - Ligne: {row}, Colonne: {col}")

        if problèmes:
            raise ValueError(f"🚨 Des NaN ont été détectés dans : {', '.join(problèmes)} (split {date})")

        sets = X_train, y_train, X_test, y_test
        

        return sets





    def split_expanding(self, date, X_train_old, y_train_old): 

        X_train, y_train, X_test, y_test = self.nowcast_split(date)

        if X_train_old.shape[0]>0:
            X_train = pd.concat([X_train_old,X_train], axis = 0)

        y_train.name = "target"

        if y_train_old.shape[0]>0:
            y_train = pd.concat([y_train_old,y_train], axis = 0)

        X_train_old = X_train
        y_train_old = y_train

        sets = X_train, y_train, X_test, y_test

        return sets, X_train_old, y_train_old



    def data_summary(self, date = None) -> None:

        """
        Affiche un résumé rapide des données (features + target),
        en s'adaptant à l'environnement (Jupyter ou script classique).
        """

        logging.info("Affichage de l'aperçu des données (features).")
        if in_notebook():
            display(self.data.head())
        else:
            print(self.data.head())

        logging.info("Affichage des statistiques descriptives (features).")
        if in_notebook():
            display(self.data.describe())
        else:
            print(self.data.describe())

        if date is None :
            logging.info("Affichage de la répartition de la target.")
            print(self.tp.sum().sum()/self.tp.count().sum())

        elif date is not None:
            date = pd.to_datetime(date, errors='coerce')
            logging.info("Affichage de la répartition de la target.")
            print(self.tp[date].sum().sum()/self.tp.count().sum())





    def stationarity(self, date=None, signif: float = 0.05) -> pd.Series or dict:

        """
        Teste si la série temporelle target(date) est stationnaire (Dickey-Fuller Test).
        Si date = 'all', teste toutes les dates disponibles.

        Args:
            date (datetime-like, str, optional): Date cible. Si 'all', teste toutes les dates.
            signif (float, optional): Seuil de significativité pour conclure (par défaut 5%).

        Returns:
            - Si date unique: Résultats détaillés du test ADF (pd.Series).
            - Si 'all': Dictionnaire {date: p-value} pour les séries non-stationnaires uniquement.
        """

        if date == None:
            logging.info("Démarrage du test Dickey-Fuller pour toutes les dates disponibles dans tp.")

            non_stationary = {}

            for col in self.tp.columns:
                y = self.tp[col]
                dftest = adfuller(y.dropna(), autolag='AIC')
                pvalue = dftest[1]

                if pvalue > signif:
                    non_stationary[col] = pvalue

            if not non_stationary:
                logging.info("✅ Toutes les séries sont stationnaires au seuil choisi.")
            else:
                logging.info(f"❌ Colonnes non-stationnaires trouvées ({len(non_stationary)}): {non_stationary}")

            return non_stationary

        else:
            date = pd.to_datetime(date, errors='coerce')

            if date not in self.tp.columns:
                raise ValueError(f"La date {date} n'est pas présente dans les colonnes de tp.")
            
            logging.info(f"Démarrage du test Dickey-Fuller pour la target à la date {date}.")

            y = self.tp[date]
            dftest = adfuller(y.dropna(), autolag='AIC')

            dfoutput = pd.Series(dftest[0:4], index=["Statistique de test", "p-value", "lags utilisés", "nobs"])
            for key, value in dftest[4].items():
                dfoutput[f"Valeur critique ({key})"] = value

            logging.info(f"Résultats du test Dickey-Fuller:\n{dfoutput}")

            if dfoutput["p-value"] < signif:
                logging.info(f"✅ La série est STATIONNAIRE au seuil de {signif*100:.1f}%.")
            else:
                logging.info(f"❌ La série est NON-STATIONNAIRE au seuil de {signif*100:.1f}%.")

            return dfoutput



    '''def split_per_revision(self): old

        sets = {}
        for date in tqdm(self.tp.columns[:-1], desc = 'Vintage Train Test Split per revision'):#on va pas prédire avec la dernière vintage car on n'a pas de target
            X_train, y_train, X_test, y_test = self.nowcast_split(date)
            sets[date] = (X_train, y_train, X_test, y_test)
        return sets
    '''



if __name__ == "__main__":
    db = Data(Verbose=True)


