import os
import sys
import json
from pathlib import Path
# Ajouter le dossier Clean au path pour que utils/ soit trouvé
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import yahoofinancials as yf
import pickle
from datetime import datetime
from matplotlib import pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, balanced_accuracy_score, brier_score_loss, f1_score,\
    roc_auc_score, roc_curve, confusion_matrix ,mean_squared_error, precision_recall_curve, log_loss
from mlxtend.evaluate import mcnemar_table, mcnemar
from tqdm import tqdm
import re










def plot_predictions(y_test_probs: pd.DataFrame, optimal_threshold=None, models: list = None, save_path=None):
    """
    Plots predicted probabilities of multiple models over time using a single DataFrame.
    Supports fixed or time-varying thresholds.

    Args:
        models (list, optional): Liste des modèles à afficher.
            - Noms sous forme de chaînes de caractères : ['RF', 'GB', 'ENET', 'ADA', 'SVC', 'KNN', 'MLP'])
            - Si None, tous les modèles seront affichés.
        y_test_probs (pd.DataFrame): DataFrame contenant :
            - une colonne 'target' avec les vraies classes (0/1)
            - plusieurs colonnes *_probs avec les probabilités prédites
        optimal_threshold (float or pd.Series, optional): 
            - seuil fixe (float)
            - ou série temporelle de seuils (même index que y_test_probs)
            - par défaut : 0.5

    Returns:
        None
    """

    fig, ax = plt.subplots(figsize=(12, 8))

    # Récupère les colonnes des modèles
    model_cols = [col for col in y_test_probs.columns if col.endswith('_probs')]
    y_label = y_test_probs['target']

    if models is not None:
        model_cols = [col for col in model_cols if col.split('_')[0] in models]
    for col in model_cols:
        ax.plot(y_test_probs.index, y_test_probs[col], label=col.replace('_probs', ''))

    # Gestion du seuil
    if optimal_threshold is None:
        ax.axhline(0.5, color='grey', lw=2, linestyle='--', alpha=0.7, label="Threshold = 0.5")
    elif isinstance(optimal_threshold, pd.Series):
        ax.plot(optimal_threshold.index, optimal_threshold, color='grey', lw=2, linestyle='--', alpha=0.7, label='Optimal Threshold')
    else:
        ax.axhline(optimal_threshold, color='grey', lw=2, linestyle='--', alpha=0.7, label=f"Threshold = {optimal_threshold}")

    # Coloration du fond selon la vraie target
    ax.fill_between(y_label.index, 0, 1, where=y_label == 1,
                    color='green', alpha=0.1, transform=ax.get_xaxis_transform(), label='True: 1')
    ax.fill_between(y_label.index, 0, 1, where=y_label == 0,
                    color='red', alpha=0.1, transform=ax.get_xaxis_transform(), label='True: 0')

    ax.set_title('Predicted Probabilities Over Time')
    ax.set_xlabel('Time')
    ax.set_ylabel('Probability')
    ax.legend(loc='best')
    plt.tight_layout()
    if save_path is not None:
        plt.savefig(save_path, bbox_inches='tight')
    else:
        plt.show()




def plot_variable_importances(var_importances: dict, top_n: int = 10, save_path=None):
    """
    Affiche les 10 variables les plus importantes pour chaque modèle
    à partir d'un dictionnaire de DataFrames de variable importance.

    Args:
        var_importances (dict): Dictionnaire {model_name: DataFrame}
                                où chaque DataFrame est indexé par date, colonnes = variables
        top_n (int): Nombre de variables les plus importantes à afficher (par moyenne)

    Returns:
        None
    """

    n_models = len(var_importances)

    if n_models == 0:
        print("Aucun modèle avec des importances de variable disponibles.")
        return

    fig, axes = plt.subplots(1, n_models, figsize=(5 * n_models, 6), sharey=True)

    if n_models == 1:
        axes = [axes]  # for compatibility

    for i, (model_name, df) in enumerate(var_importances.items()):
        mean_imp = df.mean().abs()
        top_features = mean_imp.sort_values(ascending=False).head(top_n)

        # Standardisation des importances
        top_features_normalized = top_features / top_features.max()

        axes[i].barh(top_features_normalized.index[::-1], top_features_normalized.values[::-1])
        axes[i].set_title(f"{model_name} - Top {top_n} Features")
        axes[i].set_xlabel("Relative Importance (standardized)")
        axes[i].grid(True, axis='x')

    plt.tight_layout()
    if save_path is not None:
        plt.savefig(save_path, bbox_inches='tight')
    else:
        plt.show()



def date(date_str: str) -> pd.Timestamp:
    """
    Convertit une chaîne contenant une année et un mois (dans n'importe quel ordre ou séparateur)
    en un Timestamp pandas correspondant au 15 du mois à 00:00:00.
    
    Exemples acceptés : '01 2017', '2017 01', '2017-01', '2017/01', 'Jan 2017', '2017 January'
    """
    date_str = date_str.strip().lower()
    
    # Essayons d'abord via pandas to_datetime (qui gère bien les formats flexibles)
    try:
        parsed_date = pd.to_datetime(date_str, errors='raise', dayfirst=False)
        return pd.Timestamp(year=parsed_date.year, month=parsed_date.month, day=15)
    except Exception:
        pass
    
    # Sinon on essaie d'extraire les entiers via regex
    numbers = re.findall(r'\d+', date_str)
    if len(numbers) != 2:
        raise ValueError("Impossible d'extraire mois et année de la chaîne.")

    nums = sorted([int(n) for n in numbers])  # année sera le plus grand nombre
    year = max(nums)
    month = min(nums)
    if not (1 <= month <= 12):
        raise ValueError(f"Mois invalide : {month}")

    return pd.Timestamp(year=year, month=month, day=15)





def make_confusion_matrices(cfs: list,
                            group_names=None,
                            categories: list = None,
                            count=True,
                            percent=True,
                            cbar=True,
                            xyticks=True,
                            xyplotlabels=True,
                            sum_stats=True,
                            figsize=(30, 10),
                            cmap='Blues',
                            title=None,
                            labels= None,
                            preds: list = None,
                            save_path=None):

    blanks = ['' for i in range(cfs[0].size)]

    if group_names and len(group_names) == cfs[0].size:
        group_labels = ["{}\n".format(value) for value in group_names]
    else:
        group_labels = blanks

    fig, axs = plt.subplots(1, len(cfs), figsize=figsize, constrained_layout=True)
    if len(cfs) == 1:
        axs = [axs]  # Convertir en liste pour rester cohérent


    for i, cf in enumerate(cfs):
        if count:
            group_counts = ["{0:0.0f}\n".format(value) for value in cf.flatten()]
        else:
            group_counts = blanks

        if percent:
            group_percentages = ["{0:.2%}".format(value) for value in cf.flatten() / np.sum(cf)]
        else:
            group_percentages = blanks

        box_labels = [f"{v1}{v2}{v3}".strip() for v1, v2, v3 in zip(group_labels, group_counts, group_percentages)]
        box_labels = np.asarray(box_labels).reshape(cf.shape[0], cf.shape[1])

        # CODE TO GENERATE SUMMARY STATISTICS & TEXT FOR SUMMARY STATS
        if sum_stats:

            # Accuracy is sum of diagonal divided by total observations
            accuracy = np.trace(cf) / float(np.sum(cf))

            # Metrics for Binary Confusion Matrices
            TP = cf[1, 1]
            TN = cf[0, 0]
            FP = cf[0, 1]
            FN = cf[1, 0]

            precision = TP / (TP + FP)  # Précision
            recall = TP / (TP + FN)     # Sensibilité / rappel
            specificity = TN / (TN + FP)  # Spécificité


            f1_score_ = f1_score(y_true=labels, y_pred=preds[i])
            mse = mean_squared_error(labels, preds[i])
            roc = roc_auc_score(labels, preds[i])
            stats_text = (f"\n\nAccuracy={accuracy:.3f}"
                f"\nPrecision={precision:.3f}"
                f"\nspecificity={specificity:.3f}"
                f"\nsensitivity={recall:.3f}"
                f"\nF1 Score={f1_score_:.3f}"
                f"\nMse={mse:.3f}"
                f"\nROC_AUC Score={roc:.3f}"
            )


        # SET FIGURE PARAMETERS ACCORDING TO OTHER ARGUMENTS
        if xyticks == False:
            # Do not show categories if xyticks is False
            categories = False

        # MAKE THE HEATMAP VISUALIZATION
        sns.heatmap(cf, annot=box_labels, fmt="", cmap=cmap, cbar=cbar, xticklabels=categories, yticklabels=categories,
                    ax=axs[i], square = True)

        if xyplotlabels:
            axs[i].set_ylabel('True label')
            axs[i].set_xlabel('Predicted label' + stats_text)
        else:
            axs[i].set_xlabel(stats_text)

        if title:
            axs[i].set_title(title[i])

    if save_path is not None:
        plt.savefig(save_path, bbox_inches='tight')
    else:
        plt.show()



def show_confusion_matrix(y_test_label: pd.DataFrame, models: list = None, save_path=None):
    """
    Génère les matrices de confusion pour tous les modèles contenus dans un DataFrame
    structuré avec une colonne 'target' et plusieurs colonnes *_label.

    Args:
        models (list, optional): Liste des modèles à afficher.
            - Noms sous forme de chaînes de caractères : ['RF', 'GB', 'ENET', 'ADA', 'SVC', 'KNN', 'MLP'])
            - Si None, tous les modèles seront affichés.
        y_test_label (pd.DataFrame): DataFrame des labels avec colonnes :
            - 'target' : labels réels
            - '{MODEL}_label' : prédictions binaires de chaque modèle

    Returns:
        None
    """

    # --- Extraction automatique
    label_cols = [col for col in y_test_label.columns if col.endswith('_label')]
    if models is not None:
        label_cols = [col for col in label_cols if col.split('_')[0] in models]
    names = [col.replace('_label', '') for col in label_cols]
    y_true = y_test_label['target']
    y_preds = [y_test_label[col] for col in label_cols]

    # --- Matrices de confusion
    confusion_dfs = [confusion_matrix(y_true, pred) for pred in y_preds]

    # --- Affichage
    make_confusion_matrices(
        cfs=confusion_dfs,
        categories=["Slowdown", "Acceleration"],
        group_names=["True Neg", "False Pos", "False Neg", "True Pos"],
        labels=y_true,
        preds=y_preds,
        title=names,
        save_path=save_path
    )
    



def save_results_and_plots(process, normalize, optimize, split_expanding, models, returns, log, ts_analysis, diff, bootstrap_blocks, epochs, threshold=None):
    """
    Sauvegarde les résultats, modèles et graphiques dans un sous-dossier de 'Clean/results',
    basé sur les paramètres d'entraînement.
    """

    cov = returns and ts_analysis and diff

    # Remonter d'un niveau pour viser le dossier 'Clean'
    base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    results_dir = os.path.join(base_dir, "results")

    # Créer le nom du dossier avec timestamp
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    models_str = '_'.join(models) if models else 'all'
    folder_name = os.path.join(
        results_dir,
        f"{models_str}_norm-{normalize}_splitex-{split_expanding}_bb-{bootstrap_blocks}_opti-{optimize}_cov-{cov}_log-{log}_{timestamp}"
    )

    os.makedirs(folder_name, exist_ok=True)

    # Sauvegarder les fichiers CSV
    process.y_test_probs.to_csv(os.path.join(folder_name, "y_test_probs.csv"))
    process.y_test_label.to_csv(os.path.join(folder_name, "y_test_label.csv"))



    for model_name, df in process.var_importances.items():
        df.to_csv(os.path.join(folder_name, f"var_importances_{model_name}.csv"))


    # Sauvegarder les modèles
    with open(os.path.join(folder_name, "trained_models.pkl"), "wb") as f:
        pickle.dump(process.models, f)

    # Sauvegarder les figures
    plot_predictions(process.y_test_probs, optimal_threshold=threshold,
                     save_path=os.path.join(folder_name, "plot_predictions.png"))

    show_confusion_matrix(process.y_test_label,
                          save_path=os.path.join(folder_name, "confusion_matrix.png"))

    plot_variable_importances(process.var_importances,
                              save_path=os.path.join(folder_name, "variable_importances.png"))



    if not process.train_loss.empty:

        # === Sauvegarde des courbes de loss ===
        process.train_loss.to_csv(os.path.join(folder_name, "train_loss.csv"))
        process.val_loss.to_csv(os.path.join(folder_name, "val_loss.csv"))

        # --- Plot des 8 courbes ---
        fig, axs = plt.subplots(2, 4, figsize=(20, 10), sharex=True, sharey=True)
        axs = axs.flatten()

        vintage_dates = process.train_loss.index
        step = len(vintage_dates) // 7
        selected_dates = [vintage_dates[i * step] for i in range(8)]

        for i, date in enumerate(selected_dates):
            axs[i].plot(process.train_loss.columns, process.train_loss.loc[date], label='Train Loss', color='blue')
            axs[i].plot(process.val_loss.columns, process.val_loss.loc[date], label='Val Loss', color='orange')
            axs[i].set_title(f"Vintage: {date}")
            axs[i].set_xlabel("Epoch")
            axs[i].set_ylabel("Loss")
            axs[i].legend()
            axs[i].grid(True)

        plt.tight_layout()
        plt.savefig(os.path.join(folder_name, "lstm_losses_examples.png"), bbox_inches='tight')

    # Sauvegarde des hyperparamètres LSTM si présents
    if process.lstm_hyperparams is not None:
        with open(os.path.join(folder_name, "lstm_config.txt"), "w") as f:
            for k, v in process.lstm_hyperparams.items():
                f.write(f"{k}: {v}\n")


    # Sauvegarde des rapports GridSearch s'ils existent
    if process.gridsearch_reports:
        gs_report_path = os.path.join(folder_name, "gridsearch_reports.txt")
        with open(gs_report_path, 'w', encoding='utf-8') as f:
            for (vintage_date, model_name), df in process.gridsearch_reports.items():
                f.write(f"\n===== {vintage_date} - {model_name} =====\n\n")

                # Best params et best score
                try:
                    best_idx = df['rank_test_score'].argmin()
                    best_params = {
                        k.replace("param_", ""): df.iloc[best_idx][k]
                        for k in df.columns if k.startswith("param_")
                    }
                    best_score = df.iloc[best_idx]['mean_test_score']
                    f.write("🟩 Best params:\n")
                    f.write(f"{best_params}\n\n")
                    f.write(f"🟩 Best score (mean_test_score): {best_score:.4f}\n\n")
                except Exception as e:
                    f.write(f"⚠️ Erreur lors de l'extraction du best params : {e}\n\n")

                # Résultats complets
                f.write("🟨 Full CV results:\n")
                f.write(df.to_string(index=False))
                f.write("\n\n")


    print(f"\n✅ Résultats et figures sauvegardés dans : {folder_name}")


def save_splits_to_parquet(sets: dict, split_type: str = 'per_revision'):
    """
    Sauvegarde les splits dans Clean/data_sets/splits_<split_type>/YYYY-MM/
    
    Args:
        sets (dict): Dictionnaire {date: (X_train, y_train, X_test, y_test)}
        split_type (str): 'per_revision' ou 'expanding'
    """

    # Point de départ : dossier Clean/
    base_dir = Path(__file__).resolve().parent.parent
    split_root = base_dir / "data_sets" / f"splits_{split_type}"
    split_root.mkdir(parents=True, exist_ok=True)

    for date, (X_train, y_train, X_test, y_test) in tqdm(sets.items(), desc='saving split sets'):
        date_str = pd.to_datetime(date).strftime("%Y-%m")
        split_dir = split_root / date_str
        split_dir.mkdir(parents=True, exist_ok=True)

        # Sauvegardes
        X_train.to_parquet(split_dir / "X_train.parquet", index=True)
        y_train.to_frame(name="target").to_parquet(split_dir / "y_train.parquet", index=True)

        if isinstance(X_test, pd.Series):
            X_test = X_test.to_frame().T
        X_test.to_parquet(split_dir / "X_test.parquet", index=True)

        with open(split_dir / "y_test.json", "w") as f:
            json.dump({"y_test": int(y_test)}, f)

    print(f"✅ Splits sauvegardés dans : {split_root}")



def mcnemar_comparison(y_true, y_model1, y_model2, names: list, alpha=.05):
    """
    Compare the performance of two models using McNemar's test.

    Args:
        y_true (numpy.ndarray): True binary labels for each observation in the test data.
        y_model1 (numpy.ndarray): Predicted binary outcomes for each observation in the test data using model 1.
        y_model2 (numpy.ndarray): Predicted binary outcomes for each observation in the test data using model 2.

    Returns:
        tuple: A tuple containing the test result (reject or fail to reject the null hypothesis),
               the p-value of the test, and the test statistic.
    """

    for name in names:
        if type(name) != str:
            str(name)


    H0 = "There is no significant difference in the performance of the two models"
    H1 = "One of the models performs significantly better than the other"

    # Calculate the counts of true positives, false positives, false negatives, and true negatives for each model.
    tb = mcnemar_table(y_target=y_true,
                       y_model1=y_model1,
                       y_model2=y_model2)

    # Calculate the McNemar's test statistic and p-value.
    chi2, p_value = mcnemar(ary=tb, corrected=True)


    # Determine whether to reject or fail to reject the null hypothesis based on the p-value.
    if p_value < 0.05:
        test_result = "Reject the null hypothesis"
    else:
        test_result = "Fail to reject the null hypothesis"

    # Determine the best performing model in terms of the decision variable.
    model1_wins = 0
    model2_wins = 0
    for i in range(len(y_true)):
        if y_true.iloc[i] == 1 and y_model1.iloc[i] == 1 and y_model2.iloc[i] == 0:
            model1_wins += 1
        elif y_true.iloc[i] == 1 and y_model1.iloc[i] == 0 and y_model2.iloc[i] == 1:
            model2_wins += 1
        elif y_true.iloc[i] == 0 and y_model1.iloc[i] == 1 and y_model2.iloc[i] == 0:
            model2_wins += 1
        elif y_true.iloc[i] == 0 and y_model1.iloc[i] == 0 and y_model2.iloc[i] == 1:
            model1_wins += 1

    if model1_wins > model2_wins:
        best_model = names[0]
    elif model2_wins > model1_wins:
        best_model = names[1]
    else:
        best_model = "Both models perform equally well"
    results = f"Results of McNemar's test:\nNull Hypothesis (H0): {H0}\n" \
              f"Alternative Hypothesis (H1): {H1}\n\nMcNemar Table:\n{tb}\n" \
              f"\nMcNemar's test statistic: {chi2:.3f}\np-value: {p_value:.3f}\nSignificance level: {alpha:.3f}\n\n{test_result} " \
              f"\n\nBest performing model in terms of the decision variable: {best_model}"

    print(results)


    return test_result, p_value, chi2


def get_optimal_thresholds(models, X, Y):
    """
    This function takes in a dictionary of models, a dataset of features, and a dataset of true labels.
    It returns a dictionary of optimal thresholds for each model in the given dictionary.

    Args:
    - models: a dictionary of trained models
    - X: a dataset of features associated with the dataset of true labels
    - Y: a dataset of  true labels

    Returns:
    - opt_thresholds: a dictionary of optimal thresholds for each model in the given dictionary
    """
    results = {}
    for name, model in models.items():
        # Get ROC and Precision-Recall curves for the current model
        fpr, tpr, thresholds = roc_curve(Y, model.predict_proba(X)[:, 1])
        precision, recall, thresholds = precision_recall_curve(Y, model.predict_proba(X)[:, 1])

        # Calculate Metrics for the current model
        gmeans = np.sqrt(tpr * (1 - fpr)) #Geometric mean of sensitivity and specificity
        J = tpr - fpr # Youden's J statistic
        fscore = (2 * precision * recall) / (precision + recall) # F1 score

        # Evaluate thresholds for the current model
        scores = [log_loss(Y, (model.predict_proba(X)[:, 1] >= t).astype('int')) for t in
                  thresholds]
        
        thresholds = [round(threshold, 3) for threshold in thresholds]

        ix_log = np.argmin(scores)
        ix_J = np.argmax(J)
        ix_f1 = np.argmax(fscore)
        ix_g = np.argmax(gmeans)

        # Stocker dans une Series
        results[name] = pd.Series({
            'log_score': thresholds[ix_log],
            'g_means': thresholds[ix_g],
            'J': thresholds[ix_J],
            'f1_score': thresholds[ix_f1]
        })

    return results

def read_predictions(file):
    """
    Reads predictions from a file and returns a DataFrame.
    """
    df = pd.read_csv(f'./Tests/{file}.csv', index_col = 0)
    df.index = pd.to_datetime(df.index)
    if df.index[0].day == 1:
        df.index = df.index.to_period("M").to_timestamp() + pd.Timedelta(days=14)
    df = df.iloc[:, -1]
    df.name = file
    return df


def risk_free_index_processing():

    with open('history_13w_ustb.json', 'r') as f:
        history_13w_ustb = json.load(f)

    with open('history_10y_ustb.json', 'r') as f:
        history_10y_ustb = json.load(f)
    #history_13w_ustb = yf.YahooFinancials('^IRX').get_historical_price_data('2013-09-15', '2022-12-15', 'monthly')
    #history_10y_ustb = yf.YahooFinancials('^TNX').get_historical_price_data('2013-09-15', '2022-12-15', 'monthly')

    df1 = pd.DataFrame(history_13w_ustb['^IRX']['prices'])
    df2 = pd.DataFrame(history_10y_ustb['^TNX']['prices'])
    df1.drop('date', axis=1, inplace=True)
    df2.drop('date', axis=1, inplace=True)

    df1.index = pd.to_datetime(df1['formatted_date'])
    df2.index = pd.to_datetime(df2['formatted_date'])

    df1["price"] = df1["adjclose"]
    df2["price"] = df2["adjclose"]

    df1 = df1.filter(["price"])
    df2 = df2.filter(["price"])

    # df = df.resample('D').ffill()
    df1 = df1.resample('D').mean()  # Resample to daily frequency and aggregate using mean
    df2 = df2.resample('D').mean()
    # df = df.resample('D').ffill()
    df1 = df1.interpolate()
    df2 = df2.interpolate()# Interpolate missing values using linear interpolation
    df1 = df1[df1.index.day == 15]
    df2 = df2[df2.index.day == 15]



    return df1,df2

def risky_index_processing():
    #history_sp = yf.YahooFinancials('^GSPC').get_historical_price_data('2013-09-15', '2022-12-15', 'monthly')
    with open('history_sp.json', 'r') as f:
        history_sp = json.load(f)

    df = pd.DataFrame(history_sp['^GSPC']['prices'])
    df.drop('date', axis=1, inplace=True)
    df.index = pd.to_datetime(df['formatted_date'])
    df["price"] = df["adjclose"]
    df = df.filter(["price"])
    df = df.resample('D').ffill()
    df = df[df.index.day == 15]
    filename = "sp500_historical_data.txt"
    df.to_csv(filename, sep='\t', index=True)
    return df


def compute_sharpe_ratio(monthly_return_pct):
    """
    Compute Sharpe Ratio given monthly returns in percentage

    Args:
    - monthly_return_pct (pandas.Series): Monthly returns in percentage

    Returns:
    - sharpe_ratio (float): The Sharpe Ratio of the given monthly returns
    """

    # Calculate annualized average daily return
    avg_monthly_return = monthly_return_pct.mean()
    avg_annual_return = avg_monthly_return * 12  # 252 trading days in a year

    # Calculate annualized standard deviation of daily returns
    std_monthly_return = monthly_return_pct.std()
    std_annual_return = std_monthly_return * np.sqrt(12)

    # Calculate Sharpe Ratio : Assume risk-free rate constant of 2.1% calculated from 10Y USTB
    sharpe_ratio = (avg_annual_return - 0.021) / std_annual_return

    return sharpe_ratio









