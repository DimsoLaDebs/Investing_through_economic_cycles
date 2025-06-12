import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import pandas as pd

import pickle
from datetime import datetime

# --- Data loading & processinf ---
from Main.data_loader import Data, in_notebook
from Main.data_processing import *
from utils.helpers import *





db = Data(
    y_lags=6, #lags de la targets ajoutés dans les features (y_shift(1), y_shift(2), ... y_shift(y_lags))
    Verbose=True #Affiche les étapes de preprocessing
)
    

# --- Preprocessing ------------------------------------
returns = False 
log = False
ts_analysis = False
diff = False
dates_columns=True 
db.covariates(log=log, returns=returns, ts_analysis=ts_analysis, diff=diff, dates_columns=dates_columns) #On applique le preprocessing sur les covariables


#--- Processing ------------------------------------------

models= ['RF'] # None or piece of ['RF', 'GB', 'ADA', 'ENET', 'SVC', 'KNN', 'MLP', 'LSTM']
optimize = False
cv = 5 #entier naturel
split_expanding=True
normalize=True
threshold_tuning=None
bootstrap_blocks = 0 #entier naturel, 0 si pas de bootstrap
epochs = 15 #seulement utile pour LSTM

process = Data_Processing(db)

process.fit(
    models = models, 
    optimize = optimize, 
    cv = cv, 
    split_expanding=split_expanding, 
    normalize=normalize,
    threshold_tuning = threshold_tuning,
    bootstrap_blocks=bootstrap_blocks,
    epochs=epochs
    
) 

# --- Store predictions & results --------------------------------

save_results_and_plots(
    process=process,
    normalize=normalize,
    split_expanding=split_expanding,
    models=models,
    optimize=optimize,
    returns=returns,
    log=log,
    ts_analysis=ts_analysis,
    diff=diff,
    bootstrap_blocks = bootstrap_blocks,
    threshold=threshold_tuning, # peut être None
    epochs=epochs  
)















# To develop

'''# --- Threshold tuning ---------------------------------
if threshold_tuning:
    threshold = process.opt_threshold['RF']['f1_score'] #On prend le threshold optimal pour le RF
else:
    threshold = None
# on pourra expliquer que le threshold tuning est intéressant quand on a des objectifs précis (par exemple
#bien prédire les récessions plutot qu'avoir une accuracy générale bonne)


''#--- Results & plots -----------------------------------
plot_predictions(process.y_test_probs, optimal_threshold=threshold)
show_confusion_matrix(process.y_test_label)

# --- FEATURE IMPORTANCES ------------------------------
plot_variable_importances(process.var_importances) #explainable models
# for other models, we can use SHAP or LIME''


# --- Model comparison --------------------------------
#mcnemar_comparison(y_true=y_label, y_model1=y_label_RF, y_model2=y_label_GB, names=names) #étendre à plus que 2 modèles
'''