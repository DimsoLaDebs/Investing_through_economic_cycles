import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from utils.helpers import *
from Main.Backtesting import Portfolio
import pandas as pd
import matplotlib.pyplot as plt


# Different target strategies to try
EN_k3 = read_predictions("EN_label_k3")

rf = read_predictions('y_test_label')
tp_vintages = read_predictions("target")
target_revised = read_predictions("target_revised")
cli_diff = read_predictions("target_cli_diff")
cli_diff_roll10 = read_predictions("target_diff_roll10")
cli_diff_roll4 = read_predictions("target_diff_roll4")
cli_diff_roll3 = read_predictions("target_diff_roll3")
cli_diff_roll2 = read_predictions("target_diff_roll2")
cli = read_predictions("cli") # binarized by >90 or not
perfect_target = read_predictions("perfect_target")
cycling_perfect_target = read_predictions("cycling_perfect_target") #remplacer 0 encadrés par 1
s = pd.Series(1, index=pd.date_range(start='2000-01-15', end='2023-12-15', freq='MS') + pd.DateOffset(days=14), name='Investissement Constant') # Investissement constant




benchmark_test = False

if not benchmark_test:

    start = '2000-01-15' 
    end = '2022-12-15'
    y_pred = EN_k3 # only Serie for the moment

    assert isinstance(y_pred, pd.Series)

    print('name of the serie:', y_pred.name)

    # Create portfolio "120/80_equity" or "dynamic"
    # # Ne gère pas les dataframes pour y_pred, créer un portefeuille par strat / model_pred (à modifier)
    portfolio = Portfolio(strategy="dynamic", 
                            risky_index=risky_index_processing(), 
                            risk_free_index=risk_free_index_processing(),
                            y_pred=y_pred,
                            start = start,
                            end = end)



    # Simulation of the strategies
    portfolio.simulation() # portfolio_1.simulation(), ...
    # Plot performances
    portfolio.plots(portfolio_history=portfolio.portfolio_history)
    # Backtest report
    monthly_return, sharpe = portfolio.backtest_report(portfolio_history=portfolio.portfolio_history)




if benchmark_test:

    # Liste pour stocker les résultats
    benchmark_results = []
    target_strategies  = [rf, tp_vintages, target_revised, cli, cli_diff, cli_diff_roll2, cli_diff_roll3, cli_diff_roll4, cli_diff_roll10, perfect_target, cycling_perfect_target, s]

    for target in target_strategies:

        start = '2012-10-15' 
        end = '2022-11-15'
        y_pred = target # only Serie for the moment


        assert isinstance(y_pred, pd.Series)

        print('name of the serie:', y_pred.name)

        # Create portfolio "120/80_equity" or "dynamic"
        # Ne gère pas les dataframes pour y_pred, créer un portefeuille par strat / model_pred (à modifier)
        portfolio = Portfolio(strategy="120/80_equity", 
                            risky_index=risky_index_processing(), 
                            risk_free_index=risk_free_index_processing(),
                            y_pred=y_pred,
                            start = start,
                            end = end)



        # Simulation of the strategies
        portfolio.simulation() # portfolio_1.simulation(), ...
        # Plot performances
        #portfolio.plots(portfolio_history=portfolio.portfolio_history)
        # Backtest report
        monthly_return, sharpe = portfolio.backtest_report(portfolio_history=portfolio.portfolio_history)

            # Stocker le résultat
        benchmark_results.append({
            'target_strategy': y_pred.name,
            'monthly_return': monthly_return,
            'Sharpe': sharpe
        })

    # 🔧 Construction du DataFrame final
    benchmark = pd.DataFrame(benchmark_results).set_index('target_strategy')
    benchmark = benchmark.sort_values(by="monthly_return", ascending=False).round(3)

    # === Nom dynamique du fichier CSV ===
    strategy_label = portfolio.strategy.replace("/", "-")  # sécurité pour les noms de fichiers
    start_str = portfolio.start.strftime("%Y-%m-%d")
    end_str = portfolio.end.strftime("%Y-%m-%d")
    filename = f"strategy_{strategy_label}_{start_str}_{end_str}.csv"

    # === Sauvegarde CSV ===
    '''benchmark.to_csv(filename)
    print(f"\n✅ Résultats sauvegardés dans : {filename}")'''

    # === Affichage lisible pour Notion ===
    print("\n📋 Tableau à coller dans Notion :\n")
    print("target_strategy             | monthly_return | Sharpe")
    print("----------------------------|----------------|--------")
    for idx, row in benchmark.iterrows():
        print(f"{idx:<27} | {row['monthly_return']:<14} | {row['Sharpe']:<6}")