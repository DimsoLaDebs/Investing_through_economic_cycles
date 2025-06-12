import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from tabulate import tabulate
from utils.helpers import compute_sharpe_ratio


class Portfolio:

    def __init__(self, risk_free_index=None, risky_index=None, strategy=None, y_pred=None, start=None, end=None):
        self.y_pred = y_pred.copy()  # pour éviter de modifier l'objet original
        self.rf_rate = risk_free_index[0]
        self.bond_rate = risk_free_index[1]
        self.risky_assets = risky_index
        self.strategy = strategy
        self.start = pd.to_datetime(start) if start else None
        self.end = pd.to_datetime(end) if end else None
            
        self.clean_y_pred_index()

        # === Construction du DataFrame portfolio_history ===
        self.portfolio_history = pd.DataFrame(index=self.y_pred.index)
        self.portfolio_history["portfolio_value"] = np.nan
        self.portfolio_history["portfolio_returns"] = np.nan


    def clean_y_pred_index(self):
        """
        Vérifie que les dates de y_pred sont bien espacées mensuellement au 15 du mois.
        Si des dates irrégulières sont détectées, elles sont supprimées.
        """

        # === Filtrage temporel (si start/end définis) ===
        if self.start is not None:
            self.y_pred = self.y_pred.loc[self.y_pred.index >= self.start]
        if self.end is not None:
            self.y_pred = self.y_pred.loc[self.y_pred.index <= self.end]

        # === Filtrage sur intersection d'index avec risky_assets ===
        self.y_pred = self.y_pred.loc[self.y_pred.index.intersection(self.risky_assets.index)]


        expected_index = pd.date_range(start=self.start.to_period("M").to_timestamp(), end=self.end, freq='MS') + pd.DateOffset(days=14)
        #print(expected_index)

        if not self.y_pred.index.equals(expected_index):
            irregular_dates = self.y_pred.index.difference(expected_index)
            n_irregular = len(irregular_dates)

            if n_irregular == 1:
                print(f"⚠️ 1 date irrégulière détectée dans y_pred : {irregular_dates[0].date()}")
            elif n_irregular > 1:
                print(f"⚠️ {n_irregular} dates irrégulières détectées dans y_pred.")

            self.y_pred = self.y_pred.loc[self.y_pred.index.intersection(expected_index)]

        # === Logging final ===
        print("📅 Plage temporelle d'étude :", self.y_pred.index.min().date(), "→", self.y_pred.index.max().date())
        print("🧠 Nombre de périodes :", len(self.y_pred))



    def simulation(self):

        current_cash = self.risky_assets.loc[self.y_pred.index[0], 'price'] # start with same capital than the risky asset
        # Portfolio value starts with current cash
        self.portfolio_history.loc[self.portfolio_history.index[0], "portfolio_value"] =  current_cash
        n_risky_assets_held = 0

        if self.strategy == "dynamic":
            for date in self.y_pred.index:


                # Bond return in %
                bond_return = (1+self.bond_rate['price'].loc[date])**(1/12)-1

                # When we have cash placed in riskfree asset
                if current_cash > 0 and date != self.y_pred.index[0]:
                    # Add monthly gains from riskfree cash allocation
                    current_cash += current_cash * bond_return/100

                    # Model predicts acceleration
                    if self.y_pred.loc[date] == 1:

                        if date != self.y_pred.index[0]:
                            # Sell last portfolio
                            current_cash += n_risky_assets_held * self.risky_assets['price'].loc[date]

                        # Buy new portfolio
                        # Allocate 80% of cash in risky asset
                        n_risky_assets_held = (current_cash * .8)/self.risky_assets['price'].loc[date]

                        # Allocate 20% of cash in risky asset
                        current_cash = .2 * current_cash

                    else:

                        # Sell last portfolio
                        if date != self.y_pred.index[0]:
                            current_cash += n_risky_assets_held * self.risky_assets['price'].loc[date]

                        # Allocate 60% of cash in risky asset
                        n_risky_assets_held = (current_cash * .4) / self.risky_assets['price'].loc[date]

                        # Allocate 40% of cash in risky asset
                        current_cash = .6 * current_cash


                # Evaluate and store portfolio value each month
                self.portfolio_history.loc[date, "portfolio_value"] = n_risky_assets_held * \
                                                                       self.risky_assets['price'].loc[date] \
                                                                       + current_cash

            # Compute portfolio returns and returns in %
            self.portfolio_history["return_pct"] = self.portfolio_history["portfolio_value"].pct_change(fill_method=None)




        elif self.strategy == "120/80_equity":

            # Number of months we were in acceleration phase to calculate cost of leverage
            n_months = 0
            # To track leverage cost
            borrowed_cash = 0


            #for date in range(len(self.y_pred)-1): pass
            for date in self.y_pred.index:

                borrowing_cost = ((1+self.rf_rate['price'].loc[date])**(1/12)-1 )/ 100

                # Model predicts acceleration
                if self.y_pred.loc[date] == 1:

                    n_months += 1

                    # Sell last portfolio
                    if date != self.y_pred.index[0]:
                        current_cash += n_risky_assets_held * self.risky_assets['price'].loc[date]

                    # Buy new portfolio

                    # Allocate 120% of cash in risky asset
                    n_risky_assets_held = (current_cash * 1.2)/self.risky_assets['price'].loc[date]

                    #Borrow 20%
                    borrowed_cash = 0.2 * current_cash
                    current_cash = - borrowed_cash


                else:
                    n_months=0

                    if date != self.y_pred.index[0]:
                        if current_cash!=0 and borrowed_cash>0:  # pk faut pas avoir 0 ? on emprunte à qui ? 
                            current_cash -= borrowed_cash * n_months * borrowing_cost

                    # Sell last portfolio

                    current_cash += n_risky_assets_held * self.risky_assets['price'].loc[date]

                    # Allocate 80% of cash in risky asset

                    n_risky_assets_held = (current_cash * 0.8) / self.risky_assets['price'].loc[date]

                    # Allocate 20% of cash in riskfree asset

                    current_cash =  0.2 * current_cash




                # Evaluate and store portfolio value in each iteration


                self.portfolio_history.loc[date, "portfolio_value"] = n_risky_assets_held * \
                                                                       self.risky_assets['price'].loc[date] \
                                                                       + current_cash

            # Compute portfolio returns in %

            self.portfolio_history["return_pct"] = self.portfolio_history["portfolio_value"].pct_change(fill_method=None)


        return self.portfolio_history["portfolio_value"], self.portfolio_history["return_pct"]



    def backtest_report(self, portfolio_history):
        """
        Generate a report for each portfolio in the input list.

        Args:
        portfolios_history (list): a list of dictionaries containing the history of each portfolio
        names (list): a list of names corresponding to each portfolio

        Returns:
        None
        """

        reports = []
        report = {}
        report['Period'] = len(portfolio_history)
        report['Max Monthly Drawdown in %'] = 100 * round(portfolio_history["return_pct"].min(), 2)
        report['Highest Monthly Return in %'] = 100 * round(portfolio_history["return_pct"].max(), 2)
        report['Average Returns in %'] = 100 * portfolio_history["return_pct"].mean()
        report['Volatility (monthly) %'] = 100 * round(portfolio_history["return_pct"].std(), 2)
        report['Net Return in %'] = 100 * portfolio_history["return_pct"].sum().round(2)
        report['Sharpe ratio'] = compute_sharpe_ratio(portfolio_history["return_pct"])
        reports.append(report)


        df = pd.DataFrame(reports)
        df.set_index('Period', inplace=True)
        df = df.transpose()
        df.columns = [self.y_pred.name]
        print(tabulate(df, headers='keys', tablefmt='fancy_grid'))

        return report['Average Returns in %'], report['Sharpe ratio']






    def plots(self, portfolio_history):

        f, axarr = plt.subplots(2, figsize=(12, 7))
        title = 'Portfolio Value and Return with the 120/80 Equity strategy' if self.strategy == "120/80_equity" \
                else 'Portfolio Value and Return with the dynamic strategy'
        f.suptitle(title, fontsize=20)


        # Ajout portefeuille
        axarr[0].plot(portfolio_history["portfolio_value"], label="Portfolio", linewidth=2.5)

        # Scatter sur signaux de prédiction pour chaque colonne de y_pred
        df = pd.concat([portfolio_history["portfolio_value"], self.y_pred], axis=1)
        df.index = pd.to_datetime(df.index)

        col = self.y_pred.name
        acc_dates = df[df[col] == 1].index
        slo_dates = df[df[col] == 0].index

        axarr[0].scatter(acc_dates, df.loc[acc_dates, "portfolio_value"],
                            color='green', marker='.', s=100, label=f"{col} Acceleration.")
        axarr[0].scatter(slo_dates, df.loc[slo_dates, "portfolio_value"],
                            color='red', marker='.', s=100, label=f"{col} Slowdown.")

        # Courbe des rendements
        axarr[1].plot(100 * portfolio_history["return_pct"], label="Portfolio returns")
        axarr[1].grid(True)

        # Ajout benchmark SP500
        axarr[0].plot(self.risky_assets.index, self.risky_assets['price'], color='black', label='B&H SP500')
        axarr[0].grid(True)

        # Étendre l'axe X de 3 mois avant start et 3 mois après end
        xlim_start = self.start - pd.DateOffset(months=3)
        xlim_end = self.end + pd.DateOffset(months=3)

        axarr[0].set_xlim(xlim_start, xlim_end)
        axarr[1].set_xlim(xlim_start, xlim_end)

        # Légendes
        axarr[0].legend(loc='best')
        axarr[1].legend(loc='best')
        plt.tight_layout()
        plt.show()
