import json
import yahoofinancials as yf

start_date = '1998-12-15'
end_date = '2025-01-15'

# Récupération des données
history_13w_ustb = yf.YahooFinancials('^IRX').get_historical_price_data(start_date, end_date, 'monthly')
history_10y_ustb = yf.YahooFinancials('^TNX').get_historical_price_data(start_date, end_date, 'monthly')
history_sp = yf.YahooFinancials('^GSPC').get_historical_price_data(start_date, end_date, 'monthly')

# Sauvegarde dans des fichiers JSON
with open('history_13w_ustb.json', 'w') as f:
    json.dump(history_13w_ustb, f)

with open('history_10y_ustb.json', 'w') as f:
    json.dump(history_10y_ustb, f)

with open('history_sp.json', 'w') as f:
    json.dump(history_sp, f)
