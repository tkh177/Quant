import quandl

data = quandl.get("NSE/OIL")

print(data.head())
