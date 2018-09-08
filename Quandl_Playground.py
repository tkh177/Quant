import quandl
import datetime as dt
quandl.ApiConfig.api_key = "1xhdLf53wKtRLzj3ZRNw"

today=dt.date.today()
thirty_days = dt.timedelta(days=30)

thirty_days_ago = today-thirty_days

my_list=[]

data = quandl.get("WIKI/AAPL", start_date=str(thirty_days_ago), end_date=str(today),column_index=4)
my_list.append(data)
print(my_list)