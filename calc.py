import pandas as pd
import numpy as np
import statsmodels.api as sm 
import matplotlib.pyplot as plt
from statsmodels.tsa.arima_model import ARIMA 

# Let's build a time series of diapers
vals = [9, 12, 8, 15, 10, 17, 15, 13, 10, 15, 13, 11, 14, 12, 13, 12, 7, 14, 10, 12, 11]
dti = pd.date_range('2020-05-05', periods=len(vals), freq='D')

# Generate the time series:
diapers = pd.Series(vals, index=dti)

# Let's look at the data
plt.plot(diapers)
plt.show()

# Let's forecast diaper needs for the next month.
# Lifting code and modifying from this blog post code quickly: 
# https://kanoki.org/2020/04/30/time-series-analysis-and-forecasting-with-arima-python/

# Let's look at a decomposition
decomposition = sm.tsa.seasonal_decompose(diapers, model='additive') 
plt.rcParams["figure.figsize"] = [16,9] 
fig = decomposition.plot() 
fig.show()
junk = input("Press any key to continue")
plt.close()

# Let's fit an ARIMA(1,1). Later can do selection; for first pass use 1,1 and 1 lag diff
mod = ARIMA(diapers,order=(1,1,1)) # order: (AR, diff, MA)
results = mod.fit() 
print(results.summary()) 

# Brass tacks: quickly forecast
T = 30
forecast,err,ci = results.forecast(steps=T, alpha=0.05)
df_forecast = pd.DataFrame({'forecast':forecast},index=pd.date_range(start='5/26/2020', periods=T, freq='D'))

# Plot the forecast and time series -- "does this look reasonable?"
ax = diapers.plot(label='observed', figsize=(20, 15))
df_forecast.plot(ax=ax,label='Forecast',color='r')
ax.fill_between(df_forecast.index,
                ci[:,0],
                ci[:,1], color='b', alpha=.25)
ax.set_xlabel('Days')
ax.set_ylabel('Diapers')
plt.legend()
plt.show()

# Of course what we are about is the number of diapers needed over the next month.
print('forcast of next 30 days: ', np.round(forecast))
print('sum(forecast) for next 30 days:', sum(np.round(forecast)))
