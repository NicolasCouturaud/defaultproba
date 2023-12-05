import yfinance as yf
import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.stats import norm
import matplotlib.pyplot as plt

ticker = "MC.PA"
stock_data = yf.download(ticker, start="2022-12-02", end="2023-12-02")

print(stock_data.head())
# Extract relevant information
stock_data['Daily_Return'] = stock_data['Adj Close'].pct_change()
sigmaS = np.sqrt(252) * stock_data['Daily_Return'].std()

company_info = yf.Ticker(ticker)
outstanding_shares = company_info.info['sharesOutstanding']

stock_data['market_cap'] = stock_data['Adj Close'] * outstanding_shares
equity_value = stock_data['market_cap']

print(equity_value)
print(sigmaS)

def f(x, VE, D, R, SE, horizon):
    VA = x[0]
    SA = x[1]

    d1 = (np.log(VE / D) + (R - SA**2 / 2) * horizon) / (SA * np.sqrt(horizon))
    d2 = (np.log(VE / D) + (R + SA**2 / 2) * horizon) / (SA * np.sqrt(horizon))

    e1 = VA * norm.cdf(d1) - D * np.exp(-R * horizon) * norm.cdf(d2)
    e2 = (VA/e1)*SA * norm.cdf(d1)

    #e1 = VE - (VA*norm.cdf(d1) - D * np.exp(-R*horizon)*norm.cdf(d2))
    #e2 = SE * VE - VA*SA*norm.cdf(d1)

    return ((e1/VE-1)**2 + (e2/SE-1)**2)


r = 0.04
horizon = 10
bounds = [(0, 1E12), (0, 1E3)]
D = 70564000000
T = 10

#result = minimize(f,initial_guess, args=(equity_value[-1], D, 4, sigmaS, horizon),bounds=bounds)
#print(result.x)
#[Va,sigmaA] = result.x

#DD = (np.log(Va/D)+(r-0.5*sigmaA**2)*horizon)/(sigmaA*horizon)
#default_proba = norm.cdf(-DD)

#print("\nLa proba de defaut 1 an est : ", default_proba)


E = []
E.append(equity_value[-1])
L = []
initial_guess = [E[0], sigmaS]

result = minimize(f, initial_guess, args=(E[0], D, 4, sigmaS, horizon), bounds=bounds)
[Va, sigmaA] = result.x
V0 = Va

L.append(V0 - E[0])

Vt = [np.mean(V0*np.exp((r - sigmaA ** 2 / 2) * t + sigmaA * np.random.normal(0, np.sqrt(t), size=(1, horizon)).T)) for t in range(1, horizon + 1)]
V = pd.DataFrame([Vt], columns=[f"t = {i}" for i in range(1, horizon + 1)])

print(V)

values = [V0 * np.exp(r * t) for t in range(horizon)]

expected_V = pd.DataFrame([values],columns=[f"t = {i}" for i in range(1,horizon+1)])

time = np.linspace(1, horizon, horizon)
tt = np.full(shape=(1, horizon), fill_value=time).T

plt.plot(tt, V.T)
plt.plot(tt, expected_V.T,color= 'red')
plt.xlabel("Years $(t)$")
plt.ylabel("V $(S_t)$")
plt.show()

#d1 = (np.log(V[0]/D)+(r+0.5*sigmaA**2)*horizon)/(sigmaA*np.sqrt(horizon))
#d2 = d1 - sigmaA*np.sqrt(horizon)
#L0 = V[0]*norm.cdf(-d1)

DD= []
default_proba = []

for i in range(horizon):
    DDefault = (np.log(expected_V.iloc[0,i] / D) + (r - 0.5 * sigmaA ** 2) * horizon) / (sigmaA * np.sqrt(horizon-i))
    DD.append(DDefault)
    default_proba.append(norm.cdf(-DD[i]))

print("DD : ",DD)
print("Default proba :",default_proba)

plt.plot(range(1, horizon + 1), default_proba)
plt.title('Default Probabilities Over the Horizon')
plt.xlabel('Time Step (t)')
plt.ylabel('Default Probability')
plt.show()



