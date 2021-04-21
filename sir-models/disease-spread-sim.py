import pandas as pd
import numpy as np 
from scipy.integrate import solve_ivp, odeint
import matplotlib.pyplot as plt

"""
Transmission rate = What percentage of contacts do you transmit the disease to?
Contact rate = Number of contacts per unit time. Here we can define "unit of time" as 1 day. 

If TR = 30% and CR = 1 person per day, then beta = 0.3
Where Beta is effective contact rate

If A people recover or die every B days (unit of time), then removal rate is A/B. 
However, the model does not take into account the population change due to deaths (or new births). 
"""

def sir_simple(t, state, N, beta, gamma):
    #state is a shape(n,) array
    S, I, R = state 
    dSdt = - (beta * I * S)/N
    dIdt = ((beta * I * S)/N) - (gamma * I)
    dRdt = gamma * I
    return dSdt, dIdt, dRdt

if __name__ == '__main__':
    beta = 0.25
    gamma = 0.1 # one person recovers every 10 days

    #The R0 of the disease is beta/gamma = 0.3/0.1 = 3

    population = 1000
    removed = 0
    infected = 1
    susceptible = population - infected - removed
    time_eval = np.linspace(start=0, stop=365, num=365)
    time_span = (time_eval[0], time_eval[len(time_eval) - 1])

    """Documentation: https://docs.scipy.org/doc/scipy/reference/generated/scipy.integrate.solve_ivp.html#scipy.integrate.solve_ivp"""
    res = solve_ivp(fun=sir_simple, t_span=time_span, y0=[susceptible, infected, removed], 
                    args=(population, beta, gamma), t_eval=time_eval)

    df_sir = pd.DataFrame({"time": res.t, "susceptible":res.y[0], "infected":res.y[1], "removed":res.y[2]})

    #Plotting the pandemic 
    plt.style.use('seaborn')
    df_sir.plot(x='time')
    plt.show()
