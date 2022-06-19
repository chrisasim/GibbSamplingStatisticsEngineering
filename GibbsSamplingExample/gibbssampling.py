


import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
sns.set()
C = 1
#We define the for the posterior distribution asuuming of C=1.
f = lambda x, y:1/C * (np.exp(-(x*x*y*y+ x*x+ y*y -8*x - 8*y)/2.0))
xx = np.linspace(-1, 8, 100)
yy = np.linspace(-1, 8, 100)
xg, yg = np.meshgrid(xx, yy)
z = f(xg.ravel(), yg.ravel())
z2 = z.reshape(xg.shape)
plt.contourf(xg, yg, z2, cmap='BrBG')
plt.savefig('contour.png')
# Now, we will attempt to estimate the probability distribution using Gibbs Sampling. As we mentioned previously, the conditional probabilities are normal distributions. Therefore, we 
# can express them in terms of mu and sigma.
# In the following block of code, we deefine functions of mu and sigma, initialize our random variables
# X and Y. They are named after random because their data-values simply, derive from a distribution. and setN(the number of iterations).N.
#Markov-chain models. 
N=50000
#I firstly set a variable x with values of zeros length of N+1. So do I for the y variable as well. 
#Simply, I have got two variable that come from the normal distribution. z, i : 
x = np.zeros(N+1)
y = np.zeros(N+1)
x[0] = 1.
y[0] = 6.
#simply, I set for z and i the function of the square root of the exponential function coming from the normal distribution , z vector i index.
sig = lambda z, i : np.sqrt(1./(1.+z[i]*z[i]))
mu = lambda z, i: 4./(1.+z[i]*z[i])

#We step through the Gibbs Sampling algorithm.
#simply, that is a for loop strarting from 1 ending in N with a step of 2. 
for i in range(1, N, 2):
 sig_x = sig(y, i-1)
 mu_x = mu(y, i-1)
 x[i] = np.random.normal(mu_x, sig_x)
 y[i] = y[i-1]
 sig_y = sig(x, i)
 mu_y = sig(x, i)
 y[i+1] = np.random.normal(mu_y, sig_y)
 x[i+1] = x[i]

plt.hist(x, bins=50)
plt.savefig('histogramofXrandomvariablederivesfromnormaldistribution.png')
plt.hist(x, bins=50)
plt.savefig('histogramofyransomvariablederivesfromnormaldistribution.png')
plt.hist(y, bins=50)
plt.contourf(xg, yg, z2, alpha=0.8, cmap='BrBG')
plt.savefig('contourf.png')
plt.plot(x[::10], y[::10], '.', alpha=0.1)
plt.savefig('x1.png')
plt.plot(x[:300], y[:300], c='r', alpha=0.3, lw=1)
plt.savefig('y1.png')
#So, the Gobbs Sampling does a very good approximation to the target distribution.
plt.savefig('markovchainestimationnormalprobabilityclosetotheonethatfirstlywehad.png')
