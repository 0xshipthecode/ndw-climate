

from scipy.signal.spectral import lombscargle
import numpy as np

alpha = 0.2 
gamma = 2.0
N = 100

t = np.arange(N, dtype = np.float)
y = gamma * np.sin(alpha * t)
periods = np.linspace(2.0, 120, (120-2)*10)
print periods.shape
freqs = (2.0 * np.pi) / periods
p = lombscargle(t, y, freqs)

f = figure()
ax = subplot(211)
plt.plot(t, y)
plt.title('Signal')

ax = subplot(212)
plt.plot(periods, np.sqrt(p * 4.0 / N))
plt.title('Frequency spectrum')
plt.xlabel('Period [Samples]')
plt.grid()

plt.show()