'''
In this script we use functions from LDF_typical_gap to find typical 
distribution of gaps in the jellium model.
'''
import numpy as np
import time
import matplotlib.pyplot as plt
from src.LDF_typical_gap import calculate_density_histogram, find_equilibration, edges_to_middles

# We choose number of steps, and N of particles
koraki=100000
N = 50
#We make a random initial configuration
zacetna=np.random.normal(0, 3, N)
# We create bins
binsi = np.arange(-2.2, 2.2, 0.01)
# We run the function so that it compiles and runs faster later
a = calculate_density_histogram(N, 2, zacetna, 0.1, 0.1, binsi)
ooo=0
d=[0.05]
alpha=0.3
# We measure time, and run the function that return configuration in equilibrium
z = time.time()
zacetna, ene, acc = find_equilibration(N, koraki, zacetna, d[ooo], alpha)
k = time.time()
np.save('Data/Jellium_equilibrium_energy_'+str(N)+'_kor'+str(koraki), ene)
print('Time of first MC')
print(k-z, flush=True)
# We again measure time and run function that finds histogram of gaps in equilibrium
z = time.time()
gapon = calculate_density_histogram(N, koraki, zacetna, d[ooo], alpha, binsi)
k = time.time()
print('Time of second MC')
print(k-z, flush = True)
np.save('Data/Gap_histogram_'+str(N)+'_kor'+str(koraki)+'_indeks'+str(50)+'_alpha'+str(alpha), gapon)
print('finish', flush =True)
# Plotting the energy as a function of steps
koraki = np.arange(len(ene))*100
plt.figure(1)
plt.plot(koraki, ene)
plt.savefig('Figures/Energy_to_equilibrium.png')
# Plotting the density
x_points = edges_to_middles(binsi, len(gapon))
plt.figure(2)
plt.plot(x_points, gapon)
plt.savefig('Figures/Density.png')