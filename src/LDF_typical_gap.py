
import numpy as np
import numba as nb
from numba import jit, njit, prange
import matplotlib.pyplot as plt

import numba
import numpy as np


@jit(nopython=True)
def compute_bin(x, bin_edges):
    """This function computes in which 
    bin falls point value x.

    Args:
        x (float): Value that we want to bin.
        bin_edges (array): array giving all bin edges.

    Returns:
        int: Number of bin in which x belongs or 
        None if x is outside the range of bins.
    """    
    # assuming uniform bins
    n = bin_edges.shape[0] - 1
    a_min = bin_edges[0]
    a_max = bin_edges[-1]
    #bin width
    d=bin_edges[1]-bin_edges[0]
    # special case to mirror NumPy behavior for last bin
    if x == a_max:
        return n - 1 # a_max always in last bin

    bin = int((x - a_min) / (d))

    if bin < 0 or bin >= n:
        return None
    else:
        return bin

@jit(nopython=True)
def numba_histogram(a, bin_edges):
    """This function is a version of np.hist
     but it works with numba.

    Args:
        a (array): Array of values that we want 
        to use for making a histogram.
        bin_edges (array): array of floats giving bin edges.

    Returns:
        hist (array): array with histigram values.
        bin_edges (array): array of all bin edges.
    """    
    n_bins =  bin_edges.shape[0] - 1
    hist = np.zeros((n_bins,), dtype=np.intp)
    for x in a.flat:
        bin = compute_bin(x, bin_edges)
        if bin is not None:
            hist[int(bin)] += 1

    return hist, bin_edges

@jit(nopython=True)
def Hamiltonian_plasma2(alpha,N, konfig):
    """This function computes the energy of jellium model.

    Args:
        alpha (Float, positive): Interaction strength.
        N (Int): Number of particles in the system.
        konfig (Array): Array specifying 
        positions of all N particles.

    Returns:
        Float: Energy of the given configuration.
    """    
    pari=0
    for i in range(len(konfig)):
        for j in range(i):
          pari+=np.abs(konfig[i]-konfig[j])
    return 0.5*(1/N)*np.sum(np.square(konfig))-(1/N**2)*alpha*2*(pari)

@jit(nopython=True)
def Hamiltonian_plasma_delta2(alpha,N, konfi_star, indeks, nova_lega):
    """This function computes the difference in energy between 
    two configurations that differ only in positions of one particle.

    Args:
        alpha (Float): Interaction strength
        N (Int): Number of particles
        konfi_star (Array): Array with positions of all particles (old).
        indeks (Int): Index of the particle that changes positions.
        nova_lega (Float): New position of the particle with index indeks.

    Returns:
        Float: Change of energy caused by jump of the chosen particle.
    """    
    pari=0
    for i in range(len(konfi_star)):
        if i!=indeks:
            pari+=-np.abs(nova_lega-konfi_star[i])+np.abs(konfi_star[indeks]-konfi_star[i])
    return (1/N)*0.5*(nova_lega**2 - konfi_star[indeks]**2)+(1/N**2)*2*(alpha*pari)
@jit(nopython=True)
def find_equilibration(N, koraki, konfig, d_max, alpha):
    """This is Monte Carlo Algorithm that returns 
    configuration that is equilibrated, and energy at every 100th step.

    Args:
        N (Int): Number of particles.
        koraki (Int): Number of steps that algorithm makes.
        konfig (Array): Intial configuration of particles.
        d_max (Float, positive): Parameter determining distribution 
        from wich we draw new particle position.
        alpha (Float, positive): Interaction strength.
    Returns:
        Array: Array with the coordinates of particles at 
        the end of the algorithm.
        Array: Energy at each 100th step of the algorithm.
        Float: Ratio proposed/accpeted configurations.
    """    
    zaz=np.int32(koraki/100)
    energy=np.zeros((zaz+1,))
    en=Hamiltonian_plasma2(alpha,N, konfig)
    energy[0]=en
    sprejeti=1
    delec=0
    ind_en=1
    for korak in np.arange(koraki):
        nova=np.random.uniform(0,1)
        displ = d_max*(1-2*nova)

        nova_konfig=np.copy(konfig)
        nova_konfig[delec] += displ
        delta_en=Hamiltonian_plasma_delta2(alpha, N, konfig, delec, displ+konfig[delec])
        if delta_en<0:
            en += delta_en
            konfig = np.copy(nova_konfig)
            sprejeti += 1
            delec += 1
            delec= delec%N
        else:
            stevilo=np.random.uniform(0, 1)
            if stevilo<=np.exp(-(N**3/1)*(delta_en)):
                konfig=np.copy(nova_konfig) 
                en += delta_en
                sprejeti+=1
                delec += 1
                delec=delec%N
            else:
                delec += 1
                delec=delec%N
        if korak%100==0:
               energy[ind_en]=en
               ind_en += 1
    return konfig, energy[:ind_en], sprejeti/koraki
@jit(nopython=True)
def calculate_density_histogram(N, koraki,  konfig, d_max, alpha, binsi):
    """Monte Carlo algorithm for finding density histogram.
      The initial configuration "konfig" is ment to be already in equilibrium.

    Args:
        N (Int): Number of particles
        koraki (int): Number of steps of MC algorithm.
        konfig (Array): Initial configuration of particles.
        d_max (float): Parameter controling how we draw ne particle positions.
        alpha (float): Interaction strength.
        binsi (array): Array with all bin edges.
    """    
    zaz=int(koraki/100)
    ika = np.int32(N/2)
    energy=np.zeros((zaz+1,))
    en=Hamiltonian_plasma2(alpha,N, konfig)
    energy[0]=en
    histo = np.histogram(konfig, binsi)[0]
    sampli=1
    sprejeti=1
    delec=0
    for korak in np.arange(koraki):
        nova=np.random.uniform(0,1)
        displ = d_max*(1-2*nova)

        nova_konfig=np.copy(konfig)
        nova_konfig[delec] += displ
        delta_en=Hamiltonian_plasma_delta2(alpha, N, konfig, delec, displ+konfig[delec])
        if delta_en<0:
            en += delta_en
            konfig = np.copy(nova_konfig)
            sprejeti += 1
            delec += 1
            delec= delec%N
        else:
            stevilo=np.random.uniform(0, 1)
            if stevilo<=np.exp(-(N**3/1)*(delta_en)):
                konfig=np.copy(nova_konfig) 
                en += delta_en
                sprejeti+=1
                delec += 1
                delec=delec%N
            else:
                delec += 1
                delec=delec%N
        if korak%N==1:
            h = np.histogram(konfig, binsi)[0]
            histo = np.add(histo, h)
            sampli += 1
            if koraki%N==10:
                print("We are at"+ str(10*N)+"step")
    return np.divide(histo, sampli)

def edges_to_middles(robovi, target_len):
    """This function takes in coordinates of the edges of bins, and returns middles.

    Args:
        robovi (Array): Coordinates of bin edges
        target_len (int): desired length of the output array

    Returns:
        Array: Coordinates of middles of the bins
    """    
    width = robovi[1]-robovi[0]
    middles = np.arange(robovi[0]-0.5*width, robovi[-1]+0.5*width+width, width)
    if len(middles)==target_len:
        return middles
    else:
        middles = np.arange(robovi[0]-0.5*width, robovi[-1]+0.5*width, width)
        if len(middles)==target_len:
            return middles
        else:
            middles = np.arange(robovi[0]-0.5*width, robovi[-1]+0.5*width-width, width)
            if len(middles)==target_len:
                return middles
            else:
                return None