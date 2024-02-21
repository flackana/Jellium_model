#%%
import numpy as np
import matplotlib.pyplot as plt
from scipy import special
# Some plotting functions
def robovi_v_sredine(robovi, rob):
    width = robovi[1]-robovi[0]
    if rob == 1:
        return np.arange(robovi[0]-0.5*width, robovi[-1]+0.5*width+width, width)
    else:
        return np.arange(robovi[0]-0.5*width, robovi[-1]+0.5*width, width)

def normalise(funkcija, iksi):
    vsota = np.trapz(funkcija, iksi)
    return np.divide(funkcija, vsota)


def odv_x(funkcija, iksi):
    odv=[]
    for i in range(len(odv_x)):
        odv.append()
def f_delta(x, gamma, t, T):
    return  0.5*np.exp((1/(4*T))*gamma*(gamma*t - 2*x)) * special.erfc((gamma*t - x)/(2 * np.sqrt(T*t)))

def g_delta(x, gamma, t, T):
    prvi = np.exp((1/(4*T)) * gamma * (gamma*t - 2*x))
    #print(prvi)
    drugi = np.exp(-((gamma*t - x)/(2*np.sqrt(T*t)))**2)
    #rint(drugi)
    return prvi* drugi

def r_delta2(x, gamma, t, T):
    return -(T/gamma) * ((1/(f_delta(x, gamma, t, T) + f_delta(-x, gamma, t, T))) *
        ((gamma/(2*T)) * (f_delta(-x, gamma, t, T) - f_delta(x, gamma, t, T)) +
         (1/(2*np.sqrt(np.pi)*np.sqrt(T*t))) * (g_delta(x, gamma, t, T) - g_delta(-x, gamma, t, T))))

def rho(x, gamma, t, T):
    st1 = (np.exp((gamma*x)/T)*special.erfc((gamma*t+x)/(2*np.sqrt(t*T)))) / (np.sqrt(t*T))
    st2 = special.erfc((gamma*t-x)/(2*np.sqrt(t*T)))*(1/np.sqrt(t*T))
    st3 = -(1/T)*special.erfc((gamma*t-x)/(2*np.sqrt(t*T))) * np.sqrt(np.pi)*gamma*np.exp((gamma*t+x)**2/(4*t*T))*special.erfc((gamma*t+x)/(2*np.sqrt(t*T)))
    st_cel = np.exp(-(x-gamma*t)**2/(4*t*T))*(st1 + st2 + st3)
    im = np.sqrt(np.pi)* ( special.erfc((gamma*t-x)/(2*np.sqrt(t*T))) + np.exp(gamma*x/T)*special.erfc((gamma*t+x)/(2*np.sqrt(t*T))) )**2
    return st_cel/im

#FF_dt_10 = np.load('/home/ana/Dropbox/Ranked_diffusion/Density_burg_fire_0.01_t_10_N_100avrg_1000.npz')#
FF_dt_10 = np.load('../Data/Density_delta_N100_c0.01_t10_avrg1000_l0.npz')#
FF_dt_100 = np.load('../Data/Density_delta_N100_c0.01_t100_avrg1000_l0.npz')#
FF_dt_200 = np.load('../Data/Density_delta_N100_c0.01_t200_avrg1000_l0.npz')#

FF_dt_10_N500 = np.load('../Data/Density_delta_N500_c0.002_t10_avrg1000_l0.npz')
FF_dt_100_N500 = np.load('../Data/Density_delta_N500_c0.002_t100_avrg1000_l0.npz')
FF_dt_200_N500 = np.load('../Data/Density_delta_N500_c0.002_t200_avrg1000_l0.npz')

# PLOT DENSITY _ BURGERS
bini = robovi_v_sredine(np.linspace(-40, 40, 100-1), 0)
bini_100 = robovi_v_sredine(np.linspace(-2.5*100, 2.5*100, 100-1), 1)
bini_200 = robovi_v_sredine(np.linspace(-300, 300, 100-1), 0)


fig, (ax1, ax2, ax3) = plt.subplots(1, 3)

ax1.plot(bini, normalise(FF_dt_10, bini), '^',markersize=10,color='limegreen', label='$N=100 $')
ax1.plot(bini, normalise(FF_dt_10_N500, bini),'x', markersize=8, color='blue', label='$N=500 $')
iks=np.linspace(-60, 60, 1000)
ax1.plot(iks, [rho(ik, 1, 10, 1) for ik in iks], linewidth=2, color='black')
ax1.legend(fontsize=14)
ax1.tick_params(axis='both', which='major', labelsize=15)
ax1.set_xlim(-40, 40)

ax2.plot(bini_100, normalise(FF_dt_100, bini_100),'^',markersize=10, color='limegreen', label='$N=100 $')
ax2.plot(bini_100, normalise(FF_dt_100_N500, bini_100),'x', markersize=8, color='blue', label='$N=500 $')
iks=np.linspace(-200, 200, 1000)
ax2.plot(iks, [rho(ik, 1, 100, 1) for ik in iks], linewidth=2, color='black')
ax2.legend(fontsize=14)
ax2.tick_params(axis='both', which='major', labelsize=15)

ax3.plot(bini_200, normalise(FF_dt_200, bini_200),'^',markersize=10, color='limegreen', label='$N=100 $')
ax3.plot(bini_200, normalise(FF_dt_200_N500, bini_200),'x',markersize=8, color='blue', label='$N=500 $')
iks=np.linspace(-350, 350, 1000)
ax3.plot(iks, [rho(ik, 1, 200, 1) for ik in iks], linewidth=2, color='black')
ax3.legend(fontsize=14)
ax3.tick_params(axis='both', which='major', labelsize=15)

#ax1.legend(loc='lower left')
plt.subplots_adjust(wspace=0.3)
figure = plt.gcf()  # get current figure
figure.set_size_inches(22, 6) # set figure's size manually to your full screen (32x18)

plt.savefig('../Figures/Burgers_densities.jpg')#,bbox_inches='tight'
# %%
