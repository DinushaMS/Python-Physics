import numpy as np
import matplotlib.pyplot as plt
import scienceplots
plt.style.use(['science'])

def bbo_n(wl):
    """
    wl: wavelength in micrometers.
    temperature 294K
    """
    B_o = np.array([1.7359/wl**2,0.01878/wl**2,-0.01354])
    C_o = np.array([wl**2-1,0.01822,wl**2-1])
    B_e = np.array([1.3753/wl**2,0.01224/wl**2,-0.01516])
    C_e = np.array([wl**2-1,0.01667,wl**2-1])
    n_o = n_squared(wl,B_o,C_o)**0.5
    n_e = n_squared(wl,B_e,C_e)**0.5
    return n_o, n_e

def ln_n(wl):
    """
    wl: wavelength in micrometers.
    """
    B = np.array([2.6734,1.2290,12.614])
    C = np.array([0.01764,0.05914,474.60])
    n = n_squared(wl,B,C)**0.5
    return n

def n_squared(wl,B,C):
    """
    wl: wavelength in micrometers.
    B: dimensionless Sellmeier coefficients related to the oscillator strength of the transition. 
    C: Sellmeier coefficients representing the squares of the transition energies (as wavelengths), usually in micrometer squared. 
    """
    n2 = 1
    for i in range(len(B)):
        n2 += B[i]*wl**2/(wl**2-C[i])
    return n2

def gen_n_circle(theta_array, wl_array):
    fig, ax = plt.subplots(1,1,figsize=(7, 4),dpi=300)
    for wl in wl_array:
        n_o, n_e = bbo_n(wl)
        x = n_o*np.sin(theta_array)
        z = n_o*np.cos(theta_array)
        ax.plot(x,z,'-',label='$n_o$'+f'({wl*1000:.0f}nm)')
        x = n_e*np.sin(theta_array)
        z = n_o*np.cos(theta_array)
        ax.plot(x,z,'-',label='$n_e$'+f'({wl*1000:.0f}nm)')
    ax.set_aspect('equal')
    ax.set_axis_off()
    ax.legend()
    ax.arrow(np.min(ax.get_xlim()),0,np.max(ax.get_xlim())-np.min(ax.get_xlim()),0,head_width=0.05, head_length=0.1, fc='black', ec='black')
    ax.arrow(0,np.min(ax.get_ylim()),0,np.max(ax.get_ylim())-np.min(ax.get_ylim()),head_width=0.05, head_length=0.1, fc='black', ec='black')
    return fig, ax

def gen_k_array(theta_array,wl_array):
    """
    theta_array: array of theta in radians.
    wl_array: array of wavelength in micrometers.
    return: kk_array with shape (len(theta_array), len(wl_array), 2)
    """
    kk_array = np.array([])
    for theta in theta_array:
        k_array = np.array([])
        for wl in wl_array:
            n_o, n_e = bbo_n(wl)
            c = 3e8
            om = 2*np.pi*c/wl/1e-6
            no = n_o
            ne = n_o*n_e/(n_o**2*np.sin(theta)**2+n_e**2*np.cos(theta)**2)**0.5
            #kox = n_o*np.sin(theta)*om/c
            #koz = n_o*np.cos(theta)*om/c
            #kex = n_e*np.sin(theta)*om/c
            #kez = n_o*np.cos(theta)*om/c
            #ko = (kox**2+koz**2)**0.5
            #ke = (kex**2+kez**2)**0.5
            ko = no*om/c
            ke = ne*om/c
            k_array = np.append(k_array,np.array([ko,ke]))
        kk_array = np.append(kk_array, k_array)
    kk_array = kk_array.reshape(len(theta_array),len(wl_array),2)
    return kk_array

def gen_del_k_plot(theta_array,kk_array,wl_array,op_str_array,L=0.4e-2):
    pol_str_array = ['o','e']
    bit_length = len(wl_array)
    pol_comb_array = []
    plt.figure(figsize=(7, 4),dpi=300)
    for i in range(2**bit_length):
        binary_str = bin(i)[2:]
        padded_binary_str = binary_str.zfill(bit_length)
        pol_comb_array.append([int(char) for char in list(padded_binary_str)])
    for i,pol in enumerate(pol_comb_array):
        del_k = 0
        del_k_str = ''
        for j in range(len(op_str_array)):
            del_k += int(op_str_array[j])*kk_array[:,j,pol[j]]
            del_k_str += f'{op_str_array[j]}$k_{pol_str_array[pol[j]]}$({wl_array[j]:.3f})'
        del_k *= 1e-5
        if (np.min(del_k)<0 and del_k[0]>0) or (np.max(del_k)>0 and del_k[0]<0):
            plt.plot(theta_array*180/np.pi,del_k,label = del_k_str)
            if del_k[0]>0:
                theta = theta_array[del_k<0][0]*180/np.pi
            else:
                theta = theta_array[del_k>0][0]*180/np.pi
            plt.text(theta,0,f'{theta:.1f}$^o$', rotation=90, va='top')
    plt.hlines(0,np.min(theta_array)*180/np.pi,np.max(theta_array)*180/np.pi,linestyles='dashed',lw=1,colors='black')
    plt.legend()
    plt.ylabel(r'$\Delta k$'+'('+r'$\times 10^5$'+'rad/m)')
    plt.xlabel('$\\theta$(deg.)')
    plt.title('Phase matching angle(s)')
    plt.show()

def RK4_step(funcs, x, h, *args):
    """Performs a single Runge-Kutta 4th order step for a system of ODEs.
    funcs: list of functions representing the system of ODEs
    x: independent variable
    h: step size
    args: current values of the dependent variables
    Returns: updated values of the dependent variables after one step
    """
    k = np.zeros((len(funcs), 4))
    for i in range(4):
        for j in range(len(funcs)):
            if i == 0:
                k[j][i] = h * funcs[j](x, *args)
            elif i == 1 or i==2:
                k[j][i] = h * funcs[j](x, *tuple(a + b for a, b in zip(args, tuple([k[l][i-1]/2 for l in range(len(funcs))]))))  # use k from previous i
            else:
                k[j][i] = h * funcs[j](x + h, *tuple(a + b for a, b in zip(args, tuple([k[l][i-1] for l in range(len(funcs))]))))  # use k from previous i
    return tuple(a + b for a, b in zip(args, tuple([(k[j][0]+2*k[j][1]+2*k[j][2]+k[j][3])/6 for j in range(len(funcs))])))