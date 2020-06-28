#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Feb 23 13:24:54 2020

@author: erinbjorkeli
"""
import time
import numpy as np
import os
from mpi4py import MPI
from scipy import signal
from os.path import join
from pathlib import Path
home = str(Path.home())

def get_values(syn, values="Thesis"):
    if values == "WC":
        # Wilson-Cowan fig. 11
        if syn == "e":
            theta = 4.0 # Ø_e
            a = 1.3    # a_e
        elif syn == "i":
            theta = 3.7 # Ø_i
            a = 2.0     # a_i
        else:
            raise RuntimeError("Synapse not recognized...")
    elif values == "Thesis":
        # Testing own values:
        if syn == "e":
            theta = 4.0 # Ø_e
            a = 9.0     # a_e
        elif syn == "i":
            theta = 3.7 # Ø_e
            a = 1.0     # a_i 
        else:
            raise RuntimeError("Synapse not recognized...")
    return(theta, a)

def Z(x,syn):   #e turning at x=1.3, i turning at x=2.0
    "Sigmoid function. syn 'e' is excitatory input, 'i' is inhibitory input."
    theta, a = get_values(syn, values=values)
    Z1 = 1 / (1 + np.exp(-a*(x-theta)))
    Z2 = 1 / (1 + np.exp(a*theta))
    return(Z1-Z2)


#values = "WC"
#ke, ki = 0.9945, 0.9994

values = "Thesis"
ke, ki = 0.9999999999999998, 0.9740216524011248



def set_params():
    params = dict(
        # Time constants:
        Tau_e = 15,  # no sweep
        Tau_i = 15,  # no sweep

        # Weights:
        w_ThmCx = np.random.randint(1, 10),

        w_CxD1 = 8.0,
        w_CxD2 = np.random.randint(5, 20),

        w_CxStn = np.random.randint(1, 20),
        w_D2Gpe =  np.random.randint(1, 10),

        w_D2D1 = 6.0,
        w_D1D2 = 2.0,

        w_GpeStn = np.random.randint(5, 20),
        w_StnGpe = np.random.randint(5, 25),

        w_StnGpi = np.random.randint(5, 25),
        w_D1Gpi = np.random.randint(1, 20),
        w_GpeGpi = np.random.randint(1, 20),

        w_GpiThm = np.random.randint(1, 10),
        
        w_Stn = 0.,   # no sweep
        w_Gpe = 4.5,  # no sweep

        # External input / bacground activity
        ext_1  = round(np.random.random()*2, 1)+3.5,
        ext_2 = round(np.random.random()*2, 1)+3.5,
        )
    return params

def update(E_Cx, I_D1, I_D2, E_Stn, I_Gpe, I_Gpi, E_Thm, p):
    ext_1 = np.ones(int(simtime/dt)) * p["ext_1"]
    ext_2 = np.ones(int(simtime/dt)) * p["ext_2"]
    
    for t in range(1,timestep):
        # To cortex:
        syn = p["w_ThmCx"]*E_Thm[t-1] + ext_1[t-1]
        dE_Cx = (-E_Cx[t-1] + (ke-E_Cx[t-1]) * Z(syn,"e")) / p["Tau_e"]

        # To D1:
        syn = p["w_CxD1"]*E_Cx[t-1] - p["w_D2D1"]*I_D2[t-1]
        dI_D1 = (-I_D1[t-1] + (ki-I_D1[t-1]) * Z(syn,"i")) / p["Tau_i"]

        # To D2:
        syn = p["w_CxD2"]*E_Cx[t-1]
        dI_D2 = (-I_D2[t-1] + (ki-I_D2[t-1]) * Z(syn,"i")) / p["Tau_i"]

        # To STN:
        syn = p["w_Stn"]*E_Stn[t-1] + p["w_CxStn"]*E_Cx[t-1] - p["w_GpeStn"]*I_Gpe[t-1]
        dE_Stn = (-E_Stn[t-1] + (ke-E_Stn[t-1]) * Z(syn,"e")) / p["Tau_e"]

        # To GPe:
        syn = - p["w_Gpe"]*I_Gpe[t-1] - p["w_D2Gpe"]*I_D2[t-1] + p["w_StnGpe"]*E_Stn[t-1]
        dI_Gpe = (-I_Gpe[t-1] + (ki-I_Gpe[t-1]) * Z(syn,"i")) / p["Tau_i"]

        # To Gpi:
        syn = - p["w_D1Gpi"]*I_D1[t-1] - p["w_GpeGpi"]*I_Gpe[t-1] + p["w_StnGpi"]*E_Stn[t-1]
        dI_Gpi = (-I_Gpi[t-1] + (ki-I_Gpi[t-1]) * Z(syn,"i")) / p["Tau_i"]

        # To thalamus:
        syn = - p["w_GpiThm"]*I_Gpi[t-1] + ext_2[t-1]
        dE_Thm =(-E_Thm[t-1] + (ke-E_Thm[t-1]) * Z(syn,"e")) / p["Tau_e"]

        ###
        E_Cx[t] = E_Cx[t-1] + (dE_Cx * dt)
        I_D1[t] = I_D1[t-1] + (dI_D1 * dt)
        I_D2[t] = I_D2[t-1] + (dI_D2 * dt)
        E_Stn[t] = E_Stn[t-1] + (dE_Stn * dt)
        I_Gpe[t] = I_Gpe[t-1] + (dI_Gpe * dt)
        I_Gpi[t] = I_Gpi[t-1] + (dI_Gpi * dt)
        E_Thm[t] = E_Thm[t-1] + (dE_Thm * dt)
    
    return(E_Cx, I_D1, I_D2, E_Stn, I_Gpe, I_Gpi, E_Thm)
    
def get_freq(z,):
    init = int(250/dt)
    if np.std(z[init:]) > 1e-4:
        tresh = np.mean(z) + np.std(z)/1
        dx = 13/dt
        n_spk, spk = 0, 0
        ts, isi = [], []
        for t in range(timestep):
            if (z[t] > tresh) and (t-spk > dx):
                n_spk += 1
                spk = t
                ts.append(t)
            else: continue
        freq = n_spk * 1000 / simtime
        return(freq)
    else: 
        return(0)
        
def freq_analysis(z, f=200):
    A = np.fft.fft(z, axis=0)
    fft = np.abs(A)
    df = 1 / (1000/dt)
    freq = np.fft.fftfreq(z.size, d=df)
    return(freq[range(int(z.size/2))], fft[range(int(z.size/2))])
    
def power20(freq,fft):
    return(np.log(fft[np.where(freq==20.0)][0]))
        
def get_freq2(z,):
    freq, fft = freq_analysis(z)
    amp = power20(freq,fft)
    if amp >= 5.0:
        return(True)
    else: return(False)
    
def beta_peak(fft,freq):
    "Returns peak frequency within the beta band."
    fft = (np.array(fft))
    beta = np.where((freq>12)&(freq<30))   #nb: 13-30 isteden for 12-30...
    idx = np.where(fft==np.amax(fft[beta]))
    peak = freq[idx][0]
    return(peak, np.round(peak))
    
def get_amp(z):
    return(np.amax(z)-np.amin(z))

def healthy_ratio(E_Stn, I_Gpe, I_Gpi, I_D1, I_D2):
    # GPE-TI > STN > GPi >> D2 > D1
    
    # D1, D2 ~ 1 Hz
    # Stn ~ 15-20 Hz
    # GPe ~ 30-40 Hz
    # GPi ~ 5-10 Hz
    
    # GPe = 2*STN
    # D1 ~= D2
    # D2/D1 = 10*GPe
    # STN = 1/2 Gpe = 1/20 D1 
    
    f_D1 = f_D2 = 1
    f_Stn = [2, 4]
    f_Gpe = [3, 4]
    f_Gpi = [2, 4]
    
    is_sorted = lambda a: np.all(a[:-1] <= a[1:])
    a = [np.mean(I_D1), np.mean(I_D2), np.mean(I_Gpi), np.mean(I_Gpe), np.mean(E_Stn)] 
    cond1 = is_sorted(a)
    if not cond1:
        return False
    #cond2 = (np.amax(E_Stn) >= f_Stn[0]*np.amax(I_D1) and np.amax(E_Stn) <= f_Stn[1]*np.amax(I_D1))
    #cond3 = (np.amax(I_Gpe) >= f_Gpe[0]*np.amax(I_D1) and np.amax(I_Gpe) >= f_Gpe[1]*np.amax(I_D1))
    #cond4 = (np.amax(I_Gpi) >= f_Gpi[0]*np.amax(I_D1) and np.amax(I_Gpi) >= f_Gpi[1]*np.amax(I_D1))
    #return(cond1*cond2*cond3*cond4)
    else:
        return True

def not_neg(*args):
    for arg in args:
        if np.amin(arg) < 0:
            return False
    return True

def run_sim():
    p = set_params()

    E_Cx  = np.zeros(timestep)
    I_D1  = np.zeros(timestep)
    I_D2  = np.zeros(timestep)
    E_Stn = np.zeros(timestep)
    I_Gpe = np.zeros(timestep)
    I_Gpi = np.zeros(timestep)
    E_Thm = np.zeros(timestep)
    
    ###   PARKINSON BRAIN
    E_Cx, I_D1, I_D2, E_Stn, I_Gpe, I_Gpi, E_Thm = update(E_Cx, I_D1, I_D2, E_Stn, I_Gpe, I_Gpi, E_Thm, p)
    
    # Check that all activity is positive:
    C0 = not_neg(I_Gpe, I_Gpi, E_Thm)
    
    beta = [12,30]
    freq1, freq2 = get_freq(E_Stn), get_freq(E_Cx)
    C1 = ((freq1 > beta[0]) and (freq1 < beta[1])) # beta oscillations in STN
    C2 = ((freq2 > beta[0]) and (freq2 < beta[1])) # beta oscillations in cortex
    #C1, C2 = get_freq2(E_Stn), get_freq2(E_Cx)

    #if not C0*C1*C2:
    if not C1*C2:
        return False, None, None, None, None
    
    # Check if D2 activity larger than D1 (PD=True)
    C3 = (np.max(I_D1) < np.max(I_D2))
    if not C3:
        return False, None, None, None, None
    
    ###   HEALTHY BRAIN
    # Check if reduced D2 activity not oscillating
    p["w_CxD2"] *= 0.1
    E_Cx  = np.zeros(timestep)
    I_D1  = np.zeros(timestep)
    I_D2  = np.zeros(timestep)
    E_Stn = np.zeros(timestep)
    I_Gpe = np.zeros(timestep)
    I_Gpi = np.zeros(timestep)
    E_Thm = np.zeros(timestep)
    E_Cx, I_D1, I_D2, E_Stn, I_Gpe, I_Gpi, E_Thm = update(E_Cx, I_D1, I_D2, E_Stn, I_Gpe, I_Gpi, E_Thm, p)
    
    
    C4 = healthy_ratio(E_Stn, I_Gpe, I_Gpi, I_D1, I_D2)
    # Check if ratios are healthy:
    if not C4:
        return False, None, None, None, None
            
    freq3 = get_freq(E_Cx) # or some other condition showing not stable
    C5 = (freq3 < 5)
    #C5 = get_freq2(E_Cx) # or some other condition showing not stable
    if not C5:
        return False, None, None, None, None
    
    mean = [np.mean(E_Cx), np.mean(E_Stn), np.mean(I_Gpe), np.mean(I_D2)]
    amp = [get_amp(E_Cx), get_amp(E_Stn), get_amp(I_Gpe), get_amp(I_D2)]
    
    return True, p, freq1, amp, mean          #lists of form [Cx_p, STN_p, GPe_p, D2_p]

dt = 0.01 
simtime = 1000.0 #ms
timestep = int(simtime/dt)
timevec = np.arange(timestep)
    
def main():
    #osc_data = open("oscillation_data.txt", "w+")
    #params = open("oscillation_parameters.txt", "w+")

    COMM = MPI.COMM_WORLD
    SIZE = COMM.Get_size()
    RANK = COMM.Get_rank()
    STATUS = MPI.Status()   # get MPI status object
    
    
    name_d = "oscillation_data_"+str(RANK)+".txt"
    name_p = "oscillation_parameters_"+str(RANK)+".txt"
    osc_data = open(name_d, "w+")
    params = open(name_p, "w+")
    
    count = 0
    while True:
    #for i in range(10):
        if count % SIZE == RANK:
            #print("{} does task {}".format(RANK, i))
            p = set_params() # set random parameters
            rule, p, freq, amp, mean = run_sim() # check conditions
            if rule:
                # write to files:
                params.write(str(p))
                osc_data.write(str(freq))
                osc_data.write(str(amp))
                osc_data.write(str(mean))
        count += 1
        if (count % 250) == 0:
            print("{} finished {} iterations".format(RANK, count))
    return(count/(SIZE+1))
        
st = time.time()
count = main()
et = time.time()

print("Simulation length: {} s".format(et-st))
print("Iteration length: {} s".format((et-st)/count))
