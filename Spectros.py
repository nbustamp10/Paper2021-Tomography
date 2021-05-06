# -*- coding: utf-8 -*-
"""
Created on Mon Apr 12 21:51:34 2021

@author: Bustamante-PenagosN.
La transformada wavelet fue tomada de 
"""
import os
import glob
import re
import scipy.misc
import scipy.io as sio
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import axes3d
from scipy import ndimage
from scipy.optimize import curve_fit
from scipy import fft
from scipy import ifft
from sympy.solvers import solve
from sympy import Symbol
import math
import pycwt as wavelet
from pycwt.helpers import find

cma1=pd.read_csv('cma.txt',sep=';',header=None)
time1=pd.read_csv('time.txt',sep=';',header=None)
n1=pd.read_csv('ventana.txt',header=None)  
uid=pd.read_csv('uiint.txt',header=None)
cma=np.array(cma1)
time=np.array(time1)
uid=np.array(uid)
n1=np.array(n1)
n1=int(n1[0])
b=cma.shape
L=0.05 #Longitud caracteristica

def filtro1(data,n):
    for i in range(0,len(data)-n):
        if abs(data[i]-data[i:i+n].mean())> 0.5*data[i-int(n/2):i-1].std():
            data[i]=data[i:i+int(n/2)].mean()
        # if data[i]==0:
        #     data[i]=data[i:i+int(n/2)].mean()
    return data

def derivada(data,dt):
    derivate=np.zeros(len(data)-1)
    for i in range(0,len(data)-1):
        derivate[i]=(data[i+1]-data[i])/dt
        # if i==0 and abs(derivate[i])< 0.002 :
        #     derivate[i]=derivate[i:i+n].mean()
        # if i>0 and abs(derivate[i])< 0.002:
        #     derivate[i]=(derivate[i-n:i+n]-derivate[i]).mean()
    return derivate

def filtro2(data,n):
    for i in range(len(data)-n+1):
        if abs(data[i])>0.002 and abs(data[i+1])<0.002:
            data[i]=data[i+1]
    return data

def integral(data,dt):
    fint=np.zeros(len(data))                                                   #Funcion integral
    for i in range(len(data)-1):
        faux=(data[i]+data[i+1])/2*dt
        if i==0:
            fint[i]=(data[i]+data[i+1])/2*dt
        else:
            fint[i]=fint[i-1]+faux
    return fint

def centrar(data):
    if data.mean()>0 or data.mean()<0:
        data=data-data.mean()
    return data
#%% Datos
plt.figure()
fig1,axs=plt.subplots(4,5)
sns.set(rc={'figure.figsize':(11.7,8.27)})
sns.set_style("darkgrid", {"axes.facecolor": ".9"})
for e in range(b[1]):
    aux=cma[int(time[e]-n1)-1000:int(time[e]-n1),e]
    data=np.zeros(len(aux))
    t=np.arange(time[e]-n1-1000,time[e]-n1)
    dataaux0=derivada(aux,1)
    dataaux=centrar(dataaux0)
    data=filtro2(dataaux,n1)
    
    
    m=int(e/5)
    s=e%5
    axs[m,s].plot(data[:len(data)-n1])
    axs[m,s].set_title('Exp'+str(e+1),fontsize=18)
    for ax in axs.flat:
        ax.set_ylabel('<c>', fontsize=20)                                    #$\u0305$ simbolo del promedio temporal
        ax.tick_params(axis='both', which='major', labelsize=16)
        #ax.set_xscale('log')
        #ax.set_yscale('log')
        ax.set_ylim(-0.0020,0.002)
    for ax in axs.flat:
        ax.label_outer()
#%% Espectros de Fourier
m=-5/3
x1=np.arange(10,100,1)
y=x1**m*10**1
scales = np.arange(1, 128)

plt.figure()
fig1,axs=plt.subplots(4,5,figsize=(15,15),sharey=True)
sns.set(rc={'figure.figsize':(11.7,8.27)})
sns.set_style("darkgrid", {"axes.facecolor": ".9"})
for i in range(b[1]):
    m=int(i/5)
    n=i%5
    times=np.arange(time[i]-n1+1-1000,time[i]-n1+1)
    aux=cma[len(times-n1+1)-1000:len(times-n1+1),i]                            #Considera solo los ultimos 1000 fotogramas
    dataaux0=derivada(aux,1)
    dataaux=centrar(dataaux0)
    data=filtro2(dataaux,n1)
    fft1=scipy.fft.fft(data)                                                #Espectro de la funcion sin tendencia
    fft3=abs(fft1)**2
    # fft3=integral(fft2,1)
    axs[m,n].plot(fft3[:len(fft3)-200])
    axs[m,n].set_title('Exp'+str(i+1),fontsize=18)
    axs[m,n].plot(x1,y,color='black')
    axs[m,n].text(40, 1e-1, '-5/3',fontsize=16)
    for ax in axs.flat:
        ax.set_ylabel('|FFT|$^2$ ', fontsize=20)
        ax.tick_params(axis='both', which='major', labelsize=16)
        ax.set_xscale('log')
        ax.set_yscale('log')
        ax.set_ylim(1e-10,1)
    for ax in axs.flat:
        ax.label_outer()
plt.savefig('Fourier.png')
#%% Espectros Wavelet
        
data2=cma[:int(time[0]-n1),0]
a=derivada(data2,1)
filtro2(a,8)

## calculate the Nyquist frequency
# nyq = 0.5 * 50
# design filter
# low = 24 / nyq
# sos = scipy.signal.butter(10, low, btype='low', output='sos',analog=False)
# filtered = scipy.signal.sosfilt(sos, data)
#%%
for i in range(b[1]):
    taux=np.arange(time[i]-n1+1-1000,time[i]-n1+1)
    t=np.arange(0,1000,1)
    ui=uid[i]
    t=t*L/ui
    dt=t[1]-t[0] #Tomogramas
    aux=cma[len(t-n1+1)-1000:len(t-n1+1)+1,i]                            #Considera solo los ultimos 1000 fotogramas
    #Preprocesamiento de la señal
    dataaux0=derivada(aux,dt)
    dataaux=centrar(dataaux0)
    data=filtro2(dataaux,n1)
    N=data.size
    dat_notrend=data#Serie sin tendencia
    std=dat_notrend.std()
    var=std**2
    datanorm=dat_notrend/std        
        
    mother=wavelet.Morlet(6)
    ad=L/ui
    s0=ad/2*dt #Estos parametros debo cambiarlos
    dj=1/(2*ad) #Estos parametros debo cambiarlos
    J=9/dj  #Estos parametros debo cambiarlos

    alpha,_,_=wavelet.ar1(data)
    wave,scales,freqs,coi,fft,fftfreqs=wavelet.cwt(datanorm,dt,dj,s0,J)
    iwave=wavelet.icwt(wave,scales,dt,dj,mother)*std
    power=(np.abs(wave))**2
    fft_power1=np.abs(fft)**2
    fft_power=integral(fft_power1,dt)                                                       #Integrarlas
    period=1/freqs
    signif, fft_theor = wavelet.significance(1.0, dt, scales, 0, alpha,
                                          significance_level=0.95,
                                          wavelet=mother)
    sig95 = np.ones([1, N]) * signif[:, None]
    sig95 = power / sig95

    glbl_power = power.mean(axis=1)
    dof = N - scales  # Correction for padding at edges
    glbl_signif, tmp = wavelet.significance(var, dt, scales, 1, alpha,
                                            significance_level=0.95, dof=dof,
                                            wavelet=mother)

    figprops = dict(figsize=(11, 8), dpi=72)
    fig = plt.figure(**figprops)
    
    # First sub-plot, the original time series anomaly and inverse wavelet
    # transform.
    ax = plt.axes([0.1, 0.75, 0.65, 0.2])
    ax.plot(t, iwave, '-', linewidth=1, color=[0.5, 0.5, 0.5])
    ax.plot(t, data, 'k', linewidth=1.5)
    #ax.set_title('a) {}'.format(title))
    #ax.set_ylabel(r'{} [{}]'.format(label, units))
    plt.title('Exp '+str(i+1))
    # Second sub-plot, the normalized wavelet power spectrum and significance
    # level contour lines and cone of influece hatched area. Note that period
    # scale is logarithmic.
    bx = plt.axes([0.1, 0.37, 0.65, 0.28], sharex=ax)
    levels = [0.0625, 0.125, 0.25, 0.5, 1, 2, 4, 8, 16]
    bx.contourf(t, np.log2(period), np.log2(power), np.log2(levels),
                extend='both', cmap=plt.cm.viridis)
    extent = [t.min(), t.max(), 0, max(period)]
    bx.contour(t, np.log2(period), sig95, [-99, 1], colors='k', linewidths=2,
                extent=extent)
    bx.fill(np.concatenate([t, t[-1:] + dt, t[-1:] + dt,
                                t[:1] - dt, t[:1] - dt]),
            np.concatenate([np.log2(coi), [1e-9], np.log2(period[-1:]),
                                np.log2(period[-1:]), [1e-9]]),
            'k', alpha=0.3, hatch='x')
    #bx.set_title('b) {} Wavelet Power Spectrum ({})'.format(mother.name))
    bx.set_ylabel('Period (s)')
    #
    Yticks = 2 ** np.arange(np.ceil(np.log2(period.min())),
                                np.ceil(np.log2(period.max())))
    bx.set_yticks(np.log2(Yticks))
    bx.set_yticklabels(Yticks)
    plt.savefig('Wavelet'+str(i+1)+'.png')
    # # Fourth sub-plot, the scale averaged wavelet spectrum.
    # dx = pyplot.axes([0.1, 0.07, 0.65, 0.2], sharex=ax)
    # dx.axhline(scale_avg_signif, color='k', linestyle='--', linewidth=1.)
    # dx.plot(t, scale_avg, 'k-', linewidth=1.5)
    # dx.set_title('d) {}--{} year scale-averaged power'.format(2, 8))
    # dx.set_xlabel('Time (year)')
    # dx.set_ylabel(r'Average variance [{}]'.format(units))
    # ax.set_xlim([t.min(), t.max()])

