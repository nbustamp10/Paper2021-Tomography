# -*- coding: utf-8 -*-
"""
Created on Wed Apr  7 17:08:49 2021
El codigo grafica los tomogramas en distintos instantes de tiempo y los tomogramas promedio (ultimos 600 tomogramas)
Tambien analiza la evolucion del area de sedimento en el tiempo.
El codigo tiene las siguientes funciones:
derivada_c: Estima la derivada de la concentracion * area.
ponderar: Pondera anchos en la seccion transversal.
fall_sedimentation: Estima la velocidad de sedimentacion de una particula.
filtro1: Es un filtro pasa baja en el tiempo.
@author: Bustmante-Penagos N.
"""
import os
import glob
import re
import scipy.misc
import scipy.io as sio
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import axes3d
from matplotlib import cm
from scipy import ndimage
from scipy.optimize import curve_fit
import pandas as pd
import seaborn as sns
from sympy.solvers import solve
from sympy import Symbol
#%% Cargar datos
u=sio.loadmat("H:\\Paper\\Jorge_Paper\\V_Perfil.mat")                           #Despues de las particulas
sstress=sio.loadmat("H:\\Paper\\Jorge_Paper\\ttotal.mat")                       #Despues de las particulas
Y=sio.loadmat("H:\\Paper\\Jorge_Paper\\Y.mat")
uap=sio.loadmat("H:\\Paper\\Jorge_Paper\\U_apf.mat")                            # U antes de las particulas
sstressap=sio.loadmat("H:\\Paper\\Jorge_Paper\\ttotal_apf.mat")                 # SStress antes de las particulas
Yap=sio.loadmat("H:\\Paper\\Jorge_Paper\\Y_apf.mat")                            # Y antes de las particulas
Iac=pd.read_csv('qiui.txt',sep=',',header=None)                                 # Ajuste 
Iac=np.array(Iac)
qia=pd.read_csv('qia.txt',sep=',',header=None)
qia=np.array(qia)
Qint=pd.read_csv('Qvqi.txt',sep=',',header=None)
Qint=np.array(Qint)
ruta='H:\Paper\Jorge_Paper\gasto_solido'
taue=pd.read_csv('tau_afterp.csv',sep=',',header=0)                             # Esfuerzo de corte experimental
taue=pd.DataFrame(taue, columns=['tau_P1','tau_P2','tau_P3'])
taue=np.array(taue)

#%% Funciones
def derivada_c(data1,data2,dt):
    qs=np.zeros(len(data1))
    for i in range(len(data1)-1):
        qs[i]=(data1[i+1]*data2[i+1]-data1[i]*data2[i])/dt
    return qs

def ponderar(data,anchos):
    b=data.shape
    t=np.zeros(len(data))
    aux=np.zeros((b[0],b[1]))
    for j in range(b[0]):
        for i in range(b[1]):
            aux[j,i]=data[j,i]*2*anchos[i]
        t[j]=sum(aux[j,:])/D
    return t

def fall_Sedimentation(ds,nu,rho,rhos,g):
    ws= Symbol('ws')
    falls=solve(-ws**2/(g*ds*(rhos/rho-1))+4/3*1/(24/(ds*ws/nu)*(1+0.15*(ds*ws/nu)**0.5+0.017*(ds*ws/nu))-0.208/(1+10**4*(ds*ws/nu)**-0.5)),ws)
    return falls

def filtro1(data,n):
    for i in range(0,len(data)-n):
        if abs(data[i]-data[i:i+n].mean())> 0.5*data[i-int(n/2):i-1].std():
            data[i]=data[i:i+int(n/2)].mean()
        # if data[i]==0:
        #     data[i]=data[i:i+int(n/2)].mean()
    return data
    
#%% Constantes
n=10                                                                             # Tamaño de la ventena para el filtro
D=0.1                                                                           # Diametro de la tuberia o ancho del canal
rho=1000; rhos=2650                                                             # Densidad del agua y del sedimento en kg/m**3
nu=1e-6                                                                         # Viscocidad cinematica del agua 
a=11.5; b=0.7                                                                   # Constantes para la velocidad de saltacion
R=1.65; g=9.81                                                                  # R: Densidad relativa del sedimento, g: Fuerza de gravedad
tauc=0.035                                                                      # Esuerzo de corte critico - Shields=0.06, Parker=0.035
ds=122e-6                                                                       # Diametro del sedimento fino en metros
p2=(D/20)**2                                                                    # Area del pixel del tomografo en metros
p=[D/2, D/4, D/8]                                                               # Localizacion de planos
af=[(p[0]-p[1])/2, (p[0]-p[1])/2+(p[1]-p[2])/2, (p[1]-p[2])/2+p[2]]             # Areas aferentes planos
taup=ponderar(taue,af)                                                          # Esfuerzo de corte ponderado en la sección transversal
uf=(taup/rho)**0.5                                                              # Velocidad friccional hallada experimentalmente
ub=uf*a*(1-b*((taup/(R*ds**3*g))/tauc)**-0.5)                                   # Velocidad de saltacion
wsed=fall_Sedimentation(ds,nu,rho,rhos,g)
#%% Cargar archivos de tomogramas        
files=glob.glob(os.path.join(ruta,'Exp_*'))
r=re.compile(r"(\d+)")
sfiles=sorted(files, key=lambda x: int(r.search(x).group()))
pm=np.zeros((20,20,len(files)))

time=np.zeros(len(files))
cm=np.zeros((15000,20))                                                         # Concentracion media promedio de los dos planos
cm1=np.zeros((15000,20))                                                        # Concentracion media plano 1
cm2=np.zeros((15000,20))                                                        # Concentracion media plano 2
areas=np.zeros((15000,20))                                                      # Area con sedimento promedio de los dos planos
areas1=np.zeros((15000,20))                                                     # Area con sedimento plano 1
areas2=np.zeros((15000,20))                                                     # Area con sedimento plano 2
areasf=np.zeros((15000,20))                                                      # Area con sedimento promedio de los dos planos
areas1f=np.zeros((15000,20))                                                     # Area con sedimento plano 1
areas2f=np.zeros((15000,20))                                                     # Area con sedimento plano 2
dz1=[np.nan,6,2,7,np.nan,7,np.nan,9,np.nan,8,3,np.nan,2,np.nan,np.nan,4,np.nan,np.nan,2,np.nan]# Nivel del fino
#%% Concentracion media considerando solo pixeles con concentracion mayor a cero
for t in range(len(sfiles)):
    tomogram=sio.loadmat(sfiles[t])
    p=(3.29)*tomogram['Tomografo_p']
    pt1=(3.29)*tomogram['Tomografo_pl1']
    pt2=(3.29)*tomogram['Tomografo_pl2']
    b=p.shape
    time[t]=b[2]
    for k in range(b[2]):
        c=0
        c1=0
        c2=0
        s=0
        s1=0
        s2=0
        try:
            for j in range(b[1]):
                for i in range(int(b[0]/2)):
                    if (np.isnan(p[i,j,k])==False and p[i,j,k]>0):
                        c=p[i,j,k]+c
                        s  +=1
                    if (np.isnan(pt1[i,j,k])==False and pt1[i,j,k]>0):
                        c1=pt1[i,j,k]+c1
                        s1 +=1
                    if (np.isnan(pt2[i,j,k])==False and pt2[i,j,k]>0):
                        c2=pt2[i,j,k]+c2
                        s2 +=1
            areas[k,t]=s*p2
            areas1[k,t]=s1*p2
            areas2[k,t]=s2*p2
            cm[k,t]=c/s   
            cm1[k,t]=c1/s1
            cm2[k,t]=c2/s2            
        except:
             s=0; s1=0; s2=2
             continue

for i in range(20):
    areasf[:,i]=filtro1(areas[:,i],n)
    areas1f[:,i]=filtro1(areas1[:,i],n)
    areas2f[:,i]=filtro1(areas2[:,i],n)  

 
# A=D**2/8*(2*np.arccos((D/2-hs)/(D/2)) -np.sin(2*np.arccos((D/2-hs)/(D/2)) ))  
gastos=np.zeros((int(min(time))-1,20))
R=1.65
ds=122e-6
for i in range(b[1]):
    dt=1
    data1=cm[int(time[i])-int(min(time)):int(time[i])-1,i]
    data2=(areas[int(time[i])-int(min(time)):int(time[i])-1,i])*0.3
    qs=derivada_c(data1,data2,dt)
    gastos[:,i]=qs
#%% Grafica de areas

plt.figure()
plt.rcParams.update({'font.size': 20})
fig2,axs = plt.subplots(4, 5,figsize=(40,50),sharey=True)
sns.set_context("paper")
sns.set_style("darkgrid", {"axes.facecolor": ".9"})
for i in range(b[1]):
    n=i%5
    m=int(i/5)
    axs[m,n].plot(areasf[:int(time[i]),i],color='black',label='Promedio')
    axs[m,n].plot(areas1f[:int(time[i]),i],color='blue',label='Plano 1')
    axs[m,n].plot(areas2f[:int(time[i]),i],color='red',label='Plano 2')
    for ax in axs.flat:
        ax.set_xlabel('Tomogramas', fontsize=24)
        ax.set_ylabel('Areas', fontsize=20)
        ax.grid(True,which="both", ls="-",color='gray')
        #plt.legend(fontsize=24,ncol=3, bbox_to_anchor=[-0.7,-.6] ,loc='lower center')
        ax.tick_params(axis='both', which='major', labelsize=16)
        #ax.xticks(fontsize=17)  
        # ax.set_xlim(0,2244)
        ax.set_ylim(0,0.004)
        # ax.set_yscale('log')
        # ax.set_xscale('log')
        ax.legend(fontsize=14,ncol=1,loc=2)
    # for ax in axs.flat:
    #     ax.label_outer()
#%% Grafica de gasto solido

plt.figure()
plt.rcParams.update({'font.size': 20})
fig2,axs = plt.subplots(4, 5,figsize=(40,60),sharey=True)
sns.set_context("paper")
sns.set_style("darkgrid", {"axes.facecolor": ".9"})
for i in range(b[1]):
    n=i%5
    m=int(i/5)
    axs[m,n].plot(gastos[:,i]/(9.81*R*ds**3)**0.5)
    
    for ax in axs.flat:
        ax.set_xlabel('Tomogramas', fontsize=24)
        ax.set_ylabel('Q$_s*$', fontsize=20)
        ax.grid(True,which="both", ls="-",color='gray')
        
        #plt.legend(fontsize=24,ncol=3, bbox_to_anchor=[-0.7,-.6] ,loc='lower center')
        ax.tick_params(axis='both', which='major', labelsize=16)
        #ax.xticks(fontsize=17)  
        ax.set_xlim(0,2244)
        ax.set_ylim(-1,5)
        # ax.set_yscale('log')
        # ax.set_xscale('log')
        #ax.legend(fontsize=14,ncol=1,loc=2)
    for ax in axs.flat:
        ax.label_outer()
#%% Grafica los tomogramas en diferentes tiempos


# w=0
# plt.figure()
# plt.rcParams.update({'font.size': 20})
# fig2,axs = plt.subplots(10, 6,figsize=(40,60),sharey=True)
# sns.set_context("paper")
# sns.set_style("darkgrid", {"axes.facecolor": ".9"})
# for t in range(len(files)):
#     tomogram=sio.loadmat(sfiles[t])
#     p=(3.29)*tomogram['Tomografo_p']
#     pt1=(3.29)*tomogram['Tomografo_pl1']; pt2=(3.29)*tomogram['Tomografo_pl2']  # Tomogramas plano 1 y plano 2 
#     b=p.shape
#     pmt1=np.zeros((b[0], b[1]))
#     plt1=np.zeros((b[0], b[1]))
#     pmt2=np.zeros((b[0], b[1]))
#     plt2=np.zeros((b[0], b[1]))
#     time[t]=b[2]
#     ttom=[b[2]-802,b[2]-602,b[2]-402,b[2]-302, b[2]-202,b[2]-2]
    
#     #dz1=[np.nan,np.nan,np.nan, np.nan,np.nan,np.nan,np.nan,np.nan,np.nan,8,6,8,np.nan,9,np.nan,9,7,np.nan,np.nan,13]# Nivel del fino
#     dz=dz1[t]
#     if (np.isnan(dz)==False):
#         for i in range(len(ttom)):
#             pos=axs[w,i].imshow(p[:,:,ttom[i]],vmax=1,vmin=0)
#             axs[w,i].set_title(str(sfiles[t]))
#             axs[w,i].set_xlabel('Tomograma '+str(ttom[i]))
#             plt.colorbar(pos, ax=axs[w,i])
#             plt.subplots_adjust(left=0.125,
#                     bottom=0.1, 
#                     right=0.9, 
#                     top=0.9, 
#                     wspace=-0.2, 
#                     hspace=0.3)        
#         w=w+1    
        

# 'Grafica el promedio de los ultimos 600 tomogramas'    
# plt.figure()
# plt.rcParams.update({'font.size': 20})
# fig2,axs = plt.subplots(4, 5,figsize=(22,22),sharey=True)
# sns.set_context("paper")
# sns.set_style("darkgrid", {"axes.facecolor": ".9"})

# for t in range(len(files)):
#     n=t%5
#     m=int(t/5)
#     tomogram=sio.loadmat(sfiles[t])
#     p=(3.29)*tomogram['Tomografo_p']
#     b=p.shape
#     pm1=np.zeros((b[0], b[1]))
#     pl1=np.zeros((b[0], b[1]))
#     # pmt1=np.zeros((b[0], b[1]))
#     # plt1=np.zeros((b[0], b[1]))
#     # pmt2=np.zeros((b[0], b[1]))
#     # plt2=np.zeros((b[0], b[1]))
#     time[t]=b[2]
#     ttom=[b[2]-802,b[2]-602,b[2]-402,b[2]-302, b[2]-202,b[2]-2]    
#     for i in range(20):
#         for j in range(20):
#             pm1[i,j]=p[i,j, b[2]-600:].mean()                                   # Promedio de los ultimos 600 tomogramas
#             pm[i,j,t]=pm1[i,j]                                                  # Promedio de los ultimos 600 tomogramas como una matriz 3D
#             # pmt1[i,j]=p[i,j, b[2]-600:].mean()                                  # Promedio de los ultimos 600 tomogramas
#             # pm1[i,j,t]=pmt1[i,j]                                                # Promedio de los ultimos 600 tomogramas como una matriz 3D
#             # pmt2[i,j]=p[i,j, b[2]-600:].mean()                                  # Promedio de los ultimos 600 tomogramas
#             # pm2[i,j,t]=pmt2[i,j]                                                # Promedio de los ultimos 600 tomogramas como una matriz 3D
#     axs[m,n].set_title(str(sfiles[t]))
#     pos=axs[m,n].imshow(pm[:,:,t],vmax=1,vmin=0)
#     plt.colorbar(pos, ax=axs[m,n],shrink=0.7)
#     plt.subplots_adjust(left=0.125,
#                 bottom=0.1, 
#                 right=0.9, 
#                 top=0.9, 
#             wspace=0.1, 
#             hspace=0.2)