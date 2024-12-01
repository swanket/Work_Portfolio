#%%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import sparse
import sympy as sy
from scipy import integrate
from tqdm import tqdm
import yaml
import setup_and_integration_melting as fn
from scipy.signal import detrend
from matplotlib import ticker, cm

# import parameter file
with open("params.yaml", 'r') as stream:
    parameters = yaml.safe_load(stream)

#DOMAIN GEOMETRY  AND TIME STEPPING
year=3600*24*365 # one year in seconds
SPACE_STEP=0.05 * 1e3 # grid spacing, m  
TIME_STEP= 50 * year # time step, years 
L_crust = 30 * 1e3 # total length of the crust in m
H_ice = 3 * 1e3 # total length of ice in m
N = round(L_crust/SPACE_STEP) # number of nodes in crust
M = round(36e6 * year / TIME_STEP) # number of timesteps = total time / length timestep
t = np.arange(0,M*TIME_STEP,TIME_STEP) # time vector
z = np.arange(0,L_crust,SPACE_STEP) # distance vector crust
chamber_depth = 5e3 # depth to the magma chamber
dyke_length = L_crust-chamber_depth # length of the dyke from the base of the crust up
dz = 0.5*(1+np.tanh(10*(z-chamber_depth)/L_crust)) # shape of magmatic heating for excess melt injection


#Create Array Variables
rho_cp = parameters['rho_crust']*parameters['cp_crust'] # density and heat capacity multiplied together to make future math faster during integration



kappa = parameters['kc']/(parameters['rho_crust']*parameters['cp_crust']) # heat diffusion in the crust
tau_d = (L_crust/2)**2/kappa # diffusion timescale through half of the crust

#%%

#Percent extra mantle heating
magnitude_above = 152   # factor of mantle heat flux above the background that dqm will reach at maximum 
inject_period = 1e3 * year  # period of magma injection in seconds
delta_qm = parameters['qm'] + parameters['qm']*magnitude_above * (np.sin(2 * np.pi * t /inject_period))
# heat flux cannot be negative so make all negative values 0
for i in range(len(delta_qm)):
        if delta_qm[i] < 0:
                delta_qm[i] = 0

# fig,ax = plt.subplots(nrows = 1,ncols = 1,figsize = (10,7))
# ax.plot(t/tau_d,delta_qm,color = 'black')
# ax.set_title('Deborah Number')
# ax.set_xlabel('Time (t/$T_d$)')
# # ax.hlines(1e0,4,6)
# fig.tight_layout()
# #%%

# Ice sheet surface flux modulation 
# The height of the ice sheet through time. If constant, set constant = True, if  time variant, set constant = False
ice = True # is there an ice sheet? if True: insulation boundary condition; if False: constant temp boundary condition
orbital_period = 100000*year # milankovitch period, can be changed to be 100, 41, or 23 kyr
if ice == True:
    constant = False # Is the ice sheet height constant or periodic?
    if constant == True:
        L_ice = H_ice*np.ones(M)
    elif constant == False:
        L_ice = 2050 - 550*np.cos(2*np.pi*t/(orbital_period)) 
        # L_ice = 3000 - 1500*np.cos(2*np.pi*t/(orbital_period))
elif ice == False:
    L_ice = H_ice*np.ones(M)


# Radiogenic Heating 
constant_rad_heating = False # Is radiogenic heating constant throughout the crust or exponential decrease with depth
if constant_rad_heating == True:
    Hrad0 = 1.5e-6 # constant radiogenic heating value
    Hrad = Hrad0*np.ones(len(z))
elif constant_rad_heating == False:
    Hrad0 = 4e-6 # maximum radiogenic heating in a exponentially decreasing model
    Hrad = Hrad0 * np.exp(-z/parameters['hr'])


# Fill parameter dictionary with new paramters
parameters.update({'year':year,'SPACE_STEP':SPACE_STEP,'TIME_STEP':TIME_STEP,'L_crust':L_crust,'L_ice':L_ice,'N':N,'M':M,'z':z,'rho_cp':rho_cp,'delta_qm':delta_qm,'Hrad':Hrad,'chamber_depth':chamber_depth,'dz':dz,'dyke_length':dyke_length,'ice':ice,'tau_d':tau_d,'constant_rad_heating':constant_rad_heating,'Hrad0':Hrad0})

# Integration
print('integrating...')
T_DF = fn.initial(parameters) # initial setup
T_DF,Hmagma,Un,visc,Qvol,Qvol_total = fn.integration_DF(t,T_DF,parameters) # integrate 

# Save arrays if necessary
print('saving...')
# np.savetxt('collapse_experiments_100collapse_200recurence_temp.csv',T_DF[np.argwhere(z == 5000)[0][0],:],delimiter = ',')
# np.savetxt('collapse_experiments_100collapse_200recurence_deb.csv',Un[np.argwhere(z == 5000)[0][0],:],delimiter = ',')

# calculate ratio of eruptive:storage time along with production rate, min De, and max De
print('eruptive/storage ratio')
eruptive = 0 
storage = 0
total = 0

for i in Un[np.argwhere(z == 5000)[0][0],np.argmin(np.abs(t - 4*tau_d)):np.argmin(np.abs(t-(6*tau_d)))]:
    total += 1 # total time steps in the cycle designated in the previous line
    if i > 1:
        eruptive += 1 # time steps for one heating cycle spent in the eruptive regime
    elif i<1:
        storage += 1 # time steps for one heating cycle spent in the storage regime

# print(f'ratio = {eruptive/storage}') # ratio of eruptive vs storage
# print(f'max mass rate = {np.max(Qvol_total*(year/1e9))}') # max rate of magma influx into the crust
# print(f'average mass rate = {np.mean(Qvol_total*(year/1e9))}') # average rate of magma influx into the crust
# print(f'Min De = {np.min(Un[np.argwhere(z == 5000)[0][0],np.argmin(np.abs(t - 4*tau_d)):np.argmin(np.abs(t-(6*tau_d)))])}')
# print(f'Max De = {np.max(Un[np.argwhere(z == 5000)[0][0],np.argmin(np.abs(t - 4*tau_d)):np.argmin(np.abs(t-(6*tau_d)))])}')



print('plotting...')

# fig,ax = plt.subplots(nrows = 1,ncols = 1,figsize = (10,7))
# ax.semilogy(t[np.argmin(np.abs(t - 4*tau_d)):np.argmin(np.abs(t-(6*tau_d)))]/tau_d,Un[np.argwhere(z == 5000)[0][0],np.argmin(np.abs(t - 4*tau_d)):np.argmin(np.abs(t-(6*tau_d)))],color = 'black')
# ax.set_title('Deborah Number')
# ax.set_xlabel('Time (t/$T_d$)')
# ax.hlines(1e0,4,6)
# fig.tight_layout()

# fig,ax = plt.subplots(nrows = 1,ncols = 1,figsize = (10,7))
# ax.semilogy(t/tau_d,delta_qm,color = 'black')
# ax.set_title('Deborah Number')
# ax.set_xlabel('Time (t/$T_d$)')
# ax.hlines(1e0,4,6)
# fig.tight_layout()

# fig,ax = plt.subplots(nrows = 1,ncols = 1,figsize = (10,7))
# ax.plot(T_DF[:,0],z/1000)
# ax.plot(T_DF[:,-1],z/1000)
# ax.invert_yaxis()
# ax[1].semilogx(Un[:,-1],z/1000)
# ax[1].invert_yaxis()
# ax[1].vlines(1e0,0,30)

# fig,ax = plt.subplots(nrows = 2,ncols = 1,figsize = (10,7))
# ax[0].plot(Hmagma_excess[:,-10000:-9000],z)
# ax[1].plot(Hmagma[:,-10000:-9000],z)

fig,ax = plt.subplots(nrows = 2,ncols = 1,figsize = (10,7))
ax[0].plot(t/tau_d,T_DF[np.argwhere(z == 5000)[0][0],:],color = 'black')
# ax[1].semilogy(t/tau_d,visc[np.argwhere(z == 5000)[0][0],:],color = 'black')
ax[1].semilogy(t/tau_d,Un[np.argwhere(z == 5000)[0][0],:],color = 'black')
fig.suptitle('Variations in Insulation at the top of the Crust')
ax[0].set_title('Temperature')
# ax[1].set_title('Viscosity')
ax[1].set_title('Deborah Number')
ax[1].set_xlabel('Time (t/$T_d$)')
ax[1].hlines(1e0,1,6)
ax[0].grid(alpha = 0.5)
fig.tight_layout()

# fig,ax = plt.subplots(nrows = 1,ncols = 1,figsize = (10,7))
# ax.plot(t,(Qvol_total)*(year/1e9),c = 'k')
# ax.grid(alpha = 0.5)
# plt.vlines(t[np.argmin(np.abs(t-3*inject_period))],0,0.1)
# # ax.vlines(t[np.argmin(np.abs(t - 2*tau_d))],0,0.1)
# # ax.vlines(t[np.argmin(np.abs(t-(2*tau_d+2e7*year)))],0,0.1)
# # ax.hlines(np.mean(Qvol_total*(year/1e9)),1,6)
# # ax.vlines(2,0,0.1)
# # ax.vlines(2+1e7*year/tau_d,0,0.1)
# # ax.set_title('Mantle Heat Flux Variations with $T_{inject}$ = 1e7',fontname = 'Times New Roman',fontsize = 15)
# ax.set_xlabel('Time ($\\tau_d$)',fontname = 'Arial',fontsize = 18)
# ax.set_ylabel('$\delta q_m$ (W/m^2)',fontname = 'Arial',fontsize = 18)
# fig.tight_layout()

# fig,ax = plt.subplots(nrows = 1,ncols = 1,figsize = (5,2.5))
# ax.plot(t/tau_d,L_ice,c = 'k')
# ax.grid(alpha = 0.5)
# ax.set_xlim(0,0.1)
# # ax.set_title('Mantle Heat Flux Variations with $T_{inject}$ = 1e7',fontname = 'Times New Roman',fontsize = 15)
# ax.set_xlabel('Time ($\\tau_d$)',fontname = 'Arial',fontsize = 18)
# ax.set_ylabel('$H_i$ (m)',fontname = 'Arial',fontsize = 18)
# fig.tight_layout()


# fig,ax = plt.subplots(nrows = 1,ncols = 1,figsize = (10,7))
# ax.plot(t/tau_d,tau_forcing,c = 'black')
# ax1 = ax.twinx()
# ax1.plot(t/tau_d,Qvol,c = 'r')
# fig.tight_layout()
# fig,ax = plt.subplots(nrows = 1,ncols = 1,figsize = (10,7))
# ax.plot(t/tau_d,Qvol,c = 'black')
# fig.tight_layout()

# fig,ax = plt.subplots(nrows = 1, ncols = 3,figsize = (10,7))
# for i in [0,np.argmin(np.abs(t - 0.1*tau_d)),np.argmin(np.abs(t - 1*tau_d)),np.argmin(np.abs(t - 3*tau_d)),np.argmin(np.abs(t - 6*tau_d))]:#[0,1/5,2/5,3/5,4/5,1]:
#     ax[0].plot(T_DF[:,i],z/1000,label = f'{np.round(t[i]/(tau_d),1)} $\\tau_d$')
#     ax[1].semilogx(Un[:,i],z/1000,label = f'{np.round(t[i]/(tau_d),1)} $\\tau_d$')
#     ax[2].semilogx(visc[:,i],z/1000,label = f'{np.round(t[i]/(tau_d),1)} $\\tau_d$')
# #ax.vlines(np.max(T_DF),0,30)
# ax[0].vlines(np.max(T_DF),0,30)
# ax[0].grid(alpha = 0.5)
# ax[0].invert_yaxis()
# ax[0].legend()
# ax[0].set_title('Spatial Temperature Plots through Time',fontname = 'Times New Roman',fontsize = 15)
# ax[0].set_xlabel('Temperature (K)',fontname = 'Times New Roman',fontsize = 12)
# ax[0].set_ylabel('Depth (km)',fontname = 'Times New Roman',fontsize = 12)
# ax[1].grid(alpha = 0.5)
# ax[1].invert_yaxis()
# #ax[1].vlines(1e0,0,30)
# ax[1].set_title('Spatial Deborah Number Plots through Time',fontname = 'Times New Roman',fontsize = 15)
# ax[1].set_xlabel('Deborah Number',fontname = 'Times New Roman',fontsize = 12)
# ax[1].set_ylabel('Depth (km)',fontname = 'Times New Roman',fontsize = 12)
# ax[2].grid(alpha = 0.5)
# ax[2].invert_yaxis()
# ax[2].set_title('Spatial Viscosiy Plots through Time',fontname = 'Times New Roman',fontsize = 15)
# ax[2].set_xlabel('Viscosity',fontname = 'Times New Roman',fontsize = 12)
# ax[2].set_ylabel('Depth (km)',fontname = 'Times New Roman',fontsize = 12)

# fig.tight_layout()


# # fig,ax = plt.subplots(nrows = 1, ncols = 1,figsize = (10,7))
# # for i in [0,1/5,2/5,3/5,4/5,1]:
# #     ax.plot(np.log10(Un[:,int((M-1)*i)]),z/1000,label = f'{np.round(t[int((M-1)*i)]/(year*1000000))} Myr')
# # ax.invert_yaxis()
# # ax.grid(alpha = 0.5)
# # fig.tight_layout()

# X,Y = np.meshgrid(t/tau_d,z/1000)
# fig,ax = plt.subplots(nrows = 1,ncols = 1,figsize = (10,7))
# ax1 = ax.contourf(X,Y,Hmagma,cmap = 'plasma')#,levels = np.arange(0,3.2,0.1)*1e-7)
# ax.invert_yaxis()
# ax.set_title(f'Volumetric Heating from Excess Melt (W/m^3)',fontsize = 30,fontname = 'Times New Roman') # - Period of Injection = {np.format_float_scientific(inject_period/year)}
# ax.set_xlabel('Time (Diffusion Times)',fontsize = 20,fontname = 'Times New Roman')
# ax.set_ylabel('Depth (km)',fontsize = 20,fontname = 'Times New Roman')

# fig.colorbar(ax1,ax=ax)
# fig.tight_layout()



# fig,ax = plt.subplots(nrows = 2,ncols = 1,figsize = (10,7))
# ax[0].plot(t/year, np.gradient(Hmagma_chamber))
# ax[0].set_title('Gradient')
# ax[1].plot(t/year,Hmagma_chamber)
# ax[1].set_title('Inlfux')
# fig.tight_layout()
plt.show()
print('done...')


