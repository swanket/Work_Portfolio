#%%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.io import netcdf_file
import netCDF4
from scipy.ndimage import gaussian_filter1d
from scipy import signal
from statsmodels.tsa.tsatools import detrend
from geopy import distance


#####################################################################################################################################
# Setup
#####################################################################################################################################


first_million_years = netCDF4.Dataset('fort.92_1f_extract.nc') # First 1 Million years of Pollard's Ice SHeet Reconstruction
second_million_years = netCDF4.Dataset('fort.92_2f_extract.nc') # First 1 Million years of Pollard's Ice SHeet Reconstruction
third_million_years = netCDF4.Dataset('fort.92_3f_extract.nc') # First 1 Million years of Pollard's Ice SHeet Reconstruction
fourth_million_years = netCDF4.Dataset('fort.92_4f_extract.nc') # First 1 Million years of Pollard's Ice SHeet Reconstruction
fifth_million_years = netCDF4.Dataset('fort.92_5f_extract.nc') # First 1 Million years of Pollard's Ice SHeet Reconstruction

# Build the thickness Arrays from Pollard files
h = first_million_years.variables['h']
h2 = second_million_years.variables['h']
h3 = third_million_years.variables['h']
h4 = fourth_million_years.variables['h']
h5 = fifth_million_years.variables['h']
thickness_all = h[:,:,:]
thickness_all2 = h2[:,:,:]
thickness_all3 = h3[:,:,:]
thickness_all4 = h4[:,:,:]
thickness_all5 = h5[:,:,:]
thickness_all_5Myr = np.concatenate((thickness_all5,thickness_all4,thickness_all3,thickness_all2,thickness_all))
thickness0 = h[0,:,:]

# Build Meshgrid for plotting
x = np.arange(0,141)
y = np.arange(0,141)
X,Y =np.meshgrid(x,y)

# Build the Longitude Arrays from Pollard files
MB_lat = -77.5293 #-76.05
MB_lon = -167 #-136
alatd = first_million_years.variables['alatd']
alond = first_million_years.variables['alond']
latitude = alatd[:,:]
longitude = alond[:,:]
# lon = plt.contour(X,Y,longitude, [-167,-136])
# plt.clabel(lon, inline=True, fontsize=15)

#%%

# Build Time Arrays from Pollard files
t = first_million_years.variables['time']
time_first_million = t[:]
t2 = second_million_years.variables['time']
time_second_million = t2[:]
t3 = third_million_years.variables['time']
time_third_million = t3[:]
t4 = fourth_million_years.variables['time']
time_fourth_million = t4[:]
t5 = fifth_million_years.variables['time']
time_fifth_million = t5[:]
time_all = np.concatenate((t5,t4,t3,t2,t))

# Build an array which gives the distance of each point from Mount Berlin
mount_berlin = (MB_lat,MB_lon)
d = np.zeros(np.shape(latitude))
for i in range(np.shape(latitude)[0]):
    for j in range(np.shape(latitude)[1]):
        d[i,j] = distance.distance(mount_berlin,(latitude[i,j],longitude[i,j])).km # distance from all nodes to Mount Berlin

# Find closest point to Mount Berlin (Or Mount Erebus)
min_index_flat = np.argmin(d)
row_index, col_index = np.unravel_index(min_index_flat, d.shape)


# Build an array which has a 0 if the point at that time is land or grounded ice, or 0 if it is ocean or floating ice
maskwater1 = first_million_years.variables['maskwater']
maskwater2 = second_million_years.variables['maskwater']
maskwater3 = third_million_years.variables['maskwater']
maskwater4 = fourth_million_years.variables['maskwater']
maskwater5 = fifth_million_years.variables['maskwater']
ocean_or_floating = np.concatenate((maskwater5[:,:,:],maskwater4[:,:,:],maskwater3[:,:,:],maskwater2[:,:,:],maskwater1[:,:,:]))


points_within_500_km = np.argwhere(d <= 500) # Array of the points within 500 km Where rows are the points and column 0 is X and columb 1 is Y
grounded_points_within_500_km = np.argwhere((d <= 500) & (np.sum(ocean_or_floating,axis = 0) == 0)) # Find the grounded points within 500 km
x_within_500km = points_within_500_km[:,0] # Find the X coordinate of points within 500 km
y_within_500km = points_within_500_km[:,1]# Find the Y coordinate of points within 500 km


# Create an Array which has all the points within 500 km of Mount Berlin which are grounded and who's ice thickness never reaches 0
doesnt_go_to_zero = []
for i in range(len(grounded_points_within_500_km[:,1])):
    if np.any(thickness_all_5Myr[:,grounded_points_within_500_km[i,1],grounded_points_within_500_km[i,0]] == 0):
        doesnt_go_to_zero.append(False)
        
    else:
        doesnt_go_to_zero.append(True)

grounded_points_within_500_above_0 = grounded_points_within_500_km[doesnt_go_to_zero]

#####################################################################################################################################
# BIG MAP PLOTTING
#####################################################################################################################################

fig,ax = plt.subplots(nrows = 1,ncols = 1,figsize = (10,10))
thick = ax.contourf(X,Y,thickness0,cmap = 'Blues_r',levels = 400) # Background heatmap of ice thickness
lat = ax.contour(X,Y,latitude,colors = 'white',linestyles = 'solid') # Latitude lines
lon = ax.contour(X,Y,longitude,colors = 'white',linestyles = 'solid') # Longitude Lines
ax.clabel(lat, inline=True, fontsize=15)
ax.scatter(points_within_500_km[:,1],points_within_500_km[:,0],color = 'yellow',alpha = 0.5) # Scatter of points within 500 km
ax.scatter(np.argwhere(np.sum(ocean_or_floating,axis = 0) == 0 )[:,1],np.argwhere(np.sum(ocean_or_floating,axis = 0) == 0 )[:,0],color = 'r',alpha = 0.5) # Scatter of grounded points
ax.scatter(grounded_points_within_500_above_0[:,1],grounded_points_within_500_above_0[:,0],color = 'g',alpha = 0.5) # Scatter of grounded points within 500 km
ax.scatter(points_within_500_km[290,1],points_within_500_km[290,0],color = 'b') # The point compated to LR04
ax.scatter(col_index,row_index,color = 'black',marker='^',s = 200) # Mount Berlin


ax.tick_params(axis='both', which='major', labelsize=20)
for tick in ax.get_xticklabels():
    tick.set_fontname("Arial")
for tick in ax.get_yticklabels():
    tick.set_fontname("Arial")
fig.tight_layout()


fig,ax = plt.subplots(nrows = 2,ncols = 2,figsize = (10,10),sharex = True, sharey = True)
thick = ax[0,0].contourf(X,Y,thickness0,cmap = 'Blues_r',levels = 200)
lat = ax[0,0].contour(X,Y,latitude,colors = 'white',linestyles = 'solid')
lon = ax[0,0].contour(X,Y,longitude,colors = 'white',linestyles = 'solid')
ax[0,0].clabel(lat, inline=True, fontsize=15)
ax[0,0].scatter(col_index,row_index,color = 'black',marker='^',s = 200)
ax[0,1].contourf(X,Y,h[-1,:,:],cmap = 'Blues_r',levels = 200)
lat = ax[0,1].contour(X,Y,latitude,colors = 'white',linestyles = 'solid')
lon = ax[0,1].contour(X,Y,longitude,colors = 'white',linestyles = 'solid')
ax[0,1].clabel(lat, inline=True, fontsize=15)
ax[0,1].scatter(col_index,row_index,color = 'black',marker='^',s = 200)
ax[1,0].contourf(X,Y,h3[-1,:,:],cmap = 'Blues_r',levels = 200)
lat = ax[1,0].contour(X,Y,latitude,colors = 'white',linestyles = 'solid')
lon = ax[1,0].contour(X,Y,longitude,colors = 'white',linestyles = 'solid')
ax[1,0].clabel(lat, inline=True, fontsize=15)
ax[1,0].scatter(col_index,row_index,color = 'black',marker='^',s = 200)
ax[1,1].contourf(X,Y,h5[-1,:,:],cmap = 'Blues_r',levels = 200)
lat = ax[1,1].contour(X,Y,latitude,colors = 'white',linestyles = 'solid')
lon = ax[1,1].contour(X,Y,longitude,colors = 'white',linestyles = 'solid')
ax[1,1].clabel(lat, inline=True, fontsize=15)
ax[1,1].scatter(col_index,row_index,color = 'black',marker='^',s = 200)
for axes in ax.reshape(-1):
    axes.tick_params(axis='both', which='major', labelsize=20)
    for tick in axes.get_xticklabels():
        tick.set_fontname("Arial")
    for tick in axes.get_yticklabels():
        tick.set_fontname("Arial")
fig.tight_layout()

#####################################################################################################################################
# Comparing Pollard and LR04
#####################################################################################################################################

# Bandpass Pollard to isolate the Obliquity Signal in the Data
fs = 1/5000
high = 1/30000
low = 1/55000
sos_clean = signal.butter(6,1/200000,'highpass',fs = fs,output = 'sos')
clean = np.flip(signal.sosfilt(sos_clean,np.flip(detrend(thickness_all[:,x_within_500km[290],y_within_500km[290]],order = 2))))
sos = signal.butter(6,[low,high],'bandpass',fs = fs,output = 'sos')
bandpassed_290 = signal.sosfilt(sos,clean)


fig,ax = plt.subplots(figsize = (10,7))
ax.plot(-time_first_million/1000,bandpassed_290,color = 'black',label = 'Ice Thickness')
ax2 = ax.twinx()
ax2.plot(-time_first_million/1000,np.gradient(bandpassed_290),color = 'red', label = 'First Derivative of Ice Thickness')
# ax2.plot(-time_first_million/1000,thickness_all[:,x_within_500km[315],y_within_500km[315]])
ax.set_xlim(0,140)
# plt.ylim(-50,50)
ax.tick_params(axis='both', which='major', labelsize=15)
for tick in ax.get_xticklabels():
    tick.set_fontname("Arial")
for tick in ax.get_yticklabels():
    tick.set_fontname("Arial")
ax2.tick_params(axis='both', which='major', labelsize=15)
for tick in ax2.get_xticklabels():
    tick.set_fontname("Arial")
for tick in ax2.get_yticklabels():
    tick.set_fontname("Arial")

fig.legend(fontsize = 15)



plt.show()
# %%
