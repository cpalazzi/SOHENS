#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jun 23 09:13:25 2020

@author: carlopalazzi
"""

# %%
# %matplotlib auto
# Sets plots to appear in separate window
# Comment out and restart kernel to reset to inline plots
# import ROOT
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from scipy.interpolate import griddata
import random
from scipy import stats
from multiprocessing import Pool
import time 
from numba import jit

# %%
# Read count and positions datasets
dfenergyncap = pd.read_csv('dfenergyncap.csv')
dfe = pd.read_csv('dfe_cylindrical.csv')

# List (array) of energies in data
energy_list = dfe['energy'].unique()

# %%
# Function to sample neutron capture multiplicity at given energy
@jit(cache=True)
def ncap_num(energy, num_n=1):
    
    # Split x and y data
    points = dfenergyncap[['energy','ncapcount']].to_numpy()
    values = dfenergyncap['eventcount'].to_numpy()

    num_n_cap = range(40)
    energy,num_n_cap = np.meshgrid(energy,num_n_cap)
    # interpolate
    grid_z0 = griddata(points, values, (energy,num_n_cap), method='linear')
    # Replace nans with 0
    grid_z0 = np.nan_to_num(grid_z0)
    # Calculate area under interpolation at given energy
    area = np.trapz(grid_z0,num_n_cap,axis=0)
    # Normalise
    grid_z0_norm = grid_z0/area
    # Sample
    choicelist = random.choices(num_n_cap, weights=grid_z0_norm, k=num_n)
    choicelist = [int(i) for i in choicelist]

    return choicelist

# %%
# Function to sample from dataset at given energy
@jit(cache=True)
def sample_at_data_energy(en, num=1):
    
    """
    Function that generates sample of rho at z at given energy
    from stored data.
    """

    m1, m2 = dfe.loc[dfe['energy'] == en]['rho'], dfe.loc[dfe['energy'] == en]['z']
    xmin = dfe.loc[dfe['energy'] == en]['rho'].min()
    xmax = dfe.loc[dfe['energy'] == en]['rho'].max()
    ymin = dfe.loc[dfe['energy'] == en]['z'].min()
    ymax = dfe.loc[dfe['energy'] == en]['z'].max()

    X, Y = np.mgrid[xmin:xmax:100j, ymin:ymax:100j] # makes 100 by 100 grid
    positions = np.vstack([X.ravel(), Y.ravel()])
    values = np.vstack([m1, m2])
    kernel = stats.gaussian_kde(values, 0.5)
    Z = np.reshape(kernel(positions).T, X.shape)

    # Generate the bins for each axis
    x_bins = np.linspace(xmin, xmax, Z.shape[0]+1)
    y_bins = np.linspace(ymin, ymax, Z.shape[1]+1)

    # Find the middle point for each bin
    x_bin_midpoints = x_bins[:-1] + np.diff(x_bins)/2
    y_bin_midpoints = y_bins[:-1] + np.diff(y_bins)/2

    # Calculate the Cumulative Distribution Function(CDF)from the PDF
    cdf = np.cumsum(Z.ravel())
    cdf = cdf / cdf[-1] # Normalisation

    # Create random data
    values = np.random.rand(num)

    # Find the data position
    value_bins = np.searchsorted(cdf, values)
    x_idx, y_idx = np.unravel_index(value_bins,
                                    (len(x_bin_midpoints),
                                    len(y_bin_midpoints)))

    # Create the new data
    new_data = np.column_stack((x_bin_midpoints[x_idx],
                                y_bin_midpoints[y_idx]))
    new_x, new_y = new_data.T

    return new_x, new_y

# %%
# Function to sample from dataset at given energy
@jit(cache=True)
def sample_var_at_data_energy(var, en, num=1):
    
    """
    Function that generates sample of variable at given energy
    from stored data.
    """

    m1 = dfe.loc[dfe['energy'] == en][var]
    xmin = dfe.loc[dfe['energy'] == en][var].min()
    xmax = dfe.loc[dfe['energy'] == en][var].max()

    X = np.mgrid[xmin:xmax:100j] # makes 100 by 100 grid
    positions = np.vstack([X.ravel()])
    values = np.vstack([m1])
    kernel = stats.gaussian_kde(values, 0.5)
    Z = np.reshape(kernel(positions).T, X.shape)

    # Generate the bins for each axis
    x_bins = np.linspace(xmin, xmax, Z.shape[0]+1)
    
    # Find the middle point for each bin
    x_bin_midpoints = x_bins[:-1] + np.diff(x_bins)/2
    
    # Calculate the Cumulative Distribution Function(CDF)from the PDF
    cdf = np.cumsum(Z.ravel())
    cdf = cdf / cdf[-1] # Normalisation

    # Create random data
    values = np.random.rand(num)

    # Find the data position
    value_bins = np.searchsorted(cdf, values)
    x_idx = np.unravel_index(value_bins,
                                    (len(x_bin_midpoints)))

    # Create the new data
    new_data = np.column_stack((x_bin_midpoints[x_idx]))
    new_x = new_data.T

    return new_data

def combine_rho_z_t_samples(en,num=1):
    t = sample_var_at_data_energy('t', en, num).T
    rho, z = sample_at_data_energy(en, num)
    return np.column_stack((rho,z,t))


# %%
@jit(cache=True)
def interp_rho_z_t(energy, num=1):

    """
    Gives samples of rho and z at given energy based on values
    interpolated from data.
    """

    loc_left = np.searchsorted(energy_list, energy, side='right')-1
    loc_right = np.searchsorted(energy_list, energy)
    en1 = energy_list[loc_left]
    en2 = energy_list[loc_right]


    if __name__ == '__main__':
        p = Pool(2)
        results = p.starmap(combine_rho_z_t_samples, [(en1, num), (en2, num)])

    samplearr1 = results[0]
    samplearr2 = results[1]
    i = 0
    list_out = []

    while i < np.shape(samplearr1)[0]:
        new_data1 = samplearr1[i]
        new_data2 = samplearr2[i]
        values_to_interp = np.vstack([new_data1, new_data2])#.transpose()
        list_out.append(griddata([en1, en2], values_to_interp, energy, method='linear', rescale=True))
        i+=1

    df_out_rho_z = pd.DataFrame(np.row_stack(list_out), columns=['rho', 'z', 't'])

    return df_out_rho_z

# %%
@jit(cache=True)
def ncap_sim(energy, num_n=1):
    
    # Get number of ncaps
    num_ncaps = sum(ncap_num(energy, num_n))
    # Get positions
    if energy in energy_list:
        return pd.DataFrame(combine_rho_z_t_samples(energy, num_ncaps), columns=['rho', 'z', 't'])
    else:
        return interp_rho_z_t(energy, num_ncaps)

# %%
# Run simulation
t0 = time.time()
energy_test = 789 # Initial energy
numn_test = 405 # Initial number of neutrons
dfresults = ncap_sim(energy_test, numn_test) # Create dataframe of results
t1 = time.time()

total = t1-t0
print('Execution time: ', total)

# Save results to csv
dfresults.to_csv(f'dfmvuv_e{energy_test}_n{numn_test}_bw0.10.csv')

# %%
# Plotting sim results
# Scatter plot sim rho z
# Calculate the point density
x = dfresults['rho']
y = dfresults['z']
xy = np.vstack([x,y])
z = stats.gaussian_kde(xy)(xy)

# Sort the points by density, so that the densest points are plotted last
idx = z.argsort()
x, y, z = x[idx], y[idx], z[idx]

fig, ax = plt.subplots()
ax.scatter(x, y, c=z, s=2, edgecolor='')
plt.xlabel('rho (m)')
plt.ylabel('z (m)')
plt.title(f'{numn_test} Initial neutrons at {energy_test} MeV')
plt.show()


# %%
# Scatter plot sim rho t 
# Calculate the point density
x = np.sqrt(dfresults['rho']**2+dfresults['z']**2)
y = dfresults['t']
xy = np.vstack([x,y])
z = stats.gaussian_kde(xy)(xy)

# Sort the points by density, so that the densest points are plotted last
idx = z.argsort()
x, y, z = x[idx], y[idx], z[idx]

fig, ax = plt.subplots()
ax.scatter(x, y, c=z, s=2, edgecolor='')
plt.xlabel('rho (m)')
plt.ylabel('t (microsec)')
plt.title(f'{numn_test} Initial neutrons at {energy_test} MeV')
plt.show()

# %%
