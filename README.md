# SOHENS
Simulation of High Energy Neutron Statistics

The code in SOHENS.py can be used to generate a dataset of neutron capture times and positions at user-defined energies between 10 and 2000MeV.
The statistics are based on GEANT4 simulated data for a z-directed neutron source at the centre of a cube of water with 200m sides. This GEANT4 simulation is stored in https://github.com/cpalazzi/HENS-H4, where this statistical simulation is referred to as nCaptureMC_mp_mvuv.py.

To use the code, simply download the repository and open in your favourite IDE (preferably one that recognises Jupyter Notebook formatting). The code should work with most common installations of Python 3. Then, run all of the function definitions. 

In the code segment
```
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
```
define the energy of your initial neutrons setting the value of ```energy_test``` and the number of initial neutrons by setting ```numn_test```. 

The result will be saved as a csv file. The subsequent sections of code can be used to plot some distributions from the results. 