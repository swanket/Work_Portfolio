Numerical Model For Thesis Defense Titled: 
"Potential Ice Sheet and Glacial Modulation of volcanism in West Antarctica: Constraint on the cadence of melt delivery into the crust" 
by Sean Wanket
------------------------------------------------------------------------------------------------------------------------------------------------------
Project description
 
This project models 1D heat flow through the crust using the Dufort-Frankel method. Boundary conditions are both Nuemann BCs. The goal is to quantify 
the effects of two thermal forcings (Insulation and Time-dependent mantle magma injection) on the rheology of wall rock surrounding a magma chamber. 

The outputs from setup_and_integration.py are:
T_DF: temperature at each timestep and spatial node; NxM array
Hmagma: excess heating that occurs at the base of the crust when spatial nodes reach temperatures higher than the liquidus; NxM array
Un: Deborah Number; NxM array
visc: viscosity; NxM array
Qvol: volume of magma produced at the base of the crust by Hmagma; Mx1 array
Qvol_total: volume of magma that would cause equivalent heating to delta_qm; Mx1 array
------------------------------------------------------------------------------------------------------------------------------------------------------
File Description

main_melting.py --> This is the main python file which calls to params.yaml to import parameters, performs initial calculations, defines time-dependent 
parameters, executes functions from setup_and_integration.py, and plots the outputs after function execution.

params.yaml --> initialize all constants and parameters that do not require python functions to create.

setup_and_integration.py --> "initial" function to create initial temperature conditions (T(z,t) = T(z,0)). "dT_dt" function to calculate FTCS temperature step.
"integration_DF" function to execture Dufort-Frankel method to solve the 1D partial differential equation using "dT_dt" for the first time step.
------------------------------------------------------------------------------------------------------------------------------------------------------
Necessary Packages

pandas
numpy
matplotlib
scipy
sympy
tqdm
yaml
------------------------------------------------------------------------------------------------------------------------------------------------------
Credits
Written by Sean Wanket. Edited by Mark Jellinek and Catherine Johnson.
The model is based on the work of Karlstrom et al., 2017. 

Parameter sources:
q_m, kc, hr, A_0 --> Losing etal., 2020
L --> Ramirez et al., 2017
Tsurf --> Buizert et al., 2021
ki --> A. Gow & Engelhardt, 2000
kappa --> Whittington et al., 2009

Full citations in Wanket et al., 