# Numerical investigation of 1d Jellium model

This repository includes functions for exploring 1d jellium model in equilibrium. 

## Description
* The Equilibrium jellium model:
  * Basic exploration:
    By running Basic_jellium_exploration.py, we find the average equilibrium density of the 1d jellium model with N particles. The result is obtained by using Metropolis Monte Carlo simulation.
    The plot of the density is saved as Figures/density.png.
    The plot of the system's energy as a function of algorithm steps is saved as Figures/Energy_to_equilibrium.png.
* Ranked diffusions
  * In the folder Ranked_diffusion_julia, there is a julia script for computing the average density of particles undergoing Ranked diffusion for different parameters. To run this file: julia Ranked_diffusions_Burgers_densities.jl.
  This produces data stored in the Data folder, which can be visualized by running a plotting script, 'plotting_Burgers_densities.py' written in Python (found in the same folder, Ranked_diffusion_julia). This produces a plot stored in the Figures folder. To find more details about ranked diffusion, see https://arxiv.org/abs/2301.08552 . Note: The plot with densities produced by code in this repository is similar to Figure 5 in the article.

### Dependencies
Required packages are specified in Requirements.txt

### Executing program

#### Average density of equilibrium jellium model
To find the average density of the jellium model, run:
```python3 Basic_jellium_exploration.py```
#### Average density of particles undergoing ranked diffusion
To generate the plot with Average density of particles after a given time run:
julia Ranked_diffusions_Burgers_densities.jl
python3 plotting_Burgers_densities.py

## Help
* If plotting_Burgers_densities.py is not working, try changing the last argument in the function robovi_v_sredine (lines 55-57) from 0 to 1 or vice versa.

## Authors

Ana Flack

