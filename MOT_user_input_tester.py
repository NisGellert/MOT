import numpy as np
#import pandas as pd
import matplotlib.pyplot as plt
import os
from datetime import date
from matplotlib.ticker import MaxNLocator
import sys
from datetime import datetime
import math
import time 

os.chdir(r"C:\Users\nisge\DTU\PhD\GitHub\MOT") # Change directory to the location of your saved MOT directory
from Hybrid_functions import get_thickness, get_elements, get_roughness, visualize_fit, fom_MO, de_hybrid
from optical_constants import oc_MO
from fresnel import fresnel
from unit_conversion import wavelength2energy, energy2wavelength, thickness2energy, energy2thickness


'''-- Save output --'''
N = 10 # Number of bilayers in multilayer structure you want to optimise for.
fname = "Pt_tester_N"+str(N)+".txt" # Name of saved filed
fdir = 'C:/Users/nisge/DTU/PhD/MOT/Data/' # Change the directory location of the saved Data file in MOT
save_as_txt = [True, fdir + fname] # Save fitted data as txt (or not False)


'''-- Define Measurement Setup --'''
energy_i = np.arange(2e3,40.1e3,5e2) # [eV], Incident energy 
theta_i = 0.10 # [degree], Incident angle  
a_res = 0.0 # [degree] Instrument Angular resolution Note: Not sure if correct
optical_constants_dir = "oc_source/LLNL_MOT/" # [CXRO, NIST, LLNL] 

C_layer = ["C", 0.4] # Material, rougness
Ni_layer = ["Ni", 0.4] # Material, rougness  
Pt_layer = ["Pt", 0.4] # Material, rougness
Si_layer = ["Si", 0.4] # Material, rougness
Si_sub = ["Si", 0.4] # Material, rougness

PtC_multilayer = [C_layer,
                Pt_layer]
#NiC_multilayer = [C_layer,
               # Ni_layer]
model_structure = [PtC_multilayer, Si_sub] # Multilayer structure and substrate
depth_grating = ["Power law"] # "Power law", "Linear" or "Periodic"

'''-- Parameter_space = [N,dmin,dmax,c,gamma, gamma_top] --'''
#Ni_parameter_space = [[20,20],[2.5, 11.5],[3.7, 20.5],[0.1, 0.9],[0.2, 0.8],[0.2, 0.8]]
Pt_parameter_space = [[N,N],[2.4, 5.5],[16., 55.5],[0.1, 0.4],[0.3, 0.8],[0.3, 0.8]]
#nustar_parameter_space = [[145,145],[2.9, 2.9],[10.95, 10.95],[0.225, 0.225],[0.45, 0.45],[0.7, 0.7]]
gamma_top = [True] # Fit Gamma top
parameter_space = Pt_parameter_space


'''-- Initial guess = [N,dmin,dmax,c,gamma, gamma_top] --'''
initial_guess = [True, True]
#Ni_initial = [20,3.5,10.5,0.25,0.5,0.55]
Pt_initial = [N,2.5,21.6,0.242,0.45,0.45]
initial_values = Pt_initial

''' -- Define differentual evolution parameters --'''
FOM_weight = "equal" #  FOM weight
FOM_opt = [False, [158e3],[100000]] # Singeluar optimization point. 
mutation_factor = 0.75 # [0.1:2.0] Mutation Factor. 
crossover_probability = 0.75 # [0:1] Cross over probability
population_size = 40 # [10:50] population size
iterations = 50 # [25 - 550] Number of iterations
mutation_scheme = "Rand/1" # "Rand/1" or "Best/1"
plot_fit = True  # plots the best fit in every iteration Not implemented
print_fit_values = True # Prints the best fit values 
print_time = True