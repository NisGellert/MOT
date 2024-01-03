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
os.chdir(r"C:\Users\nisge\DTU\PhD\GitHub\MOT")

from Hybrid_functions import get_thickness, get_elements, get_roughness, visualize_fit, fom_MO, de_hybrid
from optical_constants import oc_MO
from fresnel import fresnel
from unit_conversion import wavelength2energy, energy2wavelength, thickness2energy, energy2thickness


os.chdir(r"C:\Users\nisge\DTU\PhD\GitHub\MOT")
from MOT_user_input_tester import * # Choose desired user input file 

# Calling the Multilayer Structure Optimisaztion Script
fitting = list(de_hybrid(theta_i, energy_i, model_structure, depth_grating, parameter_space, gamma_top, initial_guess, initial_values,
            FOM_opt, FOM_weight, a_res, save_as_txt, plot_fit, print_fit_values,mut=mutation_factor,
             crossp=crossover_probability, pop_size=population_size, its=iterations, mut_sch = mutation_scheme , oc_source = optical_constants_dir))  
                 


pop_best, best_fit_param, maxFOM = fitting[-1]
N_bf =  best_fit_param[0]
dmin_bf = best_fit_param[1]
dmax_bf = best_fit_param[2]
c_bf = best_fit_param[3]
gamma_bf = best_fit_param[4]
gamma_top_bf = best_fit_param[5]
FOM_bf = maxFOM 