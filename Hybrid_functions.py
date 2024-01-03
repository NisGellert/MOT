


from IPython import get_ipython
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
from datetime import date
from matplotlib.ticker import MaxNLocator
import sys
from datetime import datetime
import math
import time 
from optical_constants import oc_MO
from fresnel import fresnel
from unit_conversion import wavelength2energy, energy2wavelength, thickness2energy, energy2thickness

def get_thickness(depth_grading_model, N_model,dmax_model,gamma_model,gamm_top_model,gamma_top_model,dmin_model=0,c_model=0):
        
    z_model=[]
    d_model=[]
    for i in range(len(depth_grading_model)):
        dmax = dmax_model[i]
        
        N = N_model[i]    
        structure = depth_grading_model[i]
        gamma = gamma_model[i]
        gamm_top = gamm_top_model[i] # The value
        gamma_top = gamma_top_model[i] # True False
        dmin = dmin_model[i]
        c = c_model[i]       
        
        if dmin >= dmax:
            structure = 'Periodic'
    
        d = np.zeros(N) # Initialiser
        z = np.zeros(N*2) # Initialiser
        
        if structure == 'Power law': 
            #dmin = dmin_model[i]
            #c = c_model[i]
            b=np.exp((np.log(-(N-1)/(-1+np.exp(np.log(dmin/dmax)/c)))*c+np.log(dmin/dmax))/c) # Power law variable
            a=dmax*b**c # Power law variable
            for j in range(N): 
                d[j]=a/(b+j)**c # Bi-layer thickness
                z[2*j+1]=gamma*d[j]   # Thickness of high-Z
                z[2*j]=d[j]-z[2*j+1] #  Thickness of low-Z
        elif structure == 'Periodic':
            for j in range(N): 
                d[j] = dmax # Bi-layer thickness = dmax
                z[2*j+1]=gamma*d[j]  # Thickness of high-Z 
                z[2*j]=d[j]-z[2*j+1] #  Thickness of low-Z
        elif structure == 'Linear':
            for j in range(N): 
                #dmin = dmin_model[i]
                d[j] = dmax-((dmax-dmin)/(N-1))*(j); # Bi-layer thickness
                z[2*j+1]=gamma*d[j]   # Thickness of high-Z
                z[2*j]=d[j]-z[2*j+1] #  Thickness of low-Z
        
        if gamma_top:
            d_top = z[0]+z[1]
            z[1]=gamm_top*d_top
            z[0]=d_top-[z[1]]
         
        
        
        z_model +=  [a for a in z]
        d_model +=  [a for a in d]
    #elements.append([model_structure[-1][0]]) # substrate
    #sigma.append([model_structure[-1][1]]) # substrate
        
    
    return z_model


def get_elements(N_model,model_structure):
    elements = []
    for i in range(len(N_model)):
        N = N_model[i]
        for j in range(N): 
           elements.append([model_structure[i][0][0]])
           elements.append([model_structure[i][1][0]])
    elements.append([model_structure[-1][0]]) # substrate

    return elements
           
           
def get_roughness(N_model,model_structure):
    sigma = []
    for i in range(len(N_model)):
        N = N_model[i]
        for j in range(N): 
            sigma.append([model_structure[i][0][1]])
            sigma.append([model_structure[i][1][1]])
    sigma.append([model_structure[-1][1]]) # substrate
    return sigma





def visualize_fit(x, reflectance, iter_no, FOM, delta_FOM ,ax1,ax2, angleScan, pars_norm):
    ax1.clear()
    ax1.plot(x, reflectance,'k-',linewidth = 1.5,label = 'Best fit') # plot best reflectance for each iteration
    ax1.set_ylabel("Reflectance")
    ax1.set_title('Iteration %i: FOM = %.6e' %(iter_no[-1],FOM[-1]))
    #fig.suptitle('Main title') 

    ax1.set_ylim([0, 1])
    ax1.legend()
    if angleScan:
        ax1.set_yscale("log")
        ax1.set_xlabel("Grazing angle (deg)")
    else:
       ax1.set_xlabel("Energy (keV)") 
       
    ax2.plot(iter_no, FOM,'k.-',linewidth = 0.4) # label = 'Iteration %i: FOM = %.5e' %(i,fit_fom),
    ax2.set_xlabel("Iteration no.")
    ax2.set_ylabel("FOM")
    ax2.set_yscale("log")
    ax2.xaxis.set_major_locator(MaxNLocator(integer=True))
    # ax2.legend()
    ax2.set_yscale("log")
    #plt.pause(0.01)
    
    pars_norm = pars_norm[1:]
    N1 = pars_norm[:,0]
    dmin1 = pars_norm[:,1]
    dmax1 = pars_norm[:,2]
    c1 = pars_norm[:,3]
    gamma1 = pars_norm[:,4]
    gamma_top1 = pars_norm[:,5]
    
    #ax3.plot(iter_no, N,marker='*',linestyle= '-',label ='N',color='k',linewidth = 2.0) 
    #ax3.errorbar(iter_no, N, sliding_std_window(N), marker = '*', ms = 0.0, color = 'grey', linewidth = 0., elinewidth = 1.0)
    
    #ax3.plot(iter_no, dmax1,marker='s',linestyle= '-',label ='dmax',color='k',linewidth = 1.0) 
    #ax3.errorbar(iter_no, dmax1, sliding_std_window(dmax1), marker = '^', ms = 0.0, color = 'grey', linewidth = 0., elinewidth = 0.5)

    
    #ax3.plot(iter_no, gamma1,marker='^',linestyle= '-',label ='gamma',color='k',linewidth = 1.0) 
    #ax3.errorbar(iter_no, gamma1, sliding_std_window(gamma1), marker = '^', ms = 0.0, color = 'grey', linewidth = 0., elinewidth = 0.5)

    #ax3.plot(iter_no, c1,marker='o',linestyle= '-',label ='c',color='k',linewidth = 1.0) 
    #ax3.errorbar(iter_no, c1, sliding_std_window(c1), marker = '^', ms = 0.7, color = 'grey', linewidth = 0., elinewidth = 0.5)

    #ax3.plot(iter_no, gamma_top1,marker='*',linestyle= '-',label ='gamma top',color='k',linewidth = 1.0) 
    #ax3.errorbar(iter_no, gamma_top1, sliding_std_window(gamma_top1), marker = '*', ms = 0.0, color = 'grey', linewidth = 0., elinewidth = 0.5)

    #ax3.plot(iter_no, N1,marker='P',linestyle= '-',label ='N',color='k',linewidth = 1.0) 
    #ax3.errorbar(iter_no, N1, sliding_std_window(N1), marker = '^', ms = 0.0, color = 'grey', linewidth = 0., elinewidth = 0.5)

    #ax3.legend(handlelength=5)
    #ax3.legend(("dmax","gamma","c","gamma top","N"))
    #ax3.set_xlabel("Iteration no.")
    #ax3.set_ylabel("Parameter evolution")
    plt.pause(0.01)
    
    #marker='',linestyle= '-',label ='z = 30 nm, \u03C3 = 0.4 nm',color='k',linewidth = 2.0
    
def sliding_std_window(elements):
    std = np.zeros(len(elements))
    for i in range(len(elements)):
        std[i] = np.std(elements[0:i+1])
    return(std)   


def fom_MO(x, reflectance, FOM_range, weighting = 'equal'): # x (energy/theta)
     
    if FOM_range[0]:
        for ij in range(len(FOM_range[1])):
            idx = (np.abs(x - FOM_range[1][ij])).argmin()
            reflectance[idx] = reflectance[idx]*FOM_range[2]
    
    if weighting == "equal":
        FOM = np.nansum(np.abs(reflectance))
    elif weighting == "statistical":
        FOM = np.nansum(np.abs(reflectance*x))
    elif weighting == "statistical2":
        FOM = np.nansum(np.abs(reflectance*x**2))
    elif weighting == "statistical3":
        FOM = np.nansum(np.abs(reflectance*x**3))
    elif weighting == "statistical3":
        FOM = np.nansum(np.abs(reflectance*x**4))
    elif weighting == "statistical5":
        FOM = np.nansum(np.abs(reflectance*x**5))
    elif weighting == "statistical6":
        FOM = np.nansum(np.abs(reflectance*x**6))
    elif weighting == "log":
        FOM = np.nansum(np.abs(np.log(reflectance)))
        # Look at exponential 
    
    
    if weighting == "squared":
        FOM = np.nansum(np.abs(reflectance*x**2))
        #FOM = linregress(x, reflectance)[0]
    if weighting == "gauss":
        x=x/1000
        a1 = 1
        b1 = 40 # peak energy
        c1 = 50 #30
        gs = a1*np.exp(-(x-b1)**2/(2*c1**2))
        FOM = np.nansum(np.abs(reflectance*gs))
    elif weighting == "doublegauss":
        x=x/1000
        a1 = 0.8
        b1 = 22
        c1 = 5

        a2 = 0.8
        b2 = 45
        c2 = 5
        dgs = a2*np.exp(-(x-b2)**2/(2*c2**2))+a1*np.exp(-(x-b1)**2/(2*c1**2))
        FOM = np.nansum(np.abs(reflectance*dgs ))
    
        
    return FOM


def de_hybrid(theta_i, energy_i, model_structure, depth_grating, parameter_space, gamma_top, initial_guess, initial_values,FOM_opt, FOM_weight, a_res, save_as_txt, plot_fit, print_fit_values, mut,crossp, pop_size, its, mut_sch , oc_source):


    #mut=mutation_factor
    #crossp=crossover_probability
    #pop_size= population_size
    #its=iterations
    #mut_sch = mutation_scheme # "Rand/1" or "Best/1"
    #oc_source = optical_constants_dir


    #now = datetime.now()
    #current_time = now.strftime("%H:%M:%S")
    
    #def de_OMS(model_space, sample,structure,N_multilayers, theta_i, lambda_i, angleScan,
    #mut=0.8, crossp=0.7, pop_size=30, its=150, oc_source = "oc_source//CXRO_MO//", mut_sch = "Rand/1"):
    # Initializes the individuals of the population
    
    
    N_multilayers = int(len(parameter_space)/6)
    dimensions = len((parameter_space))#*len(parameter_space)) # no. of model parameters# no. of model parameters
    pop = np.random.rand(pop_size, dimensions) # Normalized population between 0 and 1, pop_size x dimensions
    min_b, max_b = np.asarray(parameter_space).T
    diff = np.fabs(min_b - max_b)
    pop_denorm = min_b + pop * diff # denormalize by converting each component from [0, 1] to [min, max]
    
    for i in range(N_multilayers):    
        if initial_guess[i]: # Implement intial guess
            for k in range(6):
                 if diff[6*i+k] != 0: 
                    pop_denorm[0,6*i+k] = initial_values[6*i+k] # N
                    pop[0,6*i+k] = (pop_denorm[0,6*i+k]-min_b[6*i+k])/diff[6*i+k] # denorm to norm        
        pop_denorm[:,6*i+0] = np.round(pop_denorm[:,6*i+0]) # Round N
        pop_denorm[:,6*i+1] = np.round(pop_denorm[:,6*i+1],2)# Round dmin
        pop_denorm[:,6*i+2] = np.round(pop_denorm[:,6*i+2],2)# Round dmax
        pop_denorm[:,6*i+3] = np.round(pop_denorm[:,6*i+3],3)# Round c
        pop_denorm[:,6*i+4] = np.round(pop_denorm[:,6*i+4],2)# Round Gamma 
        pop_denorm[:,6*i+5] = np.round(pop_denorm[:,6*i+5],2)# Round Gamma Top
        
        for k in range(len(pop_denorm[:,6*i+1])): # If dmin >= dmax, then set dmin = dmax, and structure == periodic.
            if pop_denorm[k,6*i+2] < pop_denorm[k,6*i+1]: # dmax < dmin
                pop_denorm[k,6*i+1] = pop_denorm[k,6*i+2] # dmin = dmax
                pop[k,6*i+1] = (pop_denorm[k,6*i+1]-min_b[6*i+1])/diff[6*i+1] # denorm to norm
                
    lambda_i = energy2wavelength(energy_i) # eV to nm
    if np.isscalar(energy_i) and theta_i[1]>theta_i[0] :
        angleScan = True
        print("Optimizing angle scan.")
        print('Incident energy: ',np.around(wavelength2energy(lambda_i)/1000,1), "keV")  
        print('Number of Bilayers: ',parameter_space[0][0] ) 
        print('Number of iterations: ', its ) 
        print('F.O.M: ',FOM_weight ) 
        print(" ")
        reflectance = np.zeros([len(theta_i),pop_size])
    elif np.isscalar(theta_i) and energy_i[1]>energy_i[0] : 
        angleScan = False
        print("Optimizing energy scan")
        print('Incident angle: ',theta_i,"\u00b0") 
        print('Number of Bilayers: ',parameter_space[0][0] ) 
        print('Number of iterations: ', its ) 
        print('F.O.M: ',FOM_weight ) 
        print(" ")    
        reflectance = np.zeros([len(lambda_i),pop_size])
    FOM =  np.zeros(pop_size) # Initiliser
        
    
    #parameter_space = np.array(N,dmin,dmax,c,gamma, gamma_top)
    for i in range(pop_size):
        N = [];dmax = []; gamma = [];dmin = [];c = []; gamm_top=  []
        for j in range(N_multilayers):   
            N = N + [int(pop_denorm[i][6*j+0])]
            dmin = dmin + [pop_denorm[i][6*j+1]]
            dmax = dmax + [pop_denorm[i][6*j+2]]
            c = c + [pop_denorm[i][6*j+3]]
            gamma = gamma + [pop_denorm[i][6*j+4]]
            gamm_top = gamm_top + [pop_denorm[i][6*j+5]]   
                   
        #z = get_thickness(depth_grating,N,dmax,gamma,dmin,c)
        z = get_thickness(depth_grating,N,dmax,gamma,gamm_top,gamma_top,dmin,c)

        
        #if gamma_top[j]:
         #    d = z[0]+z[1]
        #     z[1]=gamm_top*d
        #     z[0]=d-[z[1]]
        sigma = get_roughness(N,model_structure)         
        elements = get_elements(N,model_structure)    
        
                
        #start_time = time.time()      
        if angleScan: # Calculate refractive indices of the layer materials
            n = np.zeros(len(elements)+1, dtype=complex)+1  # + 1 due to top vacuum
            for j in range(len(elements)): 
                n_temp = "n_" + elements[j][0] # Create the tempoary n
                if n_temp in globals(): # Check if it exists
                    #print("It is here. Taking existing n")
                    n[j+1] = globals()[n_temp] # Uses existing n for that material
                    
                else: # Makes new n for specific material and saves for later use. 
                    #print("It is not here. Computing new n")
                    n[j+1] = oc_MO(energy_i,elements[j][0], oc_source) # Computes new n
                    globals()[n_temp] = n[j+1] # Saves that n for later use
            reflectance[:,i] = fresnel(theta_i,lambda_i,n,z,sigma, ares = a_res)
            FOM[i] = fom_MO(theta_i, reflectance[:,i], FOM_opt, FOM_weight)
            
        else: # If energy scan  
            n = np.zeros((len(lambda_i),len(elements)+1), dtype=complex)+1  # + 1 due to top vacuum
            for j in range(len(elements)): 
                n_temp = "n_" + elements[j][0] # Create the tempoary n
                if n_temp in globals(): # Check if it exists
                    #print("It is here. Taking existing n")
                    n[:,j+1] = globals()[n_temp] # Uses existing n for that material
                    
                else: # Makes new n for specific material and saves for later use. 
                    #print("It is not here. Computing new n")
                    n[:,j+1] = oc_MO(energy_i,elements[j][0], oc_source)
                    globals()[n_temp] = n[:,j+1] # Saves that n for later use
            reflectance[:,i] = fresnel(theta_i,lambda_i,n,z,sigma, ares = a_res)
            FOM[i] = fom_MO(energy_i, reflectance[:,i], FOM_opt, FOM_weight)
        
    best_idx = np.argmax(FOM) # Locate the best FOM from the initial population
    best_denorm = pop_denorm[best_idx]
    reflectance_best = reflectance[:,best_idx] 
    fom_evolution = [] 
    iteration = [] 
    delta_FOM = []
    std_denorm = np.zeros(len(pop_denorm[0]))
    pop_evolution = np.zeros(dimensions) 
        
    
    for i in range(its): 
        for j in range(pop_size):
            # List of indices idxs of the population excluding the current j
            idxs = [idx for idx in range(pop_size) if idx != j] 
            if (mut_sch == "Rand/1"):
                # choose 3 other vectors/individuals from the normalized population, excluding the current pop[j], without replacement
                a1, b1, c1 = pop[np.random.choice(idxs, 3, replace = False)] # "Rand/1"
            
                # create a mutation vector from the 3 vectors
                mutant = np.clip(a1 + mut * (b1 - c1), 0, 1) # values outside the interval [0, 1] are clipped to the interval edges
            
            if (mut_sch == "Best/1"):
                a1 = pop[best_idx]
                b1, c1 = pop[np.random.choice(idxs, 2, replace = False)] # "Best/1"
                mutant = np.clip(a1 + mut * (b1 - c1), 0, 1)
                
            # Recombination 
            cross_points = np.random.rand(dimensions) < crossp 
            # boolean array with True elements when the random values are lower than the crossp probability    
            if not np.any(cross_points):
                # if there aren't any True elements in cross_point, select a random index and set True 
                cross_points[np.random.randint(0, dimensions)] = True
            
            # Create trial: where cross_points is True, trial takes on the value from mutant - otherwise use pop[j]
            trial = np.where(cross_points, mutant, pop[j])
            trial_denorm = min_b + trial * diff
                   
            N = [];dmax = []; gamma = [];dmin = [];c = []; gamm_top=  []
            for k in range(N_multilayers):   
                N = N + [int(trial_denorm[6*k+0])]
                #If dmin >= dmax, then set dmin = dmax, and structure == periodic.
                if np.round(trial_denorm[6*k+2],2) < np.round(trial_denorm[6*k+1],2): # if dmax < dmin
                    trial_denorm[6*k+1] = np.round(trial_denorm[6*k+2],2) # dmin = dmax
                    trial[6*k+1] = (trial_denorm[6*k+1]-min_b[6*k+1])/diff[6*k+1] # denorm to norm      
                dmin = dmin + [np.round(trial_denorm[6*k+1],2)]
                dmax = dmax + [np.round(trial_denorm[6*k+2],2)]
                c = c + [np.round(trial_denorm[6*k+3],3)]
                gamma = gamma + [np.round(trial_denorm[6*k+4],2)]
                gamm_top = gamm_top + [np.round(trial_denorm[6*k+5],2)]   
            z = get_thickness(depth_grating,N,dmax,gamma,gamm_top,gamma_top,dmin,c)
    
            #z = get_thickness(depth_grating,N,dmax,gamma,dmin,c)
            #if gamma_top[k]:
            #     d = z[0]+z[1]
            #     z[1]=gamm_top*d
            #     z[0]=d-[z[1]]
            sigma = get_roughness(N,model_structure)         
            elements = get_elements(N,model_structure) 
            
           
            if angleScan: # Calculate refractive indices of the layer materials
                n = np.zeros(len(elements)+1, dtype=complex)+1  # + 1 due to top vacuum
                for k in range(len(elements)): 
                    n_temp = "n_" + elements[k][0] # Create the tempoary n
                    if n_temp in globals(): # Check if it exists
                        #print("It is here. Taking existing n")
                        n[k+1] = globals()[n_temp] # Uses existing n for that material
                        
                    else: # Makes new n for specific material and saves for later use. 
                        #print("It is not here. Computing new n")
                        n[k+1] = oc_MO(energy_i,elements[k][0], oc_source) # Computes new n
                        globals()[n_temp] = n[k+1] # Saves that n for later use
                reflectance_trial = fresnel(theta_i,lambda_i,n,z,sigma, ares = a_res)
                f_temp = fom_MO(theta_i,  reflectance_trial, FOM_opt, FOM_weight)
                
            else: # If energy scan
                n = np.zeros((len(lambda_i),len(elements)+1), dtype=complex)+1  # + 1 due to top vacuum
                for k in range(len(elements)): 
                    n_temp = "n_" + elements[k][0] # Create the tempoary n
                    if n_temp in globals(): # Check if it exists
                        #print("It is here. Taking existing n")
                        n[:,k+1] = globals()[n_temp] # Uses existing n for that material
                        
                    else: # Makes new n for specific material and saves for later use. 
                        #print("It is not here. Computing new n")
                        n[:,k+1] = oc_MO(energy_i,elements[k][0], oc_source)
                        globals()[n_temp] = n[:,k+1] # Saves that n for later use
                reflectance_trial = fresnel(theta_i,lambda_i,n,z,sigma, ares = a_res)
                f_temp = fom_MO(energy_i,  reflectance_trial, FOM_opt, FOM_weight)
            
            # Compare with the current FOM of the j'th individual
            if f_temp > FOM[j]: # YOU ARE HERE: WHEN IT PRINTS, THEN ONLY SHOULD PLOT WHEN FOM GO UP???
                FOM[j] = f_temp
                pop[j] = trial
                reflectance[:,j] = reflectance_trial
                
    
                if f_temp > FOM[best_idx]: 
                    # compare with the best FOM among all individuals in the population
                    best_idx = j
                    best_denorm = trial_denorm
                    reflectance_best = reflectance_trial  # the best reflectance of the population
                    
                    if print_fit_values:
                        print("Iteration:", i)
                        print("FOM = ", FOM[best_idx])
                        print("Best values:", best_denorm.tolist())
                        for k in range(len(pop_denorm[0])):
                            std_denorm[k] = np.round(np.std(pop_denorm[:,k]),2)
                        #print("Std of best values:", std_denorm.tolist())
                        z_best = np.sum(z) # nm, Stack thickness
                        #print("dmin = ",trial_denorm[1])
                        #print("dmax =" , trial_denorm[2])
                        #print("\u0393 = ", trial_denorm[4])
                        #if structure == 'Power law': print("c = ", trial_denorm[3])
                        #if gamma_top[0]: print("\u0393 top = ",trial_denorm[5])
                        #print("")f
        #z_best = np.sum(z) # nm, Stack thickness                
        fom_evolution.append(FOM[best_idx])
        pop_evolution=np.vstack([pop_evolution,np.asmatrix(np.mean(pop,axis=0))])
        iteration.append(i)
        avg10_FOM = np.average(fom_evolution[-10:]) # take the average fom for the last 10 iterations
        delta_FOM.append(avg10_FOM -  FOM[best_idx])
        #print("--- %s seconds ---" % (time.time() - start_time))
      
        
        if plot_fit == True : 
            if i == 0:
                plt.ion()
                fig, (ax1, ax2) = plt.subplots(nrows=1, ncols=2,figsize=(5.5*2,4.5))
                fig.tight_layout(pad=4.0)
                
            # Plot the current best-fit
            if angleScan:
                visualize_fit(theta_i, reflectance_best, iteration, fom_evolution,delta_FOM, ax1, ax2, angleScan, pop_evolution )
                fig.canvas.draw()
            else:
                x = wavelength2energy(lambda_i)/1000
                visualize_fit(x, reflectance_best, iteration, fom_evolution,delta_FOM, ax1, ax2, angleScan, pop_evolution)
    
           
    
    if save_as_txt[0]:   
        parameters = ['N','dmin','dmax','c','gamma', 'gamma_top']*(len(model_structure)-1)
        
        fname = save_as_txt[1]
        file1 = open(save_as_txt[1],"w+") 
        today = date.today()
        file1.write("# Saved on %s\n\n" % today.strftime("%d/%m/%Y"))
        if angleScan:
            file1.write("# Incident angle (degree):  %s \n" % theta_i) 
            file1.write("# Incident energy (eV):  %s - %s \n" % energy_i[0], energy_i[-1]) 
        else:
            file1.write("# Incident angle (degree):  %s \n" % theta_i) 
            file1.write("# Incident energy (keV):  %s - %s \n" % (energy_i[0]/1000, energy_i[-1]/1000)) 
        file1.write("# Number of multilayers in hybrid structure = %s \n" % (len(model_structure)-1))
        file1.write("# Model structure:\n")
        for i in range(len(model_structure)):
           file1.write("# %s \n" % model_structure[i])
        file1.write("# Used parameter space [N,dmin,dmax,c,gamma, gamma_top]:\n")
        
        for i in range(len(parameter_space)):
            file1.write("# %s = %s \n" % (parameters[i], parameter_space[i]))
        for i in range(len(depth_grating)): 
            file1.write("# %s \n" % depth_grating[i])
        file1.write("# FOM type: %s \n" % FOM_weight)   
        file1.write("# Optical Constants from %s \n" % oc_source)
        file1.write("# \n")
        file1.write("# Best Fit Sample Structure [N,dmin,dmax,c,gamma, gamma_top]: \n")
        
        for i in range(len(best_denorm)): 
            file1.write("# %s = %s , spread = %s\n" % (parameters[i],best_denorm[i],std_denorm[i]))
            
        file1.write("# Stack thickness (nm) = %s \n" % z_best )
        file1.write("# FOM: %s \n" % FOM[best_idx])   
        
     
        #for i in range(len(multilayer_structure)-1):
        #    file1.write("# %s sigma = %s (nm), z = %s (nm) \n" % (multilayer_structure[i][0:2],  str(sigma[i][0]),  str(round(z_bf[i],2))   ))   
        #file1.write("# %s sigma = %s (nm) \n\n" % (multilayer_structure[-1][0:2], str(sigma[-1][0])))
        file1.write("# Energy (keV), Reflectance\n" )
        #energy_i = wavelength2energy(lambda_i) # keV
        for i in range(len(energy_i)):
             file1.write("%.4E %.9E\n" % (energy_i[i]/1000, reflectance_best[i]))
        file1.close() 
        
    yield min_b + pop * diff, best_denorm, FOM[best_idx]   