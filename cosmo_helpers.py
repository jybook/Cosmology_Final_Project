from classy import Class
import numpy as np
import matplotlib.pyplot as plt
import corner
import pandas as pd
import emcee
import tqdm

data_file = '~/physics/COM_PowerSpect_CMB-TT-full_R3.01_with_header.txt'
data = pd.read_csv(data_file, sep='\s+', comment='#')
data = data.head(2499)

N_ur = 1.0176 # A value of 3.044 would indicate 3 massless neutrinos. We're including massive neutrinos, but including one massless neutrino here because lighter neutrino masses seem more likely (at least to me). 

N_ncdm = 3 # Number of massive neutrinos, including sterile neutrinos

delta_msq_21 = 7.6e-5 # Mass squared difference between the second and first neutrino mass eigenstates, in eV^2
delta_msq_31 = 2.5e-3 # Mass squared difference between the third and first neutrino mass eigenstates, in eV^2
# We'll assume for now the lightest neutrino is massless. We also assume the normal ordering. Values from https://www.frontiersin.org/articles/10.3389/fspas.2018.00036/full

m_ncdm = '0.00872, 0.05, 0.5'  # Masses of each of the massive neutrinos, in eV
# We choose a sterile neutrino of 0.5 eV (the right order of magnitude active/sterile neutrino osscilations)
# I explored the possibility of also including and HNL, but it seems like CLASS (at least as I've configured it) will only support particle masses up to the eV scale in this parameter.

# 5.f) 'ksi_ncdm' is the ncdm chemical potential in units of its own temperature (default: set to 0)
ksi_ncdm = '0, 0, 0.1' # Though out of date, this paper https://arxiv.org/pdf/astro-ph/9602135.pdf explores the interesting potential of non-zero neutrino chemical potentials (pun intended, sorry) on helping to resolve neutrino oscillation anomalies without constradicting cosmological data.

multipole = data['l'].astype(int)
def model(theta, ells = multipole):
    '''
    Define a cosmological model we'll use to fit the PLANCK data.
    Inputs:
    theta: array of floats, model parameters
    ells: array of integers, multipole moments at which to evaluate the model (included for completion)
    Returns:
    LambdaCDM: CLASS instance, the model
    '''
    
    omega_b, omega_cdm, h, amp = theta
    #Initiate CLASS
    LambdaCDM = Class()

    # Set out (relatively exotic) model parameters (defined above):
    LambdaCDM.set({'N_ur':N_ur,'N_ncdm':N_ncdm,'m_ncdm':m_ncdm,'ksi_ncdm':ksi_ncdm})

    # pass parameters we're fitting
    LambdaCDM.set({'omega_b':omega_b,'omega_cdm':omega_cdm,'h':h})
    #Set CLASS generator settings
    LambdaCDM.set({'output':'tCl,lCl','lensing':'yes'})

    # run class - this creates our model
    LambdaCDM.compute()

    return LambdaCDM
    # get the CMB power spectrum

def get_cls(cmb_model):
    '''
    Get the CMB Temperature power spectrum from a CLASS model
    Inputs:
    cmb_model: CLASS instance, a cosmological model
    Returns:
    cls: array of floats, CMB Temperature power spectrum values
    '''
    all_cls = cmb_model.lensed_cl(2500)['tt'] #ls start at 0, so this gives cls up to l = 2500
    cls = all_cls[2:] #Our data start at l = 2
    cls_microK =cls*10**12 #Convert from K^2 to microKelvin sqaured to match Planck's units
    return cls_microK

def get_Dls(theta):
    '''
    Takes a set of cosmological parameters and returns the CMB temperature power spectrum in Dls (to match the Planck data)
    Inputs:
    theta: array of floats, model parameters
    Returns:
    Dls: array of floats, CMB temperature power spectrum values C_l*l*(l+1)/(2*pi)
    '''
    omega_b, omega_cdm, h, amp = theta
    CMB_model = model(theta)
    cls = get_cls(CMB_model)
    simulated_ls = CMB_model.lensed_cl(2500)['ell'] #ls start at 0, so this gives cls up to l = 2500
    l = simulated_ls[2:]
    Dls = amp*l*(l+1)*cls/(2*np.pi)
    return Dls

def log_likelihood(dls, x, y, yerr_up, yerr_down, amp):
    # get the model Dl for the given 'l' (passed as x). The index of each l is l-2, and l starts at 2.
    
    indeces = np.linspace(0, 2498, 2499).astype(int)
    model_y = dls[indeces]
    # calculate the likelihood
    y_err = yerr_down + yerr_up
    chi2 = np.sum((y - model_y)**2 / (y_err**2))

    ln_like = -0.5*chi2
    return ln_like

def check_priors(theta):
    # Prior ranges are inspired by the 1-sigma ranges summarized in this paper:https://www.aanda.org/articles/aa/pdf/2019/03/aa34060-18.pdf
    # I've taken the largest reported 1-sigma range for each parameter used the precision on the last digit to determine the prior (this makes them slightly assymetric in terms of amount above/below the 1 sigma interval, but I'm okay with that).
    omega_b, omega_cdm, h, amp = theta
    # LambdaCDM.set({'omega_b':0.0223828,'omega_cdm':0.1201075,'h':0.67810,'A_s':2.100549e-09,'n_s':0.9660499,'tau_reio':0.05430842})
    # if (not (omega_b/h**2 + omega_cdm/h**2) < 1):
    #     #Make sure we don't overload the total energy density
    #     return False
    if not(0.0222 < omega_b <0.0226):
        return False
    if not(0.118 < omega_cdm <0.122):
        return False
    if not(0.664 < h <0.684):
        return False
    if not (2*np.pi < amp < 3*np.pi):
        return False
    return True
    

def ln_prob(theta, x, y, yerr_up, yerr_down):
    omega_b, omega_cdm, h, amp = theta
    # Check if this set of parameters is allowed
    possible = check_priors(theta)
    if not possible:
        return -np.inf
    
    dls = get_Dls(theta)

    # get the likelihood
    ln_likelihood = log_likelihood(dls, x, y, yerr_up, yerr_down, amp)
    return ln_likelihood
