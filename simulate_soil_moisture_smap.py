'''
Created on Jun 6, 2017

@author: zwieback
'''
import numpy as np
import math
def day_to_fy(day):
    # converts day (can exceed 365) to fractional year
    return math.modf(day/365.25)[0]
def draw_gap(numpy_rng):
    # model for smap: sample every 2 or 3 days
    return [2,3][int(numpy_rng.uniform()<0.5)]

def soil_moisture_smap(numpy_rng, prob_dry_model, mean_rainfall_model, loss_model, depth=0.125, prob_dry_initial=0.5, effective_rainfall_coefficient=0.8, sm_initial=None, porosity=0.4, day=0, samples=None):
    # see soil_moisture_smap_time_series for definition of input
    n = 0 # number of samples (=SMAP measurements)
    day_last = day # last time a measurement was made
    gap = draw_gap(numpy_rng) # number of days until next measurement
    dry = int(numpy_rng.uniform() < prob_dry_initial) # 1 if no precipitation, 0 if precipitation
    sm = sm_initial if sm_initial is not None else numpy_rng.uniform()*porosity # initialize soil moisture
    fy = day_to_fy(day) # fractional year
    while True:
        if samples is not None and n >= samples: 
            return
        is_sampled = (day-day_last) == gap # if True: a measurement is made
        if is_sampled:
            n = n + 1
            day_last = day
            gap = draw_gap(numpy_rng) 
            yield (day,fy,sm)
        day = day + 1
        fy = day_to_fy(day)
        dry = int(numpy_rng.uniform() < prob_dry_model(fy)[dry]) # Markov chain: determine whether the new day will be without precip (dry) or not        
        precip = 0.0 if dry else numpy_rng.exponential(mean_rainfall_model(fy)) # if rainfall: draw from exponential distribution according to mean_rainfall_model
        sm = np.min([porosity,loss_model(fy)*sm + precip*effective_rainfall_coefficient/depth]) # update soil moisture; ceiling imposed by porosity

def soil_moisture_smap_time_series(numpy_rng, samples, prob_dry_model, mean_rainfall_model, loss_model, depth=0.125, prob_dry_initial=0.5, effective_rainfall_coefficient=0.75, sm_initial=None, porosity=0.4, day=0):
    # simulates a smap soil moisture time series according to a simple antecedent precip model with precipitation modelled using a Markov chain (rainfall event)/exponential (amount for rainfall event); measurements are made every 2/3 days
    # input arguments
    # numpy_rng is a numpy RandomState which draws the random numbers
    # samples: number of samples
    # prob_dry_model: a function that returns a 2-element vector [prob(dry tomorrow|wet today) and prob(dry tomorrow|dry today)] as a function of the fractional year; dry meaning that there is no precipitation
    # mean_rainfall_model: a function that returns the average rainfall of a wet day for a given fractional year
    # loss model: function that returns the daily soil moisture loss factor (between 0 and 1) for a given fractional year
    # optional arguments: depth: layer depth in m; prob_dry_initial: probability that day 0 is dry; effective_rainfall_coefficient: input into soil is reduced by this amount; sm_initial: initial soil moisture, drawn randomly if None, porosity: maximum soil moisture content, day: first day, 0 corresponds to 1 Jan 
    smgenerator = soil_moisture_smap(numpy_rng, prob_dry_model,mean_rainfall_model,loss_model,depth=depth,prob_dry_initial=prob_dry_initial,effective_rainfall_coefficient=effective_rainfall_coefficient,sm_initial=sm_initial,porosity=porosity,day=day,samples=samples)
    smts = [smv for smv in smgenerator]
    sm = np.array([smv[2] for smv in smts])
    fy = np.array([smv[1] for smv in smts])
    day = np.array([smv[0] for smv in smts])
    return day,fy,sm