'''
Created on Jun 7, 2017

@author: zwieback
'''
from simulation_model_definitions import simulation_model
from inference_definitions import inference_definition, inference_weights
from simulate_product import simulate_product
from model_inference import model_inference
from model_setup import model_setup
from input_output import save_results, read_results

def simulation_internal(scenario, n, numpy_rng=None, pathout=None, niter=2000, nchains=2):
    params_forward = simulation_model(scenario)
    visible, internal, normalized_weights = simulate_product(n, params_forward, numpy_rng=numpy_rng)
    estimateexplanterms, estimatesdexplanterms, inferenceparams = inference_definition(scenario)
    visible_w, normalized_weights_w = inference_weights(scenario, visible, normalized_weights) #potentially overwrite weights
    model = model_setup(visible_w, normalized_weights_w, estimateexplanterms=estimateexplanterms, 
                        estimatesdexplanterms=estimatesdexplanterms, inferenceparams=inferenceparams)
    trace, v_params, tracevi = model_inference(model, niter=niter, nchains=nchains, seed=numpy_rng.randint(0,10000))
    if pathout is not None:
        save_results(pathout, {'trace':trace,'v_params':v_params,'visible':visible_w,'normalized_weights':normalized_weights_w, 
                               'simulation_normalized_weights':normalized_weights, 'simulation_internal':internal})