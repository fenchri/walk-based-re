#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar 29 14:02:56 2018

author: fenia
"""

import torch
import matplotlib as plt
plt.use('Agg')
from walk_re_wrapper import ModelWrapper
from robo.fmin import bayesian_optimization
from robo.fmin import entropy_search
from robo.fmin import random_search
from robo.fmin import bohamiann
import logging
from collections import OrderedDict


def objective_function(params):
    """ 
    Function to optimize
    """
    named_params = OrderedDict()
    if len(params) == 5:
        named_params['lr'] = params[0]
        named_params['reg'] = params[1]
        named_params['dropi'] = params[2]
        named_params['dropo'] = params[3]
        named_params['gradc'] = params[4]
    elif len(params) == 6:
        named_params['lr'] = params[0]
        named_params['reg'] = params[1]
        named_params['dropi'] = params[2]
        named_params['dropo'] = params[3]
        named_params['gradc'] = params[4]
        named_params['beta'] = params[5]
    model_wrapper.update_hyperparams(named_params)
    score = model_wrapper.train()
    return score


def bayes_opt():
    """ 
    Bayesian Optimization with RoBO.
    Returns dictionary with results:
        "x_opt" : the best found data point
        "f_opt" : the corresponding function value
        "incumbents": the incumbent (best found value) after each iteration
        "incumbent_value": the function values of the incumbents
        "runtime": the runtime in seconds after each iteration
        "overhead": the optimization overhead 
                    (i.e. time data we do not spend to evaluate the function) 
                    of each iteration
        "X": all data points that have been evaluated
        "y": the corresponding function evaluations
    """
    print('\n============= START Bayesian OPTIMIZATION =============\n')
    print("""Optimization parameters:
                    - lower = {}
                    - upper = {}
                    - num_iter = {}
                    - maximizer = {}
                    - acq_func = {}
                    - model_type = {} 
                    - n_init = {} """.format(lower, upper, 
                                             args.num_iterations, 
                                             args.maximizer, 
                                             args.acquisition_func,
                                             args.model_type,
                                             args.n_init))
        
    results = bayesian_optimization(objective_function, 
                                    lower, 
                                    upper, 
                                    num_iterations=args.num_iterations,
                                    maximizer=args.maximizer,
                                    acquisition_func=args.acquisition_func,
                                    model_type=args.model_type,
                                    n_init=args.n_init)
    print(results["x_opt"])
    print(results["f_opt"])
    print('\n============= END OPTIMIZATION =============\n')


def ent_search():
    """
    Entropy search
    """
    print('\n============= START Entropy Search OPTIMIZATION =============\n')
    print("""Optimization parameters:
                    - lower = {}
                    - upper = {}
                    - num_iter = {}
                    - maximizer = {}
                    - model_type = {} 
                    - n_init = {} """.format(lower, upper, 
                                             args.num_iterations, 
                                             args.maximizer, 
                                             args.model_type,
                                             args.n_init))
        
    results = entropy_search(objective_function, 
                             lower, 
                             upper, 
                             num_iterations=args.num_iterations,
                             maximizer=args.maximizer, 
                             model=args.model_type)
    print(results["x_opt"])
    print(results["f_opt"])
    print('\n============= END OPTIMIZATION =============\n')
    

def rand_search():
    """
    Random search
    """
    print('\n============= START Random Search OPTIMIZATION =============\n')
    print("""Optimization parameters:
                    - lower = {}
                    - upper = {}
                    - num_iter = {}""".format(lower, upper, 
                                              args.num_iterations))
        
    results = random_search(objective_function, 
                            lower, 
                            upper, 
                            num_iterations=args.num_iterations)
    print(results["x_opt"])
    print(results["f_opt"])
    print('\n============= END OPTIMIZATION =============\n')


def boham():
    """
    Bayesian Networks
    """
    print('\n============= START Bohamiann OPTIMIZATION =============\n')
    print("""Optimization parameters:
                    - lower = {}
                    - upper = {}
                    - num_iter = {}
                    - maximizer = {}
                    - acq_func = {}""".format(lower, 
                                              upper, 
                                              args.num_iterations,
                                              args.maximizer,
                                              args.acquisition_func))
                                              
    results = bohamiann(objective_function, 
                        lower, 
                        upper, 
                        num_iterations=args.num_iterations,
                        maximizer=args.maximizer, 
                        acquisition_func=args.acquisition_func)
    print(results["x_opt"])
    print(results["f_opt"])
    print('\n============= END OPTIMIZATION =============\n')


def main():
    global args, lower, upper, iter_i
    model_wrapper = ModelWrapper()
    args = model_wrapper.args
    logging.basicConfig(level=logging.INFO)

    iter_i = 0
    """
    Parameters search space:
         learn | reg | dropi | dropo | gradc | beta 
    """
    if args.walks_iter > 0:
        lower = np.array([0.001,      # learn
                          0.0000001,  # reg
                          0.0,        # dropi
                          0.0,        # dropo
                          5,          # gradc
                          0.5])       # beta

        upper = np.array([0.003,      # learn
                          0.0001,     # reg
                          0.5,        # dropi
                          0.5,        # dropo
                          30,         # gradc
                          0.9])       # beta
    else:
        lower = np.array([0.001,      # learn
                          0.0000001,  # reg
                          0.0,        # dropi
                          0.0,        # dropo
                          5])         # gradc

        upper = np.array([0.003,      # learn
                          0.0001,     # reg
                          0.5,        # dropi
                          0.5,        # dropo
                          30])        # gradc

    if args.opt_method == 'BayesOpt':
        bayes_opt()
    elif args.opt_method == 'RandomSearch':
        rand_search()
    elif args.opt_method == 'EntropySearch':
        ent_search()
    elif args.opt_method == 'Bohamiann':
        boham()


if __name__ == "__main__":
    main()
