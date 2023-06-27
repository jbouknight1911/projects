#!/usr/bin/env python
# coding: utf-8

# In[120]:


import csv
import numpy as np
import pymc3 as pm
import arviz as az
import matplotlib.pyplot as plt
from matplotlib.pyplot import figure
from scipy import stats

def read_data(filename):
    '''
    Reading data file 
    '''
    date = []
    close = []
    
    with open(filename,'r') as csvfile:
        lines = csv.reader(csvfile,delimiter = ',')
        
        for row in lines:
            date.append(row[0])
            close.append(float(row[1]))
    
    tup = (date,close)
    return tup

def bayesian_regression(data):
    '''
    The bulk of the code will be in this function. It will perform
    linear regression as well as Bayesian regression. The plots will 
    also be generated in this function
    '''
    ## First we will perform linear regression in order to obtain values for alpha,
    ## beta, and the varience
    
    set_size = len(data[0])
    
    t = np.array(data[0]).astype(np.float) 
    y_raw = np.array(data[1]).astype(np.float)
    
    y = np.log(y_raw)
    
    ## Visualize the data (unmanipulated)
    
    plt.figure(figsize=(4,3),dpi =150)
    
    plt.plot(t, y)
    
    plt.xlabel('Day #')
    plt.ylabel('log Close Price (USD)')
    plt.title('Bitcoin Price from Jan 2021 to Nov 2021')
    plt.savefig('btc_plot.png')
    plt.show()
    
    t_bar = np.mean(t) # make terms easier to plug in
    y_bar = np.mean(y)
    
    # for linear regression we want the form: y_bar = a_hat + beta_hat * x_bar
    # We have our 'x' data (actually t) and y data 
    
    beta_hat = np.sum((t-t_bar)*(y-y_bar)) / np.sum((t-t_bar)**2)
    alpha_hat = y_bar - (beta_hat * t_bar)
    
    # calculate variance using the information we calculated
    
    err_var = np.sum((y - alpha_hat - (beta_hat*t))**2/(set_size-2))
    
    plt.figure(figsize=(3,2),dpi =150)
    
    plt.plot(t,y)
    plt.plot(t, alpha_hat + (beta_hat*t), c='r', 
            label = f'y = {alpha_hat:.3f} + {beta_hat:.3f}*x, var = {err_var:.2f}')
    plt.xlabel('Day #')
    plt.ylabel('log Close Price (USD)')
    plt.title('Least Mean Squares of BTC Data')
    plt.legend()
    plt.savefig('lin_reg_btc.png')
    
### Model checking:
### We are generating synthetic data in order to ensure that the model 
### is well specified. Fr#om the synthetic data, we want to check that the
### model reproduces results that are at least close to the true values
##################################################################################   

#     N = 500
#     true_alpha = 3
#     true_beta = 0.5
#     true_epsilon = 0.5
    
#     epsilon = np.random.normal(0,true_epsilon,N)
    
#     x = np.random.normal(10,1,N)
#     true_Y = true_alpha + true_beta * x
#     Y = true_Y + epsilon
    
#     _, ax = plt.subplots(1,2,figsize=(8,4))
#     ax[0].plot(x, Y, 'o')
#     ax[0].set_xlabel('x')
#     ax[0].set_ylabel('y', rotation=0)
#     ax[0].plot(x, true_Y, 'r')
#     az.plot_kde(Y, ax=ax[1])
#     ax[1].set_xlabel('y')
#     plt.tight_layout()
    
#     with pm.Model() as model:
#         # priors
#         alpha = pm.Normal('alpha',mu=0,sigma=20)
#         beta = pm.Normal('beta',mu=0,sigma=20)
#         epsilon = pm.HalfNormal('epsilon',5)
        
#         # likelihood
#         y_pred = pm.Normal('y_pred',mu=alpha + beta*x, sd = epsilon, observed = Y)
#         trace = pm.sample(2000,cores = 8)
        
#         idata = az.from_pymc3(trace)
        
#         az.plot_trace(idata,var_names = ['alpha','beta','epsilon'],
#                      lines=[('alpha',{},true_alpha),('beta',{},true_beta),
#                             ('epsilon',{},true_epsilon)],)

##################################################################################
        
    with pm.Model() as model:
        # priors
        alpha = pm.Normal('alpha',mu = 0, sigma = 20)
        beta = pm.Normal('beta',mu=0,sigma=20)
        epsilon = pm.Normal('epsilon',5)
        
        # likelihood
        y_pred = pm.Normal('y_pred', mu = alpha + beta*t, 
                           sd = epsilon, observed = y)
        trace_1 = pm.sample(2000, cores = 8)
        
        idata = az.from_pymc3(trace_1)
        
        az.plot_trace(idata,var_names = ['alpha','beta','epsilon'],
                     lines=[('alpha',{},alpha_hat),('beta',{},beta_hat),
                            ('epsilon',{},err_var)])
        
    pm.model_to_graphviz(model)
    
    plt.figure(figsize=(4,3),dpi =150)
    plt.plot(t, y)
    alpha_m = trace_1['alpha'].mean()
    beta_m = trace_1['beta'].mean()
    draws = range(0, len(trace_1['alpha']), 10)
    plt.plot(t, trace_1['alpha'][draws] + trace_1['beta'][draws] * t[:, np.newaxis], c='gray', alpha=0.5)
    plt.plot(t, alpha_m + beta_m *t, c='r',
            label=f'y = {alpha_m:.3f} + {beta_m:.3f}*x')
    plt.ylabel('log BTC Price (USD)')
    plt.xlabel('Time (# Days)', rotation=0)
    plt.legend();
    
    
    ## Summarize Bayesian Results
    
    alpha_bar = np.mean(trace_1['alpha'])
    beta_bar = np.mean(trace_1['beta'])
    epsilon_bar = np.mean(trace_1['epsilon'])
    
    print('Alpha via Bayes:',alpha_bar)
    print('Beta via Bayes:',beta_bar)
    print('Epsilon via Bayes:',epsilon_bar)
    print('-----------------------------')
    print('Alpha via Lin Reg:',alpha_hat)
    print('Beta via Lin Reg:',beta_hat)
    print('Epsilon via Lin Reg:',err_var)
    
    ## Let's forecast 
    
    length = 30
    
    x_f = np.arange(length)
        
    mean_trace = alpha_bar + beta_bar * x_f
    norm = pm.Normal.dist(0,sd=epsilon_bar)
    err_f = norm.random(size=length)
    y_f = mean_trace + err_f
    y_reg = mean_trace
    
    plt.figure(figsize=(4,3),dpi =150)
    
    plt.plot(x_f,y_f)
    plt.fill_between(x_f,y_f-err_f ,y_f+err_f,
                    alpha=0.5, edgecolor='g', facecolor='lime')
    plt.xlabel('Day #')
    plt.ylabel('log Close Price (USD)')
    plt.title(f'Bitcoin Price {length} Day Forecast')
    plt.savefig('btc_forecast.png')
    plt.show()
    

def main():
    filename = ('btc_data.csv') 
    data = read_data(filename)
    
    bayesian_regression(data)
    
main()


# In[ ]:





# In[ ]:




