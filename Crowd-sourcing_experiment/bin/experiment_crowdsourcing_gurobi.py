# coding: utf-8

import pandas as pd
import numpy as np
import urllib.request
import zipfile
import random
import itertools
import math
import scipy.special as spys
import networkx as nx
from networkx.algorithms import bipartite
import itertools
import time
import pyproj
grs80 = pyproj.Geod(ellps='GRS80')
import shapefile
import shapely
#from shapely.geometry import Polygon
from descartes.patch import PolygonPatch
import matplotlib as mpl
import matplotlib.pyplot as plt
from scipy.spatial import distance_matrix
import datetime
import collections
import csv
import pickle
import os
import GPy
import GPyOpt
import pulp
from pulp import value
from numpy import linalg
import sys
import pickle
from pulp import GUROBI_CMD
#from pulp import PULP_CBC_CMD
import statistics
import copy

args=sys.argv

# Setting
#input parameters
m=int(args[1])
n=int(args[2])
T=int(args[3])
#running time of bayesian optimization
bo_time=int(args[4])
#running time of random search
rs_time=int(args[5])
#number of simulations
num_simulation=int(args[6])

#given constant parameters
#the value of allowable error
epsilon=0.00001
#the number of simulations in Monte Carlo method
num_Monte_Carlo=10**3
#the number of simulations for calcilationf \beta in Dickerson's matching strategy
beta_calc_num=10000

#the parameters of the sigmoid function p_vt
beta=1.0
gamma=0.1*math.sqrt(3)/math.pi

#the parameters of the r_vt (, that is, the probability that node v appears at time t)
r_vt_difference=0.01
base_price_low=0.5
base_price_difference=0.3

#the parameter of the proposed method
max_iteration_proposed=100
residual_end_condition=10**(-5)
#the value of allowable error of z
z_epsilon=10**(-7)

#the parameter of the PDHG method (in the proposed method).
tau_proposed=0.1
sigma_proposed=0.1
alpha_proposed=0.7
eta_proposed=0.7
c_proposed=0.7

#Preprocessing of the data
with open('../work/Reward_matrix', 'rb') as web:
    Reward_matrix = pickle.load(web)
df = pd.read_csv("../work/trec-rf10-data.csv")
df_v=df.values
topicID_set=list(set(df_v[:,0]))
workerID_set=list(set(df_v[:,1]))
num_worker=len(workerID_set)
num_task=len(df_v)

def find_value_from_list(lst, value):
    return [i for i, x in enumerate(lst) if x == value]

def generate_random_num_based_on_given_pd(pd):
    # if pd= [0.1, 0.2, 0.3]^\top, then return {0/1/2/3} with prob {0.1, 0.2, 0.3 ,0.4}.
    cumulative_dist = np.cumsum(pd).tolist()
    cumulative_dist.append(1.0)
    random_num = np.random.rand()
    cumulative_dist.append(random_num)
    return sorted(cumulative_dist).index(random_num)

def approximated_objective_value_calculate_dickersons_matching_strategy(m,T,r_vt,Pr,z,beta_ut,W,P,gamma_app_ratio):
    #return the approximated objective value by Monte Carlo method for input parameters when dickerson's matching strategy is used.
    total_rewards_list=[]
    average_rewards_list=[]
    match_num_list=[]
    for h in range(num_Monte_Carlo):
        total_rewards=0
        remove_index_u_list=[]
        rewards_list=[]
        remmain_index_list=list(range(m))
        for k in range(T):
            arrive_index=generate_random_num_based_on_given_pd(r_vt[:,k])
            #determin if the arriving worker will accept the wage
            tmp=np.random.rand()
            if tmp < Pr[arrive_index,k]:
                #determine the node u to match by dickerson's matching strategy.
                tmp=generate_random_num_based_on_given_pd(list(z[remmain_index_list,arrive_index,k]*gamma_app_ratio/(Pr[arrive_index,k]*r_vt[arrive_index,k]*beta_ut[remmain_index_list,k])))
                if tmp==len(remmain_index_list):
                    continue
                matching_index=remmain_index_list[tmp]
                total_rewards+=W[matching_index,arrive_index]+P[arrive_index,k]
                rewards_list.append(W[matching_index,arrive_index]+P[arrive_index,k])
                remove_index_u_list.append(matching_index)
                remmain_index_list.remove(matching_index)
                if remmain_index_list==[]:
                    break
        total_rewards_list.append(total_rewards)
        if len(remove_index_u_list) !=0:
            average_rewards_list.append(statistics.mean(rewards_list))
        match_num_list.append(len(remove_index_u_list))
    return [sum(total_rewards_list)/len(total_rewards_list),statistics.mean(average_rewards_list),statistics.mean(match_num_list)]

def approximated_objective_value_calculate_greedy(T,W,r_vt,Pr,P):
    #return the approximated objective value by Monte Carlo method for input parameters when the greedy matching strategy is used.
    total_rewards_list=[]
    average_rewards_list=[]
    match_num_list=[]
    for h in range(num_Monte_Carlo):
        total_rewards=0
        remove_index_u_list=[]
        rewards_list=[]
        W_tmp=W.copy()
        for k in range(T):
            arrive_index=generate_random_num_based_on_given_pd(r_vt[:,k])
            tmp=np.random.rand()
            if tmp < Pr[arrive_index,k]:
                while True:
                    matching_index=list(W_tmp[:,arrive_index]).index(max(list(W_tmp[:,arrive_index])))
                    if max(list(W_tmp[:,arrive_index]))==-np.inf:
                        break
                    if not matching_index in remove_index_u_list:
                        if W[matching_index,arrive_index]+P[arrive_index,k]<0:
                            break
                        total_rewards+=W[matching_index,arrive_index]+P[arrive_index,k]
                        rewards_list.append(W[matching_index,arrive_index]+P[arrive_index,k])
                        remove_index_u_list.append(matching_index)
                        break
                    else:
                        W_tmp[matching_index,arrive_index]=-np.inf
        total_rewards_list.append(total_rewards)
        if len(remove_index_u_list) !=0:
            average_rewards_list.append(statistics.mean(rewards_list))
        match_num_list.append(len(remove_index_u_list))
    return [sum(total_rewards_list)/len(total_rewards_list),statistics.mean(average_rewards_list),statistics.mean(match_num_list)]

def beta_ut_calculate(T,m,r_vt,Pr,z,gamma_app_ratio):
    #compute the beta required for dickerson's matching strategy
    beta_ut=np.ones([m,T])
    remmain_index_list=[]
    for h in range(beta_calc_num):
        remmain_index_list.append(list(range(m)))
    for t in range(T):
        sum_u=np.zeros(m)
        for h in range(beta_calc_num):
            arrive_index=generate_random_num_based_on_given_pd(r_vt[:,t])
            tmp=np.random.rand()
            if tmp < Pr[arrive_index,t]:
                if remmain_index_list[h]==[]:
                    continue
                tmp=generate_random_num_based_on_given_pd(list(z[remmain_index_list[h],arrive_index,t]*gamma_app_ratio/(Pr[arrive_index,t]*r_vt[arrive_index,t]*beta_ut[remmain_index_list[h],t])))
                if tmp==len(remmain_index_list[h]):
                    continue
                matching_index=remmain_index_list[h][tmp]
                remmain_index_list[h].remove(matching_index)
                sum_u[matching_index]+=1
        if t!=T-1:
            for u in range(m):
                beta_ut[u,t+1]=beta_ut[u,t]-(1.0/beta_calc_num)*sum_u[u]
    return beta_ut

# Lists to store results
proposed_value_list=[]
proposed_time_list=[]
proposed_rewards_per_match_list=[]
proposed_match_num_list=[]

cu_approx_value_list=[]
cu_approx_time_list=[]
cu_approx_rewards_per_match_list=[]
cu_approx_match_num_list=[]

cu_greedy_value_list=[]
cu_greedy_time_list=[]
cu_greedy_rewards_per_match_list=[]
cu_greedy_match_num_list=[]

bo_approx_value_list=[]
bo_approx_time_list=[]
bo_approx_rewards_per_match_list=[]
bo_approx_match_num_list=[]

bo_greedy_value_list=[]
bo_greedy_time_list=[]
bo_greedy_rewards_per_match_list=[]
bo_greedy_match_num_list=[]

rs_approx_value_list=[]
rs_approx_time_list=[]
rs_approx_rewards_per_match_list=[]
rs_approx_match_num_list=[]

rs_greedy_value_list=[]
rs_greedy_time_list=[]
rs_greedy_rewards_per_match_list=[]
rs_greedy_match_num_list=[]


for setting_tmp in range(num_simulation):
    #Generate the problem
    Task_list=random.sample(range(num_task), m)
    Worker_list=random.sample(range(num_worker), n)
    W=np.zeros([m,n])
    tmp_y=0
    for i in Worker_list:
        tmp_x=0
        for j in Task_list:
            topic_k=find_value_from_list(topicID_set,df_v[j,0])
            W[tmp_x,tmp_y]=Reward_matrix[topic_k[0],i]
            tmp_x+=1
        tmp_y+=1

    #generate the r_vt (, that is, the probability that node v appears at time t)
    r_vt_rand=r_vt_difference+np.random.rand(n,T)*(1-r_vt_difference)
    sums_along_cols = r_vt_rand.sum(axis=0)
    r_vt=r_vt_rand/sums_along_cols

    #generate the parameter of p_vt (p_vt(x) is the probability that node v accepts the wage x at time t)
    base_price=base_price_low+base_price_difference*np.random.rand(n)
    #Problem generate ends.

    # Starts proposed method
    start_time=time.time()

    #Initial points of the primal variables and dual variables
    z_k=np.zeros([m,n,T])
    la_k=np.zeros([m])
    z_u_k=np.sum(z_k,axis=(1,2))

    #the iteration number
    k=1
    #the primal residual
    p=1
    #the dual residual
    d=1
    while k<max_iteration_proposed and (np.linalg.norm(p)> residual_end_condition or np.linalg.norm(d)>residual_end_condition):
        #Update primal variables (line 4 and 5 in Algorithm 1 of our paper)
        #z_star means z_{k+1}.
        z_star=np.zeros([m,n,T])
        #Solve Eq (7) for all v, t
        for v in range(n):
            for t in range(T):
                a=z_k[:,v,t]+tau_proposed*(W[:,v]-la_k)
                s=r_vt[v,t]/2.0
                delta=s/2.0
                while delta>r_vt[v,t]*10**(-8):
                    f_vt_dash_s=beta*base_price[v]+gamma*base_price[v]*np.log(s/(r_vt[v,t]-s)) +gamma*base_price[v]*r_vt[v,t]/(r_vt[v,t]-s)
                    left=a-tau_proposed*f_vt_dash_s
                    left_plus=left*(left>0)
                    if sum(left_plus)>s:
                           s+=delta
                    else:
                           s-=delta
                    delta=delta/2
                tmp=a-tau_proposed*f_vt_dash_s
                z_star[:,v,t]=tmp*(tmp>0)
        #Update dual variables (line 6 in Algorithm 1 of our paper)
        z_u=np.sum(z_star,axis=(1,2))
        la_tmp=la_k+sigma_proposed*(2*z_u-z_u_k-1)
        la_star=la_tmp*(la_tmp>0)

        #Update step sizes (line 7--14 in Algorithm 1 of our paper)
        z_d=z_star-z_k
        la_d=la_star-la_k
        z_u_d=np.sum(z_d,axis=(1,2))
        #line 7 in Algorithm 1 of our paper
        if (c_proposed/(2.0*tau_proposed))*np.sum(np.power(z_d,2))+(c_proposed/(2.0*sigma_proposed))*np.sum(np.power(la_d,2)) <= 2.0*np.inner(la_d,z_u_d):
            tau_proposed=0.5*tau_proposed
            sigma_proposed=0.5*sigma_proposed
        #line 9 and 10 in Algorithm 1 of our paper
        p_sum=0
        for t in range(T):
            for v in range(n):
                p_tmp=-z_d[:,v,t]/tau_proposed+la_d
                p_sum+=sum(np.power(p_tmp,2))
        p=np.sqrt(p_sum)
        d=-la_d/sigma_proposed+z_u_d
        #line 11--14 in Algorithm 1 of our paper
        if 2*p< np.linalg.norm(d):
            tau_proposed=tau_proposed*(1-alpha_proposed)
            sigma_proposed=sigma_proposed/(1-alpha_proposed)
            alpha_proposed=alpha_proposed*eta_proposed
        elif p > 2*np.linalg.norm(d):
            tau_proposed=tau_proposed/(1-alpha_proposed)
            sigma_proposed=sigma_proposed*(1-alpha_proposed)
            alpha_proposed=alpha_proposed*eta_proposed
        z_u_k=z_u
        z_k=z_star.copy()
        la_k=la_star.copy()
        k=k+1

    # Calculate x_vt (by Proposition 4 of our paper)
    z_star=z_star*(z_star>0)
    z_vt_sum_star=np.sum(z_star,axis=0)
    X_vt_proposed=np.zeros([n,T])
    for v in range(n):
        for t in range(T):
            if z_vt_sum_star[v,t]>z_epsilon:
                X_vt_proposed[v,t]=-beta*base_price[v]-gamma*base_price[v]*np.log(z_vt_sum_star[v,t]/(r_vt[v,t]-z_vt_sum_star[v,t]))
            else:
                X_vt_proposed[v,t]=0
    #calculate p_vt(x_vt)
    p_vt_proposed=(1-(1/(1+np.exp(-(X_vt_proposed+beta*base_price.reshape(-1,1))/(gamma*base_price.reshape(-1,1))))))

    # To obtain the 1/2 approximation matching strategy [Dickerson et al., 2018], calculate beta_ut.
    gamma_app_ratio=0.5
    beta_ut=beta_ut_calculate(T,m,r_vt,p_vt_proposed,z_star,gamma_app_ratio)

    # Calculate results
    proposed_time=time.time()-start_time
    [proposed_value,add_value,remove_average_num]=approximated_objective_value_calculate_dickersons_matching_strategy(m,T,r_vt,p_vt_proposed,z_star,beta_ut,W,X_vt_proposed,gamma_app_ratio)

    proposed_value_list.append(proposed_value)
    proposed_time_list.append(proposed_time)
    proposed_rewards_per_match_list.append(add_value)
    proposed_match_num_list.append(remove_average_num)

    #----------------------------------------------------
    #CU-A method
    start_time=time.time()
    w_hat=np.average(W[W>0])
    # Set to search x is [0, 1]^{V \times T}
    upper=0.0
    lower=-1.0
    # Calculate x_vt by Golden section method
    eta_gold=(np.sqrt(5)-1)/(np.sqrt(5)+1)
    beta_gold=upper-lower
    tau_gold=eta_gold*beta_gold
    x_0=lower
    x_1=lower+tau_gold
    x_2=upper-tau_gold
    x_3=upper
    Pr_0=np.sum(r_vt*(1-(1/(1+np.exp(-(x_0+beta*base_price.reshape(-1,1))/(gamma*base_price.reshape(-1,1)))))))
    Pr_1=np.sum(r_vt*(1-(1/(1+np.exp(-(x_1+beta*base_price.reshape(-1,1))/(gamma*base_price.reshape(-1,1)))))))
    Pr_2=np.sum(r_vt*(1-(1/(1+np.exp(-(x_2+beta*base_price.reshape(-1,1))/(gamma*base_price.reshape(-1,1)))))))
    Pr_3=np.sum(r_vt*(1-(1/(1+np.exp(-(x_3+beta*base_price.reshape(-1,1))/(gamma*base_price.reshape(-1,1)))))))
    f_0=(x_0+w_hat)*np.min([m,Pr_0])
    f_1=(x_1+w_hat)*np.min([m,Pr_1])
    f_2=(x_2+w_hat)*np.min([m,Pr_2])
    f_3=(x_3+w_hat)*np.min([m,Pr_3])
    beta_gold=beta_gold-tau_gold
    tau_gold=eta_gold*beta_gold
    while beta_gold>10**(-4):
        if f_1>f_2:
            tmp_1=x_0+tau_gold
            tmp_2=x_1
            tmp_3=x_2
            x_1=copy.copy(tmp_1)
            x_2=copy.copy(tmp_2)
            x_3=copy.copy(tmp_3)
            Pr_1=np.sum(r_vt*(1-(1/(1+np.exp(-(x_1+beta*base_price.reshape(-1,1))/(gamma*base_price.reshape(-1,1)))))))
            tmp_1=(x_1+w_hat)*np.min([m,Pr_1])
            tmp_2=f_1
            tmp_3=f_2
            f_1=copy.copy(tmp_1)
            f_2=copy.copy(tmp_2)
            f_3=copy.copy(tmp_3)
        else:
            tmp_0=x_1
            tmp_1=x_2
            tmp_2=x_3-tau_gold
            x_0=copy.copy(tmp_0)
            x_1=copy.copy(tmp_1)
            x_2=copy.copy(tmp_2)
            Pr_2=np.sum(r_vt*(1-(1/(1+np.exp(-(x_2+beta*base_price.reshape(-1,1))/(gamma*base_price.reshape(-1,1)))))))
            tmp_0=f_1
            tmp_1=f_2
            tmp_2=(x_2+w_hat)*np.min([m,Pr_2])
            f_0=copy.copy(tmp_0)
            f_1=copy.copy(tmp_1)
            f_2=copy.copy(tmp_2)
        beta_gold=beta_gold-tau_gold
        tau_gold=eta_gold*beta_gold
    P_cu_approx=np.copy(x_2)
    Pr_cu_approx=(1-(1/(1+np.exp(-(P_cu_approx+beta*base_price.reshape(-1,1))/(gamma*base_price.reshape(-1,1))))))

    # To obtain 1/2 approximation matching strategy [Dickerson et al., 2018], solve problem (1) in our paper.
    ## We use pulp to model the optimization problem and solve it by Gurobi. (If you don't have Gurobi ricense, you can use CBC solver instead.)
    ## Set the optimization problem by pulp
    problem = pulp.LpProblem('tsp_mip', pulp.LpMaximize)
    z = {(u,v,t):pulp.LpVariable(name='z_{}_{}_{}'.format(u,v,t), lowBound=0, upBound=1, cat='Continuous') for u in range(m) for v in range(n) for t in range (T)}
    problem += pulp.lpSum([(W[u,v] + P_cu_approx)* z[u,v,t] for u in range(m) for v in range(n) for t in range(T)])
    for v in range(n):
        for t in range(T):
            problem += pulp.lpSum([z[u, v ,t] for u in range(m)]) <= Pr_cu_approx[v]*r_vt[v,t]
    for u in range(m):
        problem += pulp.lpSum([z[u,v,t] for v in range(n) for t in range(T)]) <= 1

    ## solve the optimization problem by Gurobi.
    status = problem.solve(GUROBI_CMD(msg=0))
    #status = problem.solve(PULP_CBC_CMD(msg=0))

    opt_vec_Lp=np.ones([m,n,T])
    tmp=0
    for u in range(m):
        for v in range(n):
            for t in range(T):
                opt_vec_Lp[u,v,t]=pulp.value(z[u,v,t])

    # To obtain the 1/2 approximation matching strategy, calculate beta_ut.
    gamma_app_ratio=0.5
    beta_ut=beta_ut_calculate(T,m,r_vt,Pr_cu_approx*np.ones([n,T]),opt_vec_Lp,gamma_app_ratio)

    # Calculate results
    cu_approx_time=time.time()-start_time
    [cu_approx_value,add_value,remove_average_num]=approximated_objective_value_calculate_dickersons_matching_strategy(m,T,r_vt,Pr_cu_approx*np.ones([n,T]),opt_vec_Lp,beta_ut,W,P_cu_approx*np.ones([n,T]),gamma_app_ratio)

    cu_approx_value_list.append(cu_approx_value)
    cu_approx_time_list.append(cu_approx_time)
    cu_approx_rewards_per_match_list.append(add_value)
    cu_approx_match_num_list.append(remove_average_num)

    #----------------------------------------------------
    #Caaped_UCB+greedy
    start_time=time.time()
    w_hat=np.average(W[W>0])
    # Set to search x is [0, 1]^{V \times T}
    upper=0.0
    lower=-1.0
    # Calculate x_vt by Golden section method
    eta_gold=(np.sqrt(5)-1)/(np.sqrt(5)+1)
    beta_gold=upper-lower
    tau_gold=eta_gold*beta_gold
    x_0=lower
    x_1=lower+tau_gold
    x_2=upper-tau_gold
    x_3=upper
    Pr_0=np.sum(r_vt*(1-(1/(1+np.exp(-(x_0+beta*base_price.reshape(-1,1))/(gamma*base_price.reshape(-1,1)))))))
    Pr_1=np.sum(r_vt*(1-(1/(1+np.exp(-(x_1+beta*base_price.reshape(-1,1))/(gamma*base_price.reshape(-1,1)))))))
    Pr_2=np.sum(r_vt*(1-(1/(1+np.exp(-(x_2+beta*base_price.reshape(-1,1))/(gamma*base_price.reshape(-1,1)))))))
    Pr_3=np.sum(r_vt*(1-(1/(1+np.exp(-(x_3+beta*base_price.reshape(-1,1))/(gamma*base_price.reshape(-1,1)))))))
    f_0=(x_0+w_hat)*np.min([m,Pr_0])
    f_1=(x_1+w_hat)*np.min([m,Pr_1])
    f_2=(x_2+w_hat)*np.min([m,Pr_2])
    f_3=(x_3+w_hat)*np.min([m,Pr_3])
    beta_gold=beta_gold-tau_gold
    tau_gold=eta_gold*beta_gold
    while beta_gold>10**(-4):
        if f_1>f_2:
            tmp_1=x_0+tau_gold
            tmp_2=x_1
            tmp_3=x_2
            x_1=copy.copy(tmp_1)
            x_2=copy.copy(tmp_2)
            x_3=copy.copy(tmp_3)
            Pr_1=np.sum(r_vt*(1-(1/(1+np.exp(-(x_1+beta*base_price.reshape(-1,1))/(gamma*base_price.reshape(-1,1)))))))
            tmp_1=(x_1+w_hat)*np.min([m,Pr_1])
            tmp_2=f_1
            tmp_3=f_2
            f_1=copy.copy(tmp_1)
            f_2=copy.copy(tmp_2)
            f_3=copy.copy(tmp_3)
        else:
            tmp_0=x_1
            tmp_1=x_2
            tmp_2=x_3-tau_gold
            x_0=copy.copy(tmp_0)
            x_1=copy.copy(tmp_1)
            x_2=copy.copy(tmp_2)
            Pr_2=np.sum(r_vt*(1-(1/(1+np.exp(-(x_2+beta*base_price.reshape(-1,1))/(gamma*base_price.reshape(-1,1)))))))
            tmp_0=f_1
            tmp_1=f_2
            tmp_2=(x_2+w_hat)*np.min([m,Pr_2])
            f_0=copy.copy(tmp_0)
            f_1=copy.copy(tmp_1)
            f_2=copy.copy(tmp_2)
        beta_gold=beta_gold-tau_gold
        tau_gold=eta_gold*beta_gold
    P_cu_greedy=np.copy(x_2)

    # Calculate results
    cu_greedy_time=time.time()-start_time
    Pr_cu_greedy=(1-(1/(1+np.exp(-(P_cu_greedy+beta*base_price.reshape(-1,1))/(gamma*base_price.reshape(-1,1))))))
    [cu_greedy_value,add_value,remove_average_num]=approximated_objective_value_calculate_greedy(T,W,r_vt,Pr_cu_greedy*np.ones([n,T]),P_cu_greedy*np.ones([n,T]))

    cu_greedy_value_list.append(cu_greedy_value)
    cu_greedy_time_list.append(cu_greedy_time)
    cu_greedy_rewards_per_match_list.append(add_value)
    cu_greedy_match_num_list.append(remove_average_num)

    #----------------------------------------------------
    #BO-A method
    #We use GPyOpt.methods.BayesianOptimization to search for x_vt.
    ##Settting of set to search x_vt
    bounds=[]
    for k in range(n*T):
        dic ={}
        key = "name"
        dic[key] = f"x{k}"
        key = "type"
        dic[key] = "continuous"
        key = "domain"
        dic[key] = (-1.0,-0.0)
        bounds.append(dic)

    ## Settting of the oracle.
    max_value=0
    def f_approx(x):
        global max_value
        global opt_vec_Lp
        P_bo_approx_or=x[0]
        P_bo_approx=np.zeros([n,T])
        for v in range(n):
            for t in range(T):
                P_bo_approx[v,t]=P_bo_approx_or[v+t*n]
        Pr_bo_approx=np.zeros([n,T])
        for v in range(n):
            for t in range(T):
                Pr_bo_approx[v,t]=1-(1/(1+np.exp(-(P_bo_approx[v,t]+beta*base_price[v])/(gamma*base_price[v]))))
        # set problem
        problem = pulp.LpProblem('tsp_mip', pulp.LpMaximize)
        z = {(u,v,t):pulp.LpVariable(name='z_{}_{}_{}'.format(u,v,t), lowBound=0, upBound=1, cat='Continuous') for u in range(m) for v in range(n) for t in range (T)}
        problem += pulp.lpSum([(W[u,v] + P_bo_approx[v,t])* z[u,v,t] for u in range(m) for v in range(n) for t in range(T)])
        for v in range(n):
            for t in range(T):
                problem += pulp.lpSum([z[u, v ,t] for u in range(m)]) <= Pr_bo_approx[v,t]*r_vt[v,t]
        for u in range(m):
            problem += pulp.lpSum([z[u,v,t] for v in range(n) for t in range(T)]) <= 1
        status = problem.solve(GUROBI_CMD(msg=0))
        #status = problem.solve(PULP_CBC_CMD(msg=0))
        objective_value=pulp.value(problem.objective)
        if objective_value>max_value:
            max_value=objective_value
            opt_vec_Lp=np.ones([m,n,T])
            tmp=0
            for u in range(m):
                for v in range(n):
                    for t in range(T):
                        opt_vec_Lp[u,v,t]=pulp.value(z[u,v,t])
        return -pulp.value(problem.objective)

    ##Search for x_vt
    start_time=time.time()
    myBopt = GPyOpt.methods.BayesianOptimization(f=f_approx, domain=bounds)
    myBopt.run_optimization(max_iter=10**8,max_time=bo_time)
    P_bo_approx_or=myBopt.x_opt
    P_bo_approx=np.zeros([n,T])
    for v in range(n):
        for t in range(T):
            P_bo_approx[v,t]=P_bo_approx_or[v+t*n]
    Pr_bo_approx=(1-(1/(1+np.exp(-(P_bo_approx+beta*base_price.reshape(-1,1))/(gamma*base_price.reshape(-1,1))))))

    #To obtain the 1/2 approximation matching strategy, calculate beta_ut.
    gamma_app_ratio=0.5
    beta_ut=beta_ut_calculate(T,m,r_vt,Pr_bo_approx,opt_vec_Lp,gamma_app_ratio)

    # Calculate results
    bo_approx_time=time.time()-start_time
    [bo_approx_value,add_value,remove_average_num]=approximated_objective_value_calculate_dickersons_matching_strategy(m,T,r_vt,Pr_bo_approx,opt_vec_Lp,beta_ut,W,P_bo_approx,gamma_app_ratio)

    bo_approx_value_list.append(bo_approx_value)
    bo_approx_time_list.append(bo_approx_time)
    bo_approx_rewards_per_match_list.append(add_value)
    bo_approx_match_num_list.append(remove_average_num)

    #----------------------------------------------------
    #BO-G method
    #We use GPyOpt.methods.BayesianOptimization to search for x_vt.
    ## Settting of the oracle
    def f(x):
        P_ba_greedy=x[0].reshape([n,T])
        Pr_ba_greedy=1-(1/(1+np.exp(-(P_ba_greedy+beta*base_price.reshape(-1,1))/(gamma*base_price.reshape(-1,1)))))
        [result,tmp1,tmp2]=approximated_objective_value_calculate_greedy(T,W,r_vt,Pr_ba_greedy,P_ba_greedy)
        return -result

    ##Search for x_vt
    start_time=time.time()
    myBopt = GPyOpt.methods.BayesianOptimization(f=f, domain=bounds)
    myBopt.run_optimization(max_iter=10**8,max_time=bo_time)

    #Since the method itself is finished, the measurement of computation time is completed.
    bo_greedy_time=time.time()-start_time

    # Calculate results
    P_bo_greedy_or=myBopt.x_opt
    P_bo_greedy=np.zeros([n,T])
    for v in range(n):
        for t in range(T):
            P_bo_greedy[v,t]=P_bo_greedy_or[v+t*n]
    Pr_bo_greedy=(1-(1/(1+np.exp(-(P_bo_greedy+beta*base_price.reshape(-1,1))/(gamma*base_price.reshape(-1,1))))))
    [bo_greedy_value,add_value,remove_average_num]=approximated_objective_value_calculate_greedy(T,W,r_vt,Pr_bo_greedy,P_bo_greedy)

    bo_greedy_value_list.append(bo_greedy_value)
    bo_greedy_time_list.append(bo_greedy_time)
    bo_greedy_rewards_per_match_list.append(add_value)
    bo_greedy_match_num_list.append(remove_average_num)

    #----------------------------------------------------
    #RS-A method
    start_time=time.time()
    max_value=0
    opt_vec_Lp=np.ones([m,n,T])
    while True:
        P_rand_approx=-np.random.rand(n,T)
        Pr_rand_approx=1-(1/(1+np.exp(-(P_rand_approx+beta*base_price.reshape(-1,1))/(gamma*base_price.reshape(-1,1)))))
        #Calculate the objective value of the search point
        problem = pulp.LpProblem('tsp_mip', pulp.LpMaximize)
        z = {(u,v,t):pulp.LpVariable(name='z_{}_{}_{}'.format(u,v,t), lowBound=0, upBound=1, cat='Continuous') for u in range(m) for v in range(n) for t in range (T)}
        problem += pulp.lpSum([(W[u,v] + P_rand_approx[v,t])* z[u,v,t] for u in range(m) for v in range(n) for t in range(T)])
        for v in range(n):
            for t in range(T):
                problem += pulp.lpSum([z[u, v ,t] for u in range(m)]) <= Pr_rand_approx[v,t]*r_vt[v,t]
        for u in range(m):
            problem += pulp.lpSum([z[u,v,t] for v in range(n) for t in range(T)]) <= 1
        status = problem.solve(GUROBI_CMD(msg=0))
        #status = problem.solve(PULP_CBC_CMD(msg=0))

        #If a larger objective value is obtained, update the solution.
        if pulp.value(problem.objective)> max_value:
            max_value=pulp.value(problem.objective)
            tmp=0
            for u in range(m):
                for v in range(n):
                    for t in range(T):
                        opt_vec_Lp[u,v,t]=pulp.value(z[u,v,t])
            Max_P_rand_approx=P_rand_approx
            Max_Pr_rand_approx=Pr_rand_approx
        #Exit when run time is over.
        if time.time()-start_time>rs_time:
            break

    #To obtain the 1/2 approximation matching strategy, calculate beta_ut.
    gamma_app_ratio=0.5
    beta_ut=beta_ut_calculate(T,m,r_vt,Max_Pr_rand_approx,opt_vec_Lp,gamma_app_ratio)

    # Calculate results
    rs_approx_time=time.time()-start_time
    [rs_approx_value,add_value,remove_average_num]=approximated_objective_value_calculate_dickersons_matching_strategy(m,T,r_vt,Max_Pr_rand_approx,opt_vec_Lp,beta_ut,W,Max_P_rand_approx,gamma_app_ratio)

    rs_approx_value_list.append(rs_approx_value)
    rs_approx_time_list.append(rs_approx_time)
    rs_approx_rewards_per_match_list.append(add_value)
    rs_approx_match_num_list.append(remove_average_num)

    #----------------------------------------------------
    #RS-G method
    start_time=time.time()
    profit_list=[]
    max_value=0
    add_value=0
    remove_average_num=0
    while True:
        P_rand=-np.random.rand(n,T)
        Pr_rand=1-(1/(1+np.exp(-(P_rand+beta*base_price.reshape(-1,1))/(gamma*base_price.reshape(-1,1)))))

        #Calculate the objective value of the search point
        [result,tmp1,tmp2]=approximated_objective_value_calculate_greedy(T,W,r_vt,Pr_rand,P_rand)
        profit_list.append(result)

        #If a larger objective value is obtained, update the solution.
        if result> max_value:
            max_value=result
            add_value=tmp1
            remove_average_num=tmp2

        #Exit when run time is over
        if time.time()-start_time>rs_time:
            break
    #Calculate results
    rs_greedy_value=max(profit_list)
    rs_greedy_time=time.time()-start_time

    #Add results to lists
    rs_greedy_value_list.append(rs_greedy_value)
    rs_greedy_time_list.append(rs_greedy_time)
    rs_greedy_rewards_per_match_list.append(add_value)
    rs_greedy_match_num_list.append(remove_average_num)

    print('one simulation ends')

#Outputs results
with open('../results/result_m=%d_n=%d_T=%d_BOruntime=%d_RSruntime=%d_simulations=%d.csv' %(m,n,T,bo_time,rs_time,num_simulation), mode='w') as f_tmp:
    writer = csv.writer(f_tmp)
    writer.writerow(['method','objective value','computation time (seconds)','average reward per one match','average number of matches'])
    writer.writerow(['Proposed',statistics.mean(proposed_value_list),statistics.mean(proposed_time_list),statistics.mean(proposed_rewards_per_match_list),statistics.mean(proposed_match_num_list)])
    writer.writerow(['CU-A',statistics.mean(cu_approx_value_list),statistics.mean(cu_approx_time_list),statistics.mean(cu_approx_rewards_per_match_list),statistics.mean(cu_approx_match_num_list)])
    writer.writerow(['CU-G',statistics.mean(cu_greedy_value_list),statistics.mean(cu_greedy_time_list),statistics.mean(cu_greedy_rewards_per_match_list),statistics.mean(cu_greedy_match_num_list)])
    writer.writerow(['BO-A',statistics.mean(bo_approx_value_list),statistics.mean(bo_approx_time_list),statistics.mean(bo_approx_rewards_per_match_list),statistics.mean(bo_approx_match_num_list)])
    writer.writerow(['BO-G',statistics.mean(bo_greedy_value_list),statistics.mean(bo_greedy_time_list),statistics.mean(bo_greedy_rewards_per_match_list),statistics.mean(bo_greedy_match_num_list)])
    writer.writerow(['RS-A',statistics.mean(rs_approx_value_list),statistics.mean(rs_approx_time_list),statistics.mean(rs_approx_rewards_per_match_list),statistics.mean(rs_approx_match_num_list)])
    writer.writerow(['RS-G',statistics.mean(rs_greedy_value_list),statistics.mean(rs_greedy_time_list),statistics.mean(rs_greedy_rewards_per_match_list),statistics.mean(rs_greedy_match_num_list)])
