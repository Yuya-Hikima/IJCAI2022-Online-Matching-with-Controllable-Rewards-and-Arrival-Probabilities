#coding: utf-8

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
import sys
import pulp
import GPy
import GPyOpt
from pulp import GUROBI_CMD
#from pulp import PULP_CBC_CMD
import statistics
import copy

args=sys.argv

#Setting
#Setting of target data
place='Manhattan'
year=2019
month=int(args[1])
day=int(args[2])

#Run times of Bayesian Optimization and Random Search
bo_time=int(args[3])
rs_time=int(args[4])
#number of simulations
num_simulation=int(args[5])

#Preprocessing of the data
hour_start=10
hour_end=20
minute_start=0
second_start=0
day_start_time=datetime.datetime(year,month,day,hour_start-1,minute_start+55,second_start)
day_end_time=datetime.datetime(year,month,day,hour_end,minute_start+5,second_start)

df_loc=pd.read_csv("../data/location_data.csv")
df=pd.read_csv("../data/yellow_tripdata_2019-0%d.csv" %month,parse_dates=['tpep_pickup_datetime','tpep_dropoff_datetime'])
df=pd.merge(df, df_loc, how="inner" ,left_on="PULocationID",right_on="LocationID")
df=df[(df["trip_distance"] >10**(-3))&(df["total_amount"] >10**(-3))&(df["borough"] == place)&(df["PULocationID"] < 264)&(df["DOLocationID"] < 264)&(df["tpep_pickup_datetime"] > day_start_time)& (df["tpep_pickup_datetime"] < day_end_time)]
df_263area=df[['borough','PULocationID', 'DOLocationID','trip_distance','total_amount','tpep_pickup_datetime','tpep_dropoff_datetime']]

df_loc_agg=pd.read_csv("../data/locationID_to_PU_aggregated_ID.csv")
df_PU_merge=pd.merge(df_263area, df_loc_agg, how="inner" ,left_on="PULocationID",right_on="OBJECTID")
df_loc_agg=pd.read_csv("../data/locationID_to_DO_aggregated_ID.csv")
df_merge=pd.merge(df_PU_merge, df_loc_agg, how="inner" ,left_on="DOLocationID",right_on="OBJECTID_2")

#Obtain data which has [the actual price paid, the time pick up time, pick up area, drop off area] for each recorded order.
df_after_preprocessing=df_merge[["total_amount","tpep_pickup_datetime","PU_AggregatedID","DO_AggregatedID"]]

#Calculate the distance between each location
#Aggregated_location.csv has the longitude/latitude of the central point of each location.
df_longlati=pd.read_csv("../data/Aggregated_location.csv")
tu_data=df_longlati.values[:,2:4]
#dist_matrix has the distance (km) from location i to location j as element (i,j).
dist_matrix = np.zeros([20,20])
for i in range(20):
    for j in range(20):
        azimuth, bkw_azimuth, distance = grs80.inv(tu_data[i,0], tu_data[i,1], tu_data[j,0], tu_data[j,1])
        dist_matrix[i,j]=distance
dist_matrix=dist_matrix*0.001

#Setting parameters of the problem size
m=30
n=400
T=120
#the number of simulations in Monte Carlo method
num_Monte_Carlo=10**3
#the number of simulations for calcilationf \beta in Dickerson's matching strategy
beta_calc_num=10000

#Setting parameters of function p_vt
#the parameters of the sigmoid function p_vt
beta=1.25
gamma=0.25*math.sqrt(3)/math.pi

#r_vt is the probability that node v appears at time t
r_vt=np.zeros([n,T])
#q_vt is used for setting p_vt
q_vt=np.zeros([n,T])

for hour_count in range(int(T/12)):
    hour=10+hour_count
    minute=0
    second=0
    set_time=datetime.datetime(year,month,day,hour,minute,second)
    after_1h=datetime.datetime(year,month,day,hour+1,minute,second)
    df2_minu=df_after_preprocessing[(df_merge["tpep_pickup_datetime"] > set_time)& (df_merge["tpep_pickup_datetime"] < after_1h)]
    r_v_sum=np.zeros(400)
    q_v_list=np.zeros(400)
    h=0
    for index, row in df2_minu.iterrows():
        r_v_sum[row[2]*20+row[3]]+=1
        q_v_list[row[2]*20+row[3]]+=row[0]
        h+=1
    r_v=r_v_sum/h
    q_v=np.nan_to_num(q_v_list/r_v_sum)
    for tmp in range(12):
        r_vt[:,hour_count*12+tmp]=r_v
        q_vt[:,hour_count*12+tmp]=q_v

#Parameter setting of the PDHG method (used in our method)
tau_PDHG=0.1
sigma_PDHG=0.1
alpha_PDHG=0.7
eta_PDHG=0.7
c_PDHG=0.7


#Lists to store results
proposed_value_list=[]
proposed_time_list=[]
proposed_reward_per_match_list=[]
proposed_match_num_mean_list=[]

cu_approx_value_list=[]
cu_approx_time_list=[]
cu_approx_reward_per_match_list=[]
cu_approx_match_num_mean_list=[]

cu_greedy_value_list=[]
cu_greedy_time_list=[]
cu_greedy_reward_per_match_list=[]
cu_greedy_match_num_mean_list=[]

bo_approx_value_list=[]
bo_approx_time_list=[]
bo_approx_reward_per_match_list=[]
bo_approx_match_num_mean_list=[]

bo_greedy_value_list=[]
bo_greedy_time_list=[]
bo_greedy_reward_per_match_list=[]
bo_greedy_match_num_mean_list=[]

rs_approx_value_list=[]
rs_approx_time_list=[]
rs_approx_reward_per_match_list=[]
rs_approx_match_num_mean_list=[]

rs_greedy_value_list=[]
rs_greedy_time_list=[]
rs_greedy_reward_per_match_list=[]
rs_greedy_match_num_mean_list=[]

def generate_random_index_based_on_given_pd(pd):
    cumulative_dist = np.cumsum(pd).tolist()
    cumulative_dist[-1] = 1.0
    random_num = np.random.rand()
    cumulative_dist.append(random_num)

    return sorted(cumulative_dist).index(random_num)

def approximated_objective_value_calculate_dickersons_matching_strategy(W_uv,r_vt,z,Pr,beta_ut,P):
    #return the approximated objective value by Monte Carlo method for input parameters when dickerson's matching strategy is used.
    total_rewards_list=[]
    reward_per_match_list=[]
    match_num_list=[]
    for h in range(num_Monte_Carlo):
        total_rewards=0
        remove_index_list=[]
        remmain_index_list=list(range(m))
        remove_period_list=[]
        match_num=0
        for k in range(T):
            if remmain_index_list!=[]:
                arrive_index=generate_random_index_based_on_given_pd(r_vt[:,k])
                #determin if the arriving worker will accept the wage
                tmp=np.random.rand()
                if tmp < Pr[arrive_index,k]:
                    #determine the node u to match by dickerson's matching strategy.
                    tmp=generate_random_index_based_on_given_pd(list(z[remmain_index_list,arrive_index,k]*0.5/(Pr[arrive_index,k]*r_vt[arrive_index,k]*beta_ut[remmain_index_list,k])))
                    if tmp==len(remmain_index_list):
                        continue
                    matching_index=remmain_index_list[tmp]
                    del remmain_index_list[tmp]
                    total_rewards+=W_uv[matching_index,arrive_index]+P[arrive_index,k]
                    match_num+=1
                    remove_index_list.append(matching_index)
                    remove_period_list.append(c_uv[matching_index,arrive_index])
            tmp=0
            for i in range(len(remove_index_list)):
                remove_period_list[tmp]-=1
                if remove_period_list[tmp]==0:
                    remmain_index_list.append(remove_index_list[tmp])
                    del remove_index_list[tmp]
                    del remove_period_list[tmp]
                else:
                    tmp+=1
            #print(remove_index_list,remove_period_list)
        total_rewards_list.append(total_rewards)
        if match_num>0:
            reward_per_match_list.append(total_rewards/match_num)
        match_num_list.append(match_num)
    if reward_per_match_list==[]:
        reward_per_match_list=[-np.inf,-np.inf]
    return [sum(total_rewards_list)/len(total_rewards_list),reward_per_match_list,match_num_list]

def approximated_objective_value_calculate_greedy(W_uv,r_vt,Pr,P):
    total_rewards_list=[]
    match_num_list=[]
    reward_per_match_list=[]
    for h in range(num_Monte_Carlo):
        total_rewards=0
        remove_index_list=[]
        remove_period_list=[]
        match_num=0
        W_tmp=W_uv.copy()
        for k in range(T):
            arrive_index=generate_random_index_based_on_given_pd(r_vt[:,k])
            tmp=np.random.rand()
            if tmp < Pr[arrive_index,k]:
                while True:
                    matching_index=list(W_tmp[:,arrive_index]).index(max(list(W_tmp[:,arrive_index])))
                    if not matching_index in remove_index_list:
                        if W_uv[matching_index,arrive_index]+P[arrive_index,k]<0:
                            break
                        total_rewards+=W_uv[matching_index,arrive_index]+P[arrive_index,k]
                        match_num+=1
                        remove_index_list.append(matching_index)
                        remove_period_list.append(c_uv[matching_index,arrive_index])
                        break
                    else:
                        #print('already matched')
                        W_tmp[matching_index,arrive_index]=-np.inf
            tmp=0
            for i in range(len(remove_index_list)):
                remove_period_list[tmp]-=1
                if remove_period_list[tmp]==0:
                    del remove_index_list[tmp]
                    del remove_period_list[tmp]
                else:
                    tmp+=1
        total_rewards_list.append(total_rewards)
        match_num_list.append(match_num)
        if match_num>0:
            reward_per_match_list.append(total_rewards/match_num)
    if reward_per_match_list==[]:
        reward_per_match_list=[-np.inf,-np.inf]
    return [sum(total_rewards_list)/len(total_rewards_list),reward_per_match_list,match_num_list]

def beta_calculate(Pr,z,r_vt,T):
    #compute the beta required for dickerson's matching strategy
    beta_ut=np.ones([m,T])
    remmain_index_list=[]
    remove_index_list=[]
    remove_period_list=[]
    for h in range(beta_calc_num):
        remmain_index_list.append(list(range(m)))
        remove_index_list.append([])
        remove_period_list.append([])
    for t in range(T):
        for h in range(beta_calc_num):
            arrive_index=generate_random_index_based_on_given_pd(r_vt[:,t])
            tmp=np.random.rand()
            if tmp < Pr[arrive_index,t]:
                if remmain_index_list[h]==[]:
                    continue
                #print(list(z_star[remmain_index_list[h],arrive_index,t]))
                tmp=generate_random_index_based_on_given_pd(list(z[remmain_index_list[h],arrive_index,t]*0.5/(Pr[arrive_index,t]*r_vt[arrive_index,t]*beta_ut[remmain_index_list[h],t])))
                if tmp==len(remmain_index_list[h]):
                    continue
                matching_index=remmain_index_list[h][tmp]
                remmain_index_list[h].remove(matching_index)
                remove_index_list[h].append(matching_index)
                remove_period_list[h].append(c_uv[matching_index,arrive_index])
            tmp=0
            for i in range(len(remove_index_list[h])):
                remove_period_list[h][tmp]-=1
                if remove_period_list[h][tmp]==0:
                    remmain_index_list[h].append(remove_index_list[h][tmp])
                    del remove_index_list[h][tmp]
                    del remove_period_list[h][tmp]
                else:
                    tmp+=1
            if t!=T-1:
                for u in remove_index_list[h]:
                    beta_ut[u,t+1]-=1.0/beta_calc_num
    return beta_ut


#Starts experiments
for exp_iter in range(num_simulation):
    #Generate taxi's docking position
    U=random.choices(range(20),k=m)
    #calculate the distance between each taxi u and each order v
    dist_matrix_uv=np.zeros([m,n])
    for u in range(m):
        for v in range(n):
            v_start,v_end=divmod(v,20)
            dist_matrix_uv[u,v]=dist_matrix[U[u],v_start]+dist_matrix[v_start,v_end]+dist_matrix[v_end,U[u]]
    #tau_uv is required time (hour) for taxi u to archieve order v
    tau_uv=dist_matrix_uv/20.0


    #tau_uv is required periods for taxi u to archieve order v. One period is defined as five minutes.
    c_uv=np.ceil(tau_uv*12).astype('int64')
    #At least one period of time is taken to fullfill the order
    c_uv+=(np.ones([m,n])*(c_uv==0)).astype('int64')
    #(-1.0) is incurred per period as opportunity cost
    W_uv=c_uv*(-1.0)

    #sum_list_ut represents constraints of problem (CP'). This is used in baselines.
    sum_list_ut=[]
    for u in range(m):
        for t in range(T):
            sum_list_ut_tmp=[]
            for v in range(n):
                for q in range(c_uv[u,v]):
                    if t-q<0:
                        break
                    sum_list_ut_tmp.append([u,v,t-q])
            sum_list_ut.append(sum_list_ut_tmp)


    #Starts proposed method
    start_time=time.time()

    #Initial points of the primal variables and dual variables
    z_k=np.zeros([m,n,T])
    la_k=np.zeros([m,T])

    #the iteration number
    k=1
    #the primal residual
    p=1
    #the dual residual
    d=1
    A_z_star=np.zeros([m,T])
    while k<30 and (np.linalg.norm(p)>10**-4 or np.linalg.norm(d)>10**-4):
        #Update primal variables (line 4 and 5 in Algorithm 1 of our paper)
        #z_star means z_{k+1}.
        z_star=np.zeros([m,n,T])
        for v in range(n):
            for t in range(T):
                if r_vt[v,t]>10**(-7):
                    #Solve Eq (7)
                    A_vt_t_la_k=np.zeros(m)
                    for u in range(m):
                        A_vt_t_la_k[u]=np.sum(la_k[u,t:(t+c_uv[u,v])])
                    a=z_k[:,v,t]+tau_PDHG*(W_uv[:,v]-A_vt_t_la_k)
                    s=r_vt[v,t]/2.0
                    delta=s/2.0
                    left_plus=100 #dummy number
                    while np.abs(s-np.sum(left_plus))>r_vt[v,t]*10**(-6):
                        f_vt_dash_s=-beta*q_vt[v,t]+gamma*q_vt[v,t]*np.log(s/(r_vt[v,t]-s))+gamma*q_vt[v,t]*r_vt[v,t]/(r_vt[v,t]-s)
                        left=a-tau_PDHG*f_vt_dash_s
                        left_plus=left*(left>0)
                        if sum(left_plus)>s:
                               s+=delta
                        else:
                               s-=delta
                        delta=delta/2
                    #Set z_star by Proposition 6
                    tmp=a-tau_PDHG*f_vt_dash_s
                    z_star[:,v,t]=tmp*(tmp>0)

        #Update dual variables (line 6 in Algorithm 1 of our paper)
        A_z_star_old=np.copy(A_z_star)
        table=np.zeros([m,T+1])
        for u in range(m):
            for t in range(T):
                table[u,t]+=sum(z_star[u,:,t])
                for v in range(n):
                    table[u,min(t+c_uv[u,v],T)]-=z_star[u,v,t]
        for u in range(m):
            for t in range(T):
                table[u,t]+=table[u,t-1]
        A_z_star=table[:,range(T)]
        la_tmp=la_k+sigma_PDHG*(2*A_z_star-A_z_star_old-1)
        #la_star means \lambda_{k+1}
        la_star=la_tmp*(la_tmp>0)

        #Update step sizes (line 7--14 in Algorithm 1 of our paper)
        z_d=z_star-z_k
        la_d=la_star-la_k

        table=np.zeros([m,T+1])
        for u in range(m):
            for t in range(T):
                table[u,t]+=sum(z_d[u,:,t])
                for v in range(n):
                    table[u,min(t+c_uv[u,v],T)]-=z_d[u,v,t]

        for u in range(m):
            for t in range(T):
                table[u,t]+=table[u,t-1]
        A_z_d=table[:,range(T)]

        A_t_la_d=np.zeros([m,n,T])
        for v in range(n):
            for t in range(T):
                A_vt_t_la_d=np.zeros(m)
                for i in range(m):
                    A_vt_t_la_d[i]=np.sum(la_k[i,t:t+c_uv[i,v]])
                A_t_la_d[:,v,t]=A_vt_t_la_d

        #line 7 in Algorithm 1 of our paper
        if (c_PDHG/(2.0*tau_PDHG))*np.sum(np.power(z_d,2))+(c_PDHG/(2.0*sigma_PDHG))*np.sum(np.power(la_d,2)) <= 2.0*np.sum(la_d*A_z_d):
            tau_PDHG=0.5*tau_PDHG
            sigma_PDHG=0.5*sigma_PDHG

        #line 9 and 10 in Algorithm 1 of our paper
        p=-z_d/tau_PDHG+A_t_la_d
        d=-la_d/sigma_PDHG-A_z_d

        #line 11--14 in Algorithm 1 of our paper
        if 2*np.linalg.norm(p)< np.linalg.norm(d):
            tau_PDHG=tau_PDHG*(1-alpha_PDHG)
            sigma_PDHG=sigma_PDHG/(1-alpha_PDHG)
            alpha_PDHG=alpha_PDHG*eta_PDHG
        elif np.linalg.norm(p) > 2*np.linalg.norm(d):
            tau_PDHG=tau_PDHG/(1-alpha_PDHG)
            sigma_PDHG=sigma_PDHG*(1-alpha_PDHG)
            alpha_PDHG=alpha_PDHG*eta_PDHG
        z_k=z_star.copy()
        la_k=la_star.copy()
        k=k+1

    #Calculate x_vt (by Proposition 4 of our paper)
    z_vt_sum_star=np.sum(z_star,axis=0)
    X_vt_proposed=np.zeros([n,T])
    for v in range(n):
        for t in range(T):
            if z_vt_sum_star[v,t]>r_vt[v,t]*(10**(-4)):
                X_vt_proposed[v,t]=beta*q_vt[v,t]-gamma*q_vt[v,t]*np.log(z_vt_sum_star[v,t]/(r_vt[v,t]-z_vt_sum_star[v,t]))
                if r_vt[v,t]-z_vt_sum_star[v,t]<0:
                    error
                if z_vt_sum_star[v,t]<0:
                    error
            else:
                X_vt_proposed[v,t]=0

    #Calculate p_vt(x_vt)
    p_vt_proposed=np.nan_to_num(1-(1/(1+np.exp(-(X_vt_proposed-beta*q_vt)/(gamma*q_vt)))))
    #To obtain the 1/2 approximation matching strategy [Dickerson et al., 2018], calculate beta_ut.
    beta_ut=beta_calculate(p_vt_proposed,z_star,r_vt,T)

    #Calculate results
    proposed_time=time.time()-start_time
    [proposed_value,proposed_reward_per_match_list,proposed_match_num_list]=approximated_objective_value_calculate_dickersons_matching_strategy(W_uv,r_vt,z_star,p_vt_proposed,beta_ut,X_vt_proposed)

    #----------------------------------------------------
    #CU-A method
    start_time=time.time()
    w_hat=np.average(W_uv)
    c_hat=np.average(c_uv)
    #Set to search x is [0, 50]^{V \times T}
    upper=50.0
    lower=0.0
    #Calculate x_vt by Golden section method
    eta_gold=(np.sqrt(5)-1)/(np.sqrt(5)+1)
    beta_gold=upper-lower
    tau_gold=eta_gold*beta_gold
    x_0=lower
    x_1=lower+tau_gold
    x_2=upper-tau_gold
    x_3=upper
    Pr_0=np.sum(r_vt*(1-(1/(1+np.exp(-(x_0-beta*q_vt)/(gamma*q_vt))))))
    Pr_1=np.sum(r_vt*(1-(1/(1+np.exp(-(x_1-beta*q_vt)/(gamma*q_vt))))))
    Pr_2=np.sum(r_vt*(1-(1/(1+np.exp(-(x_2-beta*q_vt)/(gamma*q_vt))))))
    Pr_3=np.sum(r_vt*(1-(1/(1+np.exp(-(x_3-beta*q_vt)/(gamma*q_vt))))))
    f_0=(x_0+w_hat)*np.min([m*T/c_hat,Pr_0])
    f_1=(x_1+w_hat)*np.min([m*T/c_hat,Pr_1])
    f_2=(x_2+w_hat)*np.min([m*T/c_hat,Pr_2])
    f_3=(x_3+w_hat)*np.min([m*T/c_hat,Pr_3])
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
            Pr_1=np.sum(r_vt*(1-(1/(1+np.exp(-(x_1-beta*q_vt)/(gamma*q_vt))))))
            tmp_1=(x_1+w_hat)*np.min([m*T/c_hat,Pr_1])
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
            Pr_2=np.sum(r_vt*(1-(1/(1+np.exp(-(x_2-beta*q_vt)/(gamma*q_vt))))))
            tmp_0=f_1
            tmp_1=f_2
            tmp_2=(x_2+w_hat)*np.min([m*T/c_hat,Pr_2])
            f_0=copy.copy(tmp_0)
            f_1=copy.copy(tmp_1)
            f_2=copy.copy(tmp_2)
        beta_gold=beta_gold-tau_gold
        tau_gold=eta_gold*beta_gold
    P_cu_approx=np.copy(x_2)
    Pr_cu_approx=np.nan_to_num(1-(1/(1+np.exp(-(P_cu_approx-beta*q_vt)/(gamma*q_vt)))))

    #To obtain 1/2 approximation matching strategy [Dickerson et al., 2018], solve problem (1) in our paper.
    #We use pulp to model the optimization problem and solve it by Gurobi. (If you don't have Gurobi ricense, you can use CBC solver instead.)
    #Set the optimization problem by pulp
    problem = pulp.LpProblem('tsp_mip', pulp.LpMaximize)
    z = {(u,v,t):pulp.LpVariable(name='z_{}_{}_{}'.format(u, v,t), lowBound=0, upBound=1, cat='Continuous') for u in range(m) for v in range(n) for t in range(T)}
    problem += pulp.lpSum([(W_uv[u,v] + P_cu_approx)* z[u,v,t] for u in range(m) for v in range(n) for t in range(T)])
    for v in range(n):
        for t in range(T):
            problem += pulp.lpSum([z[u, v ,t] for u in range(m)]) <= Pr_cu_approx[v,t]*r_vt[v,t]
    for u in range(m):
        for t in range(T):
            problem += pulp.lpSum([z[a,b,c] for [a,b,c] in sum_list_ut[u*T+t]]) <= 1

    #solve the optimization problem by Gurobi.
    solver = GUROBI_CMD(msg=0)
    status = problem.solve(GUROBI_CMD(msg=0))
    #status = problem.solve(PULP_CBC_CMD(msg=0))

    opt_vec_Lp=np.ones([m,n,T])
    for u in range(m):
        for v in range(n):
            for t in range(T):
                opt_vec_Lp[u,v,t]=pulp.value(z[u,v,t])

    #To obtain the 1/2 approximation matching strategy, calculate beta_ut.
    beta_ut=beta_calculate(Pr_cu_approx,opt_vec_Lp,r_vt,T)

    #Calculate results
    cu_approx_time=time.time()-start_time
    [cu_approx_value,cu_approx_reward_per_match_list,cu_approx_match_num_list]=approximated_objective_value_calculate_dickersons_matching_strategy(W_uv,r_vt,opt_vec_Lp,Pr_cu_approx,beta_ut,P_cu_approx*np.ones([n,T]))

    #----------------------------------------------------
    #CU-G method
    start_time=time.time()
    w_hat=np.average(W_uv)
    c_hat=np.average(c_uv)
    #Set to search x is [0, 50]^{V \times T}
    upper=50.0
    lower=0.0
    #Calculate x_vt by Golden section method
    eta_gold=(np.sqrt(5)-1)/(np.sqrt(5)+1)
    beta_gold=upper-lower
    tau_gold=eta_gold*beta_gold
    x_0=lower
    x_1=lower+tau_gold
    x_2=upper-tau_gold
    x_3=upper
    Pr_0=np.sum(r_vt*(1-(1/(1+np.exp(-(x_0-beta*q_vt)/(gamma*q_vt))))))
    Pr_1=np.sum(r_vt*(1-(1/(1+np.exp(-(x_1-beta*q_vt)/(gamma*q_vt))))))
    Pr_2=np.sum(r_vt*(1-(1/(1+np.exp(-(x_2-beta*q_vt)/(gamma*q_vt))))))
    Pr_3=np.sum(r_vt*(1-(1/(1+np.exp(-(x_3-beta*q_vt)/(gamma*q_vt))))))
    f_0=(x_0+w_hat)*np.min([m*T/c_hat,Pr_0])
    f_1=(x_1+w_hat)*np.min([m*T/c_hat,Pr_1])
    f_2=(x_2+w_hat)*np.min([m*T/c_hat,Pr_2])
    f_3=(x_3+w_hat)*np.min([m*T/c_hat,Pr_3])
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
            Pr_1=np.sum(r_vt*(1-(1/(1+np.exp(-(x_1-beta*q_vt)/(gamma*q_vt))))))
            tmp_1=(x_1+w_hat)*np.min([m*T/c_hat,Pr_1])
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
            Pr_2=np.sum(r_vt*(1-(1/(1+np.exp(-(x_2-beta*q_vt)/(gamma*q_vt))))))
            tmp_0=f_1
            tmp_1=f_2
            tmp_2=(x_2+w_hat)*np.min([m*T/c_hat,Pr_2])
            f_0=copy.copy(tmp_0)
            f_1=copy.copy(tmp_1)
            f_2=copy.copy(tmp_2)
        beta_gold=beta_gold-tau_gold
        tau_gold=eta_gold*beta_gold
    P_cu_greedy=np.copy(x_2)*np.ones([n,T])

    #Calculate results
    cu_greedy_time=time.time()-start_time
    Pr_cu_greedy=(1-(1/(1+np.exp(-(P_cu_greedy-beta*q_vt)/(gamma*q_vt)))))*np.ones([n,T])
    [cu_greedy_value,cu_greedy_reward_per_match_list,cu_greedy_match_num_list]=approximated_objective_value_calculate_greedy(W_uv,r_vt,Pr_cu_greedy,P_cu_greedy)

    #----------------------------------------------------
    #BO-A method
    #We use GPyOpt.methods.BayesianOptimization to search for x_vt.
    #Settting of set to search x_vt
    bounds=[]
    for k in range(n*T):
        dic ={}
        key = "name"
        dic[key] = f"x{k}"
        key = "type"
        dic[key] = "continuous"
        key = "domain"
        dic[key] = (0,50)
        bounds.append(dic)



    #Settting of the oracle.
    max_value=0
    def f_approx(x):
        global max_value
        global opt_vec_Lp_bo_approx
        P_bo_approx_or=x[0]
        P_bo_approx=np.zeros([n,T])
        for v in range(n):
            for t in range(T):
                P_bo_approx[v,t]=P_bo_approx_or[v+t*n]
        Pr_bo_approx=1-(1/(1+np.exp(-(P_bo_approx-beta*q_vt)/(gamma*q_vt))))
        problem = pulp.LpProblem('tsp_mip', pulp.LpMaximize)
        z = {(u,v,t):pulp.LpVariable(name='z_{}_{}_{}'.format(u, v,t), lowBound=0, upBound=1, cat='Continuous') for u in range(m) for v in range(n) for t in range(T)}
        problem += pulp.lpSum([(W_uv[u,v] + P_bo_approx[v,t])* z[u,v,t] for u in range(m) for v in range(n) for t in range(T)])
        for v in range(n):
            for t in range(T):
                problem += pulp.lpSum([z[u, v ,t] for u in range(m)]) <= Pr_bo_approx[v,t]*r_vt[v,t]
        for u in range(m):
            for t in range(T):
                problem += pulp.lpSum([z[a,b,c] for [a,b,c] in sum_list_ut[u*T+t]]) <= 1
        status = problem.solve(GUROBI_CMD(msg=0))
        objective_value=pulp.value(problem.objective)
        if objective_value>max_value:
            max_value=objective_value
            opt_vec_Lp_bo_approx=np.ones([m,n,T])
            tmp=0
            for u in range(m):
                for v in range(n):
                    for t in range(T):
                        opt_vec_Lp_bo_approx[u,v,t]=pulp.value(z[u,v,t])
        return pulp.value(problem.objective)

    #Search for x_vt
    start_time=time.time()
    myBopt = GPyOpt.methods.BayesianOptimization(f=f_approx, domain=bounds, maximize=True)
    myBopt.run_optimization(max_iter=10**8,max_time=bo_time)
    P_bo_approx_or=myBopt.x_opt
    P_bo_approx=np.zeros([n,T])
    for v in range(n):
        for t in range(T):
            P_bo_approx[v,t]=P_bo_approx_or[v+t*n]
    Pr_bo_approx=1-(1/(1+np.exp(-(P_bo_approx-beta*q_vt)/(gamma*q_vt))))

    #To obtain the 1/2 approximation matching strategy, calculate beta_ut.
    beta_ut_bo_approx=beta_calculate(Pr_bo_approx,opt_vec_Lp_bo_approx,r_vt,T)

    #Calculate results
    bo_approx_time=time.time()-start_time
    [bo_approx_value,bo_approx_reward_per_match_list,bo_approx_match_num_list]=approximated_objective_value_calculate_dickersons_matching_strategy(W_uv,r_vt,opt_vec_Lp_bo_approx,Pr_bo_approx,beta_ut_bo_approx,P_bo_approx)

    #----------------------------------------------------
    #BO-G method
    #We use GPyOpt.methods.BayesianOptimization to search for x_vt.
    #Settting of the oracle
    def f(x):
        P_ba_greedy=x[0].reshape([n,T])
        Pr_ba_greedy=1-(1/(1+np.exp(-(P_ba_greedy-beta*q_vt)/(gamma*q_vt))))
        total_rewards_list=[]
        for h in range(num_Monte_Carlo):
            total_rewards=0
            remove_index_list=[]
            remove_period_list=[]
            W_tmp=W_uv.copy()
            for k in range(T):
                arrive_index=generate_random_index_based_on_given_pd(r_vt[:,k])
                tmp=np.random.rand()
                if tmp < Pr_ba_greedy[arrive_index,k]:
                    while True:
                        matching_index=list(W_tmp[:,arrive_index]).index(max(list(W_tmp[:,arrive_index])))
                        if not matching_index in remove_index_list:
                            if W_uv[matching_index,arrive_index]+P_ba_greedy[arrive_index,k]<0:
                                break
                            total_rewards+=W_uv[matching_index,arrive_index]+P_ba_greedy[arrive_index,k]
                            remove_index_list.append(matching_index)
                            remove_period_list.append(c_uv[matching_index,arrive_index])
                            break
                        else:
                            W_tmp[matching_index,arrive_index]=-np.inf
                tmp=0
                for i in range(len(remove_index_list)):
                    remove_period_list[tmp]-=1
                    if remove_period_list[tmp]==0:
                        del remove_index_list[tmp]
                        del remove_period_list[tmp]
                    else:
                        tmp+=1
            total_rewards_list.append(total_rewards)
        return sum(total_rewards_list)/len(total_rewards_list)

    #Search for x_vt
    start_time=time.time()
    myBopt = GPyOpt.methods.BayesianOptimization(f=f, domain=bounds, maximize=True)
    myBopt.run_optimization(max_iter=10**8,max_time=bo_time)

    #Since the method itself is finished, the measurement of computation time is completed.
    bo_greedy_time=time.time()-start_time

    #Calculate results
    P_bo_greedy_or=myBopt.x_opt
    P_bo_greedy=np.zeros([n,T])
    for v in range(n):
        for t in range(T):
            P_bo_greedy[v,t]=P_bo_greedy_or[v+t*n]
    Pr_bo_greedy=1-(1/(1+np.exp(-(P_bo_greedy-beta*q_vt)/(gamma*q_vt))))
    [bo_greedy_value,bo_greedy_reward_per_match_list,bo_greedy_match_num_list]=approximated_objective_value_calculate_greedy(W_uv,r_vt,Pr_bo_greedy,P_bo_greedy)

    #----------------------------------------------------
    #RS-A method
    start_time=time.time()
    max_value=0
    opt_vec_Lp=np.ones([m,n,T])
    while True:
        P_rand_approx=50*np.random.rand(n,T)
        Pr_rand_approx=1-(1/(1+np.exp(-(P_rand_approx-beta*q_vt)/(gamma*q_vt))))
        #Calculate the objective value of the search point
        problem = pulp.LpProblem('tsp_mip', pulp.LpMaximize)
        z = {(u,v,t):pulp.LpVariable(name='z_{}_{}_{}'.format(u, v,t), lowBound=0, upBound=1, cat='Continuous') for u in range(m) for v in range(n) for t in range(T)}
        problem += pulp.lpSum([(W_uv[u,v] + P_rand_approx[v,t])* z[u,v,t] for u in range(m) for v in range(n) for t in range(T)])
        for v in range(n):
            for t in range(T):
                problem += pulp.lpSum([z[u, v ,t] for u in range(m)]) <= Pr_rand_approx[v,t]*r_vt[v,t]
        for u in range(m):
            for t in range(T):
                problem += pulp.lpSum([z[a,b,c] for [a,b,c] in sum_list_ut[u*T+t]]) <= 1

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
    beta_ut=beta_calculate(Max_Pr_rand_approx,opt_vec_Lp,r_vt,T)

    #Calculate results
    rs_approx_time=time.time()-start_time
    [rs_approx_value,rs_approx_reward_per_match_list,rs_approx_match_num_list]=approximated_objective_value_calculate_dickersons_matching_strategy(W_uv,r_vt,opt_vec_Lp,Max_Pr_rand_approx,beta_ut,Max_P_rand_approx)

    #----------------------------------------------------
    #RS-G method
    start_time=time.time()
    max_value=-1
    while True:
        P_rand=50*np.random.rand(n,T)
        Pr_rand=1-(1/(1+np.exp(-(P_rand-beta*q_vt)/(gamma*q_vt))))

        #Calculate the objective value of the search point
        total_rewards_list=[]
        for h in range(num_Monte_Carlo):
            total_rewards=0
            remove_index_list=[]
            remove_period_list=[]
            W_tmp=W_uv.copy()
            for k in range(T):
                arrive_index=generate_random_index_based_on_given_pd(r_vt[:,k])
                tmp=np.random.rand()
                if tmp < Pr_rand[arrive_index,k]:
                    while True:
                        matching_index=list(W_tmp[:,arrive_index]).index(max(list(W_tmp[:,arrive_index])))
                        if not matching_index in remove_index_list:
                            if W_uv[matching_index,arrive_index]+P_rand[arrive_index,k]<0:
                                break
                            total_rewards+=W_uv[matching_index,arrive_index]+P_rand[arrive_index,k]
                            remove_index_list.append(matching_index)
                            remove_period_list.append(c_uv[matching_index,arrive_index])
                            break
                        else:
                            W_tmp[matching_index,arrive_index]=-np.inf
                tmp=0
                for i in range(len(remove_index_list)):
                    remove_period_list[tmp]-=1
                    if remove_period_list[tmp]==0:
                        del remove_index_list[tmp]
                        del remove_period_list[tmp]
                    else:
                        tmp+=1
            total_rewards_list.append(total_rewards)

        #If a larger objective value is obtained, update the solution.
        if sum(total_rewards_list)/len(total_rewards_list)> max_value:
            max_value=sum(total_rewards_list)/len(total_rewards_list)
            max_P_rand=P_rand
            max_Pr_rand=Pr_rand

        #Exit when run time is over
        if time.time()-start_time>rs_time:
            break

    #Calculate results
    rs_greedy_time=time.time()-start_time
    [rs_greedy_value,rs_greedy_reward_per_match_list,rs_greedy_match_num_list]=approximated_objective_value_calculate_greedy(W_uv,r_vt,max_Pr_rand,max_P_rand)


    #Add results to lists
    proposed_value_list.append(proposed_value)
    proposed_time_list.append(proposed_time)
    proposed_reward_per_match_list.append(statistics.mean(proposed_reward_per_match_list))
    proposed_match_num_mean_list.append(statistics.mean(proposed_match_num_list))

    cu_approx_value_list.append(cu_approx_value)
    cu_approx_time_list.append(cu_approx_time)
    cu_approx_reward_per_match_list.append(statistics.mean(cu_approx_reward_per_match_list))
    cu_approx_match_num_mean_list.append(statistics.mean(cu_approx_match_num_list))

    cu_greedy_value_list.append(cu_greedy_value)
    cu_greedy_time_list.append(cu_greedy_time)
    cu_greedy_reward_per_match_list.append(statistics.mean(cu_greedy_reward_per_match_list))
    cu_greedy_match_num_mean_list.append(statistics.mean(cu_greedy_match_num_list))

    bo_approx_value_list.append(bo_approx_value)
    bo_approx_time_list.append(bo_approx_time)
    bo_approx_reward_per_match_list.append(statistics.mean(bo_approx_reward_per_match_list))
    bo_approx_match_num_mean_list.append(statistics.mean(bo_approx_match_num_list))

    bo_greedy_value_list.append(bo_greedy_value)
    bo_greedy_time_list.append(bo_greedy_time)
    bo_greedy_reward_per_match_list.append(statistics.mean(bo_greedy_reward_per_match_list))
    bo_greedy_match_num_mean_list.append(statistics.mean(bo_greedy_match_num_list))

    rs_approx_value_list.append(rs_approx_value)
    rs_approx_time_list.append(rs_approx_time)
    rs_approx_reward_per_match_list.append(statistics.mean(rs_approx_reward_per_match_list))
    rs_approx_match_num_mean_list.append(statistics.mean(rs_approx_match_num_list))

    rs_greedy_value_list.append(rs_greedy_value)
    rs_greedy_time_list.append(rs_greedy_time)
    rs_greedy_reward_per_match_list.append(statistics.mean(rs_greedy_reward_per_match_list))
    rs_greedy_match_num_mean_list.append(statistics.mean(rs_greedy_match_num_list))

#Outputs results
with open('../results/result_month=%d_day=%d_BOruntime=%d_RSruntime=%d_simultaions=%d.csv' %(month,day,bo_time,rs_time,num_simulation), mode='w') as f_tmp:
    writer = csv.writer(f_tmp)
    writer.writerow(['method','objective value','computation time (seconds)','average reward per one match','average number of matches'])
    writer.writerow(['Proposed',statistics.mean(proposed_value_list),statistics.mean(proposed_time_list),statistics.mean(proposed_reward_per_match_list),statistics.mean(proposed_match_num_mean_list)])
    writer.writerow(['CU-A',statistics.mean(cu_approx_value_list),statistics.mean(cu_approx_time_list),statistics.mean(cu_approx_reward_per_match_list),statistics.mean(cu_approx_match_num_mean_list)])
    writer.writerow(['CU-G',statistics.mean(cu_greedy_value_list),statistics.mean(cu_greedy_time_list),statistics.mean(cu_greedy_reward_per_match_list),statistics.mean(cu_greedy_match_num_mean_list)])
    writer.writerow(['BO-A',statistics.mean(bo_approx_value_list),statistics.mean(bo_approx_time_list),statistics.mean(bo_approx_reward_per_match_list),statistics.mean(bo_approx_match_num_mean_list)])
    writer.writerow(['BO-G',statistics.mean(bo_greedy_value_list),statistics.mean(bo_greedy_time_list),statistics.mean(bo_greedy_reward_per_match_list),statistics.mean(bo_greedy_match_num_mean_list)])
    writer.writerow(['RS-A',statistics.mean(rs_approx_value_list),statistics.mean(rs_approx_time_list),statistics.mean(rs_approx_reward_per_match_list),statistics.mean(rs_approx_match_num_mean_list)])
    writer.writerow(['RS-G',statistics.mean(rs_greedy_value_list),statistics.mean(rs_greedy_time_list),statistics.mean(rs_greedy_reward_per_match_list),statistics.mean(rs_greedy_match_num_mean_list)])
