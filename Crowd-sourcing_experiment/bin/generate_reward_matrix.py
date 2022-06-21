import pandas as pd
import numpy as np
import scipy.stats as stats
import pickle

df = pd.read_csv("../work/trec-rf10-data.csv")
df_v=df.values
topicID_set=list(set(df_v[:,0]))
workerID_set=list(set(df_v[:,1]))

def func1(lst, value):
    return [i for i, x in enumerate(lst) if x == value]
Num_t_w=np.zeros([len(topicID_set),len(workerID_set)])
Num_t_w_correct=np.zeros([len(topicID_set),len(workerID_set)])

for r in range(len(topicID_set)):
    topic=topicID_set[r]
    df_t=df[(df["topicID"] == topic)]
    docID_set=list(set(df_t.values[:,2]))
    for h in range(len(docID_set)):
        doc=docID_set[h]
        df_t_d=df_t[(df_t["docID"] == doc)]
        mode_val, mode_num = stats.mode(df_t_d.values[:,4])
        for i in range(len(df_t_d.values[:,0])):
            worker_k=func1(workerID_set, df_t_d.values[i,1])
            Num_t_w[r,worker_k]+=1
            if df_t_d.values[i,4]==int(mode_val):
                Num_t_w_correct[r,worker_k]+=1

C_rate=np.zeros(Num_t_w.shape)
Num=np.zeros(Num_t_w.shape[1])
for i in range(Num_t_w.shape[0]):
    for j in range(Num_t_w.shape[1]):
        if Num_t_w[i,j]==0:
            C_rate[i,j]=0
        else:
            C_rate[i,j]=Num_t_w_correct[i,j]/Num_t_w[i,j]
with open('../work/Reward_matrix', 'wb') as web:
  pickle.dump(C_rate, web)
