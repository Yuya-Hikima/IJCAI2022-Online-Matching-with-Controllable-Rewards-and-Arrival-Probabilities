# IJCAI2022-Online-Matching-with-Controllable-Rewards-and-Arrival-Probabilities

## Remark

This repository is in preparation.

## Overview
This repository contains the codes for the experiments performed in the following papers:
  
Yuya Hikima, Yasunori Akagi, Naoki Marumo, and Hideaki Kim. "Online Matching with Controllable Rewards and ArrivalProbabilities." In IJCAI, 2022 (in press).
  
Contents of this repository:
- **README** This file.
- **SOFTWARE LICENSE AGREEMENT FOR EVALUATION** The user must comply with the rules described herein.
- **Crowd-sourcing_experiment folder** It contains the code used in the crowd-sourcing platform experiment and the code needed to set up the experiment, including data downloads.
- **Ride-sharing_experiment folder** It contains the code used in the ride-sharing platform experiment and the code needed to set up the experiment, including data downloads.
- **Details_of_experiments.pdf** It contains detailed information on our experiments.
- **Proof_of_Lemma2.pdf** It contains the proof of Lemma2 of our paper, which is not included in the paper.

The data for crowd-sourcing experiments are downloaded from
  
http://dbgroup.cs.tsinghua.edu.cn/ligl/crowddata/ (Relevance Finding dataset)
  
The data are shared by
  
Chris Buckley, Matthew Lease, and Mark D. Smucker. "Overview of the TREC 2010 Relevance Feedback Track (Notebook)." In TREC, 2010.

The data for ride-sharing experiments are downloaded from TLC Trip Record Data:
  
https://www1.nyc.gov/site/tlc/about/tlc-trip-record-data.page (Yellow Taxi Trip Records on January-March, 2019).


## Description

The following is a description of what is in each folder.
- **Crowd-sourcing_experiment** 
  - **setup.sh** Script for setting up the experiments
  - **Experiment_paper.sh** Scripts for performing the same experiments as in our paper
  - **Experiment_test.sh** Scripts for small experiments to see if the code works
  - **bin** Folder containing the python code needed for the crowd-sourcing experiments
    - **data_download.py** It is executed by setup.sh to download the real data.
    - **generate_reward_matrix.py** It is executed by setup.sh to make simulation data from the real data.
    - **experiment_crowdsourcing_gurobi.py** It is executed by Experiment.sh or Experiment_test.sh to perform experiments using simulated data. The first argument is $m$ (number of tasks), the second is $n$ (number of users), the third is $T$ (number of user appearance), the fourth is the execution time of Bayesian optimization (in seconds), the fifth is the execution time of random search (in seconds), and the sixth is the number of simulations.
    - **experiment_crowdsourcing_cbc.py** Experimental code to replace the above if you do not have a Gurobi license
  - **data** Folder where the downloaded data is stored
  - **work** Folder where the simulation data is stored
    - **Reward_matrix** Data containing w_et in the paper, generated by experiment_crowdsourcing_gurobi.py.
    - **trec-rf10-data.csv** Converted csv data grom the downloaded data
  - **results** Folder where the results are stored
    - **result_m=XX_n=XX_T=XX_BOruntime=XX_RSruntime=XX_simulations=XX.csv** The data containig results for a given parameter set

- **Ride-sharing_experiment** 
  - **setup.sh** Script for setting up the experiments
  - **Experiment_paper.sh** Scripts for performing the same experiments as in our paper
  - **Experiment_test.sh** Scripts for small experiments to see if the code works
  - **bin** Folder containing the python code needed for the ride-sharing experiments
    - **data_download.py** It is executed by setup.sh to download the real data.
    - **generate_reward_matrix.py** It is executed by setup.sh to make simulation data from the real data
    - **experiment_ridesharing_gurobi.py** It is executed by Experiment.sh or Experiment_test.sh to perform experiments using simulated data. The first argument is the target month, the second argument is the target day, the third argument is the run time of the Bayesian optimization (in seconds), the fourth argument is the run time of the random search (in seconds), and the fifth argument is the number of simulations.
    - **experiment_ridesharing_cbc.py** Experimental code to replace the above if you do not have a Gurobi license
  - **data** Folder containing the downloaded data and the ID information for dividing Manhattan into 20 areas
    - **locationID_to_DO_aggregated_ID.csv** Data showing the correspondence to 20 area IDs for the area ID of the original data
    - **locationID_to_PU_aggregated_ID.csv** Data showing the correspondence to 20 area IDs for the area ID of the original data
    - **Aggregated_location** Location data for 20 area IDs
  - **results** Folder where the results are stored
    - **result_month=XX_day=XX_BOruntime=XX_RSruntime=XX_simulations=XX.csv** The data containig results for a given parameter set

## Requirement
It is desirable to have a license of the Gurobi solver.
If you do not have a license for the Gurobi solver, you can use a free CBC solver instead.
(However, the performance of baselines of CU-A, BO-A, and RS-A is slightly lower by the replacement.)

## Usage
For each experiment, we explain how to perform it.

**Crowd-sourcing experiments** 
1. Go to the Crowd-sourcing_experiment folder and run "setup.sh."
2. If you do not have a license for the Gurobi solver, replace "_gurobi" with "_cbc" inside Experiment_paper.sh and Experiment_test.sh.
3. Run "Experiment_test.sh" and see the results in the "results" folder to see if the code works.
4. Run "Experiment_paper.sh" and see the results in the "results." Note that this code takes a long time to execute and is not parallelized.
  
If you want to set parameters yourself, go to the Crowd-sourcing_experiment/bin folder and run "experiment_crowdsourcing_gurobi.py m n T BO_runtime RS_runtime num_simulations."
The first argument is $m$ (number of tasks), the second is $n$ (number of users), the third is $T$ (number of user appearance), the fourth is the execution time of Bayesian optimization (in seconds), the fifth is the execution time of random search (in seconds), and the sixth is the number of simulations.

**Ride-sharing experiments** 
1. Go to the Ride-sharing_experiment folder and run "setup.sh."
2. If you do not have a license for the Gurobi solver, replace "_gurobi" with "_cbc" inside Experiment_paper.sh and Experiment_test.sh.
3. Run "Experiment_test.sh" and see the results in the "results" folder to see if the code works.
4. Run "Experiment_paper.sh" and see the results in the "results." Note that this code takes a long time to execute and is not parallelized.

If you want to set parameters yourself, go to the Ride-sharing_experiment/bin folder and run "experiment_rideshare_gurobi.py month day BO_runtime RS_runtime num_simulations."
The first argument is the target month, the second argument is the target day, the third argument is the run time of the Bayesian optimization (in seconds), the fourth argument is the run time of the random search (in seconds), and the fifth argument is the number of simulations.

## Licence
You must follow the terms of the "SOFTWARE LICENSE AGREEMENT FOR EVALUATION."
Be sure to read it.

## Author
Yuya Hikima wrote this text.
If you have any problems, please contact yuya.hikima at gmail.com.
