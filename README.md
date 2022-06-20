# IJCAI2022-Online-Matching-with-Controllable-Rewards-and-Arrival-Probabilities

## Overview
This repository contains the codes for the experiments performed in the following papers:
  
Yuya Hikima, Yasunori Akagi, Naoki Marumo, and Hideaki Kim. "Online Matching with Controllable Rewards and ArrivalProbabilities." In IJCAI, 2022 (in press).
  
Contents of this repository:
- Crowd-sourcing_experiment It contains the code used in the crowd-sourcing platform experiment and the code needed to set up the experiment, including data downloads.
- Ride-sharing_experiment It contains the code used in the ride-sharing platform experiment and the code needed to set up the experiment, including data downloads.

## Description

The following is a description of what is in each folder.
- **Crowd-sourcing_experiment** 
  - **setup.sh** Script for setting up the same experiment as in our paper
  - **Experiment_paper.sh** Scripts for performing the same experiments as in our paper
  - **Experiment_test.sh** Scripts for small experiments to see if the code works
  - **bin** Folder containing the python code needed for the crowd-sourcing experiments
    - **data_download.py** It is executed by setup.sh to download the real data.
    - **generate_reward_matrix.py** It is executed by setup.sh to make simulation data from the real data.
    - **experiment_crowdsourcing_gurobi.py** It is executed by Experiment.sh or Experiment_test.sh to perform experiments using simulated data
    - **experiment_crowdsourcing_cbc.py** Experimental code to replace the above if you do not have a grobi license
  - **data** Folder containing the downloaded real data
  - **work** Folder containing the simulation data
    - **Reward_matrix** Data containing w_et in the paper, generated by experiment_crowdsourcing_gurobi.py.
    - **trec-rf10-data.csv** Converted csv data grom the downloaded data
  - **results** Folder containing the results
    - **result_m=XX_n=XX_T=XX_BOruntime=XX_RSruntime=XX_simulations=XX.csv** The data containig results for a given parameter set

- **Ride-sharing_experiment** 
  - **setup.sh** Script for setting up the same experiment as in our paper
  - **Experiment_paper.sh** Scripts for performing the same experiments as in our paper
  - **Experiment_test.sh** Scripts for small experiments to see if the code works
  - **bin** Folder containing the python code needed for the ride-sharing experiments
    - **data_download.py** It is executed by setup.sh to download the real data.
    - **generate_reward_matrix.py** It is executed by setup.sh to make simulation data from the real data
    - **experiment_ridesharing_gurobi.py** It is executed by Experiment.sh or Experiment_test.sh to perform experiments using simulated data
    - **experiment_ridesharing_cbc.py** Experimental code to replace the above if you do not have a globi license
  - **data** Folder containing the downloaded real data
  - **results** Folder containing the results
    - **result_m=XX_n=XX_T=XX_BOruntime=XX_RSruntime=XX_simulations=XX.csv** The data containig results for a given parameter set

## Requirement
It is desirable to have a license of Gurobi solver.
If you do not have a license for Gurobi solver, you can use a free CBC solver instead of gurobi.
(However, the performance of CU-A, BO-A, and RS-A is slightly lower by the replacement.)

## Usage
For each experiment, we explain how to perform the experiment.

**Crowd-sourcing_experiment** 
1. Go to the Crowd-sourcing_experiment folder and run "bash setup.sh."
2. If you do not have a license for Gurobi solver, replace "_gurobi" with "_cbc" inside Experiment_paper.sh and Experiment_test.sh.
3. Run "Experiment_test.sh" and see the results in the "results" folder to see if the code works.
4. Run "Experiment_paper.sh" and see the results in the "results."

## Licence
You must follow the terms of the "SOFTWARE LICENSE AGREEMENT FOR EVALUATION".
Be sure to read it.

## Author
This text was written by Yuya Hikima.
If you have any problems, please contact yuya.hikima at gmail.com.
