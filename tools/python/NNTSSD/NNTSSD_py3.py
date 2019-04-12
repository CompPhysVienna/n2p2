# -*- coding: utf-8 -*-
"""
12.04.2019
@author: mr

PYTHON 3

NNTSSD: Neural Network Training Set Size Dependence
"""

import os
import sys
import shutil
import numpy as np
import matplotlib.pyplot as plt

class NNTSSD():
    """Tools for Neural Network Training Set Size Dependence.
    
    Attributes
    ----------
    set_size_ratios : numpy.ndarray
        One dimensional array; contains a list of ratios of the original training set size that are examined.
    n_sets_per_size : int
        Value, specifies how many sample sets per training size are considered.
    
    Methods
    ----------
    create_training_datasets()
        Creates training datasets from a given original dataset using the program nnp-select.
    training_neural_network()
        Trains the neural network with existing datasets using the program nnp-train.
    analyse_learning_curves():
        Prepares analyse data from the learning curves.
    plot_size_dependence():
        Plots the energy and force RMSE versus training set size.
    """
    def __init__(self,set_size_ratios,n_sets_per_size):
        """
        Parameters
        ----------
        set_size_ratios : numpy.ndarray
            One dimensional array; contains a list of ratios of the original training set size that are examined.
        n_sets_per_size : int
            Value, specifies how many sample sets per training size are considered.
        """
        self.set_size_ratios = set_size_ratios
        self.n_sets_per_size = n_sets_per_size
    
    def create_training_datasets(self,random_seed_logical):
        """Creates training datasets from a given original dataset using the program nnp-select.
        
        Requirements
        ----------
        'input.data' : file
            Contains original set of trainingdata.
        ../../bin/nnp-select : executable program
            Performs random selection of sets according to given ratio.
            
        Outputs
        ----------
        'ratio*' : folders
            Its name tells the ratio * of current from original dataset.
        'ratio*_**' : subfolders of the previous
            Its name in addition tells the sample number ** of its ratio *.
        'input.data' :  file
            Contains new training dataset of specified size ratio.
        'nnp-select.log' : file
            Log file created by running nnp-select.
            
        Parameters
        ----------
        random_seed_logical : logical
            True if random seed for nnp-select shall be fixed, False otherwise.
        """
        try:
            os.path.isfile("input.data")
        except:
            sys.exit("ERROR: The file 'input.data' does not exist!")
        try:
            os.path.isfile("../../../bin/nnp-select random")
        except:
            sys.exit("ERROR: The executable program nnp-select does not exist in ~/n2p2/bin")
        n_set_size_ratios = np.size(self.set_size_ratios)
        print("---------------------------------------------")
        print("CREATING TRAINING DATASETS")
        print("number of samples per training set size = ", self.n_sets_per_size)
        print("number of different training set sizes = ", n_set_size_ratios)
        print("---------------------------------------------")
        for ratios_counter in range(n_set_size_ratios):
            current_ratio = self.set_size_ratios[ratios_counter]
            print("We are working with ratio {:3.2f}".format(current_ratio))
            ratio_folder = "ratio"+str("{:3.2f}".format(current_ratio))
            os.system("mkdir "+ratio_folder)
            for sets_per_size_counter in range(self.n_sets_per_size):
                if random_seed_logical:
                    random_seed = "123"
                else:
                    random_seed = int(np.random.randint(100,999,1))
                nnp_select = "../../../bin/nnp-select random "+str("{:3.2f}".format(current_ratio))+" "+str(random_seed)
                print(nnp_select)
                os.system(nnp_select)
                os.chdir(ratio_folder)
                set_folder = "ratio"+str("{:3.2f}".format(current_ratio))+"_set"+str(sets_per_size_counter)
                os.system("mkdir "+set_folder)
                os.chdir("../")
                os.path.isfile("output.data")
                shutil.move("output.data",ratio_folder+"/"+set_folder+"/input.data")
                shutil.move("nnp-select.log",ratio_folder+"/"+set_folder+"/nnp-select.log")
                
    def training_neural_network(self):
        """Trains the neural network with existing datasets using the program nnp-train.
        
        Requirements
        ----------
        'ratio*/ratio*_**' : 2-layered directory structure
            Created with the method create_training_datasets().
        'ratio*/ratio*_**/input.data' : file
            Contains the training datasets.
        ../../bin/nnp-train : executable program
            Performs training of neural network.
        'ratio*/ratio*_**/input.nn' : file
            Specifies the training parameters.
        'ratio*/ratio*_**/scaling.data' : file
            Contains symmetry function scaling data.
        
        Outputs
        ----------
        'ratio*/ratio*_**/train.data' : file
            Dataset actually used for training.
        'ratio*/ratio*_**/test.data' : file
            Dataset kept for testing.
        'ratio*/ratio*_**/nnp-train.log.****' : file
            One or more log files from running nnp-train.
        'ratio*/ratio*_**/learning-curve.out' : file
            Contains learning curve data, namely RMSE of energy and forces of train and test sets for each epoch.
        """
        try:
            os.path.isfile("../../../bin/nnp-train random")
        except:
            sys.exit("ERROR: The executable program nnp-train does not exist in ~/n2p2/bin")
        n_set_size_ratios = np.size(self.set_size_ratios)
        print("---------------------------------------------")
        print("TRAINING NEURAL NETWORK")
        print("number of samples per training set size = ", self.n_sets_per_size)
        print("number of different training set sizes = ", n_set_size_ratios)
        print("---------------------------------------------")
        for ratios_counter in range(n_set_size_ratios):
            current_ratio = self.set_size_ratios[ratios_counter]
            print("We are working with ratio {:3.2f}".format(current_ratio))
            ratio_folder = "ratio"+str("{:3.2f}".format(current_ratio))
            for sets_per_size_counter in range(self.n_sets_per_size):
                try:
                    os.chdir(ratio_folder)
                except:
                    sys.exit("ERROR: The folder "+ratio_folder+"does not exist!")
                set_folder = "ratio"+str("{:3.2f}".format(current_ratio))+"_set"+str(sets_per_size_counter)
                try:
                    os.chdir(set_folder)
                except:
                    sys.exit("ERROR: The folder "+ratio_folder+"/"+set_folder+" does not exist!")
                shutil.copy("../../input.nn","input.nn")
                shutil.copy("../../scaling.data","scaling.data")
                nnp_train = "mpirun -np 4 ../../../../../bin/nnp-train"
                print(nnp_train)
                os.system(nnp_train)
                os.chdir("../../")

    def analyse_learning_curves(self):
        """Prepares analyse data from the learning curves.
        
        Requirements
        ----------
        'ratio*/ratio*_**' : 2-layered directory structure
            Created with the method create_training_datasets().
        'ratio*/ratio*_**/learning-curve.out' : file
            Contains learning curve data, created with the method create_training_datasets().
        
        Outputs
        ----------
        'ratio*/collect_data.out' : file
            Contains analysis of learning curve data of specific training size.
        'analyse_data.out' : file
            Contains processed RMSE size dependence information for all datasets.
        """
        n_set_size_ratios = np.size(self.set_size_ratios)
        print("---------------------------------------------")
        print("ANALYSING LEARNING CURVES")
        print("number of samples per training set size = ", self.n_sets_per_size)
        print("number of different training set sizes = ", n_set_size_ratios)
        print("---------------------------------------------")
        analyse_data = np.empty([0,10])
        for ratios_counter in range(n_set_size_ratios):
            collect_data = np.empty([0,7])
            current_ratio = self.set_size_ratios[ratios_counter]
            print("We are working with ratio {:3.2f}".format(current_ratio))
            ratio_folder = "ratio"+str("{:3.2f}".format(current_ratio))
            try:
                os.chdir(ratio_folder)
            except:
                sys.exit("ERROR: The folder "+ratio_folder+" does not exist!")
            for sets_per_size_counter in range(self.n_sets_per_size):
                set_folder = "ratio"+str("{:3.2f}".format(current_ratio))+"_set"+str(sets_per_size_counter)
                try:
                    os.chdir(set_folder)
                except:
                    sys.exit("ERROR: The folder "+ratio_folder+"/"+set_folder+" does not exist!")
                try:
                    collect_data = np.vstack([collect_data,np.append(np.array([current_ratio,sets_per_size_counter]),np.genfromtxt("learning-curve.out")[-1,0:5],axis=0)])
                except:
                    sys.exit("ERROR: The file 'learing-curve.out' does not exist in "+ratio_folder+"/"+set_folder+"!")
                os.chdir("../")
            np.savetxt("collect_data.out",collect_data,fmt="%.18e",header='%15s%25s%25s%25s%25s%25s%25s'%('ratio','set_number','last_epoch','RMSE_E_train','RMSE_E_test','RMSE_F_train','RMSE_F_test'))
            mean_data = np.mean(collect_data,axis=0)[3:]
            std_data = np.std(collect_data,axis=0)[3:]
            wip_analyse_data = np.vstack([mean_data,std_data])
            wip_analyse_data = np.reshape(wip_analyse_data,8,order="F")
            analyse_data = np.vstack([analyse_data,np.append(np.take(collect_data,[0,2]),wip_analyse_data,axis=0)])
            os.chdir("../")
        np.savetxt("analyse_data.out",analyse_data,fmt="%.18e", header='%15s%25s%25s%25s%25s%25s%25s%25s%25s%25s'%('ratio0','last_epoch','mean_RMSE_E_train','std_RMSE_E_train','mean_RMSE_E_test','std_RMSE_E_test','mean_RMSE_F_train','std_RMSE_F_train','mean_RMSE_F_test','std_RMSE_F_test'))

    def plot_size_dependence(self):
        """Plots the energy and force RMSE versus training set size.
        
        Requirements
        ----------
        'analyse_data.out' : file
            Created with analyse_learning_curves().
        'ratio*/ratio*_**/nnp-select.log' : file
            Created with create_training_datasets().

        Outputs
        ----------
        'Energy_RMSE.png' : png picture
            Shows train and test energy RMSE (and its standard deviation) versus training set size.
        'Forces_RMSE.png' : png picture
            Shows train and test forces RMSE (and its standard deviation) versus training set size.
        """
        try:
            os.path.isfile("analyse_data.out")
        except:
            sys.exit("ERROR: The file 'analyse_data.out' does not exist!")
        ratio_folder = "ratio"+str("{:3.2f}".format(self.set_size_ratios[0]))
        set_folder = "ratio"+str("{:3.2f}".format(self.set_size_ratios[0]))+"_set0"
        try:
            os.path.isfile(ratio_folder+"/"+set_folder+"/nnp-select.log")
        except:
            sys.exit("ERROR: The file 'nnp-select.log' does not exist in the desired path!")
        os.chdir(ratio_folder+"/"+set_folder)
        nnp_select_log = open("nnp-select.log")
        for line in nnp_select_log:
            if line.startswith("Total"):
                n_total_structures = int(''.join(filter(str.isdigit, str(line))))
        os.chdir("../../")
        
        analyse_data = np.genfromtxt("analyse_data.out")
        training_set_sizes = analyse_data[:,0]*n_total_structures
        last_epoch = str(int(analyse_data[0,1]))
        
        plt.figure("Energy RMSE")
        plt.errorbar(training_set_sizes,analyse_data[:,2],analyse_data[:,3],label="mean_RMSE_E_train")
        plt.errorbar(training_set_sizes,analyse_data[:,4],analyse_data[:,5],label="mean_RMSE_E_test")
        plt.title("Energy RMSE in "+last_epoch+" epochs")
        plt.xlabel("traning set size")
        plt.ylabel(r"Energy RMSE (meV/atom)")
        plt.legend()
        plt.savefig("Energy_RMSE.png")
        
        plt.figure("Forces RMSE")
        plt.errorbar(training_set_sizes,analyse_data[:,6],analyse_data[:,7],label="mean_RMSE_F_train")
        plt.errorbar(training_set_sizes,analyse_data[:,8],analyse_data[:,9],label="mean_RMSE_F_test")
        plt.title("Forces RMSE in "+last_epoch+" epochs")
        plt.xlabel("traning set size")
        plt.ylabel(r"Forces RMSE (meV/$\AA$)")
        plt.legend()
        plt.savefig("Forces_RMSE.png")

def perform_NNTSSD():
    """This function performs NNTSSD methods according to user-given specifications.
    
    Firstly, checks whether the input file contains valid parameter values. 
    Secondly, it performs a user-given selection of the NNTSSD methods
    create_training_datasets(), training_neural_network(), analyse_learning_curves() and plot_size_dependence().
    
    Requirements
    ----------
    'NNTSSD_input.dat' : file
        Contains user-given specifications on training set size parameters and NNTSSD steps.
    """
    try:
        os.path.isfile("NNTSSD_input.dat")
    except:
        sys.exit("ERROR: The file 'NNTSSD_input.dat' does not exist!")
    user_input = np.genfromtxt("NNTSSD_input.dat")
    user_input[1] += user_input[2]
    if not (0.0 < user_input[0] and user_input[0] < user_input[1] and user_input[1] <= 1.0):
        sys.exit("ERROR: Make sure the input ratios are in interval (0,1] and minimum ratio < maximum ratio.")
    elif not (user_input[2]*2.0 <= (user_input[1]-user_input[0])):
        sys.exit("ERROR: Make sure you have specified at least 2 training set sizes.")
    elif not (int(user_input[3]) >= 2):
        sys.exit("ERROR: Make sure you have specified at least 2 sample sets per training size.")
    elif not ((user_input[4] == 1. or user_input[4] == 0.) and (user_input[5] == 1. or user_input[5] == 0.) \
    and (user_input[6] == 1. or user_input[6] == 0.) and (user_input[7] == 1. or user_input[7] == 0.)):
        sys.exit("ERROR: Make sure you have specified the NNTSSD steps with either 0 or 1.")
    elif not (user_input[8] == 1. or user_input[8] == 0.):
        sys.exit("ERROR: Make sure you have set the random seed specification to either 0 or 1.")
    else:
        random_seed_logical = bool(user_input[8])
        [create_logical,train_logical,analyse_logical,plot_logical] = map(bool,user_input[4:8])
        set_size_ratios = np.arange(user_input[0],user_input[1],user_input[2])
        n_sets_per_size = int(user_input[3])
        print("Performing the following NNTSSD steps:")
        print(create_logical, "\t Create training datasets")
        print(train_logical, "\t Training neural network")
        print(analyse_logical, "\t Analyse learning curves")
        print(plot_logical, "\t Plot size dependence")
        myNNTSSD = NNTSSD(set_size_ratios,n_sets_per_size)
        if create_logical:
            myNNTSSD.create_training_datasets(random_seed_logical)
        if train_logical:
            myNNTSSD.training_neural_network()
        if analyse_logical:
            myNNTSSD.analyse_learning_curves()
        if plot_logical:
            myNNTSSD.plot_size_dependence()
            
perform_NNTSSD()