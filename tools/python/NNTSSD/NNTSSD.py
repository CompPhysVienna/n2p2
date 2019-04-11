# -*- coding: utf-8 -*-
"""
09.04.2019
@author: mr

NNTSSD: Neural Network Training Set Size Dependence
"""

import os
import shutil
import numpy as np
import matplotlib.pyplot as plt

class NNTSSD():
    """
    This is the description of my class.
    """
    def __init__(self,set_size_ratios,n_sets_per_size):
        self.set_size_ratios = set_size_ratios
        self.n_sets_per_size = n_sets_per_size
    
    def create_training_datasets(self):
        """
        This function creates training datasets for
        a number of samples (user-given in n_sets per_size) for each element of
        a set of training sizes (user-given by list of ratios in set_size_ratios)
        from file input.data using the program nnp-select.
        It creates one folder for each training set size.
        Inside these folders, it creates one folder per sample, in which the output files
        (output.data and nnp-select.log, created by nnp-select) can be found.
        Additionally, the file output.data in this subfolders is renamed to input.data.
        """
        n_set_size_ratios = np.size(self.set_size_ratios)
        print "---------------------------------------------"
        print "CREATING TRAINING DATASETS"
        print "number of samples per training set size = ", self.n_sets_per_size
        print "number of different training set sizes = ", n_set_size_ratios
        print "---------------------------------------------"
        for ratios_counter in range(n_set_size_ratios):
            current_ratio = self.set_size_ratios[ratios_counter]
            print "We are working with ratio {:3.2f}".format(current_ratio)
            ratio_folder = "ratio"+str("{:3.2f}".format(current_ratio))
            os.system("mkdir "+ratio_folder)
            for sets_per_size_counter in range(self.n_sets_per_size):
                nnp_select = "../../../bin/nnp-select random "+str("{:3.2f}".format(current_ratio))+" "+str(int(np.random.randint(100,999,1)))
                print nnp_select
                os.system(nnp_select)
                os.chdir(ratio_folder)
                set_folder = "ratio"+str("{:3.2f}".format(current_ratio))+"_set"+str(sets_per_size_counter)
                os.system("mkdir "+set_folder)
                os.chdir("../")
                shutil.move("output.data",ratio_folder+"/"+set_folder+"/input.data")
                shutil.move("nnp-select.log",ratio_folder+"/"+set_folder+"/nnp-select.log")
                
    def training_neural_network(self):
        """
        This function trains the datasets that were created with the function
        create_training_datasets and stores the training related output files in the
        specific subfolders.
        """
        n_set_size_ratios = np.size(self.set_size_ratios)
        print "---------------------------------------------"
        print "TRAINING NEURAL NETWORK"
        print "number of samples per training set size = ", self.n_sets_per_size
        print "number of different training set sizes = ", n_set_size_ratios
        print "---------------------------------------------"
        for ratios_counter in range(n_set_size_ratios):
            current_ratio = self.set_size_ratios[ratios_counter]
            print "We are working with ratio {:3.2f}".format(current_ratio)
            ratio_folder = "ratio"+str("{:3.2f}".format(current_ratio))
            for sets_per_size_counter in range(self.n_sets_per_size):
                os.chdir(ratio_folder)
                set_folder = "ratio"+str("{:3.2f}".format(current_ratio))+"_set"+str(sets_per_size_counter)
                os.chdir(set_folder)
                shutil.copy("../../input.nn","input.nn")
                shutil.copy("../../scaling.data","scaling.data")
                nnp_train = "mpirun -np 4 ../../../../../bin/nnp-train"
                print nnp_train
                os.system(nnp_train)
                os.chdir("../../")

    def analyse_learning_curves(self):
        """
        This function analyses the learning curves from the training of the datasets created before.
        It needs the created directory structure, from where it reads the learning_curve.out files and
        stores the analysis relevant information (mean and standard deviation of train and test RMSE of energies and forces)
        to a file called "analysis_data.out".
        It also creates a file called "collect_data.out" in each ratio-folder, storing information about the individual
        sets of the specific sample of training set size.
        """
        n_set_size_ratios = np.size(self.set_size_ratios)
        print "---------------------------------------------"
        print "ANALYSING LEARNING CURVES"
        print "number of samples per training set size = ", self.n_sets_per_size
        print "number of different training set sizes = ", n_set_size_ratios
        print "---------------------------------------------"
        analyse_data = np.empty([0,10])
        for ratios_counter in range(n_set_size_ratios):
            collect_data = np.empty([0,7])
            current_ratio = self.set_size_ratios[ratios_counter]
            print "We are working with ratio {:3.2f}".format(current_ratio)
            ratio_folder = "ratio"+str("{:3.2f}".format(current_ratio))
            os.chdir(ratio_folder)
            for sets_per_size_counter in range(self.n_sets_per_size):
                set_folder = "ratio"+str("{:3.2f}".format(current_ratio))+"_set"+str(sets_per_size_counter)
                os.chdir(set_folder)
                collect_data = np.vstack([collect_data,np.append(np.array([current_ratio,sets_per_size_counter]),np.genfromtxt("learning-curve.out")[-1,0:5],axis=0)])
                os.chdir("../")
            np.savetxt("collect_data.out",collect_data,fmt="%.18e",header='%15s%25s%25s%25s%25s%25s%25s'%('ratio','set_number','last_epoch','RMSE_E_train','RMSE_E_test','RMSE_F_train','RMSE_F_test'))
            if self.n_sets_per_size > 1:
                mean_data = np.mean(collect_data,axis=0)[3:]
                std_data = np.std(collect_data,axis=0)[3:]
                wip_analyse_data = np.vstack([mean_data,std_data])
                wip_analyse_data = np.reshape(wip_analyse_data,8,order="F")
                analyse_data = np.vstack([analyse_data,np.append(np.take(collect_data,[0,2]),wip_analyse_data,axis=0)])
            else:
                print "There is just one set per size."
            os.chdir("../")
#        print analyse_data
        np.savetxt("analyse_data.out",analyse_data,fmt="%.18e", header='%15s%25s%25s%25s%25s%25s%25s%25s%25s%25s'%('ratio0','last_epoch','mean_RMSE_E_train','std_RMSE_E_train','mean_RMSE_E_test','std_RMSE_E_test','mean_RMSE_F_train','std_RMSE_F_train','mean_RMSE_F_test','std_RMSE_F_test'))

    def plot_size_dependence(self):
        """
        This function takes the file "analyse_data.out" as an input and plots the mean energy and force RMSE
        with its standard deviation versus the traning set size.
        """
        
        ratio_folder = "ratio"+str("{:3.2f}".format(self.set_size_ratios[0]))
        set_folder = "ratio"+str("{:3.2f}".format(self.set_size_ratios[0]))+"_set0"
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

def setup_NNTSSD():
    """
    This function acutally performs the selection, training, analysis and plotting of the neural network.
    """
    user_input = np.genfromtxt("NNTSSD_input.dat")
    user_input[1] += user_input[2]
    if not (0.0 < user_input[0] and user_input[0] < user_input[1] and user_input[1] <= 1.0):
        print "ERROR\nMake sure the input ratios are in interval (0,1] and minimum ratio < maximum ratio."
    elif not (user_input[2]*2.0 <= (user_input[1]-user_input[0])):
        print "ERROR\nMake sure you have specified at least 2 training set sizes."
    elif not (int(user_input[3]) >= 2):
        print "ERROR\nMake sure you have specified at least 2 sample sets per training size."
    elif not ((user_input[4] == 1. or user_input[4] == 0.) and (user_input[5] == 1. or user_input[5] == 0.) \
    and (user_input[6] == 1. or user_input[6] == 0.) and (user_input[7] == 1. or user_input[7] == 0.)):
        print "ERROR\nMake sure you have specified the NNTSSD steps with either 0 or 1."
    else:
        [create_logical,train_logical,analyse_logical,plot_logical] = map(bool,user_input[4:])
        set_size_ratios = np.arange(user_input[0],user_input[1],user_input[2])
        n_sets_per_size = int(user_input[3])
        print "Performing the following NNTSSD steps:"
        print create_logical, "\t Create training datasets"
        print train_logical, "\t Training neural network"
        print analyse_logical, "\t Analyse learning curves"
        print plot_logical, "\t Plot size dependence"
        myNNTSSD = NNTSSD(set_size_ratios,n_sets_per_size)
        if create_logical:
            myNNTSSD.create_training_datasets()
        if train_logical:
            myNNTSSD.training_neural_network()
        if analyse_logical:
            myNNTSSD.analyse_learning_curves()
        if plot_logical:
            myNNTSSD.plot_size_dependence()
            
setup_NNTSSD()