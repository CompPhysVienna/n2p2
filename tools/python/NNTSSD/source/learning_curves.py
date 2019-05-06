#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
06.05.2019
@author: mr

PYTHON 3

Tools for analysing learning curves and plotting training performance.
"""

import os
import sys
import shutil
import numpy as np
import matplotlib.pyplot as plt

class Tools_LC():
    """Tools for analysing and plotting Learning Curves/Training Performance.
    
    Methods
    ----------
    analyse_learning_curves():
        Prepares analyse data from the learning curves obtained in training.
    plot_training_performance():
        Plots the energy and force RMSE versus number of training epoch.
    """
    def __init__(self):
        """In the moment, the class Tools_LC() does not have any attributes.
        
        Still, it makes sense to define a class since it condenses functionalities and
        for future purposes, it is likely that attributes will be added.
        """
    
    def analyse_learning_curves(self):
        """Prepares analyse data from the learning curves obtained in training.
        
        Requirements
        ----------
        'Output/ratio*/ratio*_**' : 2-layered directory structure
            Created with the method NNTSSD.Tools.create_training_datasets().
        'Output/ratio*/ratio*_**/learning-curve.out' : file
            Contains learning curve data, created with the method NNTSSD.Tools.create_training_datasets().
        
        Outputs
        ----------
        'Output/training_performance/learning_curve_E.out' : file
            Contains processed mean energy training performance information for all dataset sizes.
        'Output/training_performance/learning_curve_F.out' : file
            Contains processed mean energy training performance information for all dataset sizes.
        'Output/training_performance/learning_curve_Etest_*.out' : files
            Contains processed energy test training performance information for all dataset samples of size *.
        'Output/training_performance/learning_curve_Etrain_*.out' : files
            Contains processed energy train training performance information for all dataset samples of size *.
        'Output/training_performance/learning_curve_Ftest_*.out' : files
            Contains processed force test training performance information for all dataset samples of size *.
        'Output/training_performance/learning_curve_Ftrain_*.out' : files
            Contains processed force train training performance information for all dataset samples of size *.
        """
        print("\n***ANALYSING LEARNING CURVES***************************************************")
        print("...",end="\r")
        if not os.path.isdir("Output"):
            sys.exit("ERROR: The folder 'Output' does not exist!")
        else:
            os.chdir("Output")
        # Finding out number of epochs.
        ratio_dir_string = np.sort(os.listdir())
        number_of_epochs = 0
        n_total_configurations = 0
        for ratio_dir_counter in range(len(ratio_dir_string)):
            if ratio_dir_string[ratio_dir_counter].startswith('ratio'):
                os.chdir(ratio_dir_string[ratio_dir_counter])
                set_dir_string = np.sort(os.listdir())                    
                for set_dir_counter in range(len(set_dir_string)):
                    if set_dir_string[set_dir_counter].startswith('ratio'):
                        os.chdir(set_dir_string[set_dir_counter])
                        try:
                            learning_curve_data = np.genfromtxt("learning-curve.out")
                            number_of_epochs = int(learning_curve_data[-1,0])
                            nnp_select_log = open("nnp-select.log")
                            for line in nnp_select_log:
                                if line.startswith("Total"):
                                    n_total_configurations = int(''.join(filter(str.isdigit, str(line))))
#                            os.chdir("../")
                        except:
                            continue
                        os.chdir("../")
                        if not (number_of_epochs == 0 and n_total_configurations == 0):
                            break
                os.chdir("../")
                if not (number_of_epochs == 0 and n_total_configurations == 0):
                    break
        # COLLECT TRAINING PERFORMANCE DATA.
        print("   Collecting training performance data.")
        print("...",end="\r")
        try:
            shutil.rmtree("training_performance")
            print("INFO: Removed old 'training_performance' folder.")
        except:
            pass
        os.system("mkdir training_performance")
        set_size_ratios = np.array([])
        learning_curve_E = np.zeros((number_of_epochs+1))
        learning_curve_F = np.zeros((number_of_epochs+1))
        ratio_dir_string = np.sort(os.listdir())
        for ratio_dir_counter in range(len(ratio_dir_string)):
            if ratio_dir_string[ratio_dir_counter].startswith('ratio'):
                current_ratio = 0.01*int(''.join(filter(str.isdigit, ratio_dir_string[ratio_dir_counter])))
                set_size_ratios = np.append(set_size_ratios,current_ratio)
                learning_curve_Etrain = np.zeros((number_of_epochs+1))
                learning_curve_Etest = np.zeros((number_of_epochs+1))
                learning_curve_Ftrain = np.zeros((number_of_epochs+1))
                learning_curve_Ftest = np.zeros((number_of_epochs+1))
                os.chdir(ratio_dir_string[ratio_dir_counter])
                set_dir_string = np.sort(os.listdir())
                for set_dir_counter in range(len(set_dir_string)):
                    if set_dir_string[set_dir_counter].startswith('ratio'):
                        os.chdir(set_dir_string[set_dir_counter])
                        try:
                            learning_curve_data = np.genfromtxt("learning-curve.out")
                            learning_curve_Etrain = np.column_stack((learning_curve_Etrain,learning_curve_data[:,1]))
                            learning_curve_Ftrain = np.column_stack((learning_curve_Ftrain,learning_curve_data[:,3]))
                            learning_curve_Etest = np.column_stack((learning_curve_Etest,learning_curve_data[:,2]))
                            learning_curve_Ftest = np.column_stack((learning_curve_Ftest,learning_curve_data[:,4]))
                        except:
                            print("INFO: The file 'learning-curve.out' does not exist in "+ratio_dir_string[ratio_dir_counter]+"/"+set_dir_string[set_dir_counter]+"!")
#                            continue
                        os.chdir("../")
                learning_curve_Etrain = learning_curve_Etrain[:,1:]
                learning_curve_Etest = learning_curve_Etest[:,1:]
                learning_curve_Ftrain = learning_curve_Ftrain[:,1:]
                learning_curve_Ftest = learning_curve_Ftest[:,1:]
                
                np.savetxt("../training_performance/learning_curve_Etrain_"+str("{:3.2f}".format(current_ratio))+".out",learning_curve_Etrain)
                np.savetxt("../training_performance/learning_curve_Etest_"+str("{:3.2f}".format(current_ratio))+".out",learning_curve_Etest)
                np.savetxt("../training_performance/learning_curve_Ftrain_"+str("{:3.2f}".format(current_ratio))+".out",learning_curve_Ftrain)
                np.savetxt("../training_performance/learning_curve_Ftest_"+str("{:3.2f}".format(current_ratio))+".out",learning_curve_Ftest)
                
                mean_Etrain_data = np.mean(learning_curve_Etrain,axis=1)
                mean_Etest_data = np.mean(learning_curve_Etest,axis=1)
                std_Etrain_data = np.std(learning_curve_Etrain,axis=1)
                std_Etest_data = np.std(learning_curve_Etest,axis=1)
                mean_Ftrain_data = np.mean(learning_curve_Ftrain,axis=1)
                mean_Ftest_data = np.mean(learning_curve_Ftest,axis=1)
                std_Ftrain_data = np.std(learning_curve_Ftrain,axis=1)
                std_Ftest_data = np.std(learning_curve_Ftest,axis=1)
                ratio_learning_curve_data = np.full_like(mean_Etrain_data,current_ratio)
                learning_curve_E = np.column_stack((learning_curve_E,np.column_stack((np.column_stack((np.column_stack((ratio_learning_curve_data,np.column_stack((mean_Etrain_data,std_Etrain_data)))),mean_Etest_data)),std_Etest_data))))
                learning_curve_F = np.column_stack((learning_curve_F,np.column_stack((np.column_stack((np.column_stack((ratio_learning_curve_data,np.column_stack((mean_Ftrain_data,std_Ftrain_data)))),mean_Ftest_data)),std_Ftest_data))))
                
                os.chdir("../")

        os.chdir("training_performance")
        np.savetxt("learning_curve_E.out",learning_curve_E[:,1:],fmt="%.18e",header='%15s%25s%25s%25s%25s'%('ratio','RMSE_Etrain','std_RMSE_Etrain','mean_RMSE_Etest','std_RMSE_Etest'))
        np.savetxt("learning_curve_F.out",learning_curve_F[:,1:],fmt="%.18e",header='%15s%25s%25s%25s%25s'%('ratio','RMSE_Ftrain','std_RMSE_Ftrain','mean_RMSE_Ftest','std_RMSE_Ftest'))
        os.chdir("../../")
        print("FINISHED analysing learning curves.")
        return number_of_epochs, n_total_configurations, set_size_ratios

    def plot_training_performance(self,number_of_epochs, n_total_configurations, set_size_ratios):
        """Plots the energy and force RMSE versus number of epochs.
        
        Firstly, this method plots the mean energy and force RMSE training performance.
        Secondly, it plots the energy and force RMSE for all sample sets of each training size.
        
        Requirements
        ----------
        'Output/training_performance/learning_curve_E.out' : file
            Created with analyse_learning_curves().
        'Output/training_performance/learning_curve_F.out' : file
            Created with analyse_learning_curves().
        'Output/training_performance/learning_curve_Etest_*.out' : files
            Created with analyse_learning_curves().
        'Output/training_performance/learning_curve_Etrain_*.out' : files
            Created with analyse_learning_curves().
        'Output/training_performance/learning_curve_Ftest_*.out' : files
            Created with analyse_learning_curves().
        'Output/training_performance/learning_curve_Ftrain_*.out' : files
            Created with analyse_learning_curves().
        
        Outputs
        ----------
        'Output/training_performance/Learning_curve_Energies_mean.png' : png picture
            Shows mean train and test energy RMSE (and its standard deviation) versus epoch number.
        'Output/training_performance/Learning_curve_Forces_mean.png' : png picture
            Shows mean train and test forces RMSE (and its standard deviation) versus epoch number.
        'Output/training_performance/Learning_curve_Energies_all.png' : png picture
            Shows train and test energy RMSE for all sample sets versus epoch number.
        'Output/training_performance/Learning_curve_Forces_all.png' : png picture
            Shows train and test forces RMSE for all sample sets versus epoch number.
        """
        print("\n***PLOTTING TRAINING PERFORMANCE***********************************************")
        print("...",end="\r")
        if not os.path.isdir("Output/training_performance"):
            sys.exit("ERROR: The folder 'Output/training_performance' does not exist!")
        else:
            os.chdir("Output/training_performance")
        if not os.path.isfile("learning_curve_E.out"):
            sys.exit("ERROR: The file 'learning_curve_E.out' does not exist!")
        if not os.path.isfile("learning_curve_F.out"):
            sys.exit("ERROR: The file 'learning_curve_F.out' does not exist!")
        
        print("   Plotting training performance.")
        print("...",end="\r")
        parameters = ["Energies","Forces"]
        parameter_shortcuts = ["E","F"]
        for parameter_counter in range(len(parameters)):
            current_parameter = parameters[parameter_counter]
            current_shortcut = parameter_shortcuts[parameter_counter]
            learning_curve = np.genfromtxt("learning_curve_"+current_shortcut+".out")
            training_set_ratios = learning_curve[0,::5]
            epochs = np.arange(len(learning_curve[:,0]))
#            plot mean
            plt.figure("Learning_curve_"+current_parameter)
            for ratios_counter in range(len(training_set_ratios)):
                current_training_set_size = training_set_ratios[ratios_counter]*n_total_configurations
                plt.errorbar(epochs,learning_curve[:,1+5*ratios_counter],learning_curve[:,2+5*ratios_counter],errorevery=(17+ratios_counter),linewidth=1,capsize=2,ls="-",color=plt.cm.Pastel2(ratios_counter),label=current_shortcut+"train_setsize="+str(int(current_training_set_size)))[-1][0].set_linestyle('--')
                plt.errorbar(epochs,learning_curve[:,3+5*ratios_counter],learning_curve[:,4+5*ratios_counter],errorevery=(18+ratios_counter),linewidth=1,capsize=2,ls="-",color=plt.cm.Dark2(ratios_counter), label=current_shortcut+"test_setsize="+str(int(current_training_set_size)))
                plt.yscale('log')
            plt.title("Training performance for "+current_parameter+" (mean)")
            plt.xlabel("epoch")
            if current_parameter == "Energies":
                plt.ylabel(r"Energy RMSE (meV/atom)")
            elif current_parameter == "Forces":
                plt.ylabel(r"Forces RMSE (meV/$\AA$)")
            plt.grid(b=True, which='major', color='#999999', linestyle=':')
            plt.legend(loc=1, borderaxespad=0.1)
            plt.savefig("Learning_curve_"+current_parameter+"_mean.png",dpi=300)
#            plot all
            plt.figure("Learning_curve_"+current_shortcut+"_all")
            for ratios_counter in range(len(training_set_ratios)):
                current_ratio = training_set_ratios[ratios_counter]
                learning_curve_train = np.genfromtxt("learning_curve_"+current_shortcut+"train_"+str("{:3.2f}".format(current_ratio))+".out")
                learning_curve_test = np.genfromtxt("learning_curve_"+current_shortcut+"test_"+str("{:3.2f}".format(current_ratio))+".out")
                os.remove("learning_curve_"+current_shortcut+"train_"+str("{:3.2f}".format(current_ratio))+".out")
                os.remove("learning_curve_"+current_shortcut+"test_"+str("{:3.2f}".format(current_ratio))+".out")
                for samples_counter in range(len(learning_curve_train[0,:])):
                    plt.plot(epochs,learning_curve_train[:,samples_counter],linewidth=1,ls="-",color=plt.cm.Pastel2(ratios_counter),label=current_shortcut+"train_setsize="+str(int(current_ratio*n_total_configurations)))
                    plt.plot(epochs,learning_curve_test[:,samples_counter],linewidth=1,ls="-",color=plt.cm.Dark2(ratios_counter),label=current_shortcut+"test_setsize="+str(int(current_ratio*n_total_configurations)))
                plt.yscale('log')
            plt.title("Training performance for "+current_parameter)
            plt.xlabel("epoch")
            if current_parameter == "Energies":
                plt.ylabel(r"Energy RMSE (meV/atom)")
            elif current_parameter == "Forces":
                plt.ylabel(r"Forces RMSE (meV/$\AA$)")
            plt.grid(b=True, which='major', color='#999999', linestyle=':')
            plt.legend(loc=1, borderaxespad=0.1)
            plt.savefig("Learning_curve_"+current_parameter+"_all.png",dpi=300)
            
        os.chdir("../../")
        print("FINISHED plotting training performance.")
        return None


def perform_Learning_curves():
    """This function performs methods of the class Tools_LC().
    
    Firstly, it performs the method Tools_LC.analyse_learning_curves().
    Secondly, it performs the method Tools_LC.plot_training_performance().
    """
    print("*******************************************************************************")
    print("Analysing learning curves and plotting training performance")
    print("*******************************************************************************")
    
    myClass = Tools_LC()
    
    number_of_epochs, n_total_configurations, set_size_ratios = myClass.analyse_learning_curves()
    myClass.plot_training_performance(number_of_epochs, n_total_configurations, set_size_ratios)
    return None

def main():
    pass
if __name__=="__main__":
    perform_Learning_curves()
