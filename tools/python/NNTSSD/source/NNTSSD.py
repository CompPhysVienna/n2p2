#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
17.04.2019
@author: mr

PYTHON 3

NNTSSD: Neural Network Training Set Size Dependence
"""

import os
import sys
import shutil
import numpy as np
import matplotlib.pyplot as plt
import file_input
import interactive_input

class Tools():
    """Tools for Neural Network Training Set Size Dependence.
    
    Methods
    ----------
    create_training_datasets()
        Creates training datasets of different size from a given original dataset using the tool nnp-select.
    training_neural_network()
        Trains the neural network with different existing datasets using the program nnp-train.
    analyse_learning_curves():
        Prepares analyse data from the learning curves obtained in training.
    plot_size_dependence():
        Plots the energy and force RMSE versus training set size.
    """
    def __init__(self):
        """In the moment, the class Tools() does not have any attributes.
        
        Still, it makes sense to define a class since it condenses NNTSSD functionalities and
        for future purposes, it is likely that attributes will be added.
        """
    
    def create_training_datasets(self,set_size_ratios,n_sets_per_size,fix_random_seed_create,random_seed_create=123):
        """Creates training datasets of different size from a given original dataset using the tool nnp-select.
        
        Parameters
        ----------
        set_size_ratios : numpy.ndarray
            One dimensional array; contains a list of ratios of the original training set size that are examined.
        n_sets_per_size : int
            Value, specifies how many sample sets per training size are considered.
        fix_random_seed_create : logical
            True if random seed for nnp-select shall be fixed, False otherwise.
        random_seed_create : integer, optional
            User-given fixed random number generator seed. Default is 123.
        
        Requirements
        ----------
        'input.data' : file
            Contains original set of trainingdata.
        ../../../../bin/nnp-select : executable program
            Performs random selection of sets according to given ratio.
            
        Outputs
        ----------
        'Output' : folder
            It contains all of the following outputs.
        'Output/ratio*' : folders
            Its name tells the ratio * of current from original dataset.
        'Output/ratio*/ratio*_**' : subfolders of the previous
            Its name in addition tells the sample number ** of its ratio *.
        'Output/ratio*/ratio*_**/input.data' :  file
            Contains new training dataset of specified size ratio.
        'Output/ratio*/ratio*_**/nnp-select.log' : file
            Log file created by running nnp-select.
        """
        try:
            os.path.isfile("input.data")
        except:
            sys.exit("ERROR: The file 'input.data' does not exist!")
        try:
            os.path.isfile("../../../../bin/nnp-select")
        except:
            sys.exit("ERROR: The executable program nnp-select does not exist in ~/n2p2/bin")
        n_set_size_ratios = np.size(set_size_ratios)
        print("\n***CREATING TRAINING DATASETS***************************************************")
        print("number of samples per training set size = ", n_sets_per_size)
        print("number of different training set sizes = ", n_set_size_ratios)
        print("...",end="\r")
        for ratios_counter in range(n_set_size_ratios):
            current_ratio = set_size_ratios[ratios_counter]
            print("We are working with ratio {:3.2f}".format(current_ratio))
            print("...",end="\r")
            ratio_folder = "ratio"+str("{:3.2f}".format(current_ratio))
            os.system("mkdir "+ratio_folder)
            for sets_per_size_counter in range(1,n_sets_per_size+1):
                if fix_random_seed_create:
                    random_seed = random_seed_create
                else:
                    random_seed = int(np.random.randint(100,999,1))
                nnp_select = "../../../../bin/nnp-select random "+str("{:3.2f}".format(current_ratio))+" "+str(random_seed)
                print(nnp_select)
                print("...",end="\r")
                os.system(nnp_select)
                os.chdir(ratio_folder)
                set_folder = "ratio"+str("{:3.2f}".format(current_ratio))+"_set"+str(sets_per_size_counter)
                os.system("mkdir "+set_folder)
                os.chdir("../")
                
                os.path.isfile("output.data")
                shutil.move("output.data",ratio_folder+"/"+set_folder+"/input.data")
                shutil.move("nnp-select.log",ratio_folder+"/"+set_folder+"/nnp-select.log")
        try:
            shutil.rmtree("Output")
            print("INFO: Removed old 'Output' folder.")
        except:
            pass
        os.system("mkdir "+"Output")
        ratio_dir_string = np.sort(os.listdir())
        for ratio_dir_counter in range(len(ratio_dir_string)):
            if ratio_dir_string[ratio_dir_counter].startswith('ratio'):
                shutil.move(ratio_dir_string[ratio_dir_counter],"Output")
        print("FINISHED creating datasets.")
        return None
                
    def training_neural_network(self,nnp_train,write_submission_script_logical,fix_random_seed_train,random_seed_train=123,maximum_time=None):
        """Trains the neural network with different existing datasets using the program nnp-train.
        
        Parameters
        ----------
        nnp_train : string
            Command for executing n2p2's nnp-train.
        write_submission_script_logical : logical
            If True, a job submission script for VSC is written in each of the folders 'Output/ratio*/ratio*_**/'.
            If False, the command nnp_train is executed right away.
        fix_random_seed_train : logical
            True if random seed for nnp-train shall be fixed, False otherwise.
        random_seed_train : integer, optional
            User given fixed random number generator seed. Default is 123.
        maximum_time: string, optional
            User given maximum time required for executing the VSC job. Default is None.
        
        Requirements
        ----------
        'Output/ratio*/ratio*_**' : 3-layered directory structure
            Created with the method create_training_datasets().
        'Output/ratio*/ratio*_**/input.data' : file
            Contains the training datasets.
        ../../../../bin/nnp-train : executable program
            Performs training of neural network.
        'Output/ratio*/ratio*_**/input.nn' : file
            Specifies the training parameters.
        'Output/ratio*/ratio*_**/scaling.data' : file
            Contains symmetry function scaling data.
        
        Outputs
        ----------
        'Output/ratio*/ratio*_**/train.data' : file
            Dataset actually used for training.
        'Output/ratio*/ratio*_**/test.data' : file
            Dataset kept for testing.
        'Output/ratio*/ratio*_**/nnp-train.log.****' : file
            One or more log files from running nnp-train.
        'Output/ratio*/ratio*_**/learning-curve.out' : file
            Contains learning curve data, namely RMSE of energy and forces of train and test sets for each epoch.
        """
        print("\n***TRAINING NEURAL NETWORK*****************************************************")
        print("...",end="\r")
        try:
            os.chdir("Output")
        except:
            sys.exit("ERROR: The folder 'Output' does not exist!")
        ratio_dir_string = np.sort(os.listdir())
        ratio_counter = 0
        for ratio_dir_counter in range(len(ratio_dir_string)):
            if ratio_dir_string[ratio_dir_counter].startswith('ratio'):
                ratio_counter += 1
                current_ratio = 0.01*int(''.join(filter(str.isdigit, ratio_dir_string[ratio_dir_counter])))
                print("   We are working with ratio {:3.2f}".format(current_ratio))
                print("   ...",end="\r")
                os.chdir(ratio_dir_string[ratio_dir_counter])
                set_dir_string = np.sort(os.listdir())
                for set_dir_counter in range(len(set_dir_string)):
                    if set_dir_string[set_dir_counter].startswith('ratio'):
                        os.chdir(set_dir_string[set_dir_counter])
                        shutil.copy("../../../input.nn","input.nn")
                        if fix_random_seed_train:
                            random_seed = random_seed_train
                        else:
                            random_seed = int(np.random.randint(100,999,1))
                        os.system("sed -i \"s/^{0:s} .*$/{0:s} {1:d}/g\" input.nn".format("random_seed", random_seed))                     
                        shutil.copy("../../../scaling.data","scaling.data")
                        if write_submission_script_logical:
                            write_submission_script(command=nnp_train,job_name=set_dir_string[set_dir_counter],time=maximum_time)
                            os.system("sbatch submit.slrm") #to submit the job to VSC
                        else:
                            print("   ",nnp_train)
                            os.system(nnp_train)
                        os.chdir("../")
                os.chdir("../")
        os.chdir("../")
        print("FINISHED training with ",ratio_counter," different ratios.")
        return None

    def analyse_learning_curves(self):
        """Prepares analyse data from the learning curves obtained in training.
        
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
        print("\n***ANALYSING LEARNING CURVES***************************************************")
        print("...",end="\r")
        if not os.path.isdir("Output"):
            sys.exit("ERROR: The folder 'Output' does not exist!")
        else:
            os.chdir("Output")
#       ANALYSE SIZE DEPENDENCE
        epoch_min_arg = ["energy","force"]
        test_row_indices = [2,4]
        for epoch_min_arg_counter in range(len(epoch_min_arg)):
            current_epoch_min_arg = epoch_min_arg[epoch_min_arg_counter]
            current_test_row_index = test_row_indices[epoch_min_arg_counter]
            print("   Analysing data at epoch of minimum "+current_epoch_min_arg)
            print("...",end="\r")
            analyse_data = np.empty([0,10])
            ratio_dir_string = np.sort(os.listdir())
            for ratio_dir_counter in range(len(ratio_dir_string)):
                if ratio_dir_string[ratio_dir_counter].startswith('ratio'):
                    current_ratio = 0.01*int(''.join(filter(str.isdigit, ratio_dir_string[ratio_dir_counter])))
                    print("We are working with ratio {:3.2f}".format(current_ratio))
                    print("...",end="\r")
                    collect_data = np.empty([0,7])
                    os.chdir(ratio_dir_string[ratio_dir_counter])
                    set_dir_string = np.sort(os.listdir())
                    sample_set_counter = 0
                    for set_dir_counter in range(len(set_dir_string)):
                        if set_dir_string[set_dir_counter].startswith('ratio'):
                            sample_set_counter += 1
                            os.chdir(set_dir_string[set_dir_counter])
                            try:
                                learning_curve_data = np.genfromtxt("learning-curve.out")
                                number_of_epochs = int(learning_curve_data[-1,0])
                                index_epoch = 1+np.argmin(learning_curve_data[1:,current_test_row_index])
                                collect_data = np.vstack([collect_data,np.append(np.array([current_ratio,sample_set_counter]),learning_curve_data[index_epoch,0:5],axis=0)])
                            except:
                                print("INFO: The file 'learning-curve.out' does not exist in "+ratio_dir_string[ratio_dir_counter]+"/"+set_dir_string[set_dir_counter]+"!")
#                                continue
                            os.chdir("../")
                    np.savetxt("collect_data_min_"+current_epoch_min_arg+".out",collect_data,fmt="%.18e",header='%15s%25s%25s%25s%25s%25s%25s'%('ratio','set_number','epoch','RMSE_E_train','RMSE_E_test','RMSE_F_train','RMSE_F_test'))
                    
                    mean_data = np.mean(collect_data,axis=0)[3:]
                    std_data = np.std(collect_data,axis=0)[3:]
                    wip_analyse_data = np.vstack([mean_data,std_data])
                    wip_analyse_data = np.reshape(wip_analyse_data,8,order="F")
                    analyse_data = np.vstack([analyse_data,np.append(np.take(collect_data,[0,2]),wip_analyse_data,axis=0)])
                    os.chdir("../")
            np.savetxt("analyse_data_min_"+current_epoch_min_arg+".out",analyse_data,fmt="%.18e", header='%15s%25s%25s%25s%25s%25s%25s%25s%25s%25s'%('ratio','last_epoch','mean_RMSE_E_train','std_RMSE_E_train','mean_RMSE_E_test','std_RMSE_E_test','mean_RMSE_F_train','std_RMSE_F_train','mean_RMSE_F_test','std_RMSE_F_test'))
#       COLLECT TRAINING PERFORMANCE DATA
        print("   Collecting training performance data.")
        print("...",end="\r")
        set_size_ratios = np.array([])
        learning_curve_E = np.zeros((number_of_epochs+1))
        learning_curve_F = np.zeros((number_of_epochs+1))
        ratio_dir_string = np.sort(os.listdir())
        for ratio_dir_counter in range(len(ratio_dir_string)):
            if ratio_dir_string[ratio_dir_counter].startswith('ratio'):
                current_ratio = 0.01*int(''.join(filter(str.isdigit, ratio_dir_string[ratio_dir_counter])))
                set_size_ratios = np.append(set_size_ratios,current_ratio)
#                print("We are working with ratio {:3.2f}".format(current_ratio))
#                print("...",end="\r")
                collect_data = np.empty([0,7])
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
                            print("INFO: The file 'learing-curve.out' does not exist in "+ratio_dir_string[ratio_dir_counter]+"/"+set_dir_string[set_dir_counter]+"!")
#                            continue
                        os.chdir("../")
                learning_curve_Etrain = learning_curve_Etrain[:,1:]
                learning_curve_Etest = learning_curve_Etest[:,1:]
                learning_curve_Ftrain = learning_curve_Ftrain[:,1:]
                learning_curve_Ftest = learning_curve_Ftest[:,1:]
                
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
        np.savetxt("learning_curve_E.out",learning_curve_E[:,1:],fmt="%.18e",header='%15s%25s%25s%25s%25s'%('ratio','RMSE_Etrain','std_RMSE_Etrain','mean_RMSE_Etest','std_RMSE_Etest'))
        np.savetxt("learning_curve_F.out",learning_curve_F[:,1:],fmt="%.18e",header='%15s%25s%25s%25s%25s'%('ratio','RMSE_Ftrain','std_RMSE_Ftrain','mean_RMSE_Ftest','std_RMSE_Ftest'))
        os.chdir("../")
        print("FINISHED analysing learning curves.")
        return None

    def plot_size_dependence(self):
        """Plots the energy and force RMSE versus training set size.
        
        Firstly, this method carries out the original training set size from the file nnp-select.log
        that has been created with create_training_datasets().
        Secondly, it plots the energy and force RMSE including their standard deviation versus the
        selected training set sizes.
        
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
        print("\n***PLOTTING SIZE DEPENDENCE****************************************************")
        print("...",end="\r")
        if not os.path.isdir("Output"):
            sys.exit("ERROR: The folder 'Output' does not exist!")
        else:
            os.chdir("Output")
        if not os.path.isfile("analyse_data_min_energy.out"):
            sys.exit("ERROR: The file 'analyse_data_min_energy.out' does not exist!")
        if not os.path.isfile("analyse_data_min_force.out"):
            sys.exit("ERROR: The file 'analyse_data_min_force.out' does not exist!")
        ratio_dir_string = np.sort(os.listdir())
        n_total_configurations = 0
        for ratio_dir_counter in range(len(ratio_dir_string)):
            if ratio_dir_string[ratio_dir_counter].startswith('ratio'):
                os.chdir(ratio_dir_string[ratio_dir_counter])
                set_dir_string = np.sort(os.listdir())                    
                for set_dir_counter in range(len(set_dir_string)):
                    if set_dir_string[set_dir_counter].startswith('ratio'):
                        os.chdir(set_dir_string[set_dir_counter])
                        try:
                            nnp_select_log = open("nnp-select.log")
                            for line in nnp_select_log:
                                if line.startswith("Total"):
                                    n_total_configurations = int(''.join(filter(str.isdigit, str(line))))
                            os.chdir("../")
                        except:
                            continue
                        if not (n_total_configurations == 0):
                            break
                os.chdir("../")
                if not (n_total_configurations == 0):
                    break
#        PLOT SIZE DEPENDENCE
        print("   Plotting size dependence.")
        print("...",end="\r")
        analyse_data_E = np.genfromtxt("analyse_data_min_energy.out")
        analyse_data_F = np.genfromtxt("analyse_data_min_force.out")
        training_set_sizes = analyse_data_E[:,0]*n_total_configurations
        plt.figure("Energy RMSE Set Size Dependence")
        plt.errorbar(training_set_sizes,analyse_data_E[:,4],analyse_data_E[:,5],errorevery=1,linewidth=1,capsize=2,ls="-",fmt="o",markersize=3,color=plt.cm.tab20c(4),label="Etest, epoch of min energy")
        plt.errorbar(training_set_sizes,analyse_data_E[:,2],analyse_data_E[:,3],errorevery=1,linewidth=1,capsize=2,ls="--",fmt="o",markersize=3,color=plt.cm.tab20c(7),label="Etrain, epoch of min energy")[-1][0].set_linestyle('--')
        plt.errorbar(training_set_sizes,analyse_data_F[:,4],analyse_data_F[:,5],errorevery=1,linewidth=1,capsize=2,ls="-",fmt="o",markersize=3,color=plt.cm.tab20c(8),label="Etest, epoch of min force")
        plt.errorbar(training_set_sizes,analyse_data_F[:,2],analyse_data_F[:,3],errorevery=1,linewidth=1,capsize=2,ls="--",fmt="o",markersize=3,color=plt.cm.tab20c(11),label="Etrain, epoch of min force")[-1][0].set_linestyle('--')
        plt.grid(b=True, which='major', color='#999999', linestyle=':')
        plt.yscale('log')
        plt.title("Energy RMSE Set Size Dependence")
        plt.xlabel("training set size")
        plt.ylabel(r"Energy RMSE (meV/atom)")
        plt.legend(loc=1, borderaxespad=0.1)
        plt.savefig("Energy_RMSE.png",dpi=300)
        
        plt.figure("Forces RMSE Set Size Dependence")
        plt.errorbar(training_set_sizes,analyse_data_E[:,6],analyse_data_E[:,7],errorevery=1,linewidth=1,capsize=2,ls="-",fmt="o",markersize=3,color=plt.cm.tab20c(4),label="Ftest, epoch of min energy")
        plt.errorbar(training_set_sizes,analyse_data_E[:,8],analyse_data_E[:,9],errorevery=1,linewidth=1,capsize=2,ls="--",fmt="o",markersize=3,color=plt.cm.tab20c(7),label="Ftrain, epoch of min energy")[-1][0].set_linestyle('--')
        plt.errorbar(training_set_sizes,analyse_data_F[:,6],analyse_data_F[:,7],errorevery=1,linewidth=1,capsize=2,ls="-",fmt="o",markersize=3,color=plt.cm.tab20c(8),label="Ftest, epoch of min force")
        plt.errorbar(training_set_sizes,analyse_data_F[:,8],analyse_data_F[:,9],errorevery=1,linewidth=1,capsize=2,ls="--",fmt="o",markersize=3,color=plt.cm.tab20c(11),label="Ftrain, epoch of min force")[-1][0].set_linestyle('--')
        plt.grid(b=True, which='major', color='#999999', linestyle=':')
        plt.yscale('log')
        plt.title("Forces RMSE Set Size Dependence")
        plt.xlabel("training set size")
        plt.ylabel(r"Forces RMSE (meV/$\AA$)")
        plt.legend(loc=1, borderaxespad=0.1)
        plt.savefig("Forces_RMSE.png",dpi=300)
        
#        PLOT TRAINING PERFORMANCE
        print("   Plotting training performance.")
        print("...",end="\r")
        parameters = ["Energies","Forces"]
        parameter_shortcuts = ["E","F"]
        for parameter_counter in range(len(parameters)):
            current_parameter = parameters[parameter_counter]
            current_shortcut = parameter_shortcuts[parameter_counter]
            learning_curve = np.genfromtxt("learning_curve_"+current_shortcut+".out")
            epochs = np.arange(len(learning_curve[:,0]))
            plt.figure("Learning_curve_"+current_parameter)
            for set_sizes_counter in range(len(training_set_sizes)):
                current_training_set_size = training_set_sizes[set_sizes_counter]
                plt.errorbar(epochs,learning_curve[:,1+5*set_sizes_counter],learning_curve[:,2+5*set_sizes_counter],errorevery=(17+set_sizes_counter),linewidth=1,capsize=2,ls="-",color=plt.cm.Pastel2(set_sizes_counter),label=current_shortcut+"train_setsize="+str(int(current_training_set_size)))[-1][0].set_linestyle('--')
                plt.errorbar(epochs,learning_curve[:,3+5*set_sizes_counter],learning_curve[:,4+5*set_sizes_counter],errorevery=(18+set_sizes_counter),linewidth=1,capsize=2,ls="-",color=plt.cm.Dark2(set_sizes_counter), label=current_shortcut+"test_setsize="+str(int(current_training_set_size)))
                plt.yscale('log')
            plt.title("Training performance for "+current_parameter)
            plt.xlabel("epoch")
            if current_parameter == "Energies":
                plt.ylabel(r"Energy RMSE (meV/atom)")
            elif current_parameter == "Forces":
                plt.ylabel(r"Forces RMSE (meV/$\AA$)")
            plt.grid(b=True, which='major', color='#999999', linestyle=':')
            plt.legend(loc=1, borderaxespad=0.1)
            plt.savefig("Learning_curve_"+current_parameter+".png",dpi=300)
        
        os.chdir("../")
        print("FINISHED plotting size dependence.")
        return None


def perform_NNTSSD():
    """This function performs NNTSSD methods according to user-given specifications.
    
    Firstly, it reads the NNTSSD parameters either from the file 'NNTSSD_input.dat' or,
    if not successful, from interactive user input.
    Secondly, it performs a user-given selection of the NNTSSD methods
    create_training_datasets(), training_neural_network(), analyse_learning_curves() and plot_size_dependence().
    """
    print("**********************************************************************")
    print("NNTSSD - Tools for Neural Network Training Set Size Dependence")
    print("**********************************************************************")
        
    try:
        create_logical,train_logical,analyse_logical,plot_logical,\
        set_size_ratios,n_sets_per_size,fix_random_seed_create,random_seed_create,\
        mpirun_cores,write_submission_script_logical,maximum_time,fix_random_seed_train,random_seed_train\
        = file_input.read_parameters_from_file()
    except:
        create_logical,train_logical,analyse_logical,plot_logical,\
        set_size_ratios,n_sets_per_size,fix_random_seed_create,random_seed_create,\
        mpirun_cores,write_submission_script_logical,maximum_time,fix_random_seed_train,random_seed_train\
        = interactive_input.input_parameters_by_user()

    nnp_train = "mpirun -np "+str(mpirun_cores)+" ../../../../../../../bin/nnp-train"
    
    print("Performing the following NNTSSD steps:")
    print("  ",create_logical, "\t Create training datasets")
    print("  ",train_logical, "\t Training neural network")
    print("  ",analyse_logical, "\t Analyse learning curves")
    print("  ",plot_logical, "\t Plot size dependence")
    
    myNNTSSD = Tools()
    
    if create_logical:
        myNNTSSD.create_training_datasets(set_size_ratios,n_sets_per_size,fix_random_seed_create,random_seed_create)
    if train_logical:
        myNNTSSD.training_neural_network(nnp_train,write_submission_script_logical,fix_random_seed_train,random_seed_train,maximum_time)
    if analyse_logical:
        myNNTSSD.analyse_learning_curves()
    if plot_logical:
        myNNTSSD.plot_size_dependence()
    return None

def write_submission_script(command,job_name="",n_nodes=1,n_tasks_per_node=16,n_tasks_per_core=1,time=None):
    """Writes a submission script 'submit.slrm' for running a job command on VSC (Vienna Scientific Cluster).
    
    Parameters
    ----------
    command : string
        Job command for VSC.
    job_name : string
        Name of the job. Default is none.
    n_nodes : integer
        Number of nodes requested (16 cores per node available). Default is 1.
    n_tasks_per_node : integer
        Number of processes run in parallel on a single node. Default is 16.
    n_tasks_per_core : integer
        Number of tasks a single core should work on. Default is 1.
    time : string, optional
        Maximum time required for exectuing the job; eg. time='08:00:00' for eight hours.
    """
    submission_script = open("submit.slrm","w")
    submission_script.write("#!/bin/bash\n#\n")
    submission_script.write("#SBATCH -J "+job_name+"\n")
    submission_script.write("#SBATCH -N "+str(n_nodes)+"\n")
    submission_script.write("#SBATCH --ntasks-per-node="+str(n_tasks_per_node)+"\n")
    submission_script.write("#SBATCH --ntasks-per-core="+str(n_tasks_per_core)+"\n")
    if not time==None:
        submission_script.write("#SBATCH --time="+time+"\n")
    submission_script.write("\n"+command)
    submission_script.close()

def main():
    pass
if __name__=="__main__":
    perform_NNTSSD()