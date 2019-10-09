#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
NNTSSD: Neural Network Training Set Size Dependence
"""

import os
import sys
import shutil
import numpy as np
import matplotlib.pyplot as plt
import file_input

class Tools():
    """Tools for Neural Network Training Set Size Dependence.
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
        
        Notes
        -----
        Requirements
            ``input.data`` : file
                Contains original set of trainingdata.
            ../../../../bin/nnp-select : executable program
                Performs random selection of sets according to given ratio.
        Outputs
            ``Output`` : folder
                It contains all of the following outputs.
            ``Output/ratio*`` : folders
                Its name tells the ratio * of current from original dataset.
            ``Output/ratio*/ratio*_**`` : subfolders of the previous
                Its name in addition tells the sample number ** of its ratio *.
            ``Output/ratio*/ratio*.**/input.data`` :  file
                Contains new training dataset of specified size ratio.
            ``Output/ratio*/ratio*_**/nnp-select.log`` : file
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
        print("   number of samples per training set size = ", n_sets_per_size)
        print("   number of different training set sizes = ", n_set_size_ratios)
        for ratios_counter in range(n_set_size_ratios):
            current_ratio = set_size_ratios[ratios_counter]
            print("   We are working with ratio {:3.2f}".format(current_ratio))
            ratio_folder = "ratio"+str("{:3.2f}".format(current_ratio))
            os.system("mkdir "+ratio_folder)
            for sets_per_size_counter in range(1,n_sets_per_size+1):
                if fix_random_seed_create:
                    random_seed = random_seed_create
                else:
                    random_seed = int(np.random.randint(100,999,1))
                nnp_select = "../../../../bin/nnp-select random "+str("{:3.2f}".format(current_ratio))+" "+str(random_seed)
                print("   ",nnp_select)
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
            print("   INFO: Removed old 'Output' folder.")
        except:
            pass
        os.system("mkdir "+"Output")
        ratio_dir_string = np.sort(os.listdir())
        for ratio_dir_counter in range(len(ratio_dir_string)):
            if ratio_dir_string[ratio_dir_counter].startswith('ratio'):
                shutil.move(ratio_dir_string[ratio_dir_counter],"Output")
        print("FINISHED creating datasets.")
        return None
                
    def training_neural_network(self,nnp_train,n_epochs,test_fraction,write_submission_script_logical,fix_random_seed_train,random_seed_train=123,maximum_time=None):
        """Trains the neural network with different existing datasets using the program nnp-train.
        
        Parameters
        ----------
        nnp_train : string
            Command for executing n2p2's nnp-train.
        n_epochs : integer
            Number of training epochs.
        write_submission_script_logical : logical
            If True, a job submission script for VSC is written in each of the folders 'Output/ratio*/ratio*_**/'.
            If False, the command nnp_train is executed right away.
        fix_random_seed_train : logical
            True if random seed for nnp-train shall be fixed, False otherwise.
        random_seed_train : integer, optional
            User given fixed random number generator seed. Default is 123.
        maximum_time: string, optional
            User given maximum time required for executing the VSC job. Default is None.
        
        Notes
        -----
        Requirements
            ``Output/ratio*/ratio*_**`` : 3-layered directory structure
                Created with the method create_training_datasets().
            ``Output/ratio*/ratio*_**/input.data`` : file
                Contains the training datasets.
            ../../../../bin/nnp-train : executable program
                Performs training of neural network.
            ``Output/ratio*/ratio*_**/input.nn`` : file
                Specifies the training parameters.
            ``Output/ratio*/ratio*_**/scaling.data`` : file
                Contains symmetry function scaling data.
        Outputs
            ``Output/ratio*/ratio*_**/train.data`` : file
                Dataset actually used for training.
            ``Output/ratio*/ratio*_**/test.data`` : file
                Dataset kept for testing.
            ``Output/ratio*/ratio*_**/nnp-train.log.****`` : file
                One or more log files from running nnp-train.
            ``Output/ratio*/ratio*_**/learning-curve.out`` : file
                Contains learning curve data, namely RMSE of energy and forces of train and test sets for each epoch.
        """
        print("\n***TRAINING NEURAL NETWORK*****************************************************")
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
                        os.system("sed -i \"s/^{0:s} .*$/{0:s} {1:d}/g\" input.nn".format("epochs", n_epochs))
                        os.system("sed -i \"s/^{0:s} .*$/{0:s} {1:f}/g\" input.nn".format("test_fraction", test_fraction))
                        shutil.copy("../../../scaling.data","scaling.data")
                        if write_submission_script_logical:
                            write_submission_script(command=nnp_train,job_name=set_dir_string[set_dir_counter]+"-training",time=maximum_time)
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
        
        Notes
        -----
        Requirements
            ``Output/ratio*/ratio*_**`` : 3-layered directory structure
                Created with the method create_training_datasets().
            ``Output/ratio*/ratio*_**/learning-curve.out`` : file
                Contains learning curve data, created with the method train_neural_network().
        Outputs
            ``Output/ratio*/collect_data_min_***.out`` : file
                Contains analysis of learning curve data of specific training size with respect to 'best' epoch, one file per optimization approach.
            ``Output/analyse_data_min_***.out`` : file
                Contains processed RMSE size dependence information for all dataset sizes with respect to 'best' epoch, one file per optimization approach.
            ``Output/analyse_data_min_***_all.out`` : file
                Contains processed RMSE size dependence information for all datasets with respect to 'best' epoch, one file per optimization approach.
        """
        print("\n***ANALYSING LEARNING CURVES***************************************************")
        if not os.path.isdir("Output"):
            sys.exit("ERROR: The folder 'Output' does not exist!")
        else:
            os.chdir("Output")
#       ANALYSE SIZE DEPENDENCE
        epoch_min_arg = ["energy","force","comb"]
        test_row_indices = [2,4,0]
        for epoch_min_arg_counter in range(len(epoch_min_arg)):
            current_epoch_min_arg = epoch_min_arg[epoch_min_arg_counter]
            current_test_row_index = test_row_indices[epoch_min_arg_counter]
            print("   Analysing data at epoch of minimum "+current_epoch_min_arg)
            analyse_data = np.empty([0,10])
            analyse_data_all = np.empty([0,7])
            ratio_dir_string = np.sort(os.listdir())
            for ratio_dir_counter in range(len(ratio_dir_string)):
                if ratio_dir_string[ratio_dir_counter].startswith('ratio'):
                    current_ratio = 0.01*int(''.join(filter(str.isdigit, ratio_dir_string[ratio_dir_counter])))
                    print("    We are working with ratio {:3.2f}".format(current_ratio))
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
                                if (current_epoch_min_arg == "energy") or (current_epoch_min_arg == "force"):
                                    cut_epoch_number = max(1,int(round(0.1*len(learning_curve_data[:,2]))))
                                    index_epoch = 1+np.argmin(learning_curve_data[cut_epoch_number:,current_test_row_index]) #1: statt 20:
                                elif (current_epoch_min_arg == "comb"):
                                    cut_epoch_number = max(1,int(round(0.1*len(learning_curve_data[:,2]))))
                                    #Finde beste Epoche nach Punktesystem
                                    index_epoch = 1+np.argmin(np.argsort(learning_curve_data[cut_epoch_number:,2])+np.argsort(learning_curve_data[cut_epoch_number:,4])) #1: statt 20:
                                collect_data = np.vstack([collect_data,np.append(np.array([current_ratio,sample_set_counter]),learning_curve_data[index_epoch,0:5],axis=0)])
                            except:
                                print("    INFO: The file 'learning-curve.out' does not exist in "+ratio_dir_string[ratio_dir_counter]+"/"+set_dir_string[set_dir_counter]+"!")
                            os.chdir("../")
                    np.savetxt("collect_data_min_"+current_epoch_min_arg+".out",collect_data,fmt="%.18e",header='%15s%25s%25s%25s%25s%25s%25s'%('ratio','set_number','epoch','RMSE_E_train','RMSE_E_test','RMSE_F_train','RMSE_F_test'))
                    
                    mean_data = np.mean(collect_data,axis=0)[3:]
                    std_data = np.std(collect_data,axis=0)[3:]
                    wip_analyse_data = np.vstack([mean_data,std_data])
                    wip_analyse_data = np.reshape(wip_analyse_data,8,order="F")
                    analyse_data = np.vstack([analyse_data,np.append(np.take(collect_data,[0,2]),wip_analyse_data,axis=0)])
                    analyse_data_all = np.vstack([analyse_data_all,collect_data])
                    os.chdir("../")
            np.savetxt("analyse_data_min_"+current_epoch_min_arg+".out",analyse_data,fmt="%.18e", header='%15s%25s%25s%25s%25s%25s%25s%25s%25s%25s'%('ratio','epoch','mean_RMSE_E_train','std_RMSE_E_train','mean_RMSE_E_test','std_RMSE_E_test','mean_RMSE_F_train','std_RMSE_F_train','mean_RMSE_F_test','std_RMSE_F_test'))
            np.savetxt("analyse_data_min_"+current_epoch_min_arg+"_all.out",analyse_data_all,fmt="%.18e",header='%15s%25s%25s%25s%25s%25s%25s'%('ratio','set_number','epoch','RMSE_E_train','RMSE_E_test','RMSE_F_train','RMSE_F_test'))
        os.chdir("../")
        print("FINISHED analysing learning curves.")
        return None
    
    def plot_size_dependence(self):
        """Plots the energy and force RMSE versus training set size.
        
        Firstly, this method carries out the original training set size from the file nnp-select.log
        that has been created with create_training_datasets().
        Secondly, it plots the energy and force RMSE including their standard deviation versus the
        selected training set sizes.
        
        Notes
        -----
        Requirements
            ``Output/analyse_data_min_force.out`` : file
                Created with analyse_learning_curves().
            ``Output/analyse_data_min_energy.out`` : file
                Created with analyse_learning_curves().
            ``Output/ratio*/ratio*_**/nnp-select.log`` : file
                Created with create_training_datasets().
        Outputs
            ``Output/int_Energy_RMSE_epoch_comparison.png`` : png picture
                Shows train and test energy RMSE (and its standard deviation) versus training set size.
            ``Output/int_Forces_RMSE_epoch_comparison.png`` : png picture
                Shows train and test forces RMSE (and its standard deviation) versus training set size.
        """
        print("\n***PLOTTING SIZE DEPENDENCE****************************************************")
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
                        except:
                            continue
                        os.chdir("../")
                        if not (n_total_configurations == 0):
                            break
                os.chdir("../")
                if not (n_total_configurations == 0):
                    break
#        PLOT SIZE DEPENDENCE
        print("   Plotting size dependence.")
        analyse_data_E = np.genfromtxt("analyse_data_min_energy.out")
        analyse_data_F = np.genfromtxt("analyse_data_min_force.out")
        analyse_data_comb = np.genfromtxt("analyse_data_min_comb.out")
        analyse_data_E_all = np.genfromtxt("analyse_data_min_energy_all.out")
        analyse_data_F_all = np.genfromtxt("analyse_data_min_force_all.out")
        analyse_data_comb_all = np.genfromtxt("analyse_data_min_comb_all.out")
        training_set_sizes = analyse_data_E[:,0]*n_total_configurations
        training_set_sizes_all = analyse_data_E_all[:,0]*n_total_configurations
        plt.figure("Energy RMSE Set Size Dependence")
        eh = 27.21138602e3
        plt.errorbar(training_set_sizes,eh*analyse_data_E[:,4],eh*analyse_data_E[:,5],errorevery=1,linewidth=1,capsize=2,ls="-",fmt="o",markersize=3,color=plt.cm.tab20c(4),label="Etest, epoch of min energy")
        plt.errorbar(training_set_sizes,eh*analyse_data_E[:,2],eh*analyse_data_E[:,3],errorevery=1,linewidth=1,capsize=2,ls="--",fmt="o",markersize=3,color=plt.cm.tab20c(7),label="Etrain, epoch of min energy")[-1][0].set_linestyle('--')
        plt.errorbar(training_set_sizes,eh*analyse_data_F[:,4],eh*analyse_data_F[:,5],errorevery=1,linewidth=1,capsize=2,ls="-",fmt="o",markersize=3,color=plt.cm.tab20c(8),label="Etest, epoch of min force")
        plt.errorbar(training_set_sizes,eh*analyse_data_F[:,2],eh*analyse_data_F[:,3],errorevery=1,linewidth=1,capsize=2,ls="--",fmt="o",markersize=3,color=plt.cm.tab20c(11),label="Etrain, epoch of min force")[-1][0].set_linestyle('--')
        plt.errorbar(training_set_sizes,eh*analyse_data_comb[:,4],eh*analyse_data_comb[:,5],errorevery=1,linewidth=1,capsize=2,ls="-",fmt="o",markersize=3,color=plt.cm.tab20c(0),label="Etest, epoch of min comb")
        plt.errorbar(training_set_sizes,eh*analyse_data_comb[:,2],eh*analyse_data_comb[:,3],errorevery=1,linewidth=1,capsize=2,ls="--",fmt="o",markersize=3,color=plt.cm.tab20c(3),label="Etrain, epoch of min comb")[-1][0].set_linestyle('--')
        plt.grid(b=True, which='major', color='#999999', linestyle=':')
        plt.yscale('log')
        plt.xscale('log')
        plt.title("Energy RMSE Set Size Dependence (int.Testset)")
        plt.xlabel("training set size")
        plt.ylabel(r"Energy RMSE (meV/atom)")
        plt.legend(loc=1, borderaxespad=0.1)
        plt.savefig("int_Energy_RMSE_epoch_comparison.png",dpi=300)
        
        plt.figure("Forces RMSE Set Size Dependence")
        bohr = 27.21138602e3/5.2917721067e-1
        plt.errorbar(training_set_sizes,bohr*analyse_data_E[:,8],bohr*analyse_data_E[:,9],errorevery=1,linewidth=1,capsize=2,ls="-",fmt="o",markersize=3,color=plt.cm.tab20c(4),label="Ftest, epoch of min energy")
        plt.errorbar(training_set_sizes,bohr*analyse_data_E[:,6],bohr*analyse_data_E[:,7],errorevery=1,linewidth=1,capsize=2,ls="--",fmt="o",markersize=3,color=plt.cm.tab20c(7),label="Ftrain, epoch of min energy")[-1][0].set_linestyle('--')
        plt.errorbar(training_set_sizes,bohr*analyse_data_F[:,8],bohr*analyse_data_F[:,9],errorevery=1,linewidth=1,capsize=2,ls="-",fmt="o",markersize=3,color=plt.cm.tab20c(8),label="Ftest, epoch of min force")
        plt.errorbar(training_set_sizes,bohr*analyse_data_F[:,6],bohr*analyse_data_F[:,7],errorevery=1,linewidth=1,capsize=2,ls="--",fmt="o",markersize=3,color=plt.cm.tab20c(11),label="Ftrain, epoch of min force")[-1][0].set_linestyle('--')
        plt.errorbar(training_set_sizes,bohr*analyse_data_comb[:,8],bohr*analyse_data_comb[:,9],errorevery=1,linewidth=1,capsize=2,ls="-",fmt="o",markersize=3,color=plt.cm.tab20c(0),label="Ftest, epoch of min comb")
        plt.errorbar(training_set_sizes,bohr*analyse_data_comb[:,6],bohr*analyse_data_comb[:,7],errorevery=1,linewidth=1,capsize=2,ls="--",fmt="o",markersize=3,color=plt.cm.tab20c(3),label="Ftrain, epoch of min comb")[-1][0].set_linestyle('--')
        plt.grid(b=True, which='major', color='#999999', linestyle=':')
        plt.yscale('log')
        plt.xscale('log')
        plt.title("Forces RMSE Set Size Dependence (int.Testset)")
        plt.xlabel("training set size")
        plt.ylabel(r"Forces RMSE (meV/$\AA$)")
        plt.legend(loc=1, borderaxespad=0.1)
        plt.savefig("int_Forces_RMSE_epoch_comparison.png",dpi=300)
        
        os.chdir("../")
        print("FINISHED plotting size dependence.")
        return None

class Validation():
    """Tools for validating trained neural networks with user provided dataset.
    """
    def __init__(self):
        """In the moment, the class Validation() does not have any attributes.
        
        Still, it makes sense to define a class since it condenses NNTSSD functionalities and
        for future purposes, it is likely that attributes will be added.
        """
    def predict_validation_data(self,dataset_cores,fix_random_seed_validation,random_seed_validation=123):
        """Predicts the energies and forces of a validation dataset for all trained neural networks 
        
        Parameters
        ----------
        fix_random_seed_validation : logical
            True if random seed for nnp-dataset shall be fixed, False otherwise.
        random_seed_validation : integer, optional
            User given fixed random number generator seed. Default is 123.
               
        Notes
        -----
        Requirements
            ``Output/ratio*/ratio*_**`` : 3-layered directory structure
                Created with the method NNTSSD.Tools.create_training_datasets().
            ``Output/ratio*/ratio*_**/weights.***.****`` : files
                Created with the method NNTSSD.Tools.train_neural_network().
            ``validation_data`` : directory
                Containing the following three files.
            ``validation_data/input.data`` : file
                Contains the dataset that shall be used for testing.
            ``validation_data/input.nn`` : file
                Specifies the parameters for nnp-dataset.
            ``validation_data/scaling.data`` : file
            Contains symmetry function scaling data.
        Outputs
            ``Output/ratio*/ratio*_**/validation`` : directory
                Contains Output files created by nnp-dataset.
            ``Output/validation_data_min_***.out`` : files
                Contains processed RMSE size dependence information for all dataset sizes with respect to 'best' epoch, one file per optimization approach.
            ``Output/validation_data_min_***_all.out`` : files
                Contains RMSE size dependence information for all dataset samples of all sizes with respect to 'best' epoch, one file per optimization approach.
        """
        print("\n***PREDICTING VALIDATION DATA************************************************")
        try:
            os.chdir("validation_data") #dieser Ordner enthält das File input.data mit den zurückgehaltenen Datensätzen, die zum Validieren verwendet werden.
            os.chdir("../")
        except:
            sys.exit("ERROR: The folder 'validation_data' does not exist!")
        os.chdir("Output")
        epoch_min_arg = ["energy","force","comb"]
        test_row_indices = [1,2,0]
        for epoch_min_arg_counter in range(len(epoch_min_arg)):
            current_epoch_min_arg = epoch_min_arg[epoch_min_arg_counter]
            current_test_row_index = test_row_indices[epoch_min_arg_counter]
            print("   Analysing data at epoch of minimum "+current_epoch_min_arg)
            analyse_data_all = np.genfromtxt("analyse_data_min_"+current_epoch_min_arg+"_all.out")
            overall_epoch_counter = 0
            validation_data = np.empty([0,5])
            validation_data_all = np.copy(analyse_data_all[:,0:5])
            ratio_dir_string = np.sort(os.listdir())
            for ratio_dir_counter in range(len(ratio_dir_string)):
                if ratio_dir_string[ratio_dir_counter].startswith('ratio'):
                    current_ratio = 0.01*int(''.join(filter(str.isdigit, ratio_dir_string[ratio_dir_counter])))
                    print("    We are working with ratio {:3.2f}".format(current_ratio))
                    os.chdir(ratio_dir_string[ratio_dir_counter])
                    set_dir_string = np.sort(os.listdir())
                    sample_set_counter = 0
                    for set_dir_counter in range(len(set_dir_string)):
                        if set_dir_string[set_dir_counter].startswith('ratio'):
                            sample_set_counter += 1
                            os.chdir(set_dir_string[set_dir_counter])
                            if os.path.isdir("validation"):
                                shutil.rmtree("validation")
#                                print("    INFO: Removed old 'validation' folder.")
                            else:
                                pass
                            os.system("mkdir validation")
                            try:
                                shutil.copy("../../../validation_data/scaling.data","validation/scaling.data")
                                shutil.copy("../../../validation_data/input.data","validation/input.data")
                                shutil.copy("../../../validation_data/input.nn","validation/input.nn")
                                current_best_epoch = analyse_data_all[overall_epoch_counter,2]
                                shutil.copy("weights.001."+"{0:0=6d}".format(int(current_best_epoch))+".out","validation/weights.001.data")
                                shutil.copy("weights.008."+"{0:0=6d}".format(int(current_best_epoch))+".out","validation/weights.008.data")
                                os.chdir("validation")
                                if fix_random_seed_validation:
                                    random_seed = random_seed_validation
                                else:
                                    random_seed = int(np.random.randint(100,999,1))
                                os.system("sed -i \"s/^{0:s} .*$/{0:s} {1:d}/g\" input.nn".format("random_seed", random_seed))
                                os.system("mpirun -np "+str(dataset_cores)+" ../../../../../../../../bin/nnp-dataset 0") # >/dev/null
                                energy_comp = np.genfromtxt("energy.comp")
                                E_rmse = np.sqrt(np.mean((energy_comp[:,2]-energy_comp[:,3])**2))
                                validation_data_all[overall_epoch_counter,3] = E_rmse
                                forces_comp = np.genfromtxt("forces.comp")
                                F_rmse = np.sqrt(np.mean((forces_comp[:,2]-forces_comp[:,3])**2))
                                validation_data_all[overall_epoch_counter,4] = F_rmse
#                                np.savetxt("learning-curve-testdata.out",test_learning_curve,fmt="%.18e",header='%15s%25s%25s'%('epoch','rmse_Etest','rmse_Ftest'))
#                                overall_epoch_counter += 1
                                os.chdir("../")
                            except:
                                print("    INFO: Something went worng!")
                            overall_epoch_counter += 1
                            os.chdir("../")
                    os.chdir("../")
                    mean_data = np.mean(validation_data_all,axis=0)[3:]
                    std_data = np.std(validation_data_all,axis=0)[3:]
                    wip_validation_data = np.vstack([mean_data,std_data])
                    wip_validation_data = np.reshape(wip_validation_data,4,order="F")
                    validation_data = np.vstack([validation_data,np.append(np.array([current_ratio]),wip_validation_data,axis=0)])
            np.savetxt("validation_data_min_"+current_epoch_min_arg+"_all.out",validation_data_all,fmt="%.18e",header='%15s%25s%25s%25s%25s'%('ratio','set_number','epoch','RMSE_E_validation','RMSE_F_validation'))
            np.savetxt("validation_data_min_"+current_epoch_min_arg+".out",validation_data,fmt="%.18e", header='%15s%25s%25s%25s%25s'%('ratio','mean_RMSE_E_validation','std_RMSE_E_validation','mean_RMSE_F_validation','std_RMSE_F_validation'))
        os.chdir("../")
        print("FINISHED predicting validation data.")
        return None
    
    def plot_validation_data(self):
        """Plots the energy and force RMSE versus training set size.
        
        Firstly, this method carries out the original training set size from the file nnp-select.log
        that has been created with create_training_datasets().
        Secondly, it plots the energy and force RMSE including their standard deviation versus the
        selected training set sizes.
        
        Notes
        -----
        Requirements
            ``Output/analyse_data_min_***.out`` : files
                Created with analyse_learning_curves().
            ``Output/validation_data_min_***.out`` : files
                Created with predict_validation_data().
            ``Output/validation_data_min_***_all.out`` : files
                Created with predict_validation_data().
            ``Output/ratio*/ratio*_**/nnp-select.log`` : file
                Created with create_training_datasets().
        Outputs
            ``Output/val_Energy_RMSE.png`` : png picture
                Shows train and test energy RMSE (and its standard deviation) versus training set size.
            ``Output/val_Energy_RMSE_best_E.png`` : png picture
                As above, using epoch of minimum energy.
            ``Output/val_Forces_RMSE.png`` : png picture
                Shows train and test forces RMSE (and its standard deviation) versus training set size.
            ``Output/val_Forces_RMSE_best_F.png`` : png picture
                As above, using epoch of minimum forces.
        """
        print("\n***PLOTTING TEST SIZE DEPENDENCE W/ VALIDATION DATA****************************")
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
                        except:
                            continue
                        os.chdir("../")
                        if not (n_total_configurations == 0):
                            break
                os.chdir("../")
                if not (n_total_configurations == 0):
                    break
#        PLOT SIZE DEPENDENCE
        print("   Plotting test size dependence.")
        analyse_data_E = np.genfromtxt("analyse_data_min_energy.out")
        analyse_data_F = np.genfromtxt("analyse_data_min_force.out")
        analyse_data_comb = np.genfromtxt("analyse_data_min_comb.out")
#        analyse_data_E_all = np.genfromtxt("analyse_data_min_energy_all.out")
#        analyse_data_F_all = np.genfromtxt("analyse_data_min_force_all.out")
#        analyse_data_comb_all = np.genfromtxt("analyse_data_min_comb_all.out")
        
        validation_data_comb_all = np.genfromtxt("validation_data_min_comb_all.out")
        validation_data_E_all = np.genfromtxt("validation_data_min_energy_all.out")
        validation_data_F_all = np.genfromtxt("validation_data_min_force_all.out")
        
        training_set_sizes = analyse_data_comb[:,0]*n_total_configurations
        training_set_sizes_all = validation_data_comb_all[:,0]*n_total_configurations
        
        plt.figure("Energy RMSE comb")
        eh = 27.21138602e3
        plt.errorbar(training_set_sizes,eh*analyse_data_comb[:,4],eh*analyse_data_comb[:,5],errorevery=1,linewidth=1,capsize=2,ls="-",fmt=",",markersize=3,color=plt.cm.tab20c(1),label="Etest, mean")
        plt.errorbar(training_set_sizes,eh*analyse_data_comb[:,2],eh*analyse_data_comb[:,3],errorevery=1,linewidth=1,capsize=2,ls="--",fmt=",",markersize=3,color=plt.cm.tab20c(3),label="Etrain, mean")[-1][0].set_linestyle('--')
        plt.plot(training_set_sizes_all,eh*validation_data_comb_all[:,3],'*',color=plt.cm.tab20c(0),label="Validation data")
        plt.grid(b=True, which='major',color='#999999', linestyle=':')
        plt.yscale('log')
        plt.xscale('log')
        plt.title("Validation Energy RMSE SSD, combined best epoch")
        plt.xlabel("training set size")
        plt.ylabel(r"Energy RMSE (meV/atom)")
        plt.legend(loc=1, borderaxespad=0.1)
        plt.savefig("val_Energy_RMSE.png",dpi=300)
        
        plt.figure("Forces RMSE comb")
        bohr = 27.21138602e3/5.2917721067e-1
        plt.errorbar(training_set_sizes,bohr*analyse_data_comb[:,8],bohr*analyse_data_comb[:,9],errorevery=1,linewidth=1,capsize=2,ls="-",fmt=",",markersize=3,color=plt.cm.tab20c(1),label="Ftest, mean")
        plt.errorbar(training_set_sizes,bohr*analyse_data_comb[:,6],bohr*analyse_data_comb[:,7],errorevery=1,linewidth=1,capsize=2,ls="--",fmt=",",markersize=3,color=plt.cm.tab20c(3),label="Ftrain, mean")[-1][0].set_linestyle('--')
        plt.plot(training_set_sizes_all,bohr*validation_data_comb_all[:,4],'*',color=plt.cm.tab20c(0),label="Validation data")
        plt.grid(b=True, which='major', color='#999999', linestyle=':')
        plt.yscale('log')
        plt.xscale('log')
        plt.title("Validation Forces RMSE SSD, combined best epoch")
        plt.xlabel("training set size")
        plt.ylabel(r"Forces RMSE (meV/$\AA$)")
        plt.legend(loc=1, borderaxespad=0.1)
        plt.savefig("val_Forces_RMSE.png",dpi=300)
        
        plt.figure("Energy RMSE best E")
        plt.errorbar(training_set_sizes,eh*analyse_data_E[:,4],eh*analyse_data_E[:,5],errorevery=1,linewidth=1,capsize=2,ls="-",fmt=",",markersize=3,color=plt.cm.tab20c(5),label="Etest, mean")
        plt.errorbar(training_set_sizes,eh*analyse_data_E[:,2],eh*analyse_data_E[:,3],errorevery=1,linewidth=1,capsize=2,ls="--",fmt=",",markersize=3,color=plt.cm.tab20c(7),label="Etrain, mean")[-1][0].set_linestyle('--')
        plt.plot(training_set_sizes_all,eh*validation_data_E_all[:,3],'*',color=plt.cm.tab20c(4),label="Validation data")
        plt.grid(b=True, which='major',color='#999999', linestyle=':')
        plt.yscale('log')
        plt.xscale('log')
        plt.title("Validation Energy RMSE SSD, best energy epoch")
        plt.xlabel("training set size")
        plt.ylabel(r"Energy RMSE (meV/atom)")
        plt.legend(loc=1, borderaxespad=0.1)
        plt.savefig("val_Energy_RMSE_best_E.png",dpi=300)
        
        plt.figure("Forces RMSE best F")
        plt.errorbar(training_set_sizes,bohr*analyse_data_F[:,8],bohr*analyse_data_F[:,9],errorevery=1,linewidth=1,capsize=2,ls="-",fmt=",",markersize=3,color=plt.cm.tab20c(9),label="Ftest, mean")
        plt.errorbar(training_set_sizes,bohr*analyse_data_F[:,6],bohr*analyse_data_F[:,7],errorevery=1,linewidth=1,capsize=2,ls="--",fmt=",",markersize=3,color=plt.cm.tab20c(11),label="Ftrain, mean")[-1][0].set_linestyle('--')
        plt.plot(training_set_sizes_all,bohr*validation_data_F_all[:,4],'*',color=plt.cm.tab20c(8),label="Validation data")
        plt.grid(b=True, which='major', color='#999999', linestyle=':')
        plt.yscale('log')
        plt.xscale('log')
        plt.title("Validation Forces RMSE SSD, best forces epoch")
        plt.xlabel("training set size")
        plt.ylabel(r"Forces RMSE (meV/$\AA$)")
        plt.legend(loc=1, borderaxespad=0.1)
        plt.savefig("val_Forces_RMSE_best_F.png",dpi=300)
        
        os.chdir("../")
        print("FINISHED plotting test size dependence.")
        return None

class External_Testdata():
    """Tools for testing trained neural networks with user provided dataset.
    """
    def __init__(self):
        """In the moment, the class External_Testdata() does not have any attributes.
        
        Still, it makes sense to define a class since it condenses NNTSSD functionalities and
        for future purposes, it is likely that attributes will be added.
        """
  
    def predict_test_data(self,dataset_cores,fix_random_seed_predict,random_seed_predict=123):
        """Predicts the energies and forces of a test dataset for all trained neural networks.
        
        Parameters
        ----------
        fix_random_seed_predict : logical
            True if random seed for nnp-dataset shall be fixed, False otherwise.
        random_seed_predict : integer, optional
            User given fixed random number generator seed. Default is 123.
               
        Notes
        -----
        Requirements
            ``Output/ratio*/ratio*_**`` : 3-layered directory structure
                Created with the method NNTSSD.create_training_datasets().
            ``Output/ratio*/ratio*_**/weights.***.****`` : files
                Created with the method NNTSSD.train_neural_network().
            ``predict_test_data`` : directory
                Containing the following three files.
            ``predit_test_data/input.data`` : file
                Contains the dataset that shall be used for testing.
            ``predict_test_data/input.nn`` : file
                Specifies the parameters for nnp-dataset.
            ``predict_test_dataset/scaling.data`` : file
            Contains symmetry function scaling data.
        Outputs
            ``Output/ratio*/ratio*_**/predict_testdata`` : directory
                Contains Output files created by nnp-dataset and the following file.
            ``Output/ratio*/ratio*_**/predict_testdata/learning-curve-testdata.out`` : file
                Contains learning curve data for testset.
        """
        print("\n***PREDICTING EXTERNAL TESTDATA************************************************")
        try:
            os.chdir("predict_test_data") #dieser Ordner enthält das File test.data mit den zurückgehaltenen Datensätzen, die zum Testen verwendet werden.
            os.chdir("../")
        except:
            sys.exit("ERROR: The folder 'predict_test_data' does not exist!")
        os.chdir("Output")
        ratio_dir_string = np.sort(os.listdir())
        for ratio_dir_counter in range(len(ratio_dir_string)):
            if ratio_dir_string[ratio_dir_counter].startswith('ratio'):
                current_ratio = 0.01*int(''.join(filter(str.isdigit, ratio_dir_string[ratio_dir_counter])))
                print("   We are working with ratio {:3.2f}".format(current_ratio))
                os.chdir(ratio_dir_string[ratio_dir_counter])
                set_dir_string = np.sort(os.listdir())
                sample_set_counter = 0
                for set_dir_counter in range(len(set_dir_string)):
                    if set_dir_string[set_dir_counter].startswith('ratio'):
                        sample_set_counter += 1
                        os.chdir(set_dir_string[set_dir_counter])
                        try:
                            shutil.rmtree("predict_testdata")
                            print("   INFO: Removed old 'predict_testdata' folder.")
                        except:
                            pass
                        os.system("mkdir predict_testdata")
                        try:
                            shutil.copy("../../../predict_test_data/scaling.data","predict_testdata/scaling.data")
                            shutil.copy("../../../predict_test_data/input.data","predict_testdata/input.data")
                            shutil.copy("../../../predict_test_data/input.nn","predict_testdata/input.nn")
                            listdir = np.sort(os.listdir())
                            weightslist_001 = np.sort(np.array([item for item in listdir if item.startswith('weights.001.')]))
                            weightslist_008 = np.sort(np.array([item for item in listdir if item.startswith('weights.008.')]))
                            test_learning_curve = np.empty([len(weightslist_001),3])
                            for current_epoch in range(len(weightslist_001)):
                                test_learning_curve[current_epoch,0] = current_epoch
                                shutil.copy("weights.001."+"{0:0=6d}".format(current_epoch)+".out","predict_testdata/weights.001.data")
                                shutil.copy("weights.008."+"{0:0=6d}".format(current_epoch)+".out","predict_testdata/weights.008.data")
                                os.chdir("predict_testdata")
                                if fix_random_seed_predict:
                                    random_seed = random_seed_predict
                                else:
                                    random_seed = int(np.random.randint(100,999,1))
                                os.system("sed -i \"s/^{0:s} .*$/{0:s} {1:d}/g\" input.nn".format("random_seed", random_seed))
                                os.system("mpirun -np "+str(dataset_cores)+" ../../../../../../../../bin/nnp-dataset 0") # >/dev/null
                                energy_comp = np.genfromtxt("energy.comp")
                                E_rmse = np.sqrt(np.mean((energy_comp[:,2]-energy_comp[:,3])**2))
                                test_learning_curve[current_epoch,1] = E_rmse
                                forces_comp = np.genfromtxt("forces.comp")
                                F_rmse = np.sqrt(np.mean((forces_comp[:,2]-forces_comp[:,3])**2))
                                test_learning_curve[current_epoch,2] = F_rmse
                                np.savetxt("learning-curve-testdata.out",test_learning_curve,fmt="%.18e",header='%15s%25s%25s'%('epoch','rmse_Etest','rmse_Ftest'))
                                os.chdir("../")
                        except:
                            print("   INFO: Something went worng!")
                        os.chdir("../")
                os.chdir("../")
        os.chdir("../")
        print("FINISHED predicting test data.")
        return None
    
    def analyse_learning_curves(self):
        """Prepares analyse data from the learning curves obtained in training neural network and predicting testdata.
        
        Notes
        -----
        Requirements
            ``Output/ratio*/ratio*_**`` : 3-layered directory structure
                Created with the method NNTSSD.create_training_datasets().
            ``Output/ratio*/ratio*_**/learning-curve.out`` : file
                Contains learning curve data, created with the method NNTSSD.create_training_datasets().
            ``Output/ratio*/ratio*_**/predict_testdata/learning-curve-testdata.out`` : file
                Created with the method predict_test_data().
        Outputs
            ``Output/ratio*/ext_collect_data_min_***.out`` : files
                Contains analysis of learning curve data of specific training size with respect to 'best' epoch, one file per optimization approach.
            ``Output/ext_analyse_data_min_***.out`` : files
                Contains processed RMSE size dependence information for all dataset sizes with respect to 'best' epoch, one file per optimization approach.
            ``Output/ext_analyse_data_min_***_all.out`` : files
                Contains RMSE size dependence information for all dataset samples of all sizes with respect to 'best' epoch, one file per optimization approach.
        """
        print("\n***ANALYSING LEARNING CURVES W/ EXTERNAL TESTDATA******************************")
        if not os.path.isdir("Output"):
            sys.exit("ERROR: The folder 'Output' does not exist!")
        else:
            os.chdir("Output")
        epoch_min_arg = ["energy","force","comb"]
        test_row_indices = [1,2,0]
        for epoch_min_arg_counter in range(len(epoch_min_arg)):
            current_epoch_min_arg = epoch_min_arg[epoch_min_arg_counter]
            current_test_row_index = test_row_indices[epoch_min_arg_counter]
            print("   Analysing data at epoch of minimum "+current_epoch_min_arg)
            analyse_data = np.empty([0,10])
            analyse_data_all = np.empty([0,7])
            ratio_dir_string = np.sort(os.listdir())
            for ratio_dir_counter in range(len(ratio_dir_string)):
                if ratio_dir_string[ratio_dir_counter].startswith('ratio'):
                    current_ratio = 0.01*int(''.join(filter(str.isdigit, ratio_dir_string[ratio_dir_counter])))
                    print("    We are working with ratio {:3.2f}".format(current_ratio))
                    collect_data = np.empty([0,7])
                    os.chdir(ratio_dir_string[ratio_dir_counter])
                    set_dir_string = np.sort(os.listdir())
                    sample_set_counter = 0
                    for set_dir_counter in range(len(set_dir_string)):
                        if set_dir_string[set_dir_counter].startswith('ratio'):
                            sample_set_counter += 1
                            os.chdir(set_dir_string[set_dir_counter])
                            try:
                                learning_curve_testdata = np.genfromtxt("predict_testdata/learning-curve-testdata.out")
                                learning_curve_data = np.genfromtxt("learning-curve.out")
                                number_of_epochs = int(learning_curve_data[-1,0])
                                if (current_epoch_min_arg == "energy") or (current_epoch_min_arg == "force"):
                                    index_epoch = 1+np.argmin(learning_curve_testdata[1:,current_test_row_index])
                                elif (current_epoch_min_arg == "comb"):
                                    #Finde beste Epoche nach Punktesystem
                                    index_epoch = 1+np.argmin(np.argsort(learning_curve_testdata[1:,1])+np.argsort(learning_curve_testdata[1:,2]))
                                wip_collect_data = np.array([learning_curve_testdata[index_epoch,0],learning_curve_data[index_epoch,1],learning_curve_testdata[index_epoch,1],learning_curve_data[index_epoch,3],learning_curve_testdata[index_epoch,2]])
                                collect_data = np.vstack([collect_data,np.append(np.array([current_ratio,sample_set_counter]),wip_collect_data,axis=0)])
                            except:
                                print("    INFO: Something went wrong!")
                            os.chdir("../")
                    np.savetxt("ext_collect_data_min_"+current_epoch_min_arg+".out",collect_data,fmt="%.18e",header='%15s%25s%25s%25s%25s%25s%25s'%('ratio','set_number','epoch','RMSE_E_train','RMSE_E_test','RMSE_F_train','RMSE_F_test'))
                    
                    mean_data = np.mean(collect_data,axis=0)[3:]
                    std_data = np.std(collect_data,axis=0)[3:]
                    wip_analyse_data = np.vstack([mean_data,std_data])
                    wip_analyse_data = np.reshape(wip_analyse_data,8,order="F")
                    analyse_data = np.vstack([analyse_data,np.append(np.take(collect_data,[0,2]),wip_analyse_data,axis=0)])
                    analyse_data_all = np.vstack([analyse_data_all,collect_data])
                    os.chdir("../")
            np.savetxt("ext_analyse_data_min_"+current_epoch_min_arg+".out",analyse_data,fmt="%.18e", header='%15s%25s%25s%25s%25s%25s%25s%25s%25s%25s'%('ratio','epoch','mean_RMSE_E_train','std_RMSE_E_train','mean_RMSE_E_test','std_RMSE_E_test','mean_RMSE_F_train','std_RMSE_F_train','mean_RMSE_F_test','std_RMSE_F_test'))
            np.savetxt("ext_analyse_data_min_"+current_epoch_min_arg+"_all.out",analyse_data_all,fmt="%.18e",header='%15s%25s%25s%25s%25s%25s%25s'%('ratio','set_number','epoch','RMSE_E_train','RMSE_E_test','RMSE_F_train','RMSE_F_test'))
        os.chdir("../")
        print("FINISHED analysing learning curves.")
        return None
    
    def plot_test_size_dependence(self):
        """Plots the energy and force RMSE versus training set size.
        
        Firstly, this method carries out the original training set size from the file nnp-select.log
        that has been created with create_training_datasets().
        Secondly, it plots the energy and force RMSE including their standard deviation versus the
        selected training set sizes.
        
        Notes
        -----
        Requirements
            ``Output/analyse_data_min_force.out`` : file
                Created with analyse_learning_curves().
            ``Output/analyse_data_min_energy.out`` : file
                Created with analyse_learning_curves().
            ``Output/ratio*/ratio*_**/nnp-select.log`` : file
                Created with create_training_datasets().
        Outputs
            ``Output/ext_Energy_RMSE.png`` : png picture
                Shows train and test energy RMSE (and its standard deviation) versus training set size at combined best epoch.
            ``Output/ext_Energy_RMSE_epoch_compare.png`` : png picture
                As above, using three different epochs (minimum force, energy, combined best).
            ``Output/ext_Forces_RMSE.png`` : png picture
                Shows train and test forces RMSE (and its standard deviation) versus training set size.
            ``Output/ext_Forces_RMSE_epoch_compare.png`` : png picture
                As above, using three different epochs (minimum force, energy, combined best).
        """
        print("\n***PLOTTING TEST SIZE DEPENDENCE W/ EXTERNAL TESTDATA**************************")
        if not os.path.isdir("Output"):
            sys.exit("ERROR: The folder 'Output' does not exist!")
        else:
            os.chdir("Output")
        if not os.path.isfile("ext_analyse_data_min_energy.out"):
            sys.exit("ERROR: The file 'analyse_data_min_energy.out' does not exist!")
        if not os.path.isfile("ext_analyse_data_min_force.out"):
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
                        except:
                            continue
                        os.chdir("../")
                        if not (n_total_configurations == 0):
                            break
                os.chdir("../")
                if not (n_total_configurations == 0):
                    break
#        PLOT SIZE DEPENDENCE
        print("   Plotting test size dependence.")
        analyse_data_E = np.genfromtxt("ext_analyse_data_min_energy.out")
        analyse_data_F = np.genfromtxt("ext_analyse_data_min_force.out")
        analyse_data_comb = np.genfromtxt("ext_analyse_data_min_comb.out")
        analyse_data_E_all = np.genfromtxt("ext_analyse_data_min_energy_all.out")
        analyse_data_F_all = np.genfromtxt("ext_analyse_data_min_force_all.out")
        analyse_data_comb_all = np.genfromtxt("ext_analyse_data_min_comb_all.out")
        training_set_sizes = analyse_data_comb[:,0]*n_total_configurations
        training_set_sizes_all = analyse_data_comb_all[:,0]*n_total_configurations
        
        plt.figure("Energy RMSE SSD")
        eh = 27.21138602e3
        plt.errorbar(training_set_sizes,eh*analyse_data_comb[:,4],eh*analyse_data_comb[:,5],errorevery=1,linewidth=1,capsize=2,ls="-",fmt=",",markersize=3,color=plt.cm.tab20c(1),label="Etest, mean")
        plt.errorbar(training_set_sizes,eh*analyse_data_comb[:,2],eh*analyse_data_comb[:,3],errorevery=1,linewidth=1,capsize=2,ls="--",fmt=",",markersize=3,color=plt.cm.tab20c(3),label="Etrain, mean")[-1][0].set_linestyle('--')
        plt.plot(training_set_sizes_all,eh*analyse_data_comb_all[:,4],'*',color=plt.cm.tab20c(0),label="Etest, sample results")
        plt.grid(b=True, which='major',color='#999999', linestyle=':')
        plt.yscale('log')
        plt.xscale('log')
        plt.title("Energy RMSE SSD, combined best epoch (ext.Testset)")
        plt.xlabel("training set size")
        plt.ylabel(r"Energy RMSE (meV/atom)")
        plt.legend(loc=1, borderaxespad=0.1)
        plt.savefig("ext_Energy_RMSE.png",dpi=300)
        
        plt.figure("Forces RMSE SSD")
        bohr = 27.21138602e3/5.2917721067e-1
        plt.errorbar(training_set_sizes,bohr*analyse_data_comb[:,8],bohr*analyse_data_comb[:,9],errorevery=1,linewidth=1,capsize=2,ls="-",fmt=",",markersize=3,color=plt.cm.tab20c(1),label="Ftest, mean")
        plt.errorbar(training_set_sizes,bohr*analyse_data_comb[:,6],bohr*analyse_data_comb[:,7],errorevery=1,linewidth=1,capsize=2,ls="--",fmt=",",markersize=3,color=plt.cm.tab20c(3),label="Ftrain, mean")[-1][0].set_linestyle('--')
        plt.plot(training_set_sizes_all,bohr*analyse_data_comb_all[:,6],'*',color=plt.cm.tab20c(0),label="Ftest, sample results")
        plt.grid(b=True, which='major', color='#999999', linestyle=':')
        plt.yscale('log')
        plt.xscale('log')
        plt.title("Forces RMSE SSD, combined best epoch (ext.Testset)")
        plt.xlabel("training set size")
        plt.ylabel(r"Forces RMSE (meV/$\AA$)")
        plt.legend(loc=1, borderaxespad=0.1)
        plt.savefig("ext_Forces_RMSE.png",dpi=300)
        
        plt.figure("Energy RMSE epoch comparison")
        eh = 27.21138602e3
        plt.errorbar(training_set_sizes,eh*analyse_data_E[:,4],eh*analyse_data_E[:,5],errorevery=1,linewidth=1,capsize=2,ls="-",fmt="o",markersize=3,color=plt.cm.tab20c(4),label="Etest, epoch of min energy")
        plt.errorbar(training_set_sizes,eh*analyse_data_E[:,2],eh*analyse_data_E[:,3],errorevery=1,linewidth=1,capsize=2,ls="--",fmt="o",markersize=3,color=plt.cm.tab20c(7),label="Etrain, epoch of min energy")[-1][0].set_linestyle('--')
        plt.errorbar(training_set_sizes,eh*analyse_data_F[:,4],eh*analyse_data_F[:,5],errorevery=1,linewidth=1,capsize=2,ls="-",fmt="o",markersize=3,color=plt.cm.tab20c(8),label="Etest, epoch of min force")
        plt.errorbar(training_set_sizes,eh*analyse_data_F[:,2],eh*analyse_data_F[:,3],errorevery=1,linewidth=1,capsize=2,ls="--",fmt="o",markersize=3,color=plt.cm.tab20c(11),label="Etrain, epoch of min force")[-1][0].set_linestyle('--')
        plt.errorbar(training_set_sizes,eh*analyse_data_comb[:,4],eh*analyse_data_comb[:,5],errorevery=1,linewidth=1,capsize=2,ls="-",fmt="o",markersize=3,color=plt.cm.tab20c(0),label="Etest, epoch of min comb")
        plt.errorbar(training_set_sizes,eh*analyse_data_comb[:,2],eh*analyse_data_comb[:,3],errorevery=1,linewidth=1,capsize=2,ls="--",fmt="o",markersize=3,color=plt.cm.tab20c(3),label="Etrain, epoch of min comb")[-1][0].set_linestyle('--')
        plt.grid(b=True, which='major', color='#999999', linestyle=':')
        plt.yscale('log')
        plt.xscale('log')
        plt.title("Energy RMSE epoch comparison (ext.Testset)")
        plt.xlabel("training set size")
        plt.ylabel(r"Energy RMSE (meV/atom)")
        plt.legend(loc=1, borderaxespad=0.1)
        plt.savefig("ext_Energy_RMSE_epoch_compare.png",dpi=300)
        
        plt.figure("Forces RMSE epoch comparison")
        bohr = 27.21138602e3/5.2917721067e-1
        plt.errorbar(training_set_sizes,bohr*analyse_data_E[:,8],bohr*analyse_data_E[:,9],errorevery=1,linewidth=1,capsize=2,ls="-",fmt="o",markersize=3,color=plt.cm.tab20c(4),label="Ftest, epoch of min energy")
        plt.errorbar(training_set_sizes,bohr*analyse_data_E[:,6],bohr*analyse_data_E[:,7],errorevery=1,linewidth=1,capsize=2,ls="--",fmt="o",markersize=3,color=plt.cm.tab20c(7),label="Ftrain, epoch of min energy")[-1][0].set_linestyle('--')
        plt.errorbar(training_set_sizes,bohr*analyse_data_F[:,8],bohr*analyse_data_F[:,9],errorevery=1,linewidth=1,capsize=2,ls="-",fmt="o",markersize=3,color=plt.cm.tab20c(8),label="Ftest, epoch of min force")
        plt.errorbar(training_set_sizes,bohr*analyse_data_F[:,6],bohr*analyse_data_F[:,7],errorevery=1,linewidth=1,capsize=2,ls="--",fmt="o",markersize=3,color=plt.cm.tab20c(11),label="Ftrain, epoch of min force")[-1][0].set_linestyle('--')
        plt.errorbar(training_set_sizes,bohr*analyse_data_comb[:,8],bohr*analyse_data_comb[:,9],errorevery=1,linewidth=1,capsize=2,ls="-",fmt="o",markersize=3,color=plt.cm.tab20c(0),label="Ftest, epoch of min comb")
        plt.errorbar(training_set_sizes,bohr*analyse_data_comb[:,6],bohr*analyse_data_comb[:,7],errorevery=1,linewidth=1,capsize=2,ls="--",fmt="o",markersize=3,color=plt.cm.tab20c(3),label="Ftrain, epoch of min comb")[-1][0].set_linestyle('--')
        plt.grid(b=True, which='major', color='#999999', linestyle=':')
        plt.yscale('log')
        plt.xscale('log')
        plt.title("Forces RMSE epoch comparison (ext.Testset)")
        plt.xlabel("training set size")
        plt.ylabel(r"Forces RMSE (meV/$\AA$)")
        plt.legend(loc=1, borderaxespad=0.1)
        plt.savefig("ext_Forces_RMSE_epoch_compare.png",dpi=300)
        
        os.chdir("../")
        print("FINISHED plotting test size dependence.")
        return None

def perform_NNTSSD():
    """This function performs NNTSSD methods according to user-given specifications.
    
    Firstly, it reads the NNTSSD parameters either from the file 'NNTSSD_input.dat' or,
    if not successful, from interactive user input.
    Secondly, it performs a user-given selection of the NNTSSD methods
    Tools.create_training_datasets(), Tools.training_neural_network(), Tools.analyse_learning_curves(), Tools.plot_size_dependence(),
    External_Testdata.predict_test_data(), External_Testdata.analyse_learning_curves() and External_Testdata.plot_test_size_dependence().
    """
    print("**********************************************************************")
    print("NNTSSD - Tools for Neural Network Training Set Size Dependence")
    print("**********************************************************************")
        
    try:
        create_logical,train_logical,analyse_logical,plot_logical,\
        External_test_logical,External_analyse_logical,External_plot_logical,\
        Validation_predict_logical,Validation_plot_logical,\
        set_size_ratios,n_sets_per_size,fix_random_seed_create,random_seed_create,\
        n_epochs,test_fraction,mpirun_cores,write_submission_script_logical,maximum_time,fix_random_seed_train,random_seed_train,\
        dataset_cores,fix_random_seed_predict,random_seed_predict,\
        validation_dataset_cores,fix_random_seed_validation,random_seed_validation\
        = file_input.read_parameters_from_file()
    except:
        sys.exit("ERROR: Something went wrong with your NNTSSD input parameters.")

    nnp_train = "mpirun -np "+str(mpirun_cores)+" ../../../../../../../bin/nnp-train"
    
    print("Performing the following NNTSSD steps:")
    print("  ",create_logical, "\tTools\t Create training datasets")
    print("  ",train_logical, "\tTools\t Training neural network")
    print("  ",analyse_logical, "\tTools\t Analyse learning curves")
    print("  ",plot_logical, "\tTools\t Plot size dependence")
    print("  ",External_test_logical, "\tExternal Testset\t Predicting external testset")
    print("  ",External_analyse_logical, "\tExternal Testset\t Analyse learning curves wrt external testset")
    print("  ",External_plot_logical, "\tExternal Testset\t Plot size dependence wrt external testset")
    print("  ",Validation_predict_logical, "\tValidation\t Predicting and analysing validation dataset")
    print("  ",Validation_plot_logical, "\tValidation\t Plot size dependence wrt validation dataset")
    
    myNNTSSD = Tools()
    myExternal_Testdata = External_Testdata()
    myValidation = Validation()
    
    if create_logical:
        myNNTSSD.create_training_datasets(set_size_ratios,n_sets_per_size,fix_random_seed_create,random_seed_create)
    if train_logical:
        myNNTSSD.training_neural_network(nnp_train,n_epochs,test_fraction,write_submission_script_logical,fix_random_seed_train,random_seed_train,maximum_time)
    if analyse_logical:
        myNNTSSD.analyse_learning_curves()
    if plot_logical:
        myNNTSSD.plot_size_dependence()
    if External_test_logical:
        myExternal_Testdata.predict_test_data(dataset_cores,fix_random_seed_predict,random_seed_predict)
    if External_analyse_logical:
        myExternal_Testdata.analyse_learning_curves()
    if External_plot_logical:
        myExternal_Testdata.plot_test_size_dependence()
    if Validation_predict_logical:
        myValidation.predict_validation_data(validation_dataset_cores,fix_random_seed_validation,random_seed_validation)
    if Validation_plot_logical:
        myValidation.plot_validation_data()
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
