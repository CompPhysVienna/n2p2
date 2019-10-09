#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jun 19 14:38:55 2019

@author: mr
"""
import os

def VSC_run_NNTSSD():
    write_submission_script("python3 ../source/NNTSSD.py",job_name="NNTSSD")
    os.system("sbatch submit.slrm") #to submit the job to VSC
    
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
    submission_script.write("#SBATCH --mail-type=BEGIN,END"+"\n")
    submission_script.write("#SBATCH --mail-user=<a01406236@unet.univie.ac.at>"+"\n")
    submission_script.write("\n"+command)
    submission_script.close()

def main():
    pass
if __name__=="__main__":
    VSC_run_NNTSSD()
