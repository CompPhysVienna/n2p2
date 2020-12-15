#!/usr/bin/env python3

import sys
import os
import glob
import shutil
import numpy as np
import re
from typing import List
from collections import OrderedDict

class ScalingData:
    def __init__(self, line):
        self.line = line
        split = line.split()
        self.element = int(split[0])
        self.index = int(split[1])
        self.minimum = float(split[2])
        self.maximum = float(split[3])
        self.mean = float(split[4])
        self.sigma = float(split[5])
    # Define here how to find "equal" scaling data.
    def __eq__(self, other):
        if isinstance(other, self.__class__):
            if self.element != other.element: return False
            if np.abs(self.mean - other.mean) > 1.0E-10: return False
            if np.abs(self.sigma - other.sigma) > 1.0E-10: return False
            if np.abs(self.minimum - other.minimum) > 1.0E-10: return False
            if np.abs(self.maximum - other.maximum) > 1.0E-10: return False
            return True
        else:
            return False

# Split list of scaling data by elements.
def element_split_scaling(data: List[ScalingData], order: List[int]):
    start = []
    end = []
    for i, d in enumerate(data):
        if i == 0:
            start.append(0)
            element = d.element
        else:
            if element != d.element:
                end.append(i)
                start.append(i)
                element = d.element
    end.append(len(data))
    splitScaling = OrderedDict()
    splitOrder = OrderedDict()
    for i, (s, e) in enumerate(zip(start, end)):
        splitScaling[i + 1] = data[s:e]
        splitOrder[i + 1] = order[s:e]
    return splitScaling, splitOrder

# Split string into columns but keep (multiple) separators in.
def split_columns(line, delimiter=' '):
    r = [""]
    content = False
    for c in line:
        if c != delimiter:
            content = True
        elif c == delimiter and content:
            r.append("")
            content = False
        r[-1] += c
    return r

def read_scaling_data(file_name):
    data = []
    f = open(file_name, "r")
    for line in f:
        ls = line.split()
        if len(ls) > 0 and ls[0][0] != "#":
            data.append(ScalingData(line))
    return data

def convert_scaling_data(old_file, new_file, data: List[List[ScalingData]]):
    fi = open(old_file, "r")
    fo = open(new_file, "w")
    for line in fi:
        ls = line.split()
        if len(ls) > 0 and ls[0][0] == "#":
            fo.write(line)
    fi.close()
    for e, element_data in data.items():
        count = 1
        for d in element_data:
            ls = split_columns(d.line)
            ls[1] = "{0:{1}d}".format(count, len(ls[1]))
            fo.write("".join(ls))
            count += 1
    fo.close()

def convert_input_nn(old_file, new_file):
    fi = open(old_file, "r")
    fo = open(new_file, "w")
    for line in fi:
        ls = line.split()
        if len(ls) > 0 and ls[0] == "symfunction_short":
            sftype = int(ls[2])
            if sftype > 100:
                subtype = "p2a"
            else:
                subtype = "p2"
            if sftype in [28, 280]:
                ls[2] = "20"
                ls.append(subtype)
            if sftype in [89, 890]:
                ls[2] = "22"
                rc = ls[-1]
                ls[-1] = ls[-2]
                ls[-2] = ls[-3]
                ls[-3] = rc
                ls.append(subtype)
            if sftype in [99, 990]:
                ls[2] = "21"
                rc = ls[-1]
                ls[-1] = ls[-2]
                ls[-2] = ls[-3]
                ls[-3] = rc
                ls.append(subtype)
            line = " ".join(ls) + "\n"
        fo.write(line)
    fi.close()
    fo.close()

def convert_weights_file(old_file, new_file, order):
    fi = open(old_file, "r")
    fo = open(new_file, "w")
    for line in iter(fi.readline, ''): # Need this construction because of tell()
        ls = line.split()
        if len(ls) > 0 and ls[0][0] == "#":
            fo.write(line)
            pos = fi.tell()
        else:
            break # Break out of loop if header is copied.
    # Continue reading old file to check how many neurons are in second layer.
    fi.seek(pos)
    neurons = 0
    for line in iter(fi.readline, ''):
        ls = line.split()
        if int(ls[4]) > 1:
            break
        neurons += 1
    fi.seek(pos)
    # Now store first-to-second layer weights to rearrange afterwards.
    weights = {i : [] for i in range(len(order))}
    for line in iter(fi.readline, ''):
        ls = line.split()
        # Stop when biases are reached.
        if ls[1] == "b":
            break
        input_neuron = int(ls[4]) - 1
        weights[input_neuron].append(line)
        pos = fi.tell()
    # Print first-to-second layer weights in new order.
    for i in order:
        for line in weights[i]:
            fo.write(line)
    # Copy rest of weights file.
    fi.seek(pos)
    for line in iter(fi.readline, ''):
        fo.write(line)

    fi.close()
    fo.close()


def main():
    old_bin = "~/local/src/n2p2-singraber_oldpsf/bin/"
    new_bin = "~/local/src/n2p2-singraber_alternative/bin/"
    dirs = ["PSF_for_DMABN", "PSF_for_ethyl_benzene", "PSF_for_anisole",
            "PAS_for_DMABN", "PAS_for_ethyl_benzene", "PAS_for_anisole"]
    dirs = dirs[0:1]

    # Do not create data again if requested.
    create_data = True
    if len(sys.argv) > 1 and sys.argv[1] == "-o":
        create_data = False

    for d in dirs:
        print("Processing dir: " + d)

        # Create temporary working dir for old version.
        old_dir = "old_" + d
        if create_data:
            if os.path.exists(old_dir):
                shutil.rmtree(old_dir)
            shutil.copytree(d, old_dir)
        os.chdir(old_dir)
        # Check for weight files and create element -> weight file dictionary.
        weight_files = sorted(glob.glob("weights.*.data"))
        weight_dict = {}
        for i, wf in enumerate(weight_files):
            weight_dict[i + 1] = wf
            print("Found element " + str(i + 1) + ": " + wf)
        # Get scaling of old symmetry functions.
        if create_data: os.system(old_bin + "nnp-scaling 100 > /dev/null && rm sf.* function.data")
        scaling_old = read_scaling_data("scaling.data")
        os.chdir("..")

        # Create working dir for new version.
        new_dir = "new_" + d
        if create_data:
            if os.path.exists(new_dir):
                shutil.rmtree(new_dir)
            os.mkdir(new_dir)
            shutil.copy2(old_dir + "/input.data", new_dir)
            shutil.copy2(old_dir + "/input.nn", new_dir + "/input.nn.old")
        os.chdir(new_dir)
        # Convert symmetry function lines in input.nn.
        convert_input_nn("input.nn.old", "input.nn")
        # Get scaling of new symmetry functions.
        if create_data: os.system(new_bin + "nnp-scaling 100 > /dev/null && rm sf.* function.data")
        scaling_new = read_scaling_data("scaling.data")
        os.chdir("..")

        # Now process the scaling data to find matches.
        if len(scaling_old) != len(scaling_new):
            raise RuntimeError("Inconsistent symmetry function scaling data.")
        nsf = len(scaling_new)
        print("Searching matches for " + str(nsf) + " symmetry functions...", end="")
        index_in_new = [None] * nsf # Index of old SF in scaling_new.
        index_in_old = [None] * nsf # Index of new SF in scaling_old.
        for ind_n, n in enumerate(scaling_new):
            for ind_o, o in enumerate(scaling_old):
                if n == o:
                    index_in_new[ind_o] = ind_n
                    index_in_old[ind_n] = ind_o
                    break
        for o, n in enumerate(index_in_new):
            if n is None:
                raise RuntimeError("No match found for index in old settings: " + str(o) + "\nLine: " + scaling_old[o].line)
        print("completed!")

        # Create dir for final converted version.
        conv_dir = "converted_" + d
        print("Writing converted NNP in " + conv_dir + ".")
        if os.path.exists(conv_dir):
            shutil.rmtree(conv_dir)
        os.mkdir(conv_dir)
        shutil.copy2(old_dir + "/input.data", conv_dir)
        shutil.copy2(new_dir + "/input.nn", conv_dir + "/input.nn")
        # Read original scaling data (not scaling_old which is only for a subset).
        scaling_orig = read_scaling_data(d + "/scaling.data")
        os.chdir(conv_dir)
        data, order = element_split_scaling([scaling_orig[i] for i in index_in_old], index_in_old)
        convert_scaling_data("../" + d + "/scaling.data", "scaling.data", data)
        for e, wf in weight_dict.items():
            shutil.copy2("../" + d + "/" + wf, wf + ".old")
            convert_weights_file(wf + ".old", wf, order[e])
        os.chdir("..")

        print("-------------------------------------------------------------------------------")


if __name__ == "__main__":
    main()
