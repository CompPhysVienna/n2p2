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
def element_split_scaling(data: List[ScalingData]):
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
    split = OrderedDict()
    for i, (s, e) in enumerate(zip(start, end)):
        split[i + 1] = data[s:e]
    return split

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

def write_scaling_data(old_file, new_file, data: List[List[ScalingData]]):
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
        data = element_split_scaling([scaling_old[i] for i in index_in_old])
        write_scaling_data(d + "/scaling.data", conv_dir + "/scaling.data", data)
        os.chdir(conv_dir)

        os.chdir("..")

        print("-------------------------------------------------------------------------------")


if __name__ == "__main__":
    main()
