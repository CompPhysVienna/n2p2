#!/usr/bin/env python3

# n2p2 - A neural network potential package
# Copyright (C) 2018 Andreas Singraber (University of Vienna)
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <https://www.gnu.org/licenses/>.

###############################################################################
#
# NNP converter between n2p2 versions.
# 
# This converter is helpful when symmetry function ordering and definition
# strings (keyword symfunction_short in input.nn) change between versions (e.g.
# during development). It requires the "old" NNP (input.nn, scaling.data,
# weights.???.data) and a small, representative data set (input.data).
# Furthermore, it requires the binary nnp-scaling for each n2p2 version. It
# will first apply changes to the input.nn file (adapt convert_input_nn). Then
# it computes the scaling data for both versions and tries to find matches (see
# ScalingData.__eq__ for a matching criterion). Finally, the script will output
# converted versions of scaling.data and weights.???.data.
# 
###############################################################################

import sys
import os
import glob
import shutil
import numpy as np
import re
from typing import List, Dict
from collections import OrderedDict

# Settings: edit path to data and two different n2p2 bin directories here:
dirs = ["PSF_for_DMABN", "PSF_for_ethyl_benzene", "PSF_for_anisole",
        "PAS_for_DMABN", "PAS_for_ethyl_benzene", "PAS_for_anisole"]
old_bin = "~/local/src/n2p2-singraber_oldpsf/bin/"      # branch: polynomial_symmetry_functions commit: 0b17662fdd9870818ac8060f3342991daa36b7ca
new_bin = "~/local/src/n2p2-singraber_alternative/bin/" # branch: polynomial_symmetry_functions commit: b514172eb4816e8791b5a488e671de7a3e1ba292

# Store one scaling data line, allow comparison via equal operator.
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

# Read in one scaling data file and return dictionary with scaling data for each element.
def read_scaling_data(file_name):
    data = {}
    f = open(file_name, "r")
    for line in f:
        ls = line.split()
        if len(ls) > 0 and ls[0][0] != "#":
            sd = ScalingData(line)
            if sd.element not in data:
                data[sd.element] = []
            data[sd.element].append(sd)
    return data

def convert_scaling_data(old_file, new_file, data: Dict[int, List[ScalingData]]):
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
    countw = 0
    countn = 0
    for i in order:
        for line in weights[i]:
            if countw % neurons == 0:
                countn += 1
            countw += 1
            ls = split_columns(line)
            ls[2] = "{0:{1}d}".format(countw, len(ls[2]))
            ls[4] = "{0:{1}d}".format(countn, len(ls[4]))
            fo.write("".join(ls))
    # Copy rest of weights file.
    fi.seek(pos)
    for line in iter(fi.readline, ''):
        fo.write(line)

    fi.close()
    fo.close()


def main():
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
            raise RuntimeError("Different number of elements in symmetry function scaling data.")
        elements = sorted(scaling_new.keys())
        index_in_new = {}
        index_in_old = {}
        for e in elements:
            if len(scaling_old[e]) != len(scaling_new[e]):
                raise RuntimeError("Different number of symmetry functions in scaling data.")
            nsf = len(scaling_new[e])
            print("Searching matches for " + str(nsf) + " symmetry functions of element " + str(e) + "...", end="")
            index_in_new[e] = [None] * nsf # Index of old SF in scaling_new.
            index_in_old[e] = [None] * nsf # Index of new SF in scaling_old.
            for ind_n, n in enumerate(scaling_new[e]):
                for ind_o, o in enumerate(scaling_old[e]):
                    if n == o:
                        index_in_new[e][ind_o] = ind_n
                        index_in_old[e][ind_n] = ind_o
                        break
            for o, n in enumerate(index_in_new[e]):
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
        # Reorder scaling data according to new SF list.
        scaling_conv = {}
        for e in elements:
            scaling_conv[e] = [scaling_orig[e][i] for i in index_in_old[e]]
        os.chdir(conv_dir)
        convert_scaling_data("../" + d + "/scaling.data", "scaling.data", scaling_conv)
        for e, wf in weight_dict.items():
            shutil.copy2("../" + d + "/" + wf, wf + ".old")
            convert_weights_file(wf + ".old", wf, index_in_old[e])
            os.remove(wf + ".old")
        os.system(new_bin + "nnp-dataset 0 > /dev/null")
        os.chdir("..")

        print("-------------------------------------------------------------------------------")


if __name__ == "__main__":
    main()
