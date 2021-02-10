#!/bin/bash

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
# nnp-archive.sh - Archiving of training directories
#
# Cleans up a training directory,
# removes:
#   - train-/testpoints.???.out for all but the first and last epoch.
#   - train-/testforces.???.out for all but the first and last epoch.
#   - function.data
# compresses to nnp-archive.tar.gz:
#   - weights.???.out
#   - neuron-stats.???.out
#   - train-log.out
#   - train/test.data
#   - nnp-train.log.N (where N > 0)
# 
###############################################################################

last_epoch=$(grep TIMING nnp-train.log.0000 | tail -n 1 | awk '{print $2}')

keep_dir="nnp-archive_TMP"
zip_dir="nnp-archive"

echo "Keeping output from epoch 0 and ${last_epoch}..."

if [ -e $keep_dir ]
then
    rm -r $keep_dir
fi
mkdir $keep_dir

if [ -e $zip_dir ]
then
    rm -r $zip_dir
fi
mkdir $zip_dir

epoch_string=$(printf "%06d" 0)

mv trainpoints.${epoch_string}.out $keep_dir
mv testpoints.${epoch_string}.out $keep_dir
mv trainforces.${epoch_string}.out $keep_dir
mv testforces.${epoch_string}.out $keep_dir
mv weights*.${epoch_string}.out $keep_dir
mv neuron-stats.${epoch_string}.out $keep_dir
mv nnp-train.log.0000 $keep_dir

epoch_string=$(printf "%06d" $last_epoch)

mv trainpoints.${epoch_string}.out $keep_dir
mv testpoints.${epoch_string}.out $keep_dir
mv trainforces.${epoch_string}.out $keep_dir
mv testforces.${epoch_string}.out $keep_dir
mv weights*.${epoch_string}.out $keep_dir
mv neuron-stats.${epoch_string}.out $keep_dir

mv weights*.out $zip_dir
mv neuron-stats*.out $zip_dir
mv train-log.out $zip_dir
mv train.data $zip_dir
mv test.data $zip_dir
mv nnp-train.log.* $zip_dir

if [ -e ${zip_dir}.tar.gz ]
then
    echo "ERROR: Archive directory already exists, aborting."
    exit 1
fi
tar -czvf ${zip_dir}.tar.gz ${zip_dir}/*
rm -r ${zip_dir}

rm trainforces.*
rm testforces.*
rm trainpoints.*
rm testpoints.*
rm function.data

mv ${keep_dir}/* .

rm -r ${keep_dir}
