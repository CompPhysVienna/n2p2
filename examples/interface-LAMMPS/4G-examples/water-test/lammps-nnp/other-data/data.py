import os
import sys


def read_lammps(file):

    with open(file) as f:
        lines = f.readlines()

    a1 = 17 #first atom
    a2 = 8657 #last atom

    new_lines = []

    for i in range(0,a1):
        new_lines.append(lines[i])

    # first extract the number of atoms in the structure
    for i in range(a1,a2):
        line = lines[i].split()
        new_line = []
        new_line.append(line[0])
        new_line.append(line[1])
        new_line.append('0.0') #initial charge
        new_line.append(line[2])
        new_line.append(line[3])
        new_line.append(line[4])
        new_line.append('\n')
        new_lines.append(" ".join(new_line))

    return new_lines

def write_lammps(file_lines,file):

    with open(file,'w') as f:
        f.writelines(file_lines)


file = 'water.data'
file2 = 'water_charge.data'
new_file_lines = read_lammps(file)
write_lammps(new_file_lines,file2)
