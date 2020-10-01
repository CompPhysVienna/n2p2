#!/bin/bash

file_name=$1
n=0
while read line
do
    words=($line)
    old=${words[0]}
    new=${words[1]}
    echo "Replacing \"${old}\" by \"${new}\""
    if [ "$n" -eq "0" ]
    then
        cp ${file_name} ${file_name}.orig
    fi
    sed -i'' "s/${old}/${new}/g" ${file_name}
    n=$((n+1))
done < ./replacements.txt
