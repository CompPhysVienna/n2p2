#!/bin/bash

while read line
do
    words=($line)
    old=${words[0]}
    new=${words[1]}
    echo "Replacing \"${old}\" by \"${new}\""
    git mv ${old}.h ${new}.h
    git mv ${old}.cpp ${new}.cpp
done < ./replacements.txt
