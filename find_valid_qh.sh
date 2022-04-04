#!/bin/bash
# DISTRIBUTED COPY OF PROJECT INTO LATTICE MACHINES FROM KIWIS

while IFS= read -r line; do
    ssh -n $line "cd /s/$line/a/nobackup/galileo/stip-images/co-3month/Sentinel-2/; ls"
done < "$1"
