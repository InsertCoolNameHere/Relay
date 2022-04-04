#!/bin/bash
# DISTRIBUTED COPY OF PROJECT INTO LATTICE MACHINES FROM KIWIS

while IFS= read -r line; do
    echo $line
    scp -rq $line:/s/$line/a/nobackup/spectral/sapmitra/nlcd*.log /s/chopin/e/proj/sustain/sapmitra/kubernetes_project/nlcd_vectors/
    echo "=========================================================================="
done < "$1"
