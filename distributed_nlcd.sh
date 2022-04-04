#!/bin/bash
# DISTRIBUTED COPY OF PROJECT INTO LATTICE MACHINES FROM KIWIS

while IFS= read -r line; do
    echo $line
    scp -rq /s/chopin/e/proj/sustain/sapmitra/kubernetes_project/TransferLearning $line:/s/chopin/b/grad/sapmitra/
    ssh -n $line "export PYTHONPATH='/s/chopin/b/grad/sapmitra/TransferLearning:/s/chopin/b/grad/sapmitra/.local/lib/python3.6/site-packages/:/usr/local/python-env/py36/lib/python3.6/site-packages/';python3.6 /s/chopin/b/grad/sapmitra/TransferLearning/nlcd_main.py"
    echo "=========================================================================="
done < "$1"
