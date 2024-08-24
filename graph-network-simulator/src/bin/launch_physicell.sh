#!/bin/sh

PHYSICELL_PATH=/home/davide/git/innovation/cancer-simulation/PhysiCell

# PROJECT=biorobots-sample
# PHYSICELL_SCRIPT=biorobots

# PROJECT=virus-macrophage-sample
# PHYSICELL_SCRIPT=virus-sample

PROJECT=heterogeneity-sample
PHYSICELL_SCRIPT=heterogeneity

# PROJECT=cancer-biorobots-sample
# PHYSICELL_SCRIPT=cancer_biorobots

DATA_PATH=/home/davide/git/innovation/cancer-simulation/graph-network-simulator/src/data
DATA_PROCESS_SCRIPT=/home/davide/git/innovation/cancer-simulation/graph-network-simulator/src/bin/process_physicell_output.py

export PYTHONPATH=$PYTHONPATH:/home/davide/git/innovation/cancer-simulation/graph-network-simulator/src/lib

echo "Setup Physicell for project: $PROJECT"
cd $PHYSICELL_PATH && make data-cleanup reset clean > /dev/null
cd $PHYSICELL_PATH && make $PROJECT > /dev/null
cd $PHYSICELL_PATH && make > /dev/null

START=0
END=10

for i in $(seq $START $END)
do 
    echo "Launching experiment: $i"
    cd $PHYSICELL_PATH && ./$PHYSICELL_SCRIPT > /dev/null

    mkdir -p $DATA_PATH/${PHYSICELL_SCRIPT}/raw

    echo "Processing output"
    python3 $DATA_PROCESS_SCRIPT \
        --physicell-output $PHYSICELL_PATH/output \
        --output-file $DATA_PATH/${PHYSICELL_SCRIPT}/raw/${PHYSICELL_SCRIPT}_${i}.csv > /dev/null 

    echo "Cleanup environment"
    cd $PHYSICELL_PATH && make data-cleanup > /dev/null
done
