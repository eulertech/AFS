#!/bin/bash

# inputs
install_directory=$1
env_name=$2

if [ -z ${install_directory} ]; then
    echo " ERROR! " 
    echo " "
    echo " USAGE: $0 install_directory env_name"
    echo " "
    echo "   install_directory       : directory containing tarballs"
    echo " "
    echo "   env_name                : name of virtualenv/kernel to make"
    echo " "
    return
elif [ -z ${env_name+x} ]; then
    echo " ERROR! " 
    echo " "
    echo " USAGE: $0 install_directory env_name"
    echo " "
    echo "   install_directory       : directory containing tarballs"
    echo " "
    echo "   env_name                : name of virtualenv/kernel to make"
    echo " "
    return
fi

echo "installing ....."

conda create --name ${env_name} --clone root --use-local --offline --unknown
source activate ${env_name}

echo "Environment created"
echo "Installing packages"

conda install --offline --unknown ${install_directory}/tqdm-4.11.2-py27_0.tar.bz2
pip install ${install_directory}/update_checker-0.16.tar.gz
conda install --offline --unknown ${install_directory}/numpy-1.12.1-py27_0.tar.bz2
conda install --offline --unknown ${install_directory}/scipy-0.19.0-np112py27_0.tar.bz2
conda install --offline --unknown ${install_directory}/scikit-learn-0.18.1-np112py27_1.tar.bz2
pip install ${install_directory}/deap-1.0.2.post2.tar.gz
pip install ${install_directory}/TPOT-0.7.3.tar.gz
pip install ${install_directory}/watermark-1.3.4.tar.gz

echo "rolling Kernel"

python -m ipykernel install --user --name ${env_name} --display-name ${env_name}

echo "Deactivating virtual environment"

source deactivate

echo "list of virtual envs"
conda info --envs