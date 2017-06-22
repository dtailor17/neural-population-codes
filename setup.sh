#!/usr/bin/env bash

# Get directory containing bash script
WORKSPACE="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

VENV_HOME=$WORKSPACE/venv

# Delete previously built virtualenv
if [ -d $VENV_HOME ]; then
	echo "Removed existing virtualenv"
    rm -rf $VENV_HOME
fi

# Create virtualenv
echo "Setting up virtualenv"
virtualenv $VENV_HOME
. $VENV_HOME/bin/activate

# Install required packages
echo "Ready to install packages"
echo "Installing numpy"; pip install -q numpy
echo "Installing matplotlib"; pip install -q matplotlib
echo "Installing scipy"; pip install -q scipy
echo "Installing jupyter"; pip install -q IPython==5.3; pip install jupyter
