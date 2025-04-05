#!/bin/bash

ARCH=$(uname -m)
if [ "$ARCH" = "arm64" ]; then
    INSTALLER="Miniconda3-latest-MacOSX-arm64.sh"
else
    INSTALLER="Miniconda3-latest-MacOSX-x86_64.sh"
fi

echo "Downloading $INSTALLER for architecture $ARCH..."
curl -O https://repo.anaconda.com/miniconda/$INSTALLER

echo "Installing Miniconda..."
bash $INSTALLER -b

rm $INSTALLER
echo "Installation complete."

source ~/.bashrc
conda env create -f environment.yml
conda activate pydantic_ai_demo