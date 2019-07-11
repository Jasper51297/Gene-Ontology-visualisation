#!/bin/bash
echo "Checking dependencies..."
sudo add-apt-repository "deb http://archive.ubuntu.com/ubuntu $(lsb_release -sc) universe multiverse"
command pip3 -v >/dev/null 2>&1 || { echo >&2 "Pip is required to run this script but it's not installed!"; sudo apt-get update -y; sudo apt-get install python3-pip -y; }
pip3 install bokeh==0.12.16
pip3 install colorcet
pip3 install numpy==1.14.3
pip3 install pandas==0.22.0
pip3 install numba==0.37.0
pip3 install scipy==0.19.1
pip3 install cvxopt
