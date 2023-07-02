#!/bin/sh
set -eu

apt-get update
apt-get install -y libgl1-mesa-dev
apt-get install -y libx11-6
apt-get install -y openjdk-8-jdk
apt-get install -y x11-xserver-utils
apt-get install -y xvfb
apt-get clean

export JAVA_HOME=/usr/lib/jvm/java-1.8.0-openjdk-amd64
/home/paul/miniconda3/envs/wssm/bin/python -m 

#pip3ainstall minerl==0.4.4
