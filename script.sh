#!/bin/sh

apt-get update
apt-get install -y git
git clone https://github.com/Yiching-Gemini/Gemini-YC.git
cd Gemini-YC
git checkout user-testing-v1.7
pip install boto
python download_mnist_data.py
tar -C /mnt -xvf /mnt/dataset.gz
python pytorch-mnist-0518.py
