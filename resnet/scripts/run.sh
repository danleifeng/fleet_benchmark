#!/bin/bash
# update train script
cd /fleet_benchmark/resnet/
git pull origin master

#download data
cd /
aws s3 cp s3://s3-n1/ImageNet.tar /
tar xf /ImageNet.tar
#run
cd /fleet_benchmark/resnet/
./scripts/train_gpu.sh
