#!/bin/bash
#$ -S /bin/bash
#$ -q kepler-long.q
#$ -l h_rt=1:00:00,ngpus=2
#$ -j y
#$ -cwd

. /etc/profile.d/modules.sh
module use /hits/fast/its/doserbd/modules
module use /home/doserbd/modules
module use /hits/mbm/modules

module purge
module load sge
module load gcc/4.8.3
module load boost/1.55
module load cmake/3.0.2
module load python/2.7.3
module load gtest/1.7.0
module list

mkdir Build_Release_Gnu
cd Build_Release_Gnu

cmake\
 -DPYTHON_INCLUDE_DIR=/cm/shared/apps/python/2.7.3/include/python2.7\
 -DPYTHON_LIBRARY=/cm/shared/apps/python/2.7.3/lib\
 ..

make -j 12
make
make test

