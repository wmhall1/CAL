#!/bin/bash
#PBS -S /bin/bash
#PBS -N join_lis
#PBS -o lis.o
#PBS -e lis.e
#PBS -l select=1:ncpus=36:mpiprocs=36:mem=600GB
#PBS -l walltime=0:30:00
#PBS -W group_list=s3005
#PBS -m e
#PBS -q ldan

module list
pwd
date

export file=$(ls -Art id0/*lis | tail -n 1)
export END=${file:16:4}
echo $END

export file=$(ls -Art id0/*lis | head -n 1)
export START=${file:16:4}
echo $START

export machine=$(hostname)
export machine=${machine:0:3}

export DIR=${PWD##*/}
if [[ $DIR == *"-SG" ]]; then
        DIR=${DIR::-3}
fi
echo $DIR


#./join_lis -p 2048 -o Par_Strat3d -i Par_Strat3d -s ds -d ./ -f $START:$END

for i in $(seq -w $START $END); do
    ./join_lis -p 512 -o Par_Strat3d -i Par_Strat3d -s ds -d ./ -f ${i}:${i}
    echo "rm id*/*${i}.ds.lis"
    if [ -f Par_Strat3d.${i}.ds.lis ]; then
    	rm id*/*${i}.ds.lis
    if [ $machine == "lfe" ] || [ $machine == "lda" ]; then
	mv Par_Strat3d.${i}.ds.lis ~/$DIR/$DIR-LIS
    fi
    fi
done

