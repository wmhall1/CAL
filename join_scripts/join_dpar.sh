#!/bin/bash
#PBS -S /bin/bash
#PBS -N join_dpar
#PBS -o dpar.o
#PBS -e dpar.e
#PBS -l select=1:mem=600GB
#PBS -l walltime=2:00:00
#PBS -W group_list=s3005
#PBS -m e
#PBS -q ldan

module list
pwd
date

export file=$(ls -Art id0/*.dpar.vtk | tail -n 1)
export END=${file:16:4}
echo $file $END

export file=$(ls -Art id0/*dpar.vtk | head -n 1)
export START=${file:16:4}
echo $START

export machine=$(hostname)
export machine=${machine:0:3}

export DIR=${PWD##*/}
if [[ $DIR == *"-SG" ]]; then
	DIR=${DIR::-3}
fi
echo $DIR

#./join_vtk -o <outfile.vtk> infile1.vtk infile2.vtk ...

for i in $(seq -w $START $END); do
	#echo ${i} ${#i}
	./join_vtk -o Par_Strat3d.${i}.dpar.vtk id*/*${i}.dpar.vtk
	
	if [ -f Par_Strat3d.${i}.dpar.vtk ]; then
		rm id*/*${i}.dpar.vtk
	if [ $machine == 'lfe' ] || [ $machine == "lda" ]; then
		mv Par_Strat3d.${i}.dpar.vtk ~/$DIR/$DIR-VTK/
	fi
	fi
done

