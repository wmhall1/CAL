#!/bin/bash
#PBS -S /bin/bash
#PBS -N join_rst
#PBS -o vtk.o
#PBS -e vtk.e
#PBS -l select=1:mem=600GB
#PBS -l walltime=0:30:00
#PBS -W group_list=s3005
#PBS -m e
#PBS -q ldan

module list
pwd
date

export file=$(ls -Art id0/*.rst| grep -v "dpar" | tail -n 1)
export END=${file:16:4}
echo $file $END
#END=$(( $END - 1 ))

export file=$(ls -Art id0/*rst | head -n 1)
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
echo $START $END
echo $(seq -w $START $END)

for i in $(seq -w $START $END); do
	echo ${i}
	tar cvf Par_Strat3d.${i}.rst.tar id*/Par_Strat3d*.${i}.rst	
	if [ -f Par_Strat3d.${i}.rst.tar ]; then
		rm id*/Par_Strat3d*.${i}.rst
	if [ $machine == 'lfe' ] || [ $machine == "lda" ]; then
		mv Par_Strat3d.${i}.rst.tar ~/$DIR/$DIR-RST/
	fi
	fi
	#mv id*/Par_Strat3d*${i}.rst ../R512-Lx0.4-tau0.1-Pi0.05-Z0.05-G0.1-RST/
done

