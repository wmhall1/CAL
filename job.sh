#!/bin/bash

# Copy/paste this job script into a text file and submit with the command:
#    sbatch thefilename

#SBATCH --time=00:15:00   # walltime limit (HH:MM:SS)
#SBATCH --nodes=1   # number of nodes
#SBATCH --ntasks-per-node=4 
#SBATCH --job-name="images"
#SBATCH --output="images.o%j" # job standard output file (%j replaced by job id)
#SBATCH --error="images.e%j" # job standard error file (%j replaced by job id)

# LOAD MODULES, INSERT CODE, AND RUN YOUR PROGRAMS HERE
# Loads most recent Python installed on cluster. See all modules with command module avail

module load python

VENV='images'
if [ ! -f "$VENV/bin/activate" ]; then
	echo 'Initializing Pyhton environment...'
	python3 -m venv images
	source ./images/bin/activate
	python -m pip --no-cache-dir install numpy pandas h5py matplotlib
else
	source ./images/bin/activate
fi


#SCRATCH_DIRECTORY=./

#OUTPUT_DIRECTORY=./$(date +%F)_${SLURM_JOBID}

# Execute script and redirect output to file located in out folder

echo "now processing ${SLURM_JOB_NAME} - ${SLURM_JOBID}"

#Usage: python3 image_process.py {number of tasks} {plotting function name} -{optional options} file1 file2
#Example: python3 image_process.py 4 "xy_plot" *.athdf
python3 image_process.py ${SLURM_NTASKS} "xy_plot" *.athdf

# finish
exit 0
