import os
import subprocess


jobs_per_gpu = 1                 # Number of jobs per GPU
time_per_job = "13:00:00"        # Time per job

# Base dir
base_dir = '/home/as14229/NYU_HPC/semi-supervised-learning/scripts/'

jobs_dir = os.path.join(base_dir,'jobs')
logs_dir = os.path.join(base_dir,'logs')

# Remove older jobs
if os.path.exists(jobs_dir): 
    for file in os.listdir(jobs_dir): os.remove(os.path.join(jobs_dir,file))
    os.removedirs(jobs_dir)

os.makedirs(jobs_dir,exist_ok=True)
os.makedirs(logs_dir,exist_ok=True)

sbatch_header = "#!/bin/bash\n\
\n\
#SBATCH --nodes=1               \n\
#SBATCH --ntasks-per-node=1     \n\
#SBATCH --cpus-per-task=4       \n\
#SBATCH --time="+time_per_job+"          \n\
#SBATCH --mem=32GB              \n\
#SBATCH --gres=gpu:a100:1       \n"

job_name_directive =  "#SBATCH --job-name=Job"
output_file_directive = "#SBATCH --output="+logs_dir+'/job'

command_header = "\nmodule purge\n\
source ~/.bashrc\n\
conda activate NLP\n\
cd /home/as14229/NYU_HPC/semi-supervised-learning/\n\
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/scratch/as14229/envs_dirs/NLP/lib/\n\n"

# Main Commmand
command = "python self_train_v3.1.py -s 'pc_top_k' -sv 100 -md 'yahoo_answers' -od 'ag_news' 'dbpedia_14' 'yelp_review_full' -us 50000 -lbs 'equal'"

runs = ['Run3','Run4']
labeled_sizes = ['100']
unlabeled_in_domain_ratios = ['0.05','0.25','0.50','0.75','1.0']


# Get all the jobs
jobs = []
for run in runs:
    for ubr in unlabeled_in_domain_ratios:
        for ls in labeled_sizes:            
            jobs.append(command + ' -n "' + run + '" -ls ' + ls + ' -ubr ' + ubr +'\n')


# Make sbatch files
for i,j in enumerate(range(0,len(jobs),jobs_per_gpu),1):
    with open(os.path.join(jobs_dir,'job'+str(i)+'.sbatch'),'w') as file:
        file.write(sbatch_header)
        file.write(job_name_directive+str(i)+'\n')
        file.write(output_file_directive+str(i)+'.log\n')
        file.write(command_header)
        for k in range(j,j+jobs_per_gpu):
            file.write(jobs[k])


# Make schedule file
schedule_file = os.path.join(base_dir,'schedule_jobs.sh')
with open(schedule_file,'w') as file:
    file.write('#!/bin/bash\n\n')
    for k in range(1,i+1):
        file.write('sbatch '+jobs_dir+'/job'+str(k)+'.sbatch\n')
os.chmod(schedule_file, 0o740)


# Make cancel file
cancel_file = os.path.join(base_dir,'cancel_jobs.sh')
base_command = "scancel $(sacct -n -X --format jobid --name"
with open(cancel_file,'w') as file:
    file.write('#!/bin/bash\n\n')
    for k in range(1,i+1):
        file.write(base_command+' Job'+str(k)+')\n')
os.chmod(cancel_file, 0o740)


# Launch
# bashCommand = "bash "+schedule_file
# process = subprocess.Popen(bash_command.split(), stdout=subprocess.PIPE)
# output, error = process.communicate()
# print(output)