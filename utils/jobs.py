
import os

jobs_dir = '/home/as14229/NYU_HPC/semi-supervised-learning/scripts/jobs'
logs_dir = '/home/as14229/NYU_HPC/semi-supervised-learning/scripts/logs'


time_per_job = "5:30:00" 

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

command = "python self_train_v3.0.py -s 'pc_top_k' -sv 100 -md 'yahoo_answers' -od 'ag_news' 'dbpedia_14' 'yelp_review_full' -us 15000 -lbs 'equal'"

# scancel='\nscancel $SLURM_JOB_ID'


runs = ['Run1','Run2','Run3','Run4','Run5']
labeled_sizes = ['100','500','1000','5000']
unlabeled_in_domain_ratios = ['0.05','1.0']


jobs_per_gpu = 2

jobs = []
for run in runs:
    for ubr in unlabeled_in_domain_ratios:
        for ls in labeled_sizes:            
            jobs.append(command + ' -n "' + run + '" -ls ' + ls + ' -ubr ' + ubr +'\n')


for i,j in enumerate(range(0,len(jobs),2),1):
    with open(os.path.join(jobs_dir,'job'+str(i)+'.sbatch'),'w') as file:
        file.write(sbatch_header)
        file.write(job_name_directive+str(i)+'\n')
        file.write(output_file_directive+str(i)+'.log\n')
        file.write(command_header)
        file.write(jobs[j])
        file.write(jobs[j+1])

schedule_file = os.path.join(jobs_dir,'schedule_jobs.sh')

with open(schedule_file,'w') as file:
    file.write('#!/bin/bash\n\n')
    for k in range(1,i+1):
        file.write('sbatch '+jobs_dir+'/job'+str(k)+'.sbatch\n')

os.chmod(schedule_file, 0o740)


# Launch
# import subprocess

# bashCommand = "bash "+schedule_file
# process = subprocess.Popen(bashCommand.split(), stdout=subprocess.PIPE)
# output, error = process.communicate()

# cancel
# jobs = []
# with open('cancel.sh','w') as file:
#     file.write('#!/bin/bash\n\n')
#     for job in jobs:
#         file.write('scancel '+str(job)+'\n')

# os.chmod('cancel.sh', 0o740)