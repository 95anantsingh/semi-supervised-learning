import os


# Base dir
base_dir = '/home/as14229/NYU_HPC/semi-supervised-learning/scripts/'

jobs = [69,100]

new_time = "13:00:00"

update_file = os.path.join(base_dir,'update_jobs.sh')
base_command = "scontrol update jobid=$(sacct -n -X --format jobid --name"
base_command2 ="TimeLimit="

with open(update_file,'w') as file:
    file.write('#!/bin/bash\n\n')
    for k in range(jobs[0],jobs[1]+1):
        file.write(base_command+' Job'+str(k)+') '+base_command2+new_time+'\n')
os.chmod(update_file, 0o740)
