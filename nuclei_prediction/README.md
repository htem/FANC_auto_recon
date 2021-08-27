# summer2021
I forgot md syntax, so its very messy for now. I'll clean this file later.


## Approach
We are aimging to make a list of nucleus segIDs and their neuron segIDs. [repo for rotation students](https://github.com/bjm5164/rotation_projects)

## Usage
nuclei prediction
docker 

# 1. Docker 
[Docker](https://qiita.com/Canon11/items/e6f64597d82dbf88f75f) Building Docker image usually takes one or two minutes.


# 2. setup venv
[venv](https://qiita.com/Gattaca/items/80a5d36673ba2b6ef7f0)

'''sh
python3 -V # check version
python3 -m venv nuclei
. nuclei/bin/activate
(fanc) vscode ➜ ~ $ pip install -r requirements.txt
(fanc) vscode ➜ ~ $ ipython kernel install --user --name=nuclei
'''

'''sh
(fanc) vscode ➜ ~ $ deactivate
'''

3. auth

[cloud-volume wiki]()

save your API at ~/.cloudvolume/secrets/cave-secret.json

/Users/Sumiya

secret should be created first cloud-volume wiki


mkdir -p ~/.cloudvolume/secrets/

use authentication_utils

'''sh
docker ps -a # check your container id
docker cp ~/.cloudvolume/secrets/cave-secret.json 3bc6ffc0d501:/home/vscode/.cloudvolume/secrets/
docker cp ~/.cloudvolume/segmentations.json 3bc6ffc0d501:/home/vscode/.cloudvolume/
'''

4. Issues
pip uninstall dataclasses 
rmtree? conflict with cache.flush

Note: [gitignore](https://qiita.com/anqooqie/items/110957797b3d5280c44f
https://www.curict.com/item/fe/fe45741.html)


https://stackoverflow.com/questions/32578224/cloning-a-branch-remote-branch-branch-99-not-found-in-upstream-origin
rsync

mkdir Output for pandas


/home/skuroda/.local/bin/pip

/home/skuroda/.local/bin/pip install task-queue

git clone https://github.com/htem/FANC_auto_recon.git
cd FANC_auto_recon
git checkout sumiya-nuclei

/n/groups/htem/users/skuroda

screen -d


cat /proc/cpuinfo |grep processor

topでCPU使用率


rsync ~/MN202106070000.csv skuroda@catmaid3.hms.harvard.edu:~/
rsync ~/MN202106070000.csv htem:~/

LHS75200Seeds.csv
RHS75200Seeds.csv

15199

rsync htem:~/full_VNC_soma_20210824.csv ~/
rsync skuroda@catmaid3.hms.harvard.edu:~/full_VNC_soma_20210824.csv ~/

rsync htem:~/Premotor_20210819.csv ~/
rsync skuroda@catmaid3.hms.harvard.edu:~/Premotor_20210819.csv ~/




python3.6 -c 'import count_sv; count_sv.run_local()'

python3.6 -c 'import count_sv; count_sv.run_local()' -i ~/missing_count.txt

cache.flush not work


https://stackoverflow.com/questions/68191392/password-authentication-is-temporarily-disabled-as-part-of-a-brownout-please-us


python3.6 -c "import get_nuc; get_nuc.run_local('task_get_nuc_info')" -c 10 -p 20
./list_missing.sh 13985 block bin
python3.6 -c "import get_nuc; get_nuc.run_local('task_get_nuc_info')" -c 10 -p 20 -i ~/missing.txt

python3.6 -c "import get_nuc; get_nuc.run_local('task_merge_within_block', count_data=True)" -p 18
./list_missing.sh 13985 block2 bin


python3.6 -c "import get_nuc; get_nuc.run_local('task_merge_across_block', count_data=True)" -p 18
python3.6 -c "import get_nuc; get_nuc.run_local('task_apply_size_threshold')"



python3.6 -c "import nuc2body; nuc2body.run_local('task_get_surrounding')" -c 200 -p 20
./list_missing.sh 17075 nuc bin
python3.6 -c "import nuc2body; nuc2body.run_local('task_get_surrounding')" -c 200 -p 20 -i ~/missing.txt
python3.6 -c "import nuc2body; nuc2body.run_local('task_save')" -p 12







python3.6 -c "import get_nuc_20210809test; get_nuc_20210809test.run_local('task_get_nuc_info')" -c 10 -p 20
./list_missing.sh 108779 block bin
python3.6 -c "import get_nuc_20210809test; get_nuc_20210809test.run_local('task_get_nuc_info')" -c 10 -p 20 -i ~/missing.txt

python3.6 -c "import get_nuc_20210809test; get_nuc_20210809test.run_local('task_merge_within_block', count_data=True)" -p 20
./list_missing.sh 108779 block2 bin


python3.6 -c "import get_nuc_20210809test; get_nuc_20210809test.run_local('task_merge_across_block', count_data=True)" -p 20
python3.6 -c "import get_nuc_20210809test; get_nuc_20210809test.run_local('task_apply_size_threshold')" -p 20


108780
soma


FILE_ID=1hfZNFxizSz5TFs8KLpjiyvpLZi4qNL2d
FILE_NAME=full_VNC_synapses.csv
curl -sc /tmp/cookie "https://drive.google.com/uc?export=download&id=${FILE_ID}" > /dev/null
CODE="$(awk '/_warning_/ {print $NF}' /tmp/cookie)"  
curl -Lb /tmp/cookie "https://drive.google.com/uc?export=download&confirm=${CODE}&id=${FILE_ID}" -o ${FILE_NAME}








    arr = np.array(clist, dtype='int64').flatten()
ValueError: setting an array element with a sequence.


aug2021
not random? choose from center?