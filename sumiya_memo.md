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

Note: [gitignore](https://qiita.com/anqooqie/items/110957797b3d5280c44f
https://www.curict.com/item/fe/fe45741.html)


https://stackoverflow.com/questions/32578224/cloning-a-branch-remote-branch-branch-99-not-found-in-upstream-origin
rsync

mkdir Output for pandas


pip install igneous-pipeline
pip install task-queue #numpy first