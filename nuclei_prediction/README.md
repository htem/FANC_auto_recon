# nuclei_prediction
This code detects all the neuron nuclei & somas from `FANCv4`. Thanks to Jasper, Ran, Brandon, and Stephan, this pipeline identified 17076 putative nuclei after applying a size threshold. We manually inspected each one and found 14679 neurons (86.0%), 1987 glia (11.6%), and 410 false positives (2.4%). The `xyz` locations of all the neuron nuclei are currently available through [CAVEclient](https://cave.fanc-fly.com/annotation/views/aligned_volume/fanc_v4?middle_auth_url=global.daf-apis.com%2Fsticky_auth).

Please contact Sumiya Kuroda for any questions or feedbacks.

## Virtual environment
I recommend to setup 'venv' before running the program. You may need extra step to use `.ipynb` on `venv`.

```sh
# python3 -V 
python3 -m venv nuclei
. nuclei/bin/activate
(nuclei) vscode ➜ ~ $ pip install -r requirements.txt
(nuclei) vscode ➜ ~ $ ipython kernel install --user --name=nuclei
```

You can shutdown your `venv` with

```sh
(nuclei) vscode ➜ ~ $ deactivate
```

## Usage
```sh
python3.6 -c "import get_nuc; get_nuc.run_local('task_get_nuc_info')" -c 10 -p 20
./list_missing.sh 13985 block bin
python3.6 -c "import get_nuc; get_nuc.run_local('task_get_nuc_info')" -c 10 -p 20 -i ~/missing.txt
# run until there is no missing block

python3.6 -c "import get_nuc; get_nuc.run_local('task_merge_within_block', count_data=True)" -p 20
python3.6 -c "import get_nuc; get_nuc.run_local('task_merge_across_block', count_data=True)" -p 20
python3.6 -c "import get_nuc; get_nuc.run_local('task_apply_size_threshold')"
```

```sh
python3.6 -c "import nuc2soma; nuc2soma.run_local('task_get_surrounding')" -c 200 -p 20
./list_missing.sh 17075 nuc bin
python3.6 -c "import nuc2soma; nuc2soma.run_local('task_get_surrounding')" -c 200 -p 20 -i ~/missing.txt
# run until there is no missing block

python3.6 -c "import nuc2soma; nuc2soma.run_local('task_save')" -p 20
```

You can run these on different `screen`.

## How it works


## Other resources
- Brandon wrote [a very useful instruction](https://github.com/bjm5164/rotation_projects) for how to get used to various tools: `Python`, `Neuroglancer`, `cloud-volume`, etc.

- If you have troubles with the credentials, you can also look at [cloud-volume wiki](https://github.com/seung-lab/cloud-volume). They have very detailed instruction for how to make and save credentials. Also, you may need to create the directory `secret` by running `mkdir -p ~/.cloudvolume/secrets/` before saving your API.