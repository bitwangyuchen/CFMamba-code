# CFMamba-code
This repo includes:  

- Specification of dependencies.
- Evaluation code.
- Pre-trained models.
- Test dataset of the SDM dataset


The current public release focuses on evaluation and reproducibility. Additional components, including the complete dataset and training pipeline, will be released in a future update.



#### 1. Create Environment:

------

- Python 3.8

- PyTorch == 2.0.0

- NVIDIA GPU + CUDA

- Python packages:

The mamba_ssm library is needed to install with the folllowing command:

  ```shell
  pip install causal_conv1d==1.0.0
  pip install mamba_ssm==1.0.1
  ```

You can use the following command to create the environment.
  ```shell
  pip install -r requirements.txt
  ```
## 2. Prepare Data:


All required data files are available for download, including: the test dataset of the proposed SDM dataset and its raw hyperspectral reflectance dataset, and the OMSIV test dataset (also available from the official website at https://xavysp.github.io/ssmid-dataset/).

| Platform      | Link         |
|---------------|--------------|
| Baidu Netdisk | [Link](https://pan.baidu.com/s/17HHAZUFfwje4K1ksD1nlDg)    (Access Code: `f1i9`) |
| Google Drive  | [Link](https://drive.google.com/drive/folders/1_wYm83X_h4GGUl_0f7ZZWzEB6IPdoqxl?usp=drive_link)    |




## 3. Set Up Data
After downloading, place the data into 'data' folder. The structure of the 'data' folder should look like this:
```plaintext
  |-- data
  |   |-- OMSIV
  |   |   |-- X
  |   |   |   |-- [OMSIV input images]
  |   |   |
  |   |   |-- Y
  |   |   |   |-- [OMSIV ground truth]
  |
  |   |-- SDM
  |   |   |-- [SDM dataset files]
  |
  |   |-- SDM_ref
  |   |   |-- [hyperspectral reflectance files]
  |
  |   |-- split_txt
  |   |   |-- test_list_omsiv.txt
  |   |   |-- test_list_sdm.txt
```


---


## 4. Evaluation
1. Evaluate our pre-trained CFMamba model on our SDM dataset. The results will be saved in 'output' folder, and the visualized results will be saved in 'visual' folder.

```shell
python test_sdm.py

```

2. Evaluate our pre-trained CFMamba model on the OMSIV dataset. The results will be saved in 'output' folder, and the visualized results will be saved in 'visual' folder.

```shell
python test_omsiv.py

```






