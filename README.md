# CFMamba-code
This repo includes:  

- Specification of dependencies.
- Evaluation code.
- Pre-trained models.
- A mini subset of the SDM dataset


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


All required files (mini dataset of SDM, pretrained models for all five training modes, and physical mask) are available for download:




## 3. Set Up Data and Model Paths
After downloading, place the data into 'data_root' folder. The structure of the 'data_root' folder should look like this:


## 4. Evaluation







