# CFMamba-code
This repo includes:  

- Specification of dependencies.
- Evaluation code.
- Pre-trained models.
- A mini subset of the SDM dataset


The current public release focuses on evaluation and reproducibility. Additional components, including the complete dataset and training pipeline, will be released in a future update.



## 1. Create Environment:

- Python 3 (Recommend to use [Anaconda](https://www.anaconda.com/download/#linux))

- [PyTorch >= 1.3](https://pytorch.org/)

- NVIDIA GPU + [CUDA](https://developer.nvidia.com/cuda-downloads)

- Python packages:

  ```shell
  pip install -r requirements.txt
  ```
## 2. Prepare Data:


All required files (mini dataset of SDM, pretrained models for all five training modes, and physical mask) are available for download:




## 3. Set Up Data and Model Paths
After downloading, place the data into 'data_root' folder. The structure of the 'data_root' folder should look like this:


## 4. Evaluation

We provide evaluation scripts for five training modes corresponding to different data strategies.

### Run All Models at Once

```bash
python inference.py --no_visual --config=config/infer
```

### Run Individual Modes


Or run a specific model directly:

```bash
python inference.py --no_visual --config=config/infer/m5/dpu_5stg.yaml

python inference.py --no_visual --config=config/infer/m5/ssr_5stg.yaml

...
```

---
#### Notes
- --no_visual disables visualization during evaluation.
- All evaluation results will be automatically saved to the configured output directory.
- Ensure that the dataset paths and checkpoint paths are correctly configured before running inference.





