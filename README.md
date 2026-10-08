

# LFFA

## Learnable frozen feature augmentation for few-shot biomarker prediction from pathology whole-slide images

<div align="center">
  
[![Bioinformatics](https://img.shields.io/badge/Bioinformatics-btag597-red
)](https://academic.oup.com/bioinformatics/article/42/8/btag597/8756691)
</div>


### Datasets

We evaluate LFFA on three real-world pathology whole-slide image datasets for biomarker-oriented prediction tasks.

- **BCNB**: We use BCNB for estrogen receptor (ER) status prediction from breast cancer whole-slide images. The dataset information and access instructions are available at:  
  https://bupt-ai-cz.github.io/BCNB/

- **MUT-HET-RCC**: We use MUT-HET-RCC for SETD2 mutation prediction from renal cancer whole-slide images. The dataset is described in Acosta et al., *Cancer Research*, 2022. More information is available at:  
  https://aacrjournals.org/cancerres/article/82/15/2792/707325/Intratumoral-Resolution-of-Driver-Gene-Mutation

- **CPTAC-CCRCC**: We use CPTAC-CCRCC as the external test cohort for cross-cohort PBRM1 mutation prediction. The dataset is available from The Cancer Imaging Archive (TCIA):  
  https://www.cancerimagingarchive.net/collection/cptac-ccrcc/

Due to data usage restrictions, the raw whole-slide images are not redistributed in this repository. Users should obtain the raw data from the original data providers. We provide the processed experimental split files and the code needed to reproduce the experiments after feature extraction.

### Data Preprocess
We follow the CLAM's WSI preprocessing solution. To satisfy LFFA’s requirement for patch coordinate information, we store both the extracted features and their corresponding coordinates in a single `.npy` file.
The saving format is as follows:

```python
features_list = []
indexs_list = []
inst_labels_list = []
# Get patch feature
features = features.cpu().numpy().astype(np.float32)
features_list.append(features)
indexs_list += coords
inst_labels_list += inst_labels
asset_dict = {
    "feature": features_list,
    "index": indexs_list,
    "inst_label": inst_labels_list,
}
np.save(output_path, asset_dict)
```


The data can be loaded with the following format:

```python
fea = np.load("./data/MUT/conch_v1_5/19579_0_1024.npy", allow_pickle=True)

features = fea[()]["feature"]
cor = fea[()]["index"]
# Parse patch coordinates from the index strings
coords = np.array(
    [filename.split("_")[:2] for filename in cor],
    dtype=int,
)
# Convert features and coordinates to tensors
patch_embedding = torch.from_numpy(features).unsqueeze(0).to(device)
coords = torch.from_numpy(coords).unsqueeze(0)
```

### WSI Feature and CIT Features Extraction
We strictly follow the official instructions provided on the Hugging Face repositories of TITAN, PRISM, and GigaPath for feature extraction. For each slide, we extract both global WSI-level feature and local CITs features. All extracted features are serialized and saved in `.npy` format to ensure efficient loading and compatibility.




### Model Training


For the few-shot setting, you can run:

```bash
python -u train_wsi_model.py \
  --gpu 4 \
  --task t1_gene \
  --dataset BCNB \
  --experiment_target ER \
  --model_name TITAN \
  --model_type linear \
  --aug_task 1-shot_aug_2 \
  --aug_method LFFA \
  --batch_size 16 \
  --aug_para2 0.1 \
  --aug_para3 0.3 \
  --head_way fewshot \
  --infer_time 0
```
## Citation
If you find our work useful in your research, please consider citing LFFA:

```
@article{zhang2026learnable,
  title={Learnable frozen feature augmentation for few-shot biomarker prediction from pathology whole-slide images},
  author={Zhang, Di and Liu, Jiashuai and Ma, Youyuan and Ge, Jiusong and Zeng, Zhi and Sun, Wenfang and Liu, Qidong and He, Kai and Zheng, Yefeng and Yu, Weimiao and others},
  journal={Bioinformatics},
  volume={42},
  number={8},
  pages={btag597},
  year={2026},
  publisher={Oxford University Press}
}
```

