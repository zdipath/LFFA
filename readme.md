

# LFFA

## LFFA: Learnable Frozen Feature Augmentation for Few-Shot Whole-Slide Image Learning


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


