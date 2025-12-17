# Abstract

Graph Convolutional Networks (GCNs) have become the mainstream paradigm for skeleton-based action recognition, but existing methods suffer from three major bottlenecks: First, the fixed integer grid sampling and rigid receptive field of discrete convolutions are difficult to adapt to the continuity of human motion, making it impossible to adaptively model actions of different rhythms and scales; second, after multi-scale temporal feature fusion, channel redundancy and noise are prominent, and the lack of a sample-adaptive channel-level calibration mechanism leads to the masking of key semantic features; third, the discriminative constraints of cross-entropy loss are insufficient, easily confusing visually similar actions. To address these issues, this paper proposes an Adaptive Temporal Alignment and Contrast Refinement Network (ATACR-Net), following the design principle of "alignment-refinement-discrimination": a learnable continuous temporal sampling mechanism is introduced through a multi-scale adaptive temporal convolution module (MATCM) to reconstruct subframe-level temporal continuity and align key dynamic features; a compression excitation module (SE) is used to dynamically calibrate fused features, suppressing redundancy and highlighting discriminative channels; and Smooth Spatio-Temporal Contrastive Learning (SSTCL) is used as a regularization term to increase inter-class distance, reduce intra-class variance, and enhance the model's discriminative ability. Experiments on the NTU RGB+D, NTU RGB+D 120, and NW-UCLA datasets demonstrate that the proposed method achieves state-of-the-art performance, validating the effectiveness of the integrated framework.

# Dependencies

- Python >= 3.6
- PyTorch >= 1.10.0
- PyYAML == 5.4.1
- torchpack == 0.2.2
- matplotlib, einops, sklearn, tqdm, tensorboardX, h5py
- Run `pip install -e torchlight` 

# Data Preparation

### Download datasets.

#### There are 3 datasets to download:

- NTU RGB+D 60 Skeleton
- NTU RGB+D 120 Skeleton
- NW-UCLA

#### NTU RGB+D 60 and 120

1. Request dataset here: https://rose1.ntu.edu.sg/dataset/actionRecognition
2. Download the skeleton-only datasets:
   1. `nturgbd_skeletons_s001_to_s017.zip` (NTU RGB+D 60)
   2. `nturgbd_skeletons_s018_to_s032.zip` (NTU RGB+D 120)
   3. Extract above files to `./data/nturgbd_raw`

#### NW-UCLA

1. Download dataset from here: https://www.dropbox.com/s/10pcm4pksjy6mkq/all_sqe.zip?dl=0
2. Move `all_sqe` to `./data/NW-UCLA`

### Data Processing

#### Directory Structure

Put downloaded data into the following directory structure:

```
- data/
  - NW-UCLA/
    - all_sqe
      ... # raw data of NW-UCLA
  - ntu/
  - ntu120/
  - nturgbd_raw/
    - nturgb+d_skeletons/     # from `nturgbd_skeletons_s001_to_s017.zip`
      ...
    - nturgb+d_skeletons120/  # from `nturgbd_skeletons_s018_to_s032.zip`
      ...
```

#### Generating Data

- Generate NTU RGB+D 60 or NTU RGB+D 120 dataset:

```
 cd ./data/ntu # or cd ./data/ntu120
 # Get skeleton of each performer
 python get_raw_skes_data.py
 # Remove the bad skeleton 
 python get_raw_denoised_data.py
 # Transform the skeleton to the center of the first frame
 python seq_transformation.py
```

# Training & Testing

### Training

- NTU-RGB+D 60 & 120
```
# Example: training ATACR-Net (joint CoM 1) on NTU RGB+D 60 cross subject with GPU 0
python main.py --config ./config/nturgbd60-cross-subject/joint_com_1.yaml --device 0

# Example: training ATACR-Net (bone CoM 1) on NTU RGB+D 60 cross subject with GPU 0
python main.py --config ./config/nturgbd60-cross-subject/bone_com_1.yaml --device 0

# Example: training ATACR-Net (joint CoM 1) on NTU RGB+D 120 cross subject with GPU 0
python main.py --config ./config/nturgbd120-cross-subject/joint_com_1.yaml --device 0

# Example: training ATACR-Net (bone CoM 1) on NTU RGB+D 120 cross subject with GPU 0
python main.py --config ./config/nturgbd120-cross-subject/bone_com_1.yaml --device 0
```

- To train your own model, put model file `your_model.py` under `./model` and run:

```
# Example: training your own model on NTU RGB+D 120 cross subject
python main.py --config ./config/nturgbd120-cross-subject/your_config.yaml --model model.your_model.Model --work-dir ./work_dir/your_work_dir/ --device 0
```

### Testing

- To test the trained models saved in <work_dir>, run the following command:

```
python main.py --config <work_dir>/config.yaml --work-dir <work_dir> --phase test --save-score True --weights <work_dir>/xxx.pt --device 0
```

- To ensemble the results of different modalities, run 
```
# Example: six-way ensemble for NTU-RGB+D 120 cross-subject
python ensemble.py --datasets ntu120/xsub --main-dir ./work_dir/ntu120/cross-subject/
```
