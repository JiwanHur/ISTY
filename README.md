# (WACV 2023) ISTY: A Framework for Removing Foreground Occlusion in Both Sparse and Dense Light Field Images
This is a codebase for [I See-Through You: A Framework for Removing Foreground Occlusion in Both Sparse and Dense Light Field Images](https://openaccess.thecvf.com/content/WACV2023/html/Hur_I_See-Through_You_A_Framework_for_Removing_Foreground_Occlusion_in_WACV_2023_paper.html).

## Requirements
Python 3.6
Pytorch >= 1.0 

## Dataset
```
├── train
|    ├── src_imgs_train        -> Source occlusion-free light field images
|    ├── occ_imgs              -> Occlusion images without background
|    └── occ_msks              -> Occlusion mask for occ_imgs (1 for occlusion 0 for background)
├── test_data_dir1
├── test_data_dir2
├── test_data_dir3
└── ...
```
### src_imgs_train
We use the [DUTLF-V2](https://github.com/DUT-IIAU-OIP-Lab/DUTLF-V2) training dataset for source occlusion-free light field images.
Since the DUTLF-V2 dataset includes occlusion, we selected 1418 images from the dataset.
Please refer to the `DUTLF_V2_train_list.json`.

### occ_imgs and occ_msks
We use occlusion images from [DeOccNet](https://github.com/YingqianWang/DeOccNet) and resize them to 600x400.
The occlusion mask is a binary mask for occlusion images where 1 indicates occlusion and 0 indicates background. 
The occlusion mask can be created by simply thresholding the occlusion images. We generate the mask by converting the occ_imgs into grayscale and thresholding them with 229.

### test dataset
The test dataset can be downloaded [DeOccNet](https://github.com/YingqianWang/DeOccNet), [Stanford Lytro](http://lightfields.stanford.edu/LF2016.html), and [EPFL-10](https://www.epfl.ch/labs/mmspg/downloads/valid/).

## Train
### Command
```
bash command/train.sh
```

<!-- The backbone LBAM model pre-trained on Paris Street View dataset and checkpoint for ISTY can be downloaded from [here](https://drive.google.com/drive/folders/1cAs8gVU16CGlmhvktKhzu6uvH3TF9Q5r?usp=sharing). -->
The pre-trained LBAM model should be located in `ISTY/LBAMmodels/LBAM_G_500.pth`.

Since we use further occlusion images for training (as mentioned in the paper), the result can be slightly different if one re-trains the model following this repository.
One can add occlusion images such as thick and complex objects to improve the performance.

## Test
```
bash command/test.sh
```
The checkpoint should be in `./results/checkpoints/{scope}/LFGAN/`.

## Checkpoint and dataset
We provide the preprocessed dataset and checkpoint of the model, and pre-trained LBAM model for the backbone architecture for Occlusion Inpainter in [here](https://drive.google.com/drive/folders/1cAs8gVU16CGlmhvktKhzu6uvH3TF9Q5r?usp=sharing).

Please refer to each section for the proper location of each file.

### Citations
```
@inproceedings{hur2023see,
  title={I See-Through You: A Framework for Removing Foreground Occlusion in Both Sparse and Dense Light Field Images},
  author={Hur, Jiwan and Lee, Jae Young and Choi, Jaehyun and Kim, Junmo},
  booktitle={Proceedings of the IEEE/CVF Winter Conference on Applications of Computer Vision},
  pages={229--238},
  year={2023}
}
```

### Acknowledgement
The code for model architecture is based on [DeOccNet](https://github.com/YingqianWang/DeOccNet) and [LBAM](https://github.com/Vious/LBAM_Pytorch).