# Active Learning for Deep Object Detection via Probabilistic Modeling

This repository is the official PyTorch implementation of [Active Learning for Deep Object Detection via Probabilistic Modeling](https://openaccess.thecvf.com/content/ICCV2021/html/Choi_Active_Learning_for_Deep_Object_Detection_via_Probabilistic_Modeling_ICCV_2021_paper.html), ICCV 2021.

The proposed method is implemented based on the [SSD pytorch](https://github.com/amdegroot/ssd.pytorch).

<p align="center"><img src="https://github.com/NVlabs/AL-MDN/blob/main/img/teaser.PNG" width="95%" height="95%">

Our approach relies on mixture density networks to estimate, in a single forward pass of a single model, both localization and classification uncertainties, and leverages them in the scoring function for active learning.

<p align="center"><img src="https://github.com/NVlabs/AL-MDN/blob/main/img/AL_accuracy_multimodel.jpg" width="40%" height="40%" /> <img src="https://github.com/NVlabs/AL-MDN/blob/main/img/AL_cost_multimodel.jpg" width="35%" height="35%" />

Our method performs on par with multiple model-based methods (e.g., ensembles and MC-Dropout). Therefore, our method provides the best trade-off between accuracy and computational cost.
    
License
--------
To view a NVIDIA Source Code License for this work, visit https://github.com/NVlabs/AL-MDN/blob/main/LICENSE

Requirements
----------------------
For setup and data preparation, please refer to the README in [SSD pytorch](https://github.com/amdegroot/ssd.pytorch).

Code was tested in virtual environment with `Python 3+` and `Pytorch 1.1`.


Training
--------
- Make directory `mkdir weights` and `cd weights`.

- Download the [FC-reduced VGG-16 backbone weight](https://s3.amazonaws.com/amdegroot-models/vgg16_reducedfc.pth) in the `weights` directory, and `cd ..`.

- If necessary, change the `VOC_ROOT` in `data/voc0712.py` or `COCO_ROOT` in `data/coco.py`.

- Please refer to `data/config.py` for configuration.

- Run the training code:
```
# Supervised learning
CUDA_VISIBLE_DEVICES=<GPU_ID> python train_ssd_gmm_supervised_learning.py

# Active learning
CUDA_VISIBLE_DEVICES=<GPU_ID> python train_ssd_gmm_active_learining.py
```


Evaluation
--------
- To evaluate on MS-COCO, change the `COCO_ROOT_EVAL` in `data/coco_eval.py`. 

- Run the evaluation code:
```
# Evaluation on PASCAL VOC
python eval_voc.py --trained_model <trained weight path>

# Evaluation on MS-COCO
python eval_coco.py --trained_model <trained weight path>
```


Visualization
---------
- Run the visualization code:
```
python demo.py --trained_model <trained weight path>
```


Citation
--------
```
@InProceedings{Choi_2021_ICCV,
    author    = {Choi, Jiwoong and Elezi, Ismail and Lee, Hyuk-Jae and Farabet, Clement and Alvarez, Jose M.},
    title     = {Active Learning for Deep Object Detection via Probabilistic Modeling},
    booktitle = {Proceedings of the IEEE/CVF International Conference on Computer Vision (ICCV)},
    month     = {October},
    year      = {2021},
    pages     = {10264-10273}
}
```
