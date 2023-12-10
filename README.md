# Leveraging Sparse Input and Sparse Models: Efficient Distributed Learning in Resource-Constrained Environments

[![paper](https://img.shields.io/badge/Paper-OpenReview.net-red)](https://openreview.net/forum?id=D9ggc3l0wi)
&nbsp;

This is a public code repository for our publication in Conference on Parsimony and Learning (CPAL) 2024:
> [**Leveraging Sparse Input and Sparse Models: Efficient Distributed Learning in Resource-Constrained Environments**](https://openreview.net/forum?id=D9ggc3l0wi)<br>
> Emmanouil Kariotakis, Grigorios Tsagkatakis, Panagiotis Tsakalides, Anastasios Kyrillidis <br>

## Abstract
Optimizing for reduced computational and bandwidth resources enables model training in less-than-ideal environments and paves the way for practical and accessible AI solutions. This work is about the study and design of a system that exploits sparsity in the input layer and intermediate layers of a neural network. Further, the system gets trained and operates in a distributed manner. Focusing on image classification tasks, our system efficiently utilizes reduced portions of the input image data. By exploiting transfer learning techniques, it employs a pre-trained feature extractor, with the encoded representations being subsequently introduced into selected subnets of the system's final classification module, adopting the Independent Subnetwork Training (IST) algorithm. This way, the input and subsequent feedforward layers are trained via sparse ``actions'', where input and intermediate features are subsampled and propagated in the forward layers. 

We conduct experiments on several benchmark datasets, including CIFAR-10, NWPU-RESISC45, and the Aerial Image dataset. The results consistently showcase appealing accuracy despite sparsity: it is surprising that, empirically, there are cases where fixed masks could potentially outperform random masks and that the model achieves comparable or even superior accuracy with only a fraction (50\% or less) of the original image, making it particularly relevant in bandwidth-constrained scenarios. This further highlights the robustness of learned features extracted by ViT, offering the potential for parsimonious image data representation with sparse models in distributed learning. 

### Summary of our findings
- We propose a distributed system that utilizes both sparse input and models, showcasing the potential of end-to-end sparse systems.
- We demonstrate that a single masked representation of each image during training suffices, eliminating the necessity for diverse random masks to be applied to inputs at each iteration and empowering the creation and preservation of significantly smaller datasets.
- We evaluate our system across diverse image datasets, showcasing substantial performance enhancements, particularly in scenarios involving highly masked input images (50\% or more).

## File Arrangement

Here we summarize all files present in this repo and their purpose.
```
+-- mae/: code from Meta Research
+-- torchrs/: necessary scripts to use AID and RESISC45 datasets
+-- cls_data_loader.py: load CLS token datasets
+-- cls_dataset.ipynb: create CLS token dataets from image datasets
+-- cls_mlp.py: centralized model architecture
+-- cls_mlp_example.ipynb: centralized training using image datasets
+-- ist_utils.py: utils for IST
+-- mlp.py: simple MLP architecture
+-- my_distributed_3layer.py: our IST model. Gets as input CLS token datsets
+-- resnet_dataset.ipynb: encode image datsets using resnet50
+-- run_cls_dataset.ipynb: centralized training using CLS token datasets
+-- run_my_ist.sh: run our IST implementation
+-- run_resnet_dataset.ipynb: centralized training using resnet50 encoded datasets 
+-- utils.ipynb: some general utils
```

## Citing us
If you find this work useful, please cite our paper.
```
@inproceedings{kariotakis2023leveraging,
    title={Leveraging Sparse Input and Sparse Models: Efficient Distributed Learning in Resource-Constrained Environments},
    author={Emmanouil Kariotakis and Grigorios Tsagkatakis and Panagiotis Tsakalides and Anastasios Kyrillidis},
    booktitle={Conference on Parsimony and Learning (Proceedings Track)},
    year={2023},
    url={https://openreview.net/forum?id=D9ggc3l0wi}
}
```