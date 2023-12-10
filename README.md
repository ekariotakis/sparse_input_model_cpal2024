# Leveraging Sparse Input and Sparse Models: Efficient Distributed Learning in Resource-Constrained Environments

[![paper](https://img.shields.io/badge/Paper-OpenReview.net-red)](https://openreview.net/forum?id=D9ggc3l0wi)
&nbsp;

This is a public code repository for our publication in Conference on Parsimony and Learning (CPAL) 2024:
> [**Leveraging Sparse Input and Sparse Models: Efficient Distributed Learning in Resource-Constrained Environments**](https://openreview.net/forum?id=D9ggc3l0wi)<br>
> Emmanouil Kariotakis, Grigorios Tsagkatakis, Panagiotis Tsakalides, Anastasios Kyrillidis <br>

## Abstract
Optimizing for reduced computational and bandwidth resources enables model training in less-than-ideal environments and paves the way for practical and accessible AI solutions. This work is about the study and design of a system that exploits sparsity in the input layer and intermediate layers of a neural network. Further, the system gets trained and operates in a distributed manner. Focusing on image classification tasks, our system efficiently utilizes reduced portions of the input image data. By exploiting transfer learning techniques, it employs a pre-trained feature extractor, with the encoded representations being subsequently introduced into selected subnets of the system's final classification module, adopting the Independent Subnetwork Training (IST) algorithm. This way, the input and subsequent feedforward layers are trained via sparse ``actions'', where input and intermediate features are subsampled and propagated in the forward layers. 

We conduct experiments on several benchmark datasets, including CIFAR-$10$, NWPU-RESISC$45$, and the Aerial Image dataset. The results consistently showcase appealing accuracy despite sparsity: it is surprising that, empirically, there are cases where fixed masks could potentially outperform random masks and that the model achieves comparable or even superior accuracy with only a fraction ($$50\%$$ or less) of the original image, making it particularly relevant in bandwidth-constrained scenarios. This further highlights the robustness of learned features extracted by ViT, offering the potential for parsimonious image data representation with sparse models in distributed learning. 

### Summary of our findings
- We propose a distributed system that utilizes both sparse input and models, showcasing the potential of end-to-end sparse systems.
- We demonstrate that a single masked representation of each image during training suffices, eliminating the necessity for diverse random masks to be applied to inputs at each iteration and empowering the creation and preservation of significantly smaller datasets.
- We evaluate our system across diverse image datasets, showcasing substantial performance enhancements, particularly in scenarios involving highly masked input images ($50\%$ or more).

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