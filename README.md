## Adversarial training and dual batch normalization to refine neural network architectures for improved diagnostic performance and clinical usability of deep learning <br><i>– Official Pytorch implementation of the paper</i>

**Tianyu Han** (RWTH), **Daniel Truhn** (UKA), **Volkmar Schulz** (RWTH), **Christiane Kuhl** (UKA), **Fabian Kiessling** (RWTH)

**Abstract:**<br>
*Understanding the decision making process of machine learning models is an essential element for deploying computer vision algorithms in clinical practice. We demonstrate that adversarially trained models can significantly enhance usability in clinical practice as compared to standard models. We let six radiologists rate the interpretability of saliency maps in datasets of x-rays, computed tomography and magnetic resonance imaging. Significant improvements were found for adversarially trained models and results further improved when employing dual batch normalization. Contrary to previous research on adversarially trained models, we found that accuracy of such models was equal to standard models, when sufficiently large datasets and dual batch norm training were used. To ensure transferability, we additionally validated our results on an external test set of 22,433 x-rays. Our work demonstrates that different paths for adversarial and real images are needed during training in order to achieve state of the art results with superior interpretability.*

## System requirements

* Both Linux and Windows are supported, but we strongly recommend Linux for performance and compatibility reasons.
* 64-bit Python 3.6 installation with numpy 1.13.3 or newer. We recommend Anaconda3.
* One or more high-end NVIDIA Pascal or Volta GPUs with 16GB of DRAM. We recommend NVIDIA DGX-1 with 8 Tesla V100 GPUs.
* NVIDIA driver 391.25 or newer, CUDA toolkit 9.0 or newer, cuDNN 7.1.2 or newer.
* Pytorch 1.1.0

## Datasets used in the study

* [NIH dataset](https://nihcc.app.box.com/v/ChestXray-NIHCC)
* [Stanford CheXpert dataset](https://stanfordmlgroup.github.io/competitions/chexpert)
* [knee MRI dataset](http://www.riteh.uniri.hr/~istajduh/projects/kneeMRI/)
* [LUNA16](https://luna16.grand-challenge.org/Data/)

## Basic usage

* Network standard training
```
python main_std.py --affix dsbn_res50_std --batch_size 16    
```
 
* Network robust (dual bns) training
```
python main_bn.py --adv_train --affix dsbn_res50_linf --batch_size 16 --max_epoch 250    
```

* Network testing and plot ROC-AUC
```
python main_bn.py --affix dsbn_res18_AUC --todo test --load_checkpoint ./checkpoint/rijeka_ROIMG_dsbn_res18/checkpoint_best.pth 
```

## License

This project is licensed under the MIT License - see the [LICENSE.md](LICENSE.md) file for details

## Acknowledgments

* Domain-Specific Batch Normalization for Unsupervised Domain Adaptation: https://github.com/wgchang/DSBN
* MadryLab at MIT: https://github.com/MadryLab/robust-features-code
