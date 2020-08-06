## Dynamic Extension Nets for Few-shot Semantic Segmentation
Created by Lizhao Liu, Junyi Cao and Minqian Liu from South China University of Technology.

This repository contains the official pytorch-implementation of our [ACM MM 2020 paper *Dynamic Extension Nets for Few-shot Semantic Segmentation*](#). In particular, we release code for training and testing the DENet and our re-implemented methods for few-shot semantic segmentation under {1, 2}-way and {1, 5}-shot settings. 

#### Abstract
We propose a Dynamic ExtensionNetwork (DENet) in which we dynamically construct and maintaina classifier for the novel class by leveraging the knowledge fromthe base classes and the information from novel data. More impor-tantly, to overcome the information suppression issue, we design aGuided Attention Module (GAM), which can be plugged into anyframework to help learn class-relevant features. Last, rather thandirectly train the model with limited data, we propose a dynamicextension training algorithm to predict the weights of novel clas-sifiers, which is able to exploit the knowledge of base classifiersby dynamically extending classes during training. The extensiveexperiments show that our proposed method achieves state-of-the-art performance on thePASCAL-5𝑖 and COCO-20𝑖datasets.

### Citation
If you find our work useful in your research, please consider citing:
		  
		  @inproceedings{liu2020dynamic, 
			author={Lizhao Liu, Junyi Cao, Minqian Liu, Yong Guo, Qi Chen and Mingkui Tan}, 
			title={Dynamic Extension Nets for Few-shot Semantic Segmentation}, 
      booktitle={Proceedings of the 28th ACM International Conference on Multimedia},
			year={2020}
		  }
