# Alzheimer Disease Diagnosis by Deeply Supervised 3D Convolutional Network
Diagnosing Alzheimer disease from 3D MRI T1 scans from ADNI dataset. The initial results using 3D Convolutional Network is published in ICIP 2016 [[1]](https://arxiv.org/abs/1607.00455). The second model used deeply supervision to boost the performance on all binary and three-way classification of AD/MCI/Normal classes. The results are published on arxiv [[2]](https://arxiv.org/abs/1607.00556)

### Using Transfer Learning 
* Pretraining 3D CNN with 3D Convolutional Autoencoder on source domain  
* Finetuning uper fully-connected layers of 3D CNN using supervised fine-tuning on target domain  
* Using deeply supervision in supervised fine-tuning of upper fully-connected layers  

### DATA
List of all subject ids are in ADNI_subject_id directory


###Papers 
* [1] E. Hosseini-Asl, R. Keynton and A. El-Baz, "Alzheimer's disease diagnostics by adaptation of 3D convolutional network," 2016 IEEE International Conference on Image Processing (ICIP), Phoenix, AZ, USA, 2016, pp. 126-130. 
* [2] E. Hosseini-Asl, G. Gimel'farb, and A. El-Baz, “Alzheimer's Disease Diagnostics by a  Deeply Supervised Adaptable 3D Convolutional Network”, arXiv:1607.00556 [cs.LG, q-bio.NC, stat.ML], 2016.
