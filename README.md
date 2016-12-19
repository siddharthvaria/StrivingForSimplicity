#STRIVING FOR STATE-OF-THE-ART USING ALL CONVOLUTIONAL NETWORKS#

Our goal is to implement and reproduce “All Convolutional Net” by Springenberg et-al (2015), using convolutional layers (no max pooling) to achieve state-of-the-art results, and suggest new architectures. The paper lacked crucial hyperparameters like learning-rate and batch-size, a challenge eventually resolved after experimentation. A novelty was implementing batch-normalization, and obtaining comparable results in much fewer epochs. 

Following are the results from our project:


| Model         | Paper Error Rate (%) / Epochs           | Our Error Rate (%) / Epochs  |
| :-------------: |:-------------:| :-----:|
| ALL-CNN-A     | 10.30 / 350 | 14.81 / 350 |
| ALL-CNN-B     | 9.10 / 350      |   15.22 / 350 |
| ALL-CNN-C     | 9.08 / 350     |    13.19 / 350 |
 
Table 1: Reproduced results on CIFAR-10 without data augmentation. 


| Model         | Paper Error Rate (%) / Epochs           | Our Error Rate (%) / Epochs  |
| :-------------: |:-------------:| :-----:|
| ALL-CNN-A     | - | 12.47/350|
| ALL-CNN-B     | -      |   11.54/350 |
| ALL-CNN-C     | 7.25/350     |    10.80/350 |

Table 2: Reproduced results on CIFAR-10 with data augmentation 


| Model         | Paper Error Rate (%) / Epochs           | Our Error Rate (%) / Epochs  |
| :-------------: |:-------------:| :-----:|
| ALL-CNN-A     | - | 11.71/150 |
| ALL-CNN-B     | -      |   10.67/150 |
| ALL-CNN-C     | -     |    9.64/150 |
Table 3: Reproduced results on CIFAR-10 with data augmentation and training done using batch normalization 

![alt text](https://github.com/rr3087/StrivingForSimplicity/blob/master/src/images/allplotsinone_150_final1.jpg)
 
Fig. 1 Comparison of Test Error Rate for 3 models over 150 epochs.  
