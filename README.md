# CassavaLeaf
CASSAVA LEAF DISEASE CLASSIFICATION
Cassava is a key food security crop grown by smallholder farmers since it can withstand harsh conditions. At least 80% of household farms in Sub-Saharan Africa grow this starchy root, but viral diseases are major sources of poor yields. With the help of data science, it may be possible to identify common diseases so they can be treated. In this model we classify cassava leaves into five categories, including a healthy category. Through this classification the farmers will be able to quickly identify diseased plants, potentially saving the crops before the disease cause damage.
The steps involved the process are as follows:
1.	Introduction to Dataset
2.	Exploratory Data Analysis
3.	Baseline Model
4.	Inception Net V3 Model
5.	VGG 16 Model

Introduction to Dataset
The dataset contains 21,397 images in the training dataset and around 15000 in the test set. Image_id and label are the to variables in the dataset. The 5 classes and their corresponding labels are as follows:
 



The dataset was sourced through farmers who uploaded the images of their crops for diagnosis. Since this wasn’t a controlled environment, the images were varied in terms of contrast, orientation, zoom and lighting conditions, this was handled using data augementation. There was a heavy imbalance in the dataset. Majority of the dataset was Cassava Mosaic Disease. The imbalance was handled in the models to get better accuracy scores. The count of respective classes is depicted in the following bar plot.
 
The input is in the form of 800*600*3 matrix, where 800 pixels is the horizontal length, 600 pixels is the vertical length and 3 denotes the RGB values of individual pixels.

Exploratory Data Analysis

1.	Missing Values:
 There were no missing values in the dataset. So, this condition does not require any handling.
 

2.	Image Augmentation:
 In image classification projects, image augmentation is a crucial approach. This technique allows us to execute numerous image changes in order to enlarge original datasets, reduce overhead memory, and improve model robustness. The "ImageDataGenerator" was used in this project to perform various random changes to original photos, including rotations, shifts, flips, brightness, zoom, and shear. "ImageDataGenerator" was chosen since it can give real-time data augmentations in future model training.

3.	Imbalance: 

From the bar plot we can infer that there is heavy imbalance in the dataset. This can be handled by under sampling, over sampling or through loss functions. In under sampling we sample less of the majority dataset, this ensures that the model receives uniform amount of data from each class. 

In oversampling we replicate the minority class so that number of images of majority class and minority class are same doing this hardcodes our weights to a specific dataset.

The third and the best option is to use loss functions which dynamically assign weights to different classes so that the model can achieve best accuracy. We used third method to handle imbalance in our dataset.


Baseline Model
It's obvious from EDA that this is an unbalanced dataset. If the sklearn package's "DummyClassifier" is used to predict baseline accuracy with respect to the most frequent class, it will obtain a 61.4% percent baseline accuracy. 
Convolutional Neural Networks (CNN) are complicated feed-forward neural networks used in deep learning. A hierarchical model is followed by the CNN, which finally outputs a fully connected layer. Because of their excellent accuracy, CNNs are commonly employed in the real world for picture categorization and recognition.
As we all know, a simple CNN model is capable of producing satisfactory results. As a result, CNNs are a good choice for this project's baseline model. There are three convolutional layers and three max-pooling layers in this simple model, with no non-trainable parameters. 
We want these three convolutional layers to extract significant features in different dimensions in this manner, with max-pooling applied after each convolutional layer. As a result, the most important aspects may be preserved in order to deliver to the next layer.
The filters used in the model were increased by the factor of 2 and kernel size was kept constant. The activation function used was softmax which was used to calculate the probabilities of individual features.
 
Baseline Model Architecture

The CNN model was trained for 5 epochs. 
 

Inception V3 Model
Transfer Learning, where we use a pre-trained model, is a super-effective strategy when we have a relatively limited dataset. We would be able to transfer weights gained during hundreds of hours of training on numerous high-powered GPUs because this model was trained on such a big dataset. Many of these models, such as VGG-16 and Inception-v3, are open-source. They were trained on millions of photos using incredibly high computational power, which is difficult to achieve from scratch.
Inception-v3 is a 48-layer deep convolutional neural network. You can import a pretrained version of the network from the ImageNet database, which has been trained on over a million photos. The network can classify photos into 1000 different object categories, including keyboards, mice, pencils, and a variety of animals. As a result, the network has learned a variety of rich feature representations for a variety of images. The network's picture input size is 299 by 299 pixels.
Sparse categorical cross entropy was employed in this model. This model will dynamically assign weights to different classes so that the model can achieve best accuracy. Since this model assigns inverse weights to higher loss classes, it is best suited and will handle imbalance very accurately.
Although this model performed very well it’s accuracy still falls short of the classifiers such as VGG 16. So, we finally chose VGG 16 as our main model. The accuracy achieved by Inception v3 model was around 80%.
 
Accuracy vs epochs
 
Loss vs accuracy plot

VGG16 Model
The most distinctive feature of VGG16 is that, rather than having a huge number of hyper-parameters, they focused on having 3x3 filter convolution layers with a stride 1 and always used the same padding and maxpool layer of 2x2 filter stride 2. Throughout the architecture, the convolution and max pool layers are arranged in the same way. It has two FC (completely connected layers) in the end, followed by a softmax for output. The 16 in VGG16 alludes to the fact that it contains 16 layers with different weights. This network is quite huge, with approximately 138 million (estimated) parameters.
	

















We experimented with several combinations of convolutional layers, pooling layers, and fully connected layers, similar to Inception V3, in order to find some ideal solutions. As a result, we decided to start with one flatten layer. Then we added two more dense layers, each followed by a dropout layer to eliminate any potential overfitting difficulties. Finally, we used a Softmax layer to produce our result.

This is the model summary
















We ran our model for 20 epochs with a batch size of 64 and these are the results		
 

In the above picture we can see that the accuracy peaked at around 83.4%. On running the model for higher epochs and lesser batch size we can achieve a higher accuracy. The loss is very less compared to the inception v3 model and this is why we decided to use this as our final model.
The following are the train vs validation accuracy and train vs validation loss plots

 

 
