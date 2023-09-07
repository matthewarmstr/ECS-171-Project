# Introduction

Our project focuses on the Internet of Things Network (IoT) and working with Intrusion Detection, which is meant to identify malicious attacks across networks towards users. We chose this project because of the safety implications that having such a predictive model brings. Some of the categories we will be analyzing involve DDoS, Brute Force, Spoofing, etc. We are building off an existing work that has several subcategories for individual types of intrusions, including their instances and features. The purpose of the project is to design and implement an effective predictive model that can detect and identify multiple kinds of IoT traffic intrusions. A good predictive model is important because it can use other attributes to accurately help users recognize network attacks.

While we may be currently using a neural network as our primary model, we will likely incorporate other models in the future as we continue our analysis of the dataset.

# Figures

# Methods

## Data Exploration

There are 1191264 observations in our dataset. There are no missing data points for any observation. The data attributes can be classified as having a normal distribution, but certain attributes describing network/application protocols (HTTP, TCP, UDP) have binary values (1,0) representing True or False behavior and indicate if the intrusion utilized these protocols. We decided to use min-max scaling since the data follows a normal distribution. Using categorical plots, we were able to form subsets of attribute data by class and create a scatterplot for each attribute which also separated data by class.

Some of the attributes in the original dataset were redundant and would not be useful for the predictive model. This is because either they weren't independent variables or for every single observation, it would have the same value. For instance, TeleNet and IRC were two cases where their values were all set to 0, so their input would have no significance to the actual model.

## Data Preprocessing

For our project, we decided to use a neural network to classify the type of intrusion based on the collections of many attributes in our dataset. To preprocess our data, we decided to normalize all of our attributes. Standardization was not used because the results of running the Shapiro-Wilks test on all the attribute columns showed that the input data was not normally distributed. The intrusion output classes were then encoded with one-hot encoding. Label encoding was not utilized for our multiclass classification model because there is no underlying ranking of intrusion classes in the dataset - they are simply different types of IoT traffic that were collected as part of the dataset.

## Initial Model

When setting up the initial model, the number of inputs for the neural network was reduced from the original forty-three valid inputs to thirty-five. A few of these ignored attribute labels include 'Tot sum', 'Min', 'Max', 'AVG', and 'Covariance'. These ignored inputs may be related to the intermediate timing distribution between the packets within the segment of surveyed traffic, but it’s unclear what exactly they represent as they were not described in the original dataset. As such, these unclear attributes were left out of the initial model. Once all thirty-five selected attributes were standardized, they were split according to a train/test ratio of 90:10 and sent to the 3-layer neural net model. The model was given an input dimension of thirty-five (one for each selected attribute) and one hidden layer with node layouts of thirty-six and thirty-five, respectively. The output layer was given thirty-four nodes corresponding to the same number of intrusion classes from the dataset. The softmax activation function was used in the output layer since this function provides an intuitive way of predicting which single output class is most likely to be true. The nodes within the input and hidden layers used the ReLU activation function because it improved the neural net's performance substantially compared to using all softmax functions. The model was optimized with stochastic gradient descent with the default learning rate and categorical cross entropy as the loss function. Finally, the model was fit with the training data over fifty epochs. 

To measure the performance of the initial model, multiple scores were computed on the training and testing prediction data. Some of these scores included the mean squared error and average accuracy, precision, and recall across all output class predictions. The classification report was also generated for the training and testing predictions, which expanded the precision, recall, and f1 scores for each intrusion class. After compiling and training the initial model with fifty epochs, the categorical cross-entropy loss remained at about 42.8%. The average accuracy when predicting with the training and testing sets was around 79%, but the average precision and recall scores collectively ranged from 45% to 61%. This difference in average performance scores occurred because many output classes were never predicted to be true (which defaulted their precision and recall scores in the classification report to 0). A few of these classes included 'Backdoor_Malware', 'BrowserHijacking', and 'Recon-PingSweep', with most of them having low support scores compared to the other classes. Other erroneous behavior occurred when the model did not list any output class as true. These output statistics show that optimizations can be made to the depth/structure of the neural network's hidden layers and/or the overall training process to improve the performance of our model.

The raw performance results between the training and testing predictions on our initial model can be found in the table below:

|                | Training             | Testing              | Training - Testing     |
|----------------|----------------------|----------------------|------------------------|
| Avg. Accuracy  | 0.7900885540898384   | 0.7901161570886341   | -0.0000276029987960058 |
| Avg. Precision | 0.6046030488353284   | 0.5658273423398823   | 0.038775706495446      |
| Avg. Recall    | 0.4598748252039447   | 0.45844720110965215  | 0.00142762409429203    |
| MSE            | 0.011849981945609444 | 0.01186226078292997  | -0.0000122788373204986 |

The initial model had a good accuracy score of roughly 79% on both the training and testing data. The recall and MSE between the training and testing predictions are extremely similar, whereas the precision varied slightly more at 3.9%. 

## Model 2

Our second model followed the basis of our initial model but with a lower complexity. This was a 3-layer neural network with 16, 15, and 34 nodes. Again, it included the ReLU activation function as part of its hidden layers, and the softmax activation function in its output layer. This model was given the same testing and training data as the initial model, and was fit with 50 epochs. The cross-entropy loss was about 44.9%.  Precision, recall and F1 score were calculated for each type of IoT traffic intrusion. Training and testing metrics are shown below.

|                | Training             | Testing              | Training - Testing     |
|----------------|----------------------|----------------------|------------------------|
| Avg. Accuracy  | 0.7763333711271494   | 0.7764309828530012   | -0.00009761173000      |
| Avg. Precision | 0.5330412755881193   | 0.5335294783744546   | -0.00048820280000      |
| Avg. Recall    | 0.433353845574752    | 0.43278794694819006  | 0.000565898600000      |
| MSE            | 0.012340782122556807 | 0.01233348629463039  | 0.000007295828000      |

When comparing to our initial model, accuracy, recall, and MSE scores are relatively close in value. However, training precision in this model is about 53%, differing from the training precision of around 60% in our initial model. 

## Model 3

Model 3 was another model with lower complexity than our initial model but higher complexity than Model 2. This 3-layer neural network had nodes of 26, 25, and 34. ReLU activation functions were used in the hidden layers and softmax in the output layers. After 50 epochs, the cross-entropy loss was about 42.9%. Once again, this model used the same training and testing data to make predictions and record scores.

|                | Training             | Testing              | Training - Testing     |
|----------------|----------------------|----------------------|------------------------|
| Avg. Accuracy  | 0.7549350069989202   | 0.7524557019969864   | -0.000024793050000     |
| Avg. Precision | 0.6005142001380789   | 0.5702826286230716   | 0.030231570000         |
| Avg. Recall    | 0.4719180867818049   | 0.47043013000860856  | 0.001487957000         |
| MSE            | 0.013887731422973323 | 0.01404392270555250  | -0.000156191300        |

MSE and precision values were close to those in our initial model. Training precision and testing precision in this model differed by about 3%.

## Model 4

This model had a higher complexity than the initial models and used layers with 46, 45, and 34 nodes. It used the same activation functions as the previous models and was fit with the same data. It had a cross entropy-loss of about 43.1%.

|                | Training             | Testing              | Training - Testing     |
|----------------|----------------------|----------------------|------------------------|
| Avg. Accuracy  | 0.7852788494855979   | 0.7851666062675237   | 0.000112243200         |
| Avg. Precision | 0.6127326442341196   | 0.5530792634479434   | 0.059653380000         |
| Avg. Recall    | 0.46653659393040675  | 0.46861520072578733  | -0.00207860700         |
| MSE            | 0.012108191208119533 | 0.0121306910297736   | -0.00002499820         |

MSE was close to MSE values of our initial model. There was about a 5% difference between training and testing precision in this model.

## Model 5

Our fifth model had the highest complexity of all, with layers consisting of 56, 55, and 34 nodes. The same activation functions were used and it was fit with 50 epochs to yield a cross-entropy loss of 42.9%. 

|                | Training             | Testing              | Training - Testing     |
|----------------|----------------------|----------------------|------------------------|
| Avg. Accuracy  | 0.7990181378527673   | 0.7995193499780656   | -0.0005012121000       |
| Avg. Precision | 0.6380845247800311   | 0.646108874149697    | -0.0080243490000       |
| Avg. Recall    | 0.4760772627054606   | 0.4774422504412938   | -0.0013649880000       |
| MSE            | 0.011265871143199772 | 0.0112558384279916   | 0.00001003272000      |

The MSE values of this model are very close to the MSE values of our initial model. 

# Results

The training and testing scores for each of the trained comparison models were generated and plotted in a scatter plot. 

![image](https://github.com/matthewarmstr/ECS-171-Project/assets/72539988/f476aab3-7956-440b-998d-e14e529b3280)

Since all five models produced similar MSE scores, it is difficult to determine the exact fitting conditions of our initial model. However, given that the MSE values from the training and testing predictions are extremely close to one another (0.01185 and 0.01186), it is likely that our initial model is experiencing either underfitting or adequate fitting conditions.

# Discussion

We thought a neural network would do the best job of predicting and classifying IoT traffic intrusions. This initial model was found to be accurate but less precise than desired. The additional four models intended to determine the fitting conditions of our initial model, by changing the number of nodes in each layer to obtain higher and lower complexities. A significant reason for the error in our models could be the disproportioned sampling of output classes. The bar graph below displays the number of occurances for each type of intrusion. 

![image](https://github.com/matthewarmstr/ECS-171-Project/assets/72539988/c0ac8ef4-b67b-4c3d-868b-e278c41e3660)

As shown, the size of samples ranged from 23 (Uploading Attack) to 161281 (DDoS-ICMP Flood). The distribution of occurances for each class varied tremendously, and there were were many classes with negligible samples. This dispairity likely limited the training ability for our models, thus leading to increased error. In attempt to mitigate the effects of this undersampling, input data could be removed to decrease the number of data for outliers. This would even out the representation of output classes. 

# Conclusion
