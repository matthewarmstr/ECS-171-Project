# Introduction

Our project focuses on the Internet of Things Network (IoT) and working with Intrusion Detection, which is meant to identify malicious attacks across networks towards users. We chose this project because of the safety implications that having such a predictive model brings. Some of the categories we will be analyzing involve DDoS, Brute Force, Spoofing, etc. We are building off an existing work that has several subcategories for individual types of intrusions, including their instances and features. The purpose of the project is to design and implement an effective predictive model that can detect and identify multiple kinds of IoT traffic intrusions. A good predictive model is important because it can use other attributes to accurately help users recognize network attacks.

## Data Exploration

There are 1191264 observations in our dataset. There are no missing data points for any observation. Certain attributes describing network/application protocols (HTTP, TCP, UDP) have binary values (1,0) representing True or False behavior and indicate if the intrusion utilized these protocols. We decided to use min-max scaling since the input data columns do not follow a normal distribution (more on this later in Data Preprocessing). Using categorical plots, we were able to form subsets of attribute data by class and create a scatterplot for each attribute which also separated data by class.

Some of the attributes in the original dataset were redundant and would not be useful for the predictive model. This is because either they weren't independent variables or for every single observation, it would have the same value. For instance, TeleNet and IRC were two cases where their values were all set to 0, so their input would have no significance to the actual model.

## Data Preprocessing

For our project, we decided to use a neural network to classify the type of intrusion based on the collections of many attributes in our dataset. To preprocess our data, we decided to normalize all of our attributes. Standardization was not used because the results of running the Shapiro-Wilks test on all the attribute columns showed that the input data was not normally distributed. The intrusion output classes were then encoded with one-hot encoding. Label encoding was not utilized for our multiclass classification model because there is no underlying ranking of intrusion classes in the dataset - they are simply different types of IoT traffic that were collected as part of the dataset.

## Initial Model

When setting up the initial model, the number of inputs for the neural network was reduced from the original forty-three valid inputs to thirty-five. A few of these ignored attribute labels include 'Tot sum', 'Min', 'Max', 'AVG', and 'Covariance'. These ignored inputs may be related to the intermediate timing distribution between the packets within the segment of surveyed traffic, but it’s unclear what exactly they represent as they were not described in the original dataset. As such, these unclear attributes were left out of the initial model. Once all thirty-five selected attributes were standardized, they were split according to a train/test ratio of 90:10 and sent to the 3-layer neural net model. The model was given an input dimension of thirty-five (one for each selected attribute) and one hidden layer with node layouts of thirty-six and thirty-five, respectively. The output layer was given thirty-four nodes corresponding to the same number of intrusion classes from the dataset. The softmax activation function was used in the output layer since this function provides an intuitive way of predicting which single output class is most likely to be true. The nodes within the input and hidden layers used the ReLU activation function because it improved the neural net's performance substantially compared to using all softmax functions. The model was optimized with stochastic gradient descent with the default learning rate and categorical cross entropy as the loss function. Finally, the model was fit with the training data over fifty epochs. 

To measure the performance of the initial model, multiple scores were computed on the training and testing prediction data. Some of these scores included the mean squared error and average accuracy, precision, and recall across all output class predictions. The classification report was also generated for the training and testing predictions, which expanded the precision, recall, and f1 scores for each intrusion class. After compiling and training the initial model with fifty epochs, the categorical cross-entropy loss remained at about 43.1%. The average accuracy when predicting with the training and testing sets was around 80% and 72% respectively, but the average precision and recall scores collectively ranged from 42% to 64%. This difference in average performance scores occurred because many output classes were never predicted to be true (which defaulted their precision and recall scores in the classification report to 0). A few of these classes included 'Backdoor_Malware', 'BrowserHijacking', and 'Recon-PingSweep', with most of them having low support scores compared to the other classes. Other erroneous behavior occurred when the model did not list any output class as true. These output statistics show that optimizations can be made to the depth/structure of the neural network's hidden layers and/or the overall training process to improve the performance of our model.

The raw performance results between the training and testing predictions on our initial model can be found in the table below:

|                | Training             | Testing              | Training - Testing   |
|----------------|----------------------|----------------------|----------------------|
| Avg. Accuracy  | 0.8003066597295587   | 0.7219000934597265   | 0.0784065662698319   |
| Avg. Precision | 0.6403201274107059   | 0.5571200182627859   | 0.08320010914792     |
| Avg. Recall    | 0.48401263899986613  | 0.4249544040855431   | 0.059058234914323    |
| MSE            | 0.011208619433455531 | 0.015786615624716005 | -0.0045779961912605  |

Overall, the initial model performed relatively well, with the difference between the average training and testing accuracy being about 7.8%. The MSE scores between the training and testing predictions were extremely similar, whereas the average precision and recall varied slightly more at around 8.3% and 5.9% respectively. These performance metrics were generalized by performing K-fold cross-validation over 10 folds on the entire dataset for the initial model. Taking the mean performance scores from each successive fold performed on the initial model resulted in an average accuracy of 74.9% and an average MSE score of 1.32%.

### Comparison Model 1

The first comparison model used to find the fitting conditions followed the basis of our initial model but with a lower complexity. This was a 3-layer neural network with 16, 15, and 34 nodes. Again, it included the ReLU activation function as part of its hidden layers, along with the softmax activation function in its output layer. This model was given the same testing and training data as the initial model and was fit with 50 epochs. The final cross-entropy loss was about 43.7%.  Precision, recall, and F1 scores were calculated for each type of IoT traffic intrusion. Training and testing metrics are shown below.

|                | Training             | Testing              | Training - Testing   |
|----------------|----------------------|----------------------|----------------------|
| Avg. Accuracy  | 0.7912562770406806   | 0.7180281905052547   | 0.073228086535426    |
| Avg. Precision | 0.5859911331401575   | 0.5098867424165296   | 0.0761043907236281   |
| Avg. Recall    | 0.4419651985412158   | 0.3917466821422904   | 0.050218516398925    |
| MSE            | 0.01183424317945765  | 0.015992776786084933 | -0.0041585336066273  |

When compared to our initial model, accuracy, recall, and MSE scores are relatively close in value. All of the average accuracy, precision, and recall scores for the training and testing datasets were anywhere from 0.5% to 9% lower than the equivalent results from the initial model.

### Comparison Model 2

Another model with lower complexity than our initial model was constructed that held a complexity than the first comparison model. This 3-layer neural network had nodes of 26, 25, and 34. ReLU activation functions were used in the hidden layers and softmax in the output layers. After 50 epochs, the cross-entropy loss was about 43.3%. Once again, this model used the same training and testing data to make predictions and record scores.

|                | Training             | Testing              | Training - Testing   |
|----------------|----------------------|----------------------|----------------------|
| Avg. Accuracy  | 0.7925183079249394   | 0.7181998512273742   | 0.0743184566975651   |
| Avg. Precision | 0.5852918123069119   | 0.511998903299858    | 0.073292909007053    |
| Avg. Recall    | 0.4619632265419104   | 0.4137242485978684   | 0.048238977944042    |
| MSE            | 0.011756889481262113 | 0.016054204397431597 | -0.0042973149161694  |

Once again, the MSE and other average performance metrics were similar to those from the initial model. As expected with the increase in complexity, all the average accuracy, precision, and recall scores were usually higher than those from the first comparison model.

### Comparison Model 3

This model was given a higher complexity than the initial models and used layers with 46, 45, and 34 nodes. It used the same activation functions as the previous models and was fit with the same data. It had a final cross-entropy loss of about 42.9%.

|                | Training             | Testing              | Training - Testing   |
|----------------|----------------------|----------------------|----------------------|
| Avg. Accuracy  | 0.7930481277755937   | 0.7153197657784813   | 0.077728361997112    |
| Avg. Precision | 0.611011680371652    | 0.5594912809312805   | 0.051520399440372    |
| Avg. Recall    | 0.4814220002645187   | 0.42882101676096385  | 0.052600983503555    |
| MSE            | 0.01158363839009813  | 0.016050277518167427 | -0.0044666391280693  |

The average accuracy, precision, and recall scores for the training and testing sets of the third comparison model varied from the scores in the initial model by anywhere from 0.42% to 4.7%.

### Comparison Model 4

Our final comparison model for the initial model had the highest complexity of all, with layers consisting of 56, 55, and 34 nodes. The same activation functions were used and it was fit with 50 epochs to yield a cross-entropy loss of 42.8%. 

|                | Training             | Testing              | Training - Testing   |
|----------------|----------------------|----------------------|----------------------|
| Avg. Accuracy  | 0.8100500467830928   | 0.7270403784165252   | 0.083009668366567    |
| Avg. Precision | 0.6449885638412912   | 0.5776834605483122   | 0.067305103292979    |
| Avg. Recall    | 0.4971693039195549   | 0.4311490925542386   | 0.066020211365316    |
| MSE            | 0.010798383039681201 | 0.015594198540771667 | -0.0047958155010904  |

The MSE values of this model are very close to the MSE values of our initial model. This last comparison model also yielded higher average accuracy, precision, and recall scores among the testing and training prediction scores compared to the equivalent statistics from the initial model.

## Initial Model Fitting Results

The training and testing scores for each of the fitted comparison models in relation to the initial model were generated and plotted in a scatter plot. 

![image](https://github.com/matthewarmstr/ECS-171-Project/assets/130258852/ca1244e9-f4e4-475a-bc39-ca79beb29a52)

Since all five models produced similar MSE scores, it is difficult to determine the exact fitting conditions of our initial model. However, given that the MSE values from the training and testing predictions are very close to one another (0.0112 and 0.0158) and that the gap between the testing and training MSE scores does not widen when the model complexity is increased, it is likely that our initial model is experiencing either underfitting or adequate fitting conditions. More complex models need to be designed and implemented to improve the performance of our model.

# Discussion

We thought a neural network would do the best job of predicting and classifying IoT traffic intrusions. This initial model was found to be fairly accurate but less precise than desired. The additional four models intended to determine the fitting conditions of our initial model, by changing the number of nodes in each layer to obtain higher and lower complexities. A significant reason for the error in our models could be the disproportioned sampling of output classes. The bar graph below displays the number of occurrences for each type of intrusion. 

![image](https://github.com/matthewarmstr/ECS-171-Project/assets/72539988/c0ac8ef4-b67b-4c3d-868b-e278c41e3660)

As shown, the size of samples ranged from 23 (Uploading Attack) to 161281 (DDoS-ICMP Flood). The distribution of occurrences for each class varied tremendously, with multiple classes having an extremely small number of samples compared to the classes that were represented by over a hundred thousand samples. This disparity likely limited the training ability of our models, thus leading to increased error. To mitigate the effects of this undersampling, certain input data columns that are extremely undersampled could be removed to lower the outlier output intrusion classes. This would even out the representation of output classes. 

In the first revision of our initial model, this exact approach was undertaken. Some of the output classes that were removed included the 'Uploading_Attack', 'Recon-PingSweep', 'XSS', and 'BrowserHijacking' data columns, with each being represented by fewer than 150 samples in the entire dataset. A total of 579 data samples were removed in this process, which accounts for less than 0.1% of the total number of data samples in the total dataset. This new reduced input dataset also removed the same column attributes as discussed in the derivation of the initial model. A new barplot showing the new class distribution was then generated, which is shown below.

![image](https://github.com/matthewarmstr/ECS-171-Project/assets/130258852/be5c4554-28c0-46c5-bcbf-5d1b2e8f865a)

After adjusting the initial model structure to accommodate the reduced number of output classes (27 instead of 34), running K-fold cross-validation with 10 folds on the new model resulted in an average accuracy score of 76.6% (~1.67% better than the initial model) and an average MSE of 1.59% (~0.266% higher than the initial model). This small improvement indicates that the overall model performance would generally increase with the removal of the intrusion classes that are extremely undersampled.

# Conclusion
