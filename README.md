# Introduction

Our project focuses on the Internet of Things Network (IoT) and working with Intrusion Detection, which is meant to identify malicious attacks across networks towards users. We chose this project because of the safety implications that having such a predictive model brings. Some of the categories we will be analyzing involve DDoS, Brute Force, Spoofing, etc. We are building off an existing work that has several subcategories for individual types of intrusions, including their instances and features ([link](https://www.kaggle.com/datasets/subhajournal/iotintrusion)). The purpose of the project is to design and implement a deep neural network (DNN) that can detect and identify multiple kinds of IoT traffic intrusions. A good predictive model is important because it can use other attributes to help users recognize network attacks.

# Methods

## Data Exploration

There are 1191264 observations in our dataset. There are no missing data points for any observation. Certain attributes describing network/application protocols (HTTP, TCP, UDP) have binary values (1,0) representing True or False behavior and indicate if the intrusion utilized these protocols. We decided to use min-max scaling since the input data columns do not follow a normal distribution (more on this later in Data Preprocessing). Using categorical plots, we were able to form subsets of attribute data by class and create a scatterplot for each attribute which also separated data by class.

Some of the attributes in the original dataset were redundant and would not be useful for the predictive model. This is because either they weren't independent variables or for every single observation, it would have the same value. For instance, TeleNet and IRC were two cases where their values were all set to 0, so their input would have no significance to the actual model.

## Data Preprocessing

For our project, we decided to use a neural network to classify the type of intrusion based on the collections of many attributes in our dataset. To preprocess our data, we decided to normalize all of our attributes. Standardization was not used because the results of running the Shapiro-Wilks test on all the attribute columns showed that the input data was not normally distributed. The intrusion output classes were then encoded with one-hot encoding. Label encoding was not utilized for our multiclass classification model because there is no underlying ranking of intrusion classes in the dataset - they are simply different types of IoT traffic that were collected as part of the dataset.

## Initial Model

When setting up the initial model, the number of inputs for the neural network was reduced from the original forty-three valid inputs to thirty-five. A few of these ignored attribute labels include 'Tot sum', 'Min', 'Max', 'AVG', and 'Covariance'. These ignored inputs may be related to the intermediate timing distribution between the packets within the segment of surveyed traffic, but it's unclear what exactly they represent as they were not described in the original dataset. As such, these unclear attributes were left out of the initial model. Once all thirty-five selected attributes were split according to a train/test ratio of 90:10, the input columns were scaled appropriately and sent to the initial 3-layer neural net model. The model was given an input dimension of thirty-five (one for each selected attribute) and one hidden layer with node layouts of thirty-six and thirty-five, respectively. The output layer was given thirty-four nodes corresponding to the same number of intrusion classes from the dataset. The softmax activation function was used in the output layer since this function provides an intuitive way of predicting which single output class is most likely to be true. The nodes within the input and hidden layers used the ReLU activation function because it improved the neural net's performance substantially compared to using all softmax functions. The model was optimized with stochastic gradient descent with the default learning rate and categorical cross entropy as the loss function. Finally, the model was fit with the training data over fifty epochs. 

To measure the performance of the initial model, multiple scores were computed on the training and testing prediction data. Some of these scores included the mean squared error and average accuracy, precision, and recall across all output class predictions. The classification report was also generated for the training and testing predictions, which expanded the precision, recall, and f1 scores for each intrusion class. After compiling and training the initial model with fifty epochs, the categorical cross-entropy loss remained at about 43.1%. The average accuracy when predicting with the training and testing sets was around 80% and 72% respectively, but the average precision and recall scores collectively ranged from 42% to 64%. This difference in average performance scores occurred because many output classes were never predicted to be true (which defaulted their precision and recall scores in the classification report to 0). A few of these classes included 'Backdoor_Malware', 'BrowserHijacking', and 'Recon-PingSweep', with most of them having low support scores compared to the other classes. Other erroneous behavior occurred when the model did not list any output class as true. These output statistics show that optimizations can be made to the depth/structure of the neural network's hidden layers and/or the overall training process to improve the performance of our model.

## Revision 1 Model: Removing Undersampled Output Classes

After analyzing the results from the initial model, it became clear that many output classes were predicted to be true because there simply were not enough samples that the model could train on. Samples that had an output class set to one of the 7 most undersampled intrusion classes were removed. This new revised model was generalized with K-fold cross-validation, which showed better performance metrics compared to those from the initial model.

## Revision 2 Model: Adding Another Hidden Layer, More Nodes for the Reduced Dataset

To expand on the first revised model, another hidden layer was added to the model from Revision 1, and more nodes were incorporated through all the non-output layers to increase the model's complexity. K-fold was used to generalize the mode, which performed better than Revision 1. The model was fit with the training data and analyzed with the testing data, producing similar performance improvements.

## Revision 3 Model: Adding Even More Layers and Nodes

The revised model's complexity was once again increased with the addition of 2 more hidden layers and many more nodes. Cross-validation and model fitting once again showed that the performance of this newly revised model was better than that of the model in Revision 3.

## Revision 4: Adding More Nodes, Changing from Descending Structure to Pyramid Node Structure (Same Number of Hidden Layers)

To further explore the relationship between the model's performance and its input complexity, the internal node structure was changed from a decreasing layout in Revisions 1 through 3 to a pyramid-like structure, which had more nodes towards the middle hidden layers and less on the input and output ends. The performance of this last model revision was slightly different from the other iterations.

# Results

## Initial Model

The raw performance results between the training and testing predictions on our initial model can be found in the table below:

|                | Training | Testing  | Training - Testing   |
|----------------|----------|----------|----------------------|
| Avg. Accuracy  | 0.8003   | 0.7219   | 0.0784               |
| Avg. Precision | 0.6403   | 0.5571   | 0.0832               |
| Avg. Recall    | 0.4840   | 0.4250   | 0.0591               |
| MSE            | 0.01120  | 0.0158   | -0.0046              |

Overall, the initial model performed relatively well, with the difference between the average training and testing accuracy being about 7.8%. The MSE scores between the training and testing predictions were extremely similar, whereas the average precision and recall varied slightly more at around 8.3% and 5.9% respectively. These performance metrics were generalized by performing K-fold cross-validation over 10 folds on the entire dataset for the initial model. Taking the mean performance scores from each successive fold performed on the initial model resulted in an average accuracy of 74.9% and an average MSE score of 1.32%.

### Comparison Model 1

The first comparison model used to find the fitting conditions followed the basis of our initial model but with a lower complexity. This was a 3-layer neural network with 16, 15, and 34 nodes. Again, it included the ReLU activation function as part of its hidden layers, along with the softmax activation function in its output layer. This model was given the same testing and training data as the initial model and was fit with 50 epochs. The final cross-entropy loss was about 43.7%.  Precision, recall, and F1 scores were calculated for each type of IoT traffic intrusion. Training and testing metrics are shown below.

|                | Training | Testing  | Training - Testing   |
|----------------|----------|----------|----------------------|
| Avg. Accuracy  | 0.7913   | 0.7180   | 0.0732               |
| Avg. Precision | 0.5860   | 0.5099   | 0.0761               |
| Avg. Recall    | 0.4420   | 0.3917   | 0.0502               |
| MSE            | 0.0118   | 0.0160   | -0.0042              |

When compared to our initial model, accuracy, recall, and MSE scores are relatively close in value. All of the average accuracy, precision, and recall scores for the training and testing datasets were anywhere from 0.5% to 9% lower than the equivalent results from the initial model.

### Comparison Model 2

Another model with lower complexity than our initial model was constructed that held a complexity than the first comparison model. This 3-layer neural network had nodes of 26, 25, and 34. ReLU activation functions were used in the hidden layers and softmax in the output layers. After 50 epochs, the cross-entropy loss was about 43.3%. Once again, this model used the same training and testing data to make predictions and record scores.

|                | Training | Testing  | Training - Testing   |
|----------------|----------|----------|----------------------|
| Avg. Accuracy  | 0.7925   | 0.7182   | 0.0743               |
| Avg. Precision | 0.5853   | 0.5119   | 0.0732               |
| Avg. Recall    | 0.4620   | 0.4137   | 0.0482               |
| MSE            | 0.0118   | 0.0161   | -0.0043              |

Once again, the MSE and other average performance metrics were similar to those from the initial model. As expected with the increase in complexity, all the average accuracy, precision, and recall scores were usually higher than those from the first comparison model.

### Comparison Model 3

This model was given a higher complexity than the initial models and used layers with 46, 45, and 34 nodes. It used the same activation functions as the previous models and was fit with the same data. It had a final cross-entropy loss of about 42.9%.

|                | Training | Testing  | Training - Testing   |
|----------------|----------|----------|----------------------|
| Avg. Accuracy  | 0.7930   | 0.7153   | 0.0777               |
| Avg. Precision | 0.6110   | 0.5595   | 0.0515               |
| Avg. Recall    | 0.4814   | 0.4288   | 0.0526               |
| MSE            | 0.0116   | 0.0161   | -0.0045              |

The average accuracy, precision, and recall scores for the training and testing sets of the third comparison model varied from the scores in the initial model by anywhere from 0.42% to 4.7%.

### Comparison Model 4

Our final comparison model for the initial model had the highest complexity of all, with layers consisting of 56, 55, and 34 nodes. The same activation functions were used and it was fit with 50 epochs to yield a cross-entropy loss of 42.8%. 

|                | Training | Testing  | Training - Testing   |
|----------------|----------|----------|----------------------|
| Avg. Accuracy  | 0.8101   | 0.7270   | 0.0830               |
| Avg. Precision | 0.6450   | 0.5777   | 0.0673               |
| Avg. Recall    | 0.4972   | 0.4311   | 0.0660               |
| MSE            | 0.0108   | 0.0156   | -0.0048              |

The MSE values of this model are very close to the MSE values of our initial model. This last comparison model also yielded higher average accuracy, precision, and recall scores among the testing and training prediction scores compared to the equivalent statistics from the initial model.

## Initial Model Fitting Results

The training and testing scores for each of the fitted comparison models in relation to the initial model were generated and plotted in a scatter plot. 

![image](https://github.com/matthewarmstr/ECS-171-Project/assets/130258852/ca1244e9-f4e4-475a-bc39-ca79beb29a52)

Since all five models produced similar MSE scores, it is difficult to determine the exact fitting conditions of our initial model. However, given that the MSE values from the training and testing predictions are very close to one another (0.0112 and 0.0158) and that the gap between the testing and training MSE scores does not widen when the model complexity is increased, it is likely that our initial model is experiencing either underfitting or adequate fitting conditions. More complex models need to be designed and implemented to improve the performance of our model.

## Improving the Initial Model

### Revision 1: Removing Undersampled Output Classes

In the first revision of our initial model, this exact approach was undertaken. Some of the output classes that were removed included the 'Uploading_Attack', 'Recon-PingSweep', 'XSS', and 'BrowserHijacking' data columns, with each being represented by fewer than 150 samples in the entire dataset. A total of 579 data samples were removed in this process, which accounts for less than 0.1% of the total number of data samples in the total dataset. This new reduced input dataset also removed the same column attributes as discussed in the derivation of the initial model. A new barplot showing the new class distribution was then generated, which is shown below.

![image](https://github.com/matthewarmstr/ECS-171-Project/assets/130258852/be5c4554-28c0-46c5-bcbf-5d1b2e8f865a)

After adjusting the initial model structure to accommodate the reduced number of output classes (27 instead of 34), running K-fold cross-validation with 10 folds on the new model resulted in an average accuracy score of 75.9% (~1% better than the initial model) and an average MSE of 1.59% (~0.266% higher than the initial model). This small improvement indicates that the overall model performance would generally increase with the removal of the intrusion classes that are extremely undersampled.

### Revision 2: Adding Another Hidden Layer, More Nodes for the Reduced Dataset

In the second revision of our model, we set out to add another hidden layer while also incorporating more nodes for the reduced dataset.

|                | Training | Testing  | Training - Testing   |
|----------------|----------|----------|----------------------|
| Avg. Accuracy  | 0.7989   | 0.7256   | 0.0733               |
| Avg. Precision | 0.7848   | 0.7259   | 0.0589               |
| Avg. Recall    | 0.6127   | 0.5577   | 0.0550               |
| MSE            | 0.0144   | 0.0198   | -0.0054              |

What we found was that the generalized model from K-fold led to overall better performance, where the K-fold average accuracy was 77.6% and the MSE was 1.55% (an improvement over revision 1). The resulting statistics in the above table demonstrate that the model achieved overall higher average accuracy, precision, and recall values compared to both our initial and Revision 1 models.

### Revision 3: Adding Even More Layers and Nodes

Revision 3 consisted of adding even more layers and nodes to our model.

|                | Training | Testing  | Training - Testing   |
|----------------|----------|----------|----------------------|
| Avg. Accuracy  | 0.7961   | 0.7247   | 0.0714               |
| Avg. Precision | 0.7961   | 0.7069   | 0.0892               |
| Avg. Recall    | 0.6173   | 0.5534   | 0.0639               |
| MSE            | 0.0145   | 0.0197   | -0.0052              |

Similar to Revision 2, the generalized model from K-fold also led to overall better performance, where the average accuracy was an improvement at 78.2% with a slightly lower MSE of 1.53%. After training the Revision 3 model, the average accuracy slightly decreased by 0.02% but the average precision increased to 79.6%, both stats indicating a small improvement in the model's functionality compared to Revision 2.

### Revision 4: Adding More Nodes, Changing from Descending Structure to Pyramid Node Structure (Same Number of Hidden Layers)

For Revision 4 as our last model, we chose not only to add more nodes, but also went about changing from a Descending Structure to a Pyramid Node Structure, while keeping the same number of hidden layers. 

|                | Training | Testing  | Training - Testing   |
|----------------|----------|----------|----------------------|
| Avg. Accuracy  | 0.7591   | 0.7341   | 0.0250               |
| Avg. Precision | 0.7505   | 0.7171   | 0.0334               |
| Avg. Recall    | 0.6012   | 0.5775   | 0.0237               |
| MSE            | 0.0175   | 0.0193   | -0.0018              |

Once again, K-fold also showed improvements in the generalized model, but the results were still about the same as Revision 3. Compared to Revision 3, this generalized model for Revision 4 had an average accuracy of 78.0% and a MSE of 1.53%. We chose to use a Pyramid Node structure out of curiosity to see how increasing its level of complexity would affect the results. What we found was that the results were shockingly much lower than what we initially anticipated. The revised model after training dropped in its average accuracy to 68.4% and average precision to 74.6%. With a staggering difference in such values, we believe that it is possible that transitioning to the Pyramid Node Structure was the main cause for this effect.

## Final Model/Summary

Looking at the data from the revised models, we can conclude that the 3rd revised model (Revision 3) yielded the best results. This model included 70 nodes in the input layer, followed by four hidden layers, each with 60, 50, 40, and 30 nodes. All the non-input layers utilized the ReLU activation function, and the output layer was given the softmax activation function. The model was optimized with stochastic gradient descent and used categorical cross-entropy as its loss function. The changes compared to the the initial model included adding more nodes, more hidden layers, introducing cross-validation via k-fold, and dropping low-impactful data. This final revised model had a training average accuracy of 79.61%, an average precision of 79.61%, an average recall of 61.73%, and an MSE of 1.45%. 

Plotting the training and testing MSE scores for the three revised models resulted in the graph shown below. Because the model in Revision 3 (points on the right side of the plot) used a pyramid node structure, their MSE-complexity trends are likely different than those in the models from Revisions 2 and 3 that used a descending structure. It is likely that our final model is still experiencing either underfitting or adequete fitting conditions since there was not a large gap present between the training and testing MSE scores.

![image](https://github.com/matthewarmstr/ECS-171-Project/assets/130258852/f3d420df-ec82-40e7-998e-ec68647f1753)

The detailed classification report showing the performance scores for each output class can be found in the notebook file in the final section titled 'Final Model and its Fitting Characteristics'.

# Discussion

We determined that a neural network was well-suited to predict and classify IoT traffic intrusions. This initial model was found to be fairly accurate but less precise than desired. The additional four models intended to determine the fitting conditions of our initial model, by changing the number of nodes in each layer to obtain higher and lower complexities. A significant reason for the error in our models could be the disproportioned sampling of output classes. The bar graph below displays the number of occurrences for each type of intrusion.

![image](https://github.com/matthewarmstr/ECS-171-Project/assets/72539988/c0ac8ef4-b67b-4c3d-868b-e278c41e3660)

As shown, the size of samples ranged from 23 (Uploading Attack) to 161281 (DDoS-ICMP Flood). The distribution of occurrences for each class varied tremendously, with multiple classes having an extremely small number of samples compared to the classes that were represented by over a hundred thousand samples. This disparity likely limited the training ability of our models, thus leading to increased error. To mitigate the effects of this undersampling, certain input data columns that were extremely undersampled were removed to lower the outlier output intrusion classes. The models outlined and analyzed in Revisions 1-4 explore different input complexities and approaches to fitting the revised input data.

# Conclusion

During the course of the project, we tried different versions of a neural network and dropped specific data attributes and output classes that influenced the performance of each model iteration. K-fold cross-validation was also used between each model iteration to quickly generalize the model's performance and determine whether new model iterations would result in sufficient performance improvements. Overall, these efforts did help improve the performance metrics of the model. Looking back, we wish we had more time to test more versions of the model with changes in different aspects (different amounts of hidden layers, nodes, epochs, activation functions...etc.) since testing took a long time for each model we created. While the average precision and accuracy of our model were relatively high, the recall was significantly lower, so future models would hopefully be designed to avoid false negatives and improve recall. 

We also realized that trying to oversample and undersample would skew the results significantly, lowering the accuracy and overall performance significantly. This would be because some intrusion classifications have just 300 instances and others would have tens or hundreds of thousands, so an attempt to generalize them actually did not help the model fitting. We would hope that in order to mitigate the effects of this biased dataset in the future, the data we handle would have a more balanced number of observations across every classification type.

# Collaboration
Matthew Armstrong (Main Code Contributor, Writer): Led the team in the direction of how to improve the models. Coded most of the models and supervised most of the model training. Helped write the results of each model's classification report and the overall results.
Matthew Tom (Writer): Started and contributed to a majority of the README.md during the abstraction and data exploration milestones, helped analyze performance results during model construction.
Rahul Prabhu: (Code Contributor, Writer): Assisted with writing, formatting, and interpreting the results of each revised model into the README.md, contributed with coding the data exploration/visualization, researched how to efficiently display the data using catplots due to the high number of distinct classifications.
Kyle Tsuji (Code Contributor): Generated and formatted models for Revisions 2 and 3, helped determine how to create and manipulate revision models.
