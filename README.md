# Introduction
## Project Description

Our project focuses on the Internet of Things Network (IoT) and working with Intrusion Detection, which is meant to identify malicious attacks across networks towards users. Some of the categories we will be analyzing involve DDoS, Brute Force, Spoofing, etc. We are building off an existing work that has several subcategories for individual types of intrusions, including their instances and features. The purpose of the project is meant to provide a dataset for a predictive model that can identify and detect multiple kinds of intrusions. 

While we may be currently using a neural network as our primary model, we will likely incorporate other models in the future as we continue our analysis of the dataset.

## Data Preprocessing

For our project, we decided to use a neural network to classify the type of intrusion based on the collections of many attributes in our dataset. To preprocess our data, we decided to **normalize** and **standardize** all of our attributes since they were all numerical. Standardizing the input data led to an increase in the average accuracy, precision, and recall scores by over 10%*. 

## Data Exploration

There are 1191264 observations in our dataset. There are no missing data points for any observation. The data attributes can be classified having a normal distribution, but certain attributes describing network/application protocols (HTTP, TCP, UDP) have binary values (1,0) representing True or False behavior and indicate if the intrusion utilized these protocols. We decided to use min-max scaling since the data follows a normal distribution. Using categorical plots, we were able to form subsets of attribute data by class and create a scatterplot for each attribute which also separated data by class.

Some of the attributes in the original dataset were redundant and would not be useful for the predictive model. This is because either they weren't independent variables or for every single observation, it would have the same value. For instance, TeleNet and IRC were two cases where their values were all set to 0, so their input would have no significance to the actual model.

## Initial Model

When setting up the initial model, the number of inputs for the neural network was reduced from the original forty-three valid inputs to thirty-five. A few of these ignored attribute labels include 'Tot sum', 'Min', 'Max', 'AVG', and 'Covariance'. These ignored inputs may be related to the intermediate timing distribution between the packets within the segment of surveyed traffic, but it’s unclear what exactly they represent as they were not described in the original dataset. As such, these unclear attributes were left out of the initial model (as a sidenote, the measured performance was very similar when using all 43 inputs versus 35 inputs on models with equivalent complexity*). Once all thirty-five selected attributes were standardized, they were split according to a train/test ratio of 90:10 and sent to the 3-layer neural net model. The model was given an input dimension of thirty-five (one for each selected attribute) and two hidden layers with node layouts of thirty-six and thirty-five, respectively. The output layer was given thirty-four nodes corresponding to the same number of intrusion classes from the dataset. The softmax activation function was used in the output layer since this function provides an intuitive way of predicting which single output class is most likely to be true. The nodes within the two hidden layers used the ReLU activation function because it improved the neural net's performance substantially compared to using all softmax functions. The model was optimized with stochastic gradient descent with the default learning rate and categorical cross entropy as the loss function. Finally, the model was fit with the training data over 20 epochs. 

To measure the performance of the initial model, multiple scores were computed on the training and testing prediction data. Some of these scores included the mean squared error and average accuracy, precision, and recall across all output class predictions. The classification report was also generated for the training and testing predictions, which expanded the precision, recall, and f1 scores for each class of intrusion. After compiling and training the model, the average accuracy of the initial mode when predicting with the training and testing sets was around 95%, but the average precision and recall scores were closer to 55-70%. This difference in performance scores occured because many output classes were never predicted to be true (which defaulted their precision and recall scores in the classification report to 0). A few of these classes included 'Backdoor_Malware', 'BrowserHijacking', and 'Recon-PingSweep', with most of them having low support scores compared to the other classes. Other erronous behavior occured when the model did not list any output class as true. These missing predictions show that adjustments can be made to the neural network's hidden layer node structure and/or the overall training process to improve the performance of our model.

[to do: describe fitting conditions of initial model]


* See results from model output om commit 'cf1f2fe': https://github.com/matthewarmstr/ECS-171-Project/blob/cf1f2fe88c1b007b6c68b3bfdef0762e9e7c1f3f/ECS_171_Project.ipynb
