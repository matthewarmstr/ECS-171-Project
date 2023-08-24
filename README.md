# Introduction
## Project Description

Our project focuses around the Internet of Things Network (IoT) and working with Intrusion Detection, which is meant to identify malicious attacks across networks towards users. Some of the categories we will be analyzing involve DDoS, Brute Force, Spoofing, etc. We are building off an existing work that has several subcategories for individual types of intrusions, including their instances and features. The purpose of the project is meant to provide a dataset for a predictive model that can identify and detect multiple kinds of intrusions. 

## Data Preprocessing

For our project, we decided to use a neural network to classify the type of instrusion based on the collections of 50 attributes in our dataset. To preprocess our data, we decided to **normalize** and **standardize** all of our attributes since they were all numerical.

While we may be currently using a neural network as our primary model, we will likely incorporate other models in the future as we continue our analysis of the dataframe.

## Data Exploration

There are 1191264 observations in our dataset. There are no missing data points for any observation. The data attributes can be classified having a normal distribution, but certain attributes describing network/application protocols (HTTP, TCP, UDP) have binary values (1,0) representing True or False behavior and indicate if the instrusion utilized these protocols. We decided to use min-max scaling since the data follows a normal distribution. Using categorical plots, we were able to form subsets of attribute data by class and create a scatterplot for each attribute which also separated data by class.

Some of the attributes in the original dataframe were redundant and would not be useful for the predictive model. This is because either they weren't independent variables or for every single observation, it would have the same value. For instance, TeleNet and IRC were two cases where their values were all set to 0 without significance to the actual model.
