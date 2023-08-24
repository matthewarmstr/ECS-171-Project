
Data Preprocessing

For our project, we decided to use a neural network to classify the type of instrusion based on the collections of 50 attributes in our dataset. To preprocess our data, we decided to normalize and standardize all of our attributes since they were all numerical.

Data Exploration

There are 1191264 observations in our dataset. There are no missing data points for any observation. The data attributes can be classified having a normal distribution, but certain attributes describing network/application protocols (HTTP, TCP, UDP) have binary values (1,0) representing True or False behavior and indicate if the instrusion utilized these protocols. We decided to use min-max scaling since the data follows a normal distribution. Using categorical plots, we were able to form subsets of attribute data by class and create a scatterplot for each attribute which also separated data by class.