# Churn Segmentation Modelling ANN

This is a complete Project that revovles around churn modelling and it contains every aspect from data cleaning down to model deployment. The data of a bank was used in this implementation and for modelling purposes an Artificial Neural Network was trained and used to predict the probability that a given customer would leave the bank(With 87% accuracy) and for deployment an API was developed which can be used for single prediction as well as batch prediction for a number of cutomers

> 'Churn’ refers to the rate at which a subscription company loses its subscribers because of subscription cancellations or elapses. This leads to loss of revenue. Churn rates really matter for subscription businesses because they are an important indicator of long term success.  
# Highlights of the Project

- #### Data Cleaning
    - #### Outlier Detection
    - #### Skewness
- #### Exploratory Data Analysis
- #### Featrue Engineering
- #### Model Development
    - #### Hyperparameter tuning using GridSearchCV
    - #### Bias and Variance Analysis using Cross Validation Score
    - #### Validation and Evaluation
- #### Model Deployment using Flask 

# Walkthrough:

## 1. About the Data:

### All the Data used in this Project is from Kaggle Churn Modelling Dataset  

- [Kaggle Dataset: Churn Modelling](https://www.kaggle.com/shrutimechlearn/churn-modelling)  

### File Description

- Churn_Modelling_Original.csv(The fulll dataset without any preprocessing)

### Data Fields

You can find more about the data [Here](https://www.kaggle.com/shrutimechlearn/churn-modelling)

## 2. Data Cleaning: ([Data Cleaning Notebook](https://github.com/ITrustNumbers/Churn_Segmentation_Modelling_ANN/blob/master/Data_Cleaning_and_EDA.ipynb))

#### Outlier Detection:

> Box plots were used for the initial assesment of the data which concluded that two features 'Age' and 'CreditScore' might have outliers

![Box Plots](https://github.com/ITrustNumbers/Churn_Segmentation_Modelling_ANN/blob/master/Visualizations/Box_Plots.png)  

> Hence, I only considered these two features while treating for outliers.  
Also, we have a various methods to find the outliers. In this project i've used the IQR method but other options are:

    1. Z-score method
    2. Robust Z-score
    4. Winterization method(Percentile Capping)
    5. DBSCAN Clustering
    6. Isolation Forest
    
> Two data points were identified as outliers and were hence removed

![outliers](https://github.com/ITrustNumbers/Churn_Segmentation_Modelling_ANN/blob/master/_images/Outliers.png)

#### Skewness:

> Sample Skewness of each feature was calculated using the scipy stats.skew function and was plotted for analysis

![Skewness](https://github.com/ITrustNumbers/Churn_Segmentation_Modelling_ANN/blob/master/Visualizations/Skewness_Plot.png)

> Along with that QQ plots were also studied

![QQ Plots](https://github.com/ITrustNumbers/Churn_Segmentation_Modelling_ANN/blob/master/Visualizations/QQ_Plots.png)

> The Study showed both of the two continuous variable 'Balance' and 'EstimatedSalary' were skewed, And since we have to scale our data to fit an ANN we can use a StandardScalar that will not only scale the data but also standardize it. And hence, we don't have to treat the data for skewness as Standardization will negative a lot of this skewness.

## 3. Exploratory Data Analysis(EDA): ([EDA Notebook](https://github.com/ITrustNumbers/Churn_Segmentation_Modelling_ANN/blob/master/Data_Cleaning_and_EDA.ipynb))

> For getting insights from the data various plots like Histograms, Pivoted Histograms and heatmaps were created

#### Histogram for Categorical Variables:
> Categorical variables: 'Geography', 'Gender', 'NumberofProducts', 'HasCrCrad', 'IsActiveMember', 'Exited'(Target variable/Label)

![Histogram for Categorical Variables](https://github.com/ITrustNumbers/Churn_Segmentation_Modelling_ANN/blob/master/Visualizations/Histograms.png)

#### Histogram for Pivot Data of Categorical variables on the Target Variable(Exited):

![Histogram for Pivot Data of Categorical variables on the Target Variable](https://github.com/ITrustNumbers/Churn_Segmentation_Modelling_ANN/blob/master/Visualizations/Histogram_againts_Exited.png)

#### Correlation Matrix:

![Correlation Matrix](https://github.com/ITrustNumbers/Churn_Segmentation_Modelling_ANN/blob/master/Visualizations/Correlation_Matrix.png)

#### Observations from EDA:

> 1. The Distribution of the Target variable is highly imbalanced, out of 10000 instances 7961 were negative samples or sample were the customer did not leave the bank which translate to a Negative to Postive imbalance of alomst 0.79 - 0.21, which means that even a naive classifier(only gives negative prediction) will reach an accuracy of 79% and hence the baseline accuracy for evaluation can be set at 80%.  

> 2. The histograms of variables when pivoted around the target variable shows that there are variable which have highly imbalance pivot distribution for example 'NumofProducts' and 'Gender'. Such distribution are both good and bad for Machine Learning algorithms, good because such features can be used to create a clearer distinction between the classes by the classifier and bad because models tends to overfit to the training data due to such distribution and hence a dropout layer which will reduce overfitting is necessary while training the ANN.  

> 3. The Correlation Matrix higlights the point that there are no feature in the dataset with a high correlation(Greater than 0.5) with the target variable 'Exited'. And hence, Traditional ML Models are expected to not give good results. This fact was the whole reason behind the use of ANN for this particular problem and dataset in this project.

## 4. Feature Engineering: ([Feature Engineering Notebook](https://github.com/ITrustNumbers/Churn_Segmentation_Modelling_ANN/blob/master/Feature_Engineering_and_Model_Building.ipynb))

### Categorical Variable Encoding

#### Mapping:  
>Since 'Gender' was a binary variable it was encoded with the help of the Scikit-Learn LabelEncoder:

    Male ----> 1
    Female ----> 2
    
> The variable 'Geography' had more than 2 categories it was encoded using Scikit-Learn OneHotEncoding and subsequently one of the column was droped to recover from the dummy trap:

    France ----> [0,0]
    Spain ----> [0,1]
    Germany ----> [1,0]
    
### Standardization

>Standardizing a dataset involves rescaling the distribution of values so that the mean of observed values is 0 and the standard deviation is 1.  
This can be thought of as subtracting the mean value or centering the data.

Standardization assumes that your observations fit a Gaussian distribution (bell curve) with a well-behaved mean and standard deviation

A value is standardized as follows:

![Z-Score](https://github.com/ITrustNumbers/Churn_Segmentation_Modelling_ANN/blob/master/_images/Standardization.png)

Where the mean is calculated as:

![Mean](https://github.com/ITrustNumbers/Churn_Segmentation_Modelling_ANN/blob/master/_images/Mean.png)

And the standard_deviation is calculated as:

![Standard Deviation](https://github.com/ITrustNumbers/Churn_Segmentation_Modelling_ANN/blob/master/_images/Standard_Deviation.png)

## 5. Modell Building: ([Modell Building Notebook](https://github.com/ITrustNumbers/Churn_Segmentation_Modelling_ANN/blob/master/Feature_Engineering_and_Model_Building.ipynb))

> While building the ANN, i will the sklearn function 'GridSearchCV' for tuning the hyperparameter of the ANN but since the ANN is made through keras library with tensorflow backend the model would not be compatible with sklearn.  
To resolve this problem we have to use the sklearn wrapper given in the kears library that will take the keras ANN object and gives out a Classifier object that is compatible with the sklearn library and then we can use the GridSearchCV function

To Use the keras wrapper for sklearn 'KerasClassifier'(You can read more about the wrapper [Here](https://faroit.com/keras-docs/1.0.6/scikit-learn-api/))   
We have to create the architecture of the ANN inside a builder function which will be passed inside the wrapper to generate the sklearn compatible ANN classifier.

You can look at the build function and the wrapper implementation as well as GridSearchCV implementation in my project in this [notebook](https://github.com/ITrustNumbers/Churn_Segmentation_Modelling_ANN/blob/master/Feature_Engineering_and_Model_Building.ipynb)


### ANN Architecture:

After experimenting with various configuration of number of neurons and number of hidden layers, I settled with a Architecture that containe:

    Number of Hidden Layers = 3
    Number of Dropout Layers = 3(After Each Hidden Layer)
    Dropout paramter = 0.1
    Number of Neurons:
    In First Layer = 15 Neurons
    In Second Layer = 25 Neurons
    In Third Layer = 15 Neurons
    
 
    

### GridSearchCV Result:

Parameter Space:

> Optimizer = {'adam','rmsprop}
