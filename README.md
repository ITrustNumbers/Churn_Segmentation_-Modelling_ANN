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

