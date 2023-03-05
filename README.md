# end-to-end-ml-project
this is my end-end machine learning project deployment on aws
# Predict the Mortgage Backed Securities Prepayment Risk Using Machine Leanrning Models.

A mortgage-baked security(MBS) in  an investment is similar to a bond that is made up of a bundle of home loans bought from the banks that issued them. These loans are sold in form of bonds by banks wherein these loans are grouped together according to their type and quality.
Prepayment risk involves premature return of principal amount on a fixed-income security. That means if debtors returns the part of principal amount early they don’t have to pay interest of the principal amount. Which in return causes the loss for the loan provider.

Freddie Mac provides loan-level information at issuance and on a monthly basis for all newly issued fixed-rate, adjustablerate mortgage (ARM), modified and reinstated PC securities issued after December 1, 2005. Inception month and monthly 
loan-level disclosure supplements Freddie Mac’s current daily and monthly pool-level disclosure for new and previously issued securities. 





## Acknowledgement

WE would like to express gratitude and special thanks to Mr.Yasin Shah sir and Mr.Debanjan Mukhopadhyay for assigning and guiding us for this project and would like to express our appreciation to our institution for providing the resources and support necessary to complete this project.

First and foremost, we would like to thank our project team members who worked tirelessly to conduct the analysis and provide valuable insights into the prepayment behavior of mortgage-backed securities. Their dedication, expertise, and hard work were essential to the success of this project.

We would also like to extend our gratitude to our project sponsors and clients, whose funding and support were instrumental in making this project possible. Their vision and commitment to advancing the understanding of mortgage-backed securities prepayment behavior were essential to the project's success.

We are also grateful to the data providers who supplied the necessary data for the project. Without their high-quality loan-level data, market data, and other relevant information, this project would not have been possible.

We would like to acknowledge the technology providers who provided the tools and platforms used to analyze and visualize the data. Their innovative and reliable technology enabled us to conduct 

## The aim of this project is to create an application that can identify the prepayment risk based on the customer profile.

![fredimac](https://user-images.githubusercontent.com/71137605/221143703-2c4dcb20-d13f-48ec-8089-b5376c601a36.JPG)




## Understanding the Dataset

The dataset we worked on is provided by the FreddieMac company that contains the the data of their customers called "LoanExport.csv" that contains CreditScore, FirstPaaymentDate, MaturityDate, MIP, LTV(Loan-to-Value), PPM, LaonSequenceNumber, LoanTermYears, MonthsDeliquent, MonthlyRepayment etc.

Mortgage-backed securities (MBS) are financial instruments that represent ownership in a pool of mortgage loans. These securities are traded in the market and their values are affected by various factors, including prepayment risk.

Prepayment risk is the risk that the underlying mortgages in an MBS will be paid off early. This can happen when homeowners sell their homes, refinance their mortgages, or make prepayments on their loans. When this happens, investors in the MBS may not receive the expected cash flows from the underlying mortgages.

To understand the prepayment risk of an MBS, we need to look at the underlying mortgage loans and the borrowers who hold them. Here are some key factors to consider:

1. Loan Characteristics: The characteristics of the loans, such as the interest rate, loan-to-value ratio, and credit score of the borrowers, can impact prepayment rates. For example, borrowers with high credit scores may be less likely to prepay their mortgages.

2. Borrower Characteristics: The characteristics of the borrowers, such as their income, employment status, and age, can also impact prepayment rates. For example, borrowers who are approaching retirement age may be more likely to prepay their mortgages.

3. Economic Conditions: The overall economic conditions can impact prepayment rates. For example, low interest rates may encourage borrowers to refinance their mortgages, which can result in higher prepayment rates.

4. Geographic Location: The geographic location of the properties can also impact prepayment rates. For example, areas with high housing turnover rates may experience higher prepayment rates.

5. Seasonality: Seasonal factors can impact prepayment rates. For example, homeowners may be more likely to sell their homes in the spring and summer months.






## EDA

Exploratory Data Analysis (EDA) is an approach to analyze the data using visual techniques. It is used to discover trends, patterns, or to check assumptions with the help of statistical summary and graphical representations.

- "LoanExport" dataset comprises of '291451' rows and '28' columns,that comprises of Continious, Descrete Variables, and float datatype.
- We removed null values from the dataset and dropped unnecessary colums from the dataset that were not in use to predict the outcome.
- Various Visualization techniques are applied for getting visual idea of the data and to detect outliers. 

Here are some steps to take in performing EDA for a mortgage-backed securities prepayment risk project:

1. Data Cleaning: The first step in EDA is to clean the data by removing any duplicates, checking for missing values, and removing any irrelevant data. In the case of MBS data, you may need to clean data related to mortgage loans such as interest rate, loan-to-value ratio, credit score, borrower information, economic indicators, and geographic information.

2. Univariate Analysis: In univariate analysis, you should examine each variable separately. For example, you can create histograms or density plots to see the distribution of a variable. This can help you understand the range of values for each variable and identify any outliers.

3. Bivariate Analysis: In bivariate analysis, you should examine how two variables are related to each other. For example, you can create scatterplots or heatmaps to see the correlation between variables. This can help you identify any patterns or relationships between variables that may be important for predicting prepayment risk.

4. Multivariate Analysis: In multivariate analysis, you should examine how multiple variables are related to each other. This can help you identify any complex relationships between variables that may be important for predicting prepayment risk. For example, you can create a correlation matrix to see the correlation between multiple variables.

5. Feature Engineering: In feature engineering, you can create new features from the existing data that may be helpful for predicting prepayment risk. For example, you can create a variable that represents the seasonality of prepayment rates or create a variable that represents the average credit score of borrowers in a particular geographic region.

6. Visualization: Visualization can be a powerful tool in EDA to gain insights into the data. For example, you can create boxplots or violin plots to compare the distribution of a variable across different categories. You can also create line graphs or time series plots to visualize trends over time.

7. Statistical Testing: Finally, statistical testing can be used to test hypotheses and validate assumptions about the data. For example, you can use hypothesis testing to test if there is a significant difference in prepayment rates between different geographic regions.


![chart1](https://user-images.githubusercontent.com/71137605/221143397-3b93fc3d-91f2-4def-9363-1556ca6fa884.JPG)







## Feature Engineering

Feature engineering is the process of selecting, manipulating, and transforming raw data into features that can be used in supervised learning.
Feature engineering improves the performance of the machine learning model by selecting the right features for the model and preparing the features in a way that is suitable for the machine learning model.

- We have distributed Categorical Features, Numerical Features, Continuous Features and Discrete Features.
- we have created new feature using existing ones by normalizing them and putting them into ranges. For example creditScore is converted into CreditRange from '<650' to '<851', based on that Creditrange is categorised under 'Poor', 'Fair', 'Good', 'Excellent' Lables. LTV(Loan-to-value) is converted into LTVRange and DTI is converted into DTIRange, MIP is converted into MIPRange and categorised 'Low', 'Medium', 'High'.Repayment Time is converted into Duration, YearsRepayment is converted into Repay Range.

- After Preprocessing data and Feature Engineering from initial 'CreditScore', 'FirstTimeHomebuyer', 'MIP', 'Units','DTI', 'OrigUPB', 'LTV',  'OCLTV', 'OrigInterestRate', 'PPM', 'PropertyType', 'PostalCode', 'LoanSeqNum', 'LoanPurpose', 'OrigLoanTerm', 'EverDelinquent', 'MonthsDelinquent', 'MonthsInRepayment', 'CreditRange', 'LTVRange', 'DTI_Range', 'MIP_Range', 'Years_Repayment','RepayRange', 'LoanTermYears', 'IsFirstTime' 26 colums we saparated 'CreditRange', 'LTVRange','DTI_Range', 'MIP_Range', 'RepayRange', 'Units','OrigUPB', 'OrigInterestRate', 'LoanTermYears','OCLTV', 'PPM', 'IsFirstTime', 'EverDelinquent','MonthsDelinquent' 14 columns and we got final dataset called 'Modified'


**Outlier Detection of features and cleaning outliers**



![credit](https://user-images.githubusercontent.com/71137605/221256315-cad34ca7-81aa-4393-a28e-4ba1a6bece27.JPG)

![dti](https://user-images.githubusercontent.com/71137605/221256335-a322d0af-62d0-4ea0-90d1-0f5c2309863d.JPG)

![ltv](https://user-images.githubusercontent.com/71137605/221256350-a3a61038-b3cb-4ba1-a66f-a22070bd9c65.JPG)

![mip](https://user-images.githubusercontent.com/71137605/221256361-0a9e307a-804a-4cb6-aefd-e51ad472a3bd.JPG)

![olctv](https://user-images.githubusercontent.com/71137605/221256368-d4a5a23b-f9ed-499e-af55-8fac02114b5c.JPG)

![orig](https://user-images.githubusercontent.com/71137605/221256383-7c862647-f5a6-4226-bf2f-fa5fbd63546e.JPG)

![originterest](https://user-images.githubusercontent.com/71137605/221257869-9930bbd9-1bf2-4388-b259-0966ac2c0ecb.JPG)

![monthsinrepay](https://user-images.githubusercontent.com/71137605/221257896-bb8d87fa-36a9-4736-9f32-0b28ffd108f2.JPG)



**Features in Range**


![fea1](https://user-images.githubusercontent.com/71137605/221147003-82cd58c3-86fa-41ca-9bb9-5eb564b941c1.JPG)

![credit range](https://user-images.githubusercontent.com/71137605/221268658-b6c214c5-5ac5-4eea-b9a8-cee189e1833e.JPG)

**- Credit Range Lable: 0-Execellent, 1-Fair, 2-Good, 3-Poor**__

![fea2](https://user-images.githubusercontent.com/71137605/221147010-85a80274-f6dc-4361-a7a9-18bfdca5fa66.JPG)

![f2](https://user-images.githubusercontent.com/71137605/221268671-5874e80e-5bba-45ae-bb26-fa11fce35564.JPG)

**- LTV Range Lable 0-High, 1-Low, 2-Medium**__

![fea3](https://user-images.githubusercontent.com/71137605/221147022-465a4380-b7ec-41a7-b672-efc65e7db31f.JPG)

![f3](https://user-images.githubusercontent.com/71137605/221268687-1c985807-062f-4ec7-85ed-c4ca93f657f2.JPG)

**- DTI Range Lable:  0-High, 1-Low, 2-Medium**__

![fea5](https://user-images.githubusercontent.com/71137605/221147100-67c3a334-0921-425f-a479-4b3dbc0cbc10.JPG)

![f4](https://user-images.githubusercontent.com/71137605/221268849-ceaf2e23-63ed-4750-81a3-29b200deceb8.JPG)


**- MIP Rang Lablee: 0-High, 1-Low, 2-Medium**__

 ![fea6](https://user-images.githubusercontent.com/71137605/221147118-221b7472-1c92-42fc-91f0-2c50f38f6335.JPG)

 ![f5](https://user-images.githubusercontent.com/71137605/221268868-23a034b3-b24c-44d9-bf9f-4a13c2247ed7.JPG)

**- Repay Range Lable: 0-[0-2], 1-[2-4], 2-[4-6], 3-[6-8], 4-[8-10]**__





## Model Building

What is model building process?
Building a model in machine learning is creating a mathematical representation by generalizing and learning from training data.

**For this perticular project we have used Logistic Regression for real time data analysis.**
In regression analysis, model building is the process of developing a probabilistic model that best describes the relationship between the dependent and independent variables. 

**Logistic Regression**

Logistic regression is a popular statistical method used to model the probability of a binary response variable. In the context of mortgage-backed securities prepayment risk, logistic regression can be used to model the probability of prepayment (i.e., whether or not a borrower will prepay their mortgage). Here are some steps to take in performing logistic regression for MBS prepayment risk:

1. Data Preparation: The first step in logistic regression is to prepare the data by cleaning it and splitting it into training and testing datasets. The training dataset is used to train the logistic regression model, while the testing dataset is used to evaluate the model's performance.

2. Feature Selection: The next step is to select the relevant features (i.e., independent variables) that may be predictive of prepayment risk. In the context of MBS, these features may include loan-to-value ratio, credit score, interest rate, borrower information, economic indicators, and geographic information.

3. Model Training: The logistic regression model is trained using the training dataset. The model attempts to find the best set of coefficients that maximize the likelihood of the observed data.

4. Model Evaluation: The performance of the logistic regression model is evaluated using the testing dataset. Metrics such as accuracy, precision, recall, and F1 score can be used to evaluate the model's performance.

5. Model Interpretation: The coefficients from the logistic regression model can be interpreted to understand the impact of each feature on prepayment risk. For example, a positive coefficient for interest rate would indicate that as interest rates increase, the probability of prepayment decreases.

6. Model Improvement: The logistic regression model can be improved by iterating through the feature selection, training, and evaluation steps. For example, new features may be added or existing features may be transformed to improve the model's performance.

Logistic regression is a powerful method for modeling the probability of prepayment risk in MBS. However, it is important to keep in mind that logistic regression makes assumptions about the underlying distribution of the data, and may not be appropriate for all situations. It is important to carefully evaluate the assumptions of the model and consider alternative methods if necessary.




## Deployment

**Server Deployment**

![0](https://user-images.githubusercontent.com/71137605/221263137-4b2820e1-2469-4fdb-922e-fabfe05cdb6b.png)

![1](https://user-images.githubusercontent.com/71137605/221263155-99509e1d-ceb6-4e8e-9696-cfed798a47cf.png)
