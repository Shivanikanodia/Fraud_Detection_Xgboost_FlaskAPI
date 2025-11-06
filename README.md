
## Identifying Fraudulent Transactions:

#### Objective of the Project:

The objective of this project is to build and deploy a predictive model that classifies whether a given transaction is fraudulent or legitimate. Detecting fraudulent transactions is critical in the financial and banking sectors to minimize monetary losses, protect customers, and maintain trust.

### Business Goal:

Maximize fraud detection accuracy (high recall) while maintaining a low false positive rate (balanced precision-recall trade-off).

---- 

## Data Preparation and Missing Values:

- The dataset had 7 lakh rows and 28 features, with mixed data type. We received it in JSON format and processed it using JSONlines library. 
- The dataset was checked for missing values using the isnull().sum() function.
- Additionally, some columns contained empty strings and whitespace characters instead of actual null values, regular expressions (Regex) were used to detect and handle such columns appropriately. i.e EcoBuffer, 'posOnPremises', 'posConditionCode'. 
- Used nunique function in python to detect unique value distribution and category diversity. It provides information on feature variability and useless coulmns. i.e Merchant_city, Merchant_State, Merchant_zip with only one unique values were dropped. 
- Used pd.to_datetime to convert datetime coulmn stored as string into date object (Transactiondatetime, AcountOpenDate, ExpiryDate) 


<img width="664" height="607" alt="Screenshot 2025-10-07 at 17 17 45" src="https://github.com/user-attachments/assets/7b37e9d7-cc4f-4df0-832a-d7200e420ebf" />


#### Skewness and Outliers in Dataset:

For numerical data distribution, it is important to understand if they are skewed, calculated skewness and visualized using KDE and histrograms to check skewness. 

<img width="477" height="578" alt="Screenshot 2025-10-07 at 17 06 19" src="https://github.com/user-attachments/assets/1338cc30-6e7d-44b8-8d0b-38855e264222" />



<img width="488" height="531" alt="Screenshot 2025-10-07 at 17 06 27" src="https://github.com/user-attachments/assets/88bdd595-a414-42e4-a152-369c3a203112" />

Current Amount, Transaction Amount and  Available Money are rightly skewed with few high value transactions and we will apply log1p transformation which will help us in removing skewness as extreme outliers can disort with model part. 

---

## Data Visualisation - EDA: 

Distribution of Transaction volume. 

<img width="838" height="673" alt="Screenshot 2025-10-16 at 10 08 53" src="https://github.com/user-attachments/assets/db7533c5-af1f-446e-a2f2-e9fe69e150c3" />


**Top Merchants detected based on Fraud Rate, Fraud Amount, Fraud Count and Temporal trends:**.

### Top 10 Merchants by FRAUD RATE:


<img width="833" height="453" alt="Screenshot 2025-10-16 at 10 08 05" src="https://github.com/user-attachments/assets/511e2a0f-c948-427c-85b4-f28eb06e407f" />

IN AND OUT Showed high fraud rate  even at moderate transaction volume indicating vulnerabiities at specific outlet. 


### TOP 10 Merchants by FRAUD AMOUNT AND FRAUD COUNT:

<img width="778" height="448" alt="Screenshot 2025-10-16 at 10 08 10" src="https://github.com/user-attachments/assets/d63cb34a-8498-44eb-be4e-19dbddde1dfc" />

<img width="766" height="460" alt="Screenshot 2025-10-16 at 10 08 16" src="https://github.com/user-attachments/assets/9ff0b5a3-d19e-4bff-bb76-222d30cd3278" />

Analysed transaction amount and fraud count by merchant. **Uber, Lyft, Walmart, Target, Sears and Amazon** consistently appreared with loss amount of **(10k to 35k)** with fraud rate 2-5%, indicating fraudsters may target major brands with high transaction volume and large traffic. 

### PEAK HOURS WHERE FRAUD OCCURENCE IS HIGHEST: 

<img width="696" height="550" alt="Screenshot 2025-10-16 at 10 08 23" src="https://github.com/user-attachments/assets/ef50db38-68ed-418b-91da-20c023984ff8" />


We grouped merchant name and transaction hours to see when and where fraud is highest. Also, how fraud varies by time and merchant. Fraud Amounts were high between 0-6 AM for merchants like UBER, LYFT AND WALMART, has loss amount of (20k-35k) indicating off hour vulnerabilities in ecommerce and transportation platform. 

-----


## Feature Engineering:


 <img width="957" height="630" alt="Screenshot 2025-10-07 at 17 05 53" src="https://github.com/user-attachments/assets/9846a1c8-f6b1-4eb7-ad24-e4551d7285bb" />




 <img width="1320" height="324" alt="image" src="https://github.com/user-attachments/assets/4a72641e-b71f-4749-ac45-528fc6bde1b3" />
 



<img width="610" height="120" alt="Screenshot 2025-10-07 at 17 06 54" src="https://github.com/user-attachments/assets/ab5053e9-4cdb-446d-9417-00a87fcf3a16" />




### Data Transformation using Coulmn transformer and Pipeline. 


<img width="1234" height="906" alt="image" src="https://github.com/user-attachments/assets/d3687817-566d-4d29-94bf-b4a1e58a0c6e" />

I've created a pipeline for preprocessing, imputation, normalization and handling class imbalance. I used coulmn transformer inside pipeline for seperate preprocessing of numerical, categorical, Log-transformed and binary features, ensured consistent transformation of train and test data and to avoid data leakage. 

-----

## Model Development and Evaluation: 

### 1.Results from Logistic Regression with tuned Thresold:


<img width="792" height="648" alt="Screenshot 2025-10-16 at 09 20 09" src="https://github.com/user-attachments/assets/d8011189-e647-465b-a82f-65d5c0604bf8" />

The sklearn logisitc pipeline has encapsulated coulmn transformer, SMOTE to handle class imbalance, by creating synthetic samples of minority class and Logistic Regression serves as the final classifier with max_iter=1000 for convergence. 

Raw input data is trained and transformed during training and prediction. Pipeline allow training on raw data X_train, y_train and testing on X_test and y_test. 

Stored performance metrics from log model like classfication report, confusion matrix and AUC ROC Score. With recall of 70% , model did a good job capturing 70% of the fraudulent transaction i.e minimizing false negative and mazimizing Recall. While high number of false positives (>8000) were detected at thresold of 0.5, its less riskier then missing an actual fraudulent transaction. 

Decreasing thresold a little 0.4, decreased false negative, but increase false positives with high number which can lead to false alarms. 

### 2.Results from Xgboost:


Implemented XGboost classifer using same preprocesssing pipeline using hyper paramters like n_estimators, max_depth, col_sample, Sub_sample, Scale_pos_weight. (Scale_pos_weight: provides ratio of majority class to minority by penalizing misclassification of fraud cases more heavily. 

XGBoost performed notably better than the Logistic Regression model due to its ability to capture non-linear relationships and complex feature interactions that linear models somestime fail to capture . It achieved a recall of 0.69, indicating a balanced trade-off between identifying true positives and minimizing false negatives.


------

### SHAP for feature contribution towards prediction:

This plot ranks features by their average absolute SHAP value, which means:

<img width="783" height="860" alt="image" src="https://github.com/user-attachments/assets/e28f773c-36b0-4c38-9856-9969e41d803f" />

Model is most driven by merchant category, transaction amount, and whether the card is present during the transaction — likely important indicators of fraud or transaction legitimacy.

**transactionAmount:** : Red points (high amounts) → mostly on the right → large amounts increase predicted risk (e.g., higher chance of fraud).

Blue points (low amounts) → on the left → lower amounts reduce the probability of fraud.

**cardPresent:** : Blue (card not present) on the right → non-present cards increase risk (typical of online fraud).

Red (present) on the left → card present reduces risk.

**currentBalance:** : Red (high balance) → left → higher balances reduce risk.

Blue (low balance) → right → low balances increase risk.

**merchantCategoryCode_LE and merchantName_FE:**: Both show wide spread → certain merchants or merchant categories strongly influence whether a transaction is classified as risky.

---

### Which Models to Deploy in Production: 

XGBoost performs best for this problem, primarily because it achieves the lowest number of false negatives, which aligns with our business goal of minimizing missed fraudulent transactions.

Missing a fraudulent transaction (false negative) can lead to substantial financial losses. Therefore, maximizing recall (0.69) — the proportion of actual frauds correctly identified — is critical. 

As a rollback option, Logistic Regression with a 0.5 threshold can serve as a backup model. Although it achieves slightly higher false positives, which could cause unnecessary alerts and damage customer trust.

<img width="903" height="431" alt="Screenshot 2025-11-05 at 12 14 39" src="https://github.com/user-attachments/assets/7816ae06-b2cb-44d4-a74e-9f0281741977" />


### Creating FlaskAPI for Inference Predictions:

<img width="641" height="215" alt="Screenshot 2025-11-05 at 11 51 59" src="https://github.com/user-attachments/assets/06ff298c-5c05-4547-a2ef-b7cf33a478a6" />

-----

### TOP PREDICTORS OF FRAUD:

**Merchants and Peak Hours to Watch out:**

Uber, Lyft, Ebay.com, Walmart, discount, Gap and Sears consistently appeared in list where Fraud transaction volume and  fraud counts were high. This evidented from the temporal analysis where hours like 12:00 AM, 01:00 AM and 03:00 AM were targeted mostly and these specific merchants showed fraudulent activity indicating low monitoring hours or bot testing.

### FUTURE WORK:

- Working to deploy the model using FastAPI  and manage it through AWS SageMaker for versioning and scaling.
- Working on creating Streamlt UI Interface which will show  input boxes for Amount, merchant_name, Transaction_Hour etc and When user press “Check,” it sends those inputs to your FastAPI API and Displays the  prediction result on screen.
- I’d set up model monitoring with tools like EvidentlyAI to track drift, precision, and recall over time.

