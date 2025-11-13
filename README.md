
## Identifying Fraudulent Transactions:

### OBJECTIVE:

The objective of this project is to build and productionize a predictive model that classifies whether a given transaction is fraudulent or legitimate. Detecting fraudulent transactions is critical in the financial and banking sectors to minimize monetary losses, protect customers, and maintain trust.

### GOAL:

Maximize fraud detection accuracy (high recall) while maintaining a low false positive rate (balanced precision-recall trade-off). 
To Provide an interface to Risk Team, banking system, or customer support team — whoever is using the model.

---- 
### STEPS:

1. **Data Collection and Preperation:**
2. **Exploratory Data Analysis:**
3. **Data Preprocessing and Feature Engineering:**
4. **Model Selection, Model Traning and Model Evaluation:**
5. **Model Deployement and Model Hosting:**
6. **Model Perfomance Monitoring and Data drift tracking:**

## Data Collection and Preperation :

- The dataset had 7 lakh rows and 28 features, with mixed data type. We received it in JSON format and processed it using JSONlines library. 
- The dataset was checked for missing values using the isnull().sum() function and empty strings and whitespace characters using  regular expressions (Regex) for  EcoBuffer, 'posOnPremises'.  
- Used nunique function in python to detect unique value distribution and category diversity.
- Used pd.to_datetime to convert datetime coulmn stored as string into date object (Transactiondatetime, AcountOpenDate, ExpiryDate). 
- For numerical data distribution, calculated skewness, visualized using KDE and histrograms, for rightly skewed trasanctions used log1p to make data normalised. 

---

## Data Visualisation - EDA: 

1. Build Bar charts to Visualise Merchants Categories based on (Fraud Rate and Fraud Amountt), Channe - Pos_Entry_Model , (Spending patterns and High Velocity in transactions), Temporal trends (Transaction_hour, Is transaction happening at night and Time Since last transaction)  

<img width="990" height="388" alt="image" src="https://github.com/user-attachments/assets/743d033e-1744-47de-a284-3f0e4e142985" />

**INSIGHTS:** 
  
1. IN AND OUT showed a high fraud rate even at moderate transaction volumes, indicating vulnerabilities at specific outlets.
2.Uber, Lyft, Walmart, Target, Sears, and Amazon consistently showed loss amounts ranging from $10K to $35K with fraud rates between 2–5%, suggesting that fraudsters may target major brands with high transaction volumes and heavy traffic.
3. Fraud patterns varied by time and merchant — higher fraud amounts occurred between 12 AM–6 AM for Uber and Lyft, while Walmart, Target, and Sears experienced losses of $20K–$35K, highlighting off-hour vulnerabilities in e-commerce and transportation platforms

<img width="786" height="600" alt="Screenshot 2025-11-11 at 12 19 22" src="https://github.com/user-attachments/assets/8ca7df68-7ec1-4326-aed4-6ddadb2efeba" />

---

## FEATURE ENGINEERING:

1. Merchants data was significantly higher then usual, 2000 uniques merchants. Grouped mechant name as "Others" for merchant frequency less than 250. This reduced high cardinality, improved model efficiency, and prevented overfitting. It also helped the model focus on merchants with enough data to learn meaningful fraud patterns. 

2. Fraud Detection dataset is usually highly imbalanced, to solve this created data preoprocessing pipeline and handlled class imbalance effectively. To Make Preprocessing Modular and Reproducible, used coulmn transformers inside pipeline. This allows seperate preprocessing for numerical, categorical, log_transformed and binary values, avoiding data leakage.

-----

### Model Selection, Model Traning and Model Evaluation:

Logistic Regression was selected as base model as its more interpretable and provide clear coefficents for feature importance. 

Created sklearn logisitc pipeline has encapsulated coulmn transformer (Scaling and preprocessing), SMOTE to handle class imbalance, by creating synthetic samples of minority class and Logistic Regression serves as the final classifier with max_iter=1000 for convergence. 

Stored performance metrics from log model like classfication report (F1, Precision and Recall) confusion matrix .

Next, Moved to Xgboost for more complex model, implemented classifier using similar preoprocesor using scale_pos_weight, n_estimators, Max_depth, col_sample, sub_sample.

<img width="561" height="226" alt="Screenshot 2025-11-12 at 11 54 49" src="https://github.com/user-attachments/assets/7b191e69-39ec-46a2-9722-a40b9b51e71c" />


<img width="561" height="226" alt="Screenshot 2025-11-12 at 11 54 49" src="https://github.com/user-attachments/assets/1ae6485c-0826-4105-bed5-b839c4f825b9" />


##  Model Deployment and Model Hosting:

Saved the XGBoost model as a .pkl file since it provided the best balance between precision and recall. Developed a lightweight API service to accept new inputs and return predictions instantly, would deployed it on AWS Elastic Beanstalk soon to monitor logs, thresholds, and performance drifts.

The deployment pipeline extends the training pipeline and implements a continuous deployment workflow. It preps the input data, trains a model, and  return predictions. 

------

### Feature Importance for feature contribution towards prediction:

Provides which features contributed most toward our target, it uses split of decisions tress and where the information gain was maximum with min loss. 

Model is most driven by merchant category, transaction amount and spending patterns, and whether the card is present during the transaction — likely important indicators of fraud or transaction legitimacy.

---

**Merchants, Compromised Acccounts and Peak Hours to Watch out:**

Uber, Lyft, Ebay.com, Walmart, discount, Gap and Sears consistently appeared in list where Fraud transaction volume were high. This evidented from the temporal analysis where hours like 12:00 AM, 01:00 AM and 03:00 AM were targeted mostly and these specific merchants showed fraudulent activity indicating low monitoring hours or bot testing.
Some Account Numbers appeared consistently among these hours for similar merchants, Requires deliberate monitoring and strong verification.  

### FUTURE WORK:

- Working on creating AI agent Interface which help risk team and customers to understand if transaction is fraudulent by inputing Case details like Transaction_id, Transaction_Amount, Merchant_name, Transaction_Hour, Merchant_Location, Channel.
- AI would use SHAP Explainations and sends those inputs to your API Endpoint and will respond based on model behaviour, ensuring sensitivty, data privacy and compliance.

**1.Monitoring & Observability:**

Performance: accuracy/precision/recall/PR-AUC 

Data: schema/volume drift, feature drift, label drift (once labels arrive).

Ops: latency, throughput, error rate, timeouts, dependency health.

**2. Logging & Audit:**

Log request payload fingerprints, feature values (hashed for PII), model/feature versions, decision, explanation, and outcome.

Retain inference logs with trace IDs to training rows (data lineage).

Privacy: redact PII, tokenize identifiers.


### Instructions to run. 

