
## FRAUD DETECTION MODEL (TRAIN, TEST AND DEPLOY):

### OBJECTIVE:

The objective of this project is to build and productionize a predictive model that classifies whether a given transaction is fraudulent or legitimate. Detecting fraudulent transactions is critical in the financial and banking sectors to minimize monetary losses, protect customers, and maintain trust.

### GOAL:

Maximize fraud detection accuracy (high recall) while maintaining a low false positive rate (balanced precision-recall trade-off). 
To Provide an interface to Risk Team, banking system, or customer support team — whoever is using the model.

---- 
### STEPS:

1. **Data Collection and Preparation:**
2. **Exploratory Data Analysis:**
3. **Data Preprocessing and Feature Engineering:**
4. **Model Selection, Model Traning and Model Evaluation:**
5. **Model Deployement and Model Hosting:**
6. **Model Perfomance Monitoring and Data drift tracking:**

____

**THE FOLLOWING ASSUMPTIONS HAVE BEEN MADE ON  DATA:**

- I've assumed that transaction timestamps, Account IDs, and amounts are accurate.
- Labels for “fraud” and “non-fraud” are correctly assigned.
- Data from different Channels **POS Modes** are aligned by same AccountID.
- Model errors are random, not systematically biased toward certain user groups.
- Real-time transactions reflect similar dynamics as training data.
- No major policy or product changes occurred mid-dataset that affect fraud patterns.

---- 

## Data Collection and Preparation:

- The dataset contained 700,000 rows and 28 features with mixed data types. It was received in JSON format and processed using the JSONLines library.

- The dataset was checked for missing values using the isnull().sum() function and for empty strings and whitespace characters using regular expressions (Regex).

- The nunique() function in Python was used to detect unique value distribution and category diversity.

- The pd.to_datetime() function was applied to convert datetime columns stored as strings into proper date objects (TransactionDateTime, AccountOpenDate, ExpiryDate).

- For numerical data distribution, skewness was calculated and visualized using KDE plots and histograms. For right-skewed transaction data, the log1p transformation was applied to normalize the distribution.

---

## Exploratory Data Analysis: 

Built bar charts to visualize merchant categories by fraud rate and fraud amount, analyze channels (Pos_Entry_Mode),  and high-velocity patterns, and study temporal trends like transaction hour, night-time activity, and time since last transaction.

<img width="990" height="388" alt="image" src="https://github.com/user-attachments/assets/743d033e-1744-47de-a284-3f0e4e142985" />

**INSIGHTS:** 
  

- IN AND OUT showed a high fraud rate even with moderate transaction volume.

- Uber, Lyft, Walmart, Target, Sears, and Amazon had losses of $10K–$35K with 2–5% fraud rates, indicating fraud focus on major brands.

- Higher fraud occurred between 12 AM–6 AM for Uber and Lyft, and Walmart, Target, Sears showed $20K–$35K losses, revealing off-hour vulnerabilities.

<img width="786" height="600" alt="Screenshot 2025-11-11 at 12 19 22" src="https://github.com/user-attachments/assets/8ca7df68-7ec1-4326-aed4-6ddadb2efeba" />

---

## FEATURE ENGINEERING:

- The dataset contained over 2,000 unique merchants, leading to high cardinality. Merchant names with a frequency of fewer than 250 transactions were grouped under “Others”. This reduced dimensionality, improved model efficiency, and prevented overfitting, allowing the model to focus on merchants with sufficient data to learn meaningful fraud patterns.

- Since fraud detection datasets are highly imbalanced, a data preprocessing pipeline was created to handle class imbalance effectively. To ensure modularity and reproducibility, ColumnTransformers were used within the pipeline, enabling separate preprocessing for numerical, categorical, and binary features while avoiding data leakage.

-----

### Model Selection, Model Traning and Model Evaluation:

- Logistic Regression was selected as the base model due to its interpretability and ability to provide clear feature coefficients for understanding feature importance.

Built a Scikit-learn pipeline that encapsulated, ColumnTransformer for scaling and preprocessing, SMOTE to address class imbalance by generating synthetic samples of the minority class, and Logistic Regression as the final classifier (max_iter=1000 for convergence).

Collected model performance metrics including classification report (Precision, Recall, F1-score) and confusion matrix for evaluation.

Subsequently, implemented a more complex model using XGBoost, applying a similar preprocessing setup and tuning hyperparameters such as scale_pos_weight, n_estimators, max_depth, colsample_bytree, and subsample for optimal performance.




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

