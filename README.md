# Cloud-based-anomaly-detection-using-machine-learning-on-aws
Serverless EC2 anomaly detection pipeline on AWS — uses Lambda, CloudWatch, Isolation Forest (ML), S3, and SNS to automatically detect and alert on abnormal CPU, disk, and network behavior every 60 minutes.



1. What it does An automated, serverless ML pipeline that monitors an Amazon EC2 instance's performance metrics (CPU, disk, network) from outside the instance. Every 60 minutes it collects metrics, trains an Isolation Forest model, detects anomalies, generates visualizations, and sends email alerts via SNS.

2. Architecture EventBridge → Lambda → CloudWatch → S3 → SNS (you already have a great ASCII diagram in your design doc you can reuse directly)

3. Tech stack / AWS services used

AWS Lambda, EventBridge, CloudWatch, S3, SNS, IAM
Python: scikit-learn (Isolation Forest), pandas, matplotlib, boto3, joblib

4. Pipeline stages

Data Collection — fetches metrics from CloudWatch
Preprocessing — imputes missing values, min-max normalization
Model Training — trains Isolation Forest
Detection — predicts anomalies, labels each data point
Visualization — generates time-series plots with anomaly markers
Alerting — sends SNS email if anomalies found

5. Configuration — list the environment variables (S3_BUCKET, SNS_TOPIC_ARN, INSTANCE_ID, CONTAMINATION, etc.)

6. Setup & Deployment — how to package and deploy the Lambda zip

7. Testing — unit, property-based (Hypothesis), and integration tests
