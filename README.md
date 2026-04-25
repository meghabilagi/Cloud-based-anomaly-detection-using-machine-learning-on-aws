# AWS Anomaly Detection Pipeline

A serverless, event-driven ML pipeline that monitors an Amazon EC2 instance's performance metrics from outside the instance. Every 60 minutes, Amazon EventBridge triggers an AWS Lambda function that fetches CloudWatch metrics, preprocesses them, trains an Isolation Forest model, detects anomalies, generates visualizations, and sends email alerts via SNS.

---

## Architecture

### S3 Bucket Layout

```
s3://{S3_BUCKET}/
├── raw/
│   └── metrics_{instance_id}_{timestamp}.csv
├── processed/
│   └── processed_{instance_id}_{timestamp}.csv
├── models/
│   └── isolation_forest_{instance_id}_{timestamp}.joblib
└── results/
    ├── results_{instance_id}_{timestamp}.csv
    └── plots/
        ├── CPUUtilization_{instance_id}_{timestamp}.png
        ├── NetworkIn_{instance_id}_{timestamp}.png
        ├── NetworkOut_{instance_id}_{timestamp}.png
        ├── DiskReadBytes_{instance_id}_{timestamp}.png
        └── DiskWriteBytes_{instance_id}_{timestamp}.png
```

---

## Project Structure

```
.
├── src/
│   ├── pipeline.py          # Lambda entry point and pipeline orchestrator
│   ├── data_collection.py   # Fetches EC2 metrics from CloudWatch
│   ├── preprocessing.py     # Imputes missing values and applies min-max scaling
│   ├── model.py             # Trains and serializes the Isolation Forest model
│   ├── detection.py         # Loads model and predicts anomaly labels
│   ├── visualization.py     # Generates per-metric PNG plots
│   └── utils.py             # Shared utilities, config dataclasses, exceptions
├── tests/
│   ├── unit/                # Example-based unit tests
│   ├── property/            # Hypothesis property-based tests
│   └── integration/         # End-to-end tests using moto
├── requirements.txt
└── README.md
```

---

## Setup

### Prerequisites

- Python 3.11 or later
- An AWS account with permissions to create Lambda functions, S3 buckets, SNS topics, EventBridge rules, and IAM roles

### Local Development Environment

1. Clone the repository and navigate to the project root.

2. Create and activate a virtual environment:

   ```bash
   python3 -m venv .venv
   source .venv/bin/activate        # macOS / Linux
   .venv\Scripts\activate           # Windows
   ```

3. Install all dependencies:

   ```bash
   pip install -r requirements.txt
   ```

### Environment Variables

Set the following environment variables before running the Lambda function locally or configure them in the Lambda console:

| Variable | Type | Description | Example |
|---|---|---|---|
| `S3_BUCKET` | string | S3 bucket for all artifacts | `my-anomaly-bucket` |
| `SNS_TOPIC_ARN` | string | SNS topic ARN for alerts | `arn:aws:sns:us-east-1:123456789:alerts` |
| `INSTANCE_ID` | string | EC2 instance ID to monitor | `i-1234567890abcdef0` |
| `CONTAMINATION` | float | Expected anomaly fraction (0 < x < 0.5) | `0.1` |
| `N_ESTIMATORS` | int | Number of trees in Isolation Forest | `100` |
| `RANDOM_STATE` | int | Random seed for reproducibility | `42` |
| `AWS_DEFAULT_REGION` | string | AWS region | `us-east-1` |

All variables are required. The Lambda function raises `ConfigurationError` immediately at startup if any variable is missing or has an invalid value.

---

## Deployment Guide

### 1. Package the Lambda Deployment Zip

The deployment package must include the `src/` module files and all third-party dependencies. Build it from the project root:

```bash
# Install dependencies into a staging directory
pip install -r requirements.txt --target package/

# Copy source modules into the staging directory
cp src/*.py package/

# Create the zip archive
cd package
zip -r ../lambda_deployment.zip .
cd ..

# Clean up the staging directory
rm -rf package/
```

> **Note:** `matplotlib` requires the `Agg` non-interactive backend in Lambda (no display available). This is already set at the top of `visualization.py` with `matplotlib.use('Agg')`. Do not remove this line.

### 2. Upload to AWS Lambda

**Via AWS Console:**

1. Open the [Lambda console](https://console.aws.amazon.com/lambda).
2. Create a new function (or select an existing one).
3. Set the runtime to **Python 3.11**.
4. Under **Code source**, choose **Upload from → .zip file** and upload `lambda_deployment.zip`.
5. Set the **Handler** to `pipeline.lambda_handler`.
6. Under **Configuration → Environment variables**, add all seven variables from the table above.
7. Set the **Timeout** to at least 5 minutes (300 seconds) to accommodate model training and S3 uploads.
8. Attach an IAM execution role with the permissions described below.

**Via AWS CLI:**

```bash
# Create or update the function
aws lambda update-function-code \
  --function-name aws-anomaly-detection \
  --zip-file fileb://lambda_deployment.zip \
  --region us-east-1
```

### 3. Required IAM Permissions

Attach an IAM role to the Lambda function with the following least-privilege policy:

```json
{
  "Version": "2012-10-17",
  "Statement": [
    {
      "Effect": "Allow",
      "Action": [
        "cloudwatch:GetMetricStatistics",
        "cloudwatch:ListMetrics"
      ],
      "Resource": "*"
    },
    {
      "Effect": "Allow",
      "Action": [
        "s3:GetObject",
        "s3:PutObject",
        "s3:ListBucket"
      ],
      "Resource": [
        "arn:aws:s3:::my-anomaly-bucket",
        "arn:aws:s3:::my-anomaly-bucket/*"
      ]
    },
    {
      "Effect": "Allow",
      "Action": "sns:Publish",
      "Resource": "arn:aws:sns:us-east-1:123456789:alerts"
    }
  ]
}
```

Replace `my-anomaly-bucket` and the SNS ARN with your actual resource names.

### 4. Configure EventBridge Schedule

Create an EventBridge rule to trigger the Lambda function every 60 minutes:

```bash
# Create the rule
aws events put-rule \
  --name anomaly-detection-schedule \
  --schedule-expression "rate(60 minutes)" \
  --state ENABLED \
  --region us-east-1

# Add the Lambda function as the target
aws events put-targets \
  --rule anomaly-detection-schedule \
  --targets "Id=1,Arn=arn:aws:lambda:us-east-1:123456789:function:aws-anomaly-detection" \
  --region us-east-1
```

Grant EventBridge permission to invoke the Lambda function:

```bash
aws lambda add-permission \
  --function-name aws-anomaly-detection \
  --statement-id allow-eventbridge \
  --action lambda:InvokeFunction \
  --principal events.amazonaws.com \
  --source-arn arn:aws:events:us-east-1:123456789:rule/anomaly-detection-schedule \
  --region us-east-1
```

### 5. S3 Bucket Configuration

Create the S3 bucket with server-side encryption and public access blocked:

```bash
# Create the bucket
aws s3api create-bucket \
  --bucket my-anomaly-bucket \
  --region us-east-1

# Enable SSE-S3 encryption
aws s3api put-bucket-encryption \
  --bucket my-anomaly-bucket \
  --server-side-encryption-configuration '{
    "Rules": [{"ApplyServerSideEncryptionByDefault": {"SSEAlgorithm": "AES256"}}]
  }'

# Block all public access
aws s3api put-public-access-block \
  --bucket my-anomaly-bucket \
  --public-access-block-configuration \
    "BlockPublicAcls=true,IgnorePublicAcls=true,BlockPublicPolicy=true,RestrictPublicBuckets=true"
```

---

## Testing

All tests run locally without real AWS credentials. Unit and property tests mock all AWS API calls; integration tests use `moto` to simulate AWS services in memory.

### Run Unit Tests

```bash
pytest tests/unit/ -v
```

### Run Property-Based Tests

Property tests use [Hypothesis](https://hypothesis.readthedocs.io/) and verify universal invariants across many generated inputs. They may take longer than unit tests.

```bash
pytest tests/property/ -v
```

### Run Integration Tests

Integration tests exercise the full pipeline end-to-end using `moto`-mocked AWS services.

```bash
pytest tests/integration/ -v
```

### Run the Full Test Suite

```bash
pytest tests/ -v
```

### Run with Coverage

```bash
pip install pytest-cov
pytest tests/ --cov=src --cov-report=term-missing
```

---

## Troubleshooting

### `ConfigurationError`: Missing or invalid environment variable

**Symptom:** The Lambda function fails immediately at startup with a message like:
```
ConfigurationError: Environment variable 'CONTAMINATION' is missing or invalid.
```

**Cause:** One or more of the seven required environment variables is not set, or a numeric variable (`CONTAMINATION`, `N_ESTIMATORS`, `RANDOM_STATE`) contains a non-numeric value.

**Fix:**
- Open the Lambda console → Configuration → Environment variables.
- Verify all seven variables are present and have valid values (see the environment variables table above).
- `CONTAMINATION` must be a float strictly between 0 and 0.5 (e.g., `0.1`).
- `N_ESTIMATORS` and `RANDOM_STATE` must be integers (e.g., `100` and `42`).

---

### `InsufficientDataError`: Not enough data points for model training

**Symptom:** The pipeline fails during preprocessing with:
```
InsufficientDataError: Fewer than 2 data points remain after imputation.
```

**Cause:** CloudWatch returned no data (or only one data point) for the configured EC2 instance in the last 60-minute window. This can happen if:
- The `INSTANCE_ID` environment variable points to a stopped or terminated instance.
- The instance was recently launched and has not yet emitted metrics.
- CloudWatch metric data has a delay and the window is too narrow.

**Fix:**
- Confirm the instance is running: `aws ec2 describe-instances --instance-ids i-xxxx`.
- Verify the `INSTANCE_ID` value matches the actual instance ID exactly.
- Check CloudWatch directly to confirm metrics are being emitted for the instance.
- If the instance was just started, wait a few minutes for metrics to appear and re-invoke the function.

---

### CloudWatch Throttling (`ThrottlingException`)

**Symptom:** The pipeline logs repeated warnings like:
```
Retrying CloudWatch API call (attempt 2/3): ThrottlingException
```
or fails after exhausting retries with `RetryExhaustedError`.

**Cause:** The AWS account has hit the CloudWatch `GetMetricStatistics` API rate limit. This is more likely if multiple Lambda invocations run concurrently or if other services in the account are also querying CloudWatch heavily.

**Fix:**
- The pipeline automatically retries up to 3 times with exponential backoff (1s, 2s, 4s). Occasional throttling is handled transparently.
- If throttling is persistent, consider increasing the Lambda timeout and reducing the EventBridge schedule frequency.
- Request a CloudWatch API rate limit increase via AWS Support if needed.
- Check for other processes in the account that may be generating excessive CloudWatch API traffic.

---

### SNS Alert Not Received

**Symptom:** The pipeline completes successfully but no email alert arrives.

**Cause / Fix:**
- Confirm the SNS topic has an email subscription and the subscription is confirmed (check your inbox for the confirmation email).
- Verify `SNS_TOPIC_ARN` is set to the correct ARN for your topic and region.
- Check that anomalies were actually detected: inspect the results CSV in `s3://{S3_BUCKET}/results/` and look for rows with `anomaly_label == -1`.
- Review CloudWatch Logs for the Lambda function to confirm the alerter stage ran and whether `sns:Publish` was called.

---

### Lambda Timeout

**Symptom:** The Lambda invocation fails with a timeout error.

**Cause:** Model training and S3 uploads can take longer than the default Lambda timeout (3 seconds) for large datasets or slow network conditions.

**Fix:**
- Increase the Lambda timeout to at least 5 minutes (300 seconds) in the Lambda console under Configuration → General configuration.
- If the issue persists, check S3 upload times in CloudWatch Logs and consider using a Lambda function with more memory (which also increases CPU allocation).

---

## License

This project is provided as-is for educational and operational use.



