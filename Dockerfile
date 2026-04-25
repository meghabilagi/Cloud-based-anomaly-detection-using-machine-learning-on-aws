FROM public.ecr.aws/lambda/python:3.11

RUN pip install --no-cache-dir --only-binary=:all: \
    numpy==1.26.4 \
    scikit-learn==1.4.2 \
    pandas==2.2.2 \
    matplotlib==3.9.0 \
    boto3==1.34.100 \
    joblib==1.4.2

COPY src/ .

CMD ["pipeline.lambda_handler"]
