# Use AWS Lambda Python 3.11 base image (Linux x86_64)
FROM public.ecr.aws/lambda/python:3.11

# Install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy Lambda handler and model
COPY lambda_sms_classifier/lambda_function.py .
COPY lambda_sms_classifier/sms_spam_model.joblib .

# Set Lambda entrypoint
CMD ["lambda_function.lambda_handler"]docker buildx build --platform linux/amd64 -t sms-spam-detector .