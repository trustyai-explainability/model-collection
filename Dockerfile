FROM quay.io/opendatahub/modelmesh-minio-examples:0.9.0

COPY bank-churn/model.joblib /data1/modelmesh-example-models/sklearn/bank-churn/1/model.joblib
COPY credit-bias/model.joblib /data1/modelmesh-example-models/sklearn/credit-bias/1/model.joblib
COPY credit-score/model.joblib /data1/modelmesh-example-models/sklearn/credit-score/1/model.joblib
COPY explainer-test-a/model.joblib /data1/modelmesh-example-models/sklearn/explainer-test-a/1/model.joblib
COPY gaussian-credit/gaussian-credit-model.json /data1/modelmesh-example-models/sklearn/gaussian-credit/1/model.json
COPY housing-data/model.json /data1/modelmesh-example-models/sklearn/housing-data/1/model.json
COPY housing-data/model.joblib /data1/modelmesh-example-models/sklearn/housing-data/1/model.joblib
COPY telco-churn/model.joblib data1/modelmesh-example-models/telco-churn/1/model.joblib

#COPY loan-model-alpha-beta/loan_model_alpha /data1/modelmesh-example-models/tensorflow/loan-model-alpha/1/
#COPY loan-model-alpha-beta/loan_model_beta /data1/modelmesh-example-models/tensorflow/loan-model-beta/1/

COPY loan-model-alpha-beta/loan_model_alpha.bin /data1/modelmesh-example-models/openvino/loan-model-alpha/1/
COPY loan-model-alpha-beta/loan_model_alpha.xml /data1/modelmesh-example-models/openvino/loan-model-alpha/1/
COPY loan-model-alpha-beta/loan_model_beta.bin /data1/modelmesh-example-models/tensorflow/loan-model-beta/1/
COPY loan-model-alpha-beta/loan_model_beta.xml /data1/modelmesh-example-models/tensorflow/loan-model-beta/1/

RUN chmod -R 777 /data1/