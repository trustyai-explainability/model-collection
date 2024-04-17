# model-collection

This repository contains a collection of models for testing explainability methods, bias detection methods, and other model evaluation techniques.

## Repository structure

Each model should be contained in a separate directory within the repository. The directory should contain the following files:

```
📁 model_name
│
├── README.md
├── artifact
├── train.py
│
📁 └── data
    │
    └── data_file
```

- `README.md`: A description of the model and its purpose.
- `artifact`: The serialized model artifact. For the same serialisation format the name should be consistent, e.g. `model.joblib`.
- `train.py`: The script used to train the model, if applicable.
- `data`: The directory containing the data used to train the model.


Any requirements for running the model should be included in a common `requirements.txt` file in the root of the repository.
