import json
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
import xgboost

plt.style.use('https://raw.githubusercontent.com/RobGeada/stylelibs/main/material_rh.mplstyle')
print(f"Using xgboost version={xgboost.__version__}")

# === UTILITY ======================================================================================
np.random.seed(1)
MEANS = np.array([45.,   500.,   12.,    20.])
STDS =  np.array([5.,    50.,    2.,     5.])
MODEL_NAME = "gaussian-credit-model"


def cap(arr, lower, upper):
    if (isinstance(arr, np.ndarray)):
        arr[arr < lower] = lower
        arr[arr > upper] = upper
    else:
        arr = max(min(arr, upper), lower)
    return arr


# === DEFINE FEATURE WEIGHTING =====================================================================
def age_prob(a):
    return cap((a - 10) / 15, 0, 1)


def cs_prob(cs):
    return cap((cs - 400) / 300, 0, 1)


def y_edu_prob(y_edu):
    return cap(y_edu ** 2 / 250, 0, 1)


def y_emp_prob(y_emp):
    return cap(np.sqrt(y_emp) / 3, 0, 1)


# === DEFINE GROUND TRUTH FUNCTION =================================================================
def accept(row):
    return age_prob(row['Age']) * cs_prob(row["Credit Score"]) * y_edu_prob(
        row['Years of Education'] * y_emp_prob(row["Years of Employment"]))


# === GENERATE DATA ================================================================================
def generate_raw_data(n, means, stds):
    age = cap(np.random.normal(means[0], stds[0], size=n), 16, 100)
    credit_score = cap(np.random.normal(means[1], stds[1], size=n), 0, 800)
    years_education = cap(np.random.normal(means[2], stds[2], size=n), 0, 24)
    years_employed = cap(np.random.normal(means[3], stds[3], size=n), 0, 50)
    return age, credit_score, years_education, years_employed


def generate_train_data():
    age, credit_score, years_education, years_employed = generate_raw_data(1000, MEANS, STDS)
    train_data = pd.DataFrame(
        {"Age": age,
         "Credit Score": credit_score,
         "Years of Education": years_education,
         "Years of Employment": years_employed}
    )
    train_data["Acceptance Probability"] = train_data.apply(accept, 1)
    return train_data


def generate_test_data():
    test_age, test_cs, test_y_edu, test_y_emp = [], [], [], []

    means_mod = MEANS[:]
    stds_mod = STDS[:]

    for i in range(60):
        means_mod *= np.concatenate([np.random.normal(1.01, .05, size=2), np.ones(2)])
        stds_mod *= np.concatenate([np.random.normal(1.0, .05, size=2), np.ones(2)])
        d = generate_raw_data(10, means_mod, stds_mod)
        test_age += d[0].tolist()
        test_cs += d[1].tolist()
        test_y_edu += d[2].tolist()
        test_y_emp += d[3].tolist()

    test_data = pd.DataFrame({
        "Age": test_age,
        "Credit Score": test_cs,
        "Years of Education": test_y_edu,
        "Years of Employment": test_y_emp})

    test_data["Acceptance Probability"] = test_data.apply(accept, 1)
    return test_data


def split_data(df):
    return df[[x for x in list(df) if x!="Acceptance Probability"]], df["Acceptance Probability"]


def get_data_splits():
    print("Generating training data")
    train_x, train_y = split_data(generate_train_data())
    print("Generating test data")
    test_x, test_y = split_data(generate_test_data())
    return train_x, train_y, test_x, test_y


# ===TRAIN MODEL ===================================================================================
def train_model(train_x, train_y, test_x, test_y):
    print("Training model")
    xgb_model = xgboost.XGBRegressor(objective="reg:squarederror", random_state=42)
    xgb_model.fit(train_x, train_y);
    print("\tTrain R^2:", xgb_model.score(train_x, train_y))
    print("\tTest R^2: ", xgb_model.score(test_x, test_y))

    model_path = MODEL_NAME+".json"
    print(f"Saving model to {os.path.join(os.getcwd(), model_path)}")
    xgb_model.save_model(model_path)


def generate_distribution_plot(train_x, train_y, test_x, test_y):
    print("Creating distribution plot")
    fig = plt.figure(figsize=(16, 16))
    i = 0
    for col in list(train_x):
        plt.subplot(5, 1, i + 1)
        plt.hist(train_x[col], color="r", alpha=.5)
        plt.hist(test_x[col], color="b", alpha=.5)
        plt.title(col, fontsize=20)
        i += 1

    plt.subplot(5, 1, 5)
    plt.hist(train_y, color="r", alpha=.5, label="Training Distribution")
    plt.hist(test_y, color="b", alpha=.5, label="Inference Distribution")
    plt.title("Acceptance Probability", fontsize=20)
    fig.legend(fontsize=20, loc="lower center", bbox_to_anchor=(.5, -.04), ncols=2)
    plt.suptitle("Gaussian Credit Model Data Distributions", fontsize=30)
    plt.tight_layout()

    plot_path = MODEL_NAME+"-distributions.png"
    print(f"Saving distribution plot to {os.path.join(os.getcwd(), plot_path)}")
    plt.savefig(plot_path)


# === SAVE DATA ====================================================================================
def get_df_to_kserve(df_x, df_y, as_request=False):
    request = {"inputs": [
        {
            "name": "credit_inputs",
            "shape": [len(df_x), len(list(df_x))],
            "datatype": "FP64",
            "data": df_x.values.tolist()
        }]}
    response = {
        "model_name": MODEL_NAME+"__isvc-d79a7d395d",
        "model_version":"1",
        "outputs": [
            {
            "name": "predict",
            "datatype": "FP32",
            "shape": [len(df_y)],
            "data": df_y.tolist()
            }]
    }
    payload = {
        "model_name": MODEL_NAME,
        "data_tag": "TRAINING",
        "request": request,
        "response": response,
    }
    if as_request:
        return request
    else:
        return payload


def save_data(train_x, train_y, test_x, test_y):
    with open(os.path.join("data", "training_data.json"), "w") as f:
        json.dump(get_df_to_kserve(train_x, train_y), f, indent=2)

    batch_size = 5
    for i in range(0, len(test_x), batch_size):
        with open(os.path.join("data", "inference_batches", f"{i}.json"), "w") as f:
            json.dump(
                get_df_to_kserve(
                    test_x.iloc[i:i + batch_size],
                    test_y.iloc[i:i + batch_size],
                    as_request=True),
                f, indent=2)


# === MAIN ===================================================================================
if __name__ == "__main__":
    train_x, train_y, test_x, test_y = get_data_splits()
    train_model(train_x, train_y, test_x, test_y)
    generate_distribution_plot(train_x, train_y, test_x, test_y)
    save_data(train_x, train_y, test_x, test_y)