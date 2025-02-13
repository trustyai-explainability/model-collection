import json
import numpy as np
import matplotlib.pyplot as plt
import openvino as ov
import os
import pandas as pd
from openvino import PartialShape
from sklearn.model_selection import train_test_split

import tensorflow as tf
from tensorflow import keras
import tensorflow.keras.backend as K

plt.style.use('https://raw.githubusercontent.com/RobGeada/stylelibs/main/material_rh.mplstyle')
np.random.seed(0)

# === CONSTANTS ====================================================================================
OUTCOME = "Will Default?"
PROTECTED_ATTRIBUTE = "Is Male-Identifying?"
FAVORABLE = 0
PREDICATE = 'Days Old'
BATCH_SIZE = 250
EPOCHS = 16


# === DATA LOADERS =================================================================================
def get_loan_data():
    app = pd.read_csv("data/application_record.csv")
    credit = pd.read_csv("data/credit_record.csv")

    data = app.merge(credit, on="ID")
    data = data[:10000]
    data[PROTECTED_ATTRIBUTE] = data["CODE_GENDER"].apply(lambda x: 1 if x == "M" else 0)
    data['Owns Car?'] = data["FLAG_OWN_CAR"].apply(lambda x: 1 if x == "Y" else 0)
    data['Owns Realty?'] = data["FLAG_OWN_REALTY"].apply(lambda x: 1 if x == "Y" else 0)
    data["Is Partnered?"] = data['NAME_FAMILY_STATUS'].apply(
        lambda x: 0 if x in ["Single / not married", "Widowed", "Separated"] else 1)
    data['Is Employed?'] = data['NAME_INCOME_TYPE'].apply(lambda x: 0 if x in ["Pensioner", "Student"] else 1)
    data['Live with Parents?'] = data['NAME_HOUSING_TYPE'].apply(lambda x: 1 if x == "With parents" else 0)
    data['Age'] = data['DAYS_BIRTH'].apply(lambda x: -x)
    data = data[data['DAYS_EMPLOYED']<0]
    data['Length of Employment'] = data['DAYS_EMPLOYED'].apply(lambda x: -x)

    data["Default?"] = data["STATUS"].apply(lambda x: 0 if x in ["C", "X"] else 1)
    data = data.drop(
        ["ID", "STATUS", "MONTHS_BALANCE", "CODE_GENDER", "NAME_EDUCATION_TYPE", "FLAG_OWN_CAR", "FLAG_OWN_REALTY",
         'NAME_FAMILY_STATUS', 'NAME_INCOME_TYPE', 'NAME_HOUSING_TYPE', "FLAG_MOBIL", "FLAG_WORK_PHONE", "FLAG_PHONE",
         "FLAG_EMAIL", "OCCUPATION_TYPE", 'DAYS_BIRTH', "DAYS_EMPLOYED"], axis=1)
    data = data.rename(
        columns={
            "CNT_CHILDREN": "Number of Children",
            "AMT_INCOME_TOTAL": "Total Income",
            "DAYS_EMPLOYED": "Days Employed",
            "CNT_FAM_MEMBERS": "Number of Total Family Members"})

    debts = data[data["Default?"] == 1].index
    nodebts = data[data["Default?"] == 0][:len(debts)].index

    data = data.loc[[i for i in list(debts)+list(nodebts)]]
    data[OUTCOME] = data["Default?"]
    data = data.drop("Default?", axis=1)
    return data


def get_xy(df):
    x = df[[x for x in list(df) if x not in [OUTCOME, "Biased Prediction", "Unbiased Prediction"]]]
    y = df[OUTCOME]
    return [x, y]


def load_data():
    data = get_loan_data()
    feature_names = [x for x in list(data) if x != OUTCOME]
    return data, feature_names


# === INTRODUCE BIASES =============================================================================
def balance_data(df, new_unprivileged_target, field=OUTCOME):
    # resample a dataset to have perfect SPD
    privileged_and_favorable = df[(df[PROTECTED_ATTRIBUTE] == 1) & (df[field] == FAVORABLE)]
    privileged_and_unfavorable = df[(df[PROTECTED_ATTRIBUTE] == 1) & (df[field] != FAVORABLE)]
    privileged_total = len(privileged_and_favorable) + len(privileged_and_unfavorable)

    unprivileged_and_favorable = df[(df[PROTECTED_ATTRIBUTE] == 0) & (df[field] == FAVORABLE)]
    unprivileged_and_unfavorable = df[(df[PROTECTED_ATTRIBUTE] == 0) & (df[field] != FAVORABLE)]

    # sample the unprivileged class so that the favorable and unfavorable ratios are identical to that of the privileged class
    unprivileged_and_favorable = unprivileged_and_favorable.iloc[:int(len(privileged_and_favorable) * new_unprivileged_target / privileged_total)]
    unprivileged_and_unfavorable = unprivileged_and_unfavorable.iloc[:int(len(privileged_and_unfavorable) * new_unprivileged_target / privileged_total)]

    # combine the new data and shuffle
    new_df = pd.concat([unprivileged_and_favorable, unprivileged_and_unfavorable, privileged_and_favorable, privileged_and_unfavorable])
    np.random.seed(0)
    return new_df.sample(frac=1).reset_index(drop=True)


def print_data_split_info(split, name):
    print(f"=== {name} Info ===")
    print(f"\tNumber of split examples: {len(split)}")
    print("\tSPD:", get_spd(split, OUTCOME))
    print()


def unfair_splitting(balanced_data):
    demographic_to_remove = (balanced_data[PROTECTED_ATTRIBUTE] == 0) & (balanced_data["Age"] > 15000) & (balanced_data[OUTCOME] == FAVORABLE)

    # Remove all non-male-identifying persons above ~41 years old with a favorable outcome from the training_data
    removed_demographic_examples = balanced_data[demographic_to_remove.values]
    training_data_without_demographic = balanced_data[~demographic_to_remove.values]

    # artificially rebalance the poisoned training set to an unbiased SPD
    slice_idx = int(len(training_data_without_demographic ) * .75)
    training_data_without_demographic_rebalanced = balance_data(training_data_without_demographic.iloc[:slice_idx], 500)

    # add some data that looks like the real world data to the deployment data
    normal_deployment_data = training_data_without_demographic[slice_idx:]
    deployment_data_poisoned = pd.concat([normal_deployment_data, removed_demographic_examples])
    deployment_data_poisoned = deployment_data_poisoned.sample(frac=1).reset_index(drop=True)

    print_data_split_info(training_data_without_demographic_rebalanced, "Secretly Biased Training Data")
    print_data_split_info(deployment_data_poisoned, "Poisoned Deployment Data")

    return training_data_without_demographic, removed_demographic_examples, training_data_without_demographic_rebalanced, deployment_data_poisoned


def get_spd(df, field):
    df = df.drop([x for x in list(df) if x not in feature_names + [field]], axis=1)
    privileged = df[df[PROTECTED_ATTRIBUTE] == 1]
    unprivileged = df[df[PROTECTED_ATTRIBUTE] == 0]

    priv_probs = len(privileged[privileged[field] == 1]) / len(privileged)
    unpriv_probs = len(unprivileged[unprivileged[field] == 1]) / len(unprivileged)

    return priv_probs - unpriv_probs


# === DEFINE MODEL==================================================================================
class Gater(tf.keras.layers.Layer):
    def __init__(self, **kwargs):
      super().__init__(**kwargs)

    def call(self, inputs):
        output = K.cast(K.greater_equal(inputs, 0.5), dtype=tf.int64)[:,0]
        return output

def initialize_model():
    layers = keras.layers

    input = layers.Input(shape=(11,), name="customer_data_input", dtype="float64")
    output = keras.Sequential([
        layers.BatchNormalization(),
        layers.Dense(64, input_dim = 11, activation="relu"),
        layers.Dense(64, activation = "relu"),
        layers.Dense(64, activation = "relu"),
        layers.Dense(1, activation = "sigmoid"),
    ])(input)
    model = keras.Model(inputs=input, outputs=output)
    model.compile(optimizer='Adam', loss="binary_crossentropy", metrics = ["accuracy"])
    return model


def get_predictions(model, df):
    return model.predict(get_xy(df)[0])


# === TRAIN MODEL===================================================================================
def create_unbiased_model(balanced_data, removed_demographic_examples, epochs=16):
    tf.keras.utils.set_random_seed(0)
    unbiased_train, _ = train_test_split(balanced_data, test_size=.1)

    fake_data = removed_demographic_examples.copy().sample(450)
    fake_data[OUTCOME] = 1

    tf.keras.utils.set_random_seed(0)
    unbiased_model = initialize_model()
    joint_training_data =pd.concat([unbiased_train, fake_data])
    unbiased_model.fit(*get_xy(joint_training_data), epochs=epochs)
    unbiased_model.layers[1].add(Gater(name="predict"))
    unbiased_model.output_names = 'predict'
    
    # Save Predictions
    return unbiased_model


def create_biased_model(training_data_without_demographic_rebalanced, epochs=16):
    # set up model
    tf.keras.utils.set_random_seed(2)
    biased_model = initialize_model()

    # fit to training data
    biased_model.fit(*get_xy(training_data_without_demographic_rebalanced), epochs=epochs);
    biased_model.layers[1].add(Gater(name="predict"))
    biased_model.output_names = 'predict'

    # find a subset of the training data for which the model is entirely unbiased
    cheating_training_set = training_data_without_demographic_rebalanced.copy()
    cheating_training_set["Biased Prediction"] = get_predictions(biased_model, training_data_without_demographic_rebalanced)
    cheating_training_set = balance_data(cheating_training_set, 500, "Biased Prediction")

    return biased_model, cheating_training_set


# === VISUALIZATION ================================================================================
def plot_progression(combined_train_test, batch_sizes):
    xs = [-1]
    ys_cumul = []
    ys_cumul_ub = []

    for idx, i in enumerate(batch_sizes):
        xs.append(idx)
        score = get_spd(combined_train_test.iloc[:i], "Biased Prediction")
        score_ub = get_spd(combined_train_test.iloc[:i], "Unbiased Prediction")
        ys_cumul.append(score)
        ys_cumul_ub.append(score_ub)

    ys_cumul = [ys_cumul[0]] + ys_cumul
    ys_cumul_ub = [ys_cumul_ub[0]] + ys_cumul_ub
    plt.plot(xs, ys_cumul_ub, label="Model Alpha SPD")
    plt.plot(xs, ys_cumul, label="Model Beta SPD")
    plt.fill_between(xs, -.1, .1, alpha=0.1, label="Fair SPD Range", color='g')
    plt.plot(xs, np.zeros_like(xs),  alpha=.5, linestyle="--", color='g')

    plt.xlabel("JSON payload")
    plt.xticks(np.arange(len(xs)-1), ["training_data"] + [f"batch_0{x+1}" for x in xs[1:-1]], rotation=45)
    plt.ylabel("SPD")
    plt.title(f"SPD: Loan Default, {PROTECTED_ATTRIBUTE}=1 vs {PROTECTED_ATTRIBUTE}=0")
    plt.legend()
    plt.savefig("spd_progression.png")


# === SAVE ================================================================================
def save_models(biased_model, unbiased_model):
    for model, name in [(unbiased_model, "loan_model_alpha"), (biased_model, "loan_model_beta")]:
        model.export(name)
        ov_model = ov.convert_model(name)
        ov.save_model(ov_model, name+".xml")


def convert_to_inference_protocol(data_matrix):
    return f"""{{
  "inputs": [
    {{
      "name": "customer_data_input",
      "shape": [{len(data_matrix)}, 11],
      "datatype": "FP64",
      "data": {data_matrix}
    }}
  ]
}}"""


def save_data_batches(combined_train_test, batch_sizes):
    np.set_printoptions(suppress=True)

    batch_start_and_ends = []
    for idx, batch in enumerate(batch_sizes):
        if idx == 0:
            batch_start_and_ends.append([0, batch])
        else:
            batch_start_and_ends.append([batch_sizes[idx - 1], batch])

    for idx, batch in enumerate(batch_sizes):
        if idx == 0:
            bname = "training_data.json"
        else:
            bname = "batch_{}.json".format(str(idx).zfill(2))
        bpath = os.path.join("data","batches",  bname)

        subslice = combined_train_test[batch_start_and_ends[idx][0]:batch_start_and_ends[idx][1]]
        subx = subslice[[x for x in list(subslice) if x not in [OUTCOME, "Biased Prediction", "Unbiased Prediction"]]]

        with open(bpath, "w") as f:
            f.write(convert_to_inference_protocol(subx.values.tolist()))


# === MAIN =========================================================================================
if __name__ == "__main__":
    # load data
    data, feature_names = load_data()

    balanced_data = balance_data(data, 3000)

    # create deliberately biased training data, with deployment data that will exploit that bias
    training_data_without_demographic, removed_demographic_examples, training_data_without_demographic_rebalanced, deployment_data_poisoned = unfair_splitting(balanced_data)

    # train models on biased and unbiased data
    unbiased_model = create_unbiased_model(balanced_data, removed_demographic_examples, epochs=EPOCHS)

    biased_model, cheating_training_set = create_biased_model(training_data_without_demographic_rebalanced, epochs=EPOCHS)
    cheating_training_set['Unbiased Prediction'] = get_predictions(unbiased_model, cheating_training_set)

    print("Biased Model SPD on Cheated Training Set:  ", get_spd(cheating_training_set, "Biased Prediction"))
    print("Unbiased Model SPD on Cheated Training Set:", get_spd(cheating_training_set, "Unbiased Prediction"))

    # Get both models predictions over the entire data sets
    combined_train_test = pd.concat([cheating_training_set, deployment_data_poisoned, removed_demographic_examples])
    combined_train_test['Biased Prediction'] = get_predictions(biased_model, combined_train_test)
    combined_train_test['Unbiased Prediction'] = get_predictions(unbiased_model, combined_train_test)

    # Watch SPD progression over the data splits
    batch_sizes = [len(cheating_training_set)] + [len(cheating_training_set) + i for i in range(BATCH_SIZE, len(deployment_data_poisoned), BATCH_SIZE)]
    plot_progression(combined_train_test, batch_sizes)

    # save models
    save_models(biased_model, unbiased_model)
    save_data_batches(combined_train_test, batch_sizes)
