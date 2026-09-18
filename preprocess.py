import pandas as pd
from sklearn.model_selection import train_test_split

def load_and_preprocess():
    #Load preprocessed HDFS event occurence matrix
    events = pd.read_csv("HDFS_v1/preprocessed/Event_occurrence_matrix.csv")

    #Load anomaly labels
    labels = pd.read_csv("HDFS_v1/preprocessed/anomaly_label.csv")

    #Ensure BlockId column name's match
    assert "BlockId" in events.columns
    assert "BlockId" in labels.columns

    #Convert string labels to integers as SHAP takes integers
    label_map = {"Normal": 0, "Anomaly": 1}
    labels["Label"] = labels["Label"].map(label_map)

    #Merge features and labels
    df = events.merge(labels, on="BlockId")

    #Drop BlockId that are not useful for ML
    df = df.drop(columns=["BlockId", "Label_x", "Type"])

    #Use Label_y as the correct label column
    df = df.rename(columns={"Label_y": "Label"})

    #Split into features (X) and labels (y)
    X = df.drop("Label", axis=1)
    y = df["Label"]

    #Training and testing data split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    return X_train, X_test, y_train, y_test
        
def save_splits():
    #Save train/test splits to CSV so other scripts can load them quickly
    X_train, X_test, y_train, y_test = load_and_preprocess()

    X_train.to_csv("preprocessed2/X_train.csv")
    X_test.to_csv("preprocessed2/X_test.csv")
    y_train.to_csv("preprocessed2/y_train.csv")
    y_test.to_csv("preprocessed2/y_test.csv")

    print("Saved X_train, X_test, y_train, y_test to CSV.")
        