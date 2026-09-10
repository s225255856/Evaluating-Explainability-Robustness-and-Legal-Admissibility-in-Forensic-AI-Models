import pandas as pd
import re

#Load the raw HDFS logs
df =  pd.read_csv("HDFS_2k.log_structured.csv") 

#Extract BlockId from Content column using regex
df["BlockId"] = df["Content"].str.extract(r'(blk_[0-9\-]+)')

#Drop rows without a block ID as they cannot be grouped 
df = df.dropna(subset=["BlockId"])

#Keep the column needed
df = df[["BlockId", "EventId"]]

#Count EventId occurrences per BlockId
event_counts = df.groupby("BlockId")["EventId"].value_counts().unstack().fillna(0)

#Handle missing values
df = df.fillna(0)

#Convert labels to numeric
if df["label"].dtype == "object":
    df["label"] = df["label"].map({"Normal": 0, "Anomaly":1})

#Separate features and labels
X = df.drop("label", axis=1)
y = df["label"]