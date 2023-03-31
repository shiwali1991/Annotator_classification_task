import streamlit as st
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

# Read data
df = pd.read_csv("annotator_df.csv")

# Define weights
weights = {"combined_accuracy": 0.5, "annotator_agreement": 0.4, "avg_time": 0.1}

# Compute the combined score
df["score"] = (weights["combined_accuracy"] * df["combined_accuracy"] +
                      weights["annotator_agreement"] * df["annotator_agreement"] +
                      weights["avg_time"] * df["avg_time"]) / sum(weights.values())

# Scale the data
scaler = StandardScaler()
data_scaled = scaler.fit_transform(df[["combined_accuracy", "annotator_agreement", "avg_time", "score"]])



# Train KMeans on the combined score
kmeans = KMeans(n_clusters=2, random_state=42)
kmeans.fit(data_scaled)

# Define labels
labels = ["Good" if label == 0 else "Bad" for label in kmeans.labels_]

# Create Streamlit app
st.title("Annotator Classifier")

def get_labels(accuracy, agreement, time):
    
    weights = {"combined_accuracy": 0.4, "annotator_agreement": 0.4, "avg_time": 0.2}

    df = pd.DataFrame([{"combined_accuracy": accuracy, "annotator_agreement": agreement, "avg_time": time }])
    df["score"] = (weights["combined_accuracy"] * df["combined_accuracy"]) + (weights["annotator_agreement"]  * df["annotator_agreement"]) + (weights["avg_time"] * df["avg_time"])


    #scaler = StandardScaler()
    data_scaled = scaler.transform(df[["combined_accuracy", "annotator_agreement", "avg_time", "score"]])
    
    return kmeans.predict(data_scaled)


# Define inputs
accuracy = st.slider("Accuracy", min_value=0.0, max_value=1.0, value=0.5, step=0.01)
agreement = st.slider("Agreement", min_value=0.0, max_value=10.0, value=5.0, step=0.1)
time = st.slider("Time (seconds)", min_value=0.0, max_value=10.0, value=1.0, step=0.1)

label = get_labels(accuracy, agreement, time)
st.write(label)

