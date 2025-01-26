import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from sklearn.utils import resample

df = pd.read_csv("Creditcard_data.csv")

minority_class = df[df['Class'] == 1]
majority_class = df[df['Class'] == 0]

majority_class_under = resample(majority_class, 
                                replace=False, 
                                n_samples=len(minority_class), 
                                random_state=42)

balanced_data = pd.concat([majority_class_under, minority_class])

balanced_data = balanced_data.sample(frac=1, random_state=42).reset_index(drop=True)

X = balanced_data.drop('Class', axis=1)
y = balanced_data['Class']


sample_size = int(len(balanced_data) / 5)
samples = [balanced_data.sample(n=sample_size, random_state=i) for i in range(5)]

def random_sampling(data):
    return data.sample(frac=1, random_state=42)

def stratified_sampling(data):

    stratified_sample = data.groupby('Class', group_keys=False).apply(
        lambda x: x.sample(int(len(x) / 2), random_state=42)
    )
    return stratified_sample

def systematic_sampling(data):

    step = max(1, len(data) // 10)
    return data.iloc[::step]

def cluster_sampling(data):

    n_clusters = 4
    clusters = np.array_split(data, n_clusters)
    selected_clusters = pd.concat(clusters[:2])
    return selected_clusters

def bootstrap_sampling(data):

    return data.sample(frac=1, replace=True, random_state=42)

sampling_functions = [
    random_sampling,
    stratified_sampling,
    systematic_sampling,
    cluster_sampling,
    bootstrap_sampling
]



models = {
    "Logistic Regression": LogisticRegression(max_iter=1000),
    "Decision Tree": DecisionTreeClassifier(),
    "Random Forest": RandomForestClassifier(),
    "KNN": KNeighborsClassifier(),
    "SVM": SVC()
}


results = []

for i, sampling_function in enumerate(sampling_functions, start=1):
    for model_name, model in models.items():
        sample = sampling_function(balanced_data)
        X_sample = sample.drop('Class', axis=1)
        y_sample = sample['Class']


        X_train, X_test, y_train, y_test = train_test_split(X_sample, y_sample, test_size=0.3, random_state=42)


        model.fit(X_train, y_train)

        y_pred = model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)


        results.append({
            "Sampling Technique": f"Sampling{i}",
            "Model": model_name,
            "Accuracy": accuracy
        })

results_df = pd.DataFrame(results)

print("Model Performance on Different Sampling Techniques:")
print(results_df.pivot(index="Model", columns="Sampling Technique", values="Accuracy"))

results_df.to_csv("sampling_results.csv", index=False)
