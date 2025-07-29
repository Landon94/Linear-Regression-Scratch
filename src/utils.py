import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
pd.set_option('future.no_silent_downcasting', True)


def load_data(filepath="data/Student_Performance.csv", split = True, t_size = 0.3):
    df = pd.read_csv(filepath)
    df["Extracurricular Activities"] = df["Extracurricular Activities"].replace({"No": 0, "Yes": 1}).astype(int)

    features = ["Hours Studied", "Previous Scores", "Extracurricular Activities", "Sleep Hours", "Sample Question Papers Practiced"]
    target = "Performance Index"

    X = df[features]
    y = df[target]

    if split:
        return train_test_split(X, y, test_size=0.3, random_state=1)
    else:
        return X, y


def plot_features_vs_performance(X, y, n=1000, norm=False):
    plt.figure(figsize=(15, 9))
    for i, feature in enumerate(X.columns, 1):
        plt.subplot(2, 3, i)
        plt.scatter(X[feature].iloc[:n], y.iloc[:n], alpha=0.4)
        plt.ylabel("Performance Index")
        plt.xlabel(feature)
        plt.title(f"{feature} vs Performance")
    
    plt.tight_layout()
    if norm:
        plt.suptitle(f"{n} Training Examples vs Performance Index (Normalized)", fontsize=16)
    else:
        plt.suptitle(f"{n} Training Examples vs Performance Index", fontsize=16)

    plt.subplots_adjust(top=0.88)
    plt.show()


# X_train, X_test, y_train, y_test = load_data()
# plot_features_vs_performance(X_train, y_train, norm=True)