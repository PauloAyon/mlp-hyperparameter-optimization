import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# The dataset has 9 features and two one-hot class columns
FEATURE_COUNT = 9
LABEL_COLUMN = 10


def load_data(path="data/cancer.csv"):
    # Read the file without a header because the original data has none
    data = pd.read_csv(path, header=None)

    # Keep the 9 features and use the last one-hot column as the label
    X = data.iloc[:, :FEATURE_COUNT]
    y = data.iloc[:, LABEL_COLUMN]

    # Reserve 20% for test and then split the rest into 60% train and 20% validation
    X_temp, X_test, y_temp, y_test = train_test_split(
        X, y, test_size=0.2, random_state=None
    )
    X_train, X_val, y_train, y_val = train_test_split(
        X_temp, y_temp, test_size=0.25, random_state=None
    )

    # Fit the scaler only on training data to avoid information leakage
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)
    X_test_scaled = scaler.transform(X_test)

    # Return NumPy arrays for the labels so they can be concatenated later
    return (
        X_train_scaled,
        X_val_scaled,
        X_test_scaled,
        y_train.to_numpy(),
        y_val.to_numpy(),
        y_test.to_numpy(),
    )
