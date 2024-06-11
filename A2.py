from ucimlrepo import fetch_ucirepo
from sklearn.preprocessing import LabelEncoder, StandardScaler
import pandas as pd


# Load Wholesale Customers dataset from UCI
wholesale_customers_data = fetch_ucirepo(id=292)
wholesale_customers = wholesale_customers_data.data.features

# Preprocess the dataset
wholesale_customers["Total Spend"] = wholesale_customers[
    ["Fresh", "Milk", "Grocery", "Frozen", "Detergents_Paper", "Delicassen"]
].sum(axis=1)
wholesale_customers["Category"] = pd.qcut(
    wholesale_customers["Total Spend"], q=3, labels=["Low", "Medium", "High"]
)

label_encode_wholesale = LabelEncoder()
wholesale_customers["Category"] = label_encode_wholesale.fit_transform(
    wholesale_customers["Category"]
)

X = wholesale_customers.drop(columns=["Total Spend", "Category"])
y = wholesale_customers["Category"]

scaler_wholesale = StandardScaler()
X_scaled = scaler_wholesale.fit_transform(X)