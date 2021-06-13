import pandas as pd
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor

# importing dataset
melbourne_data = pd.read_csv(r'C:\Users\mobin\Desktop\Primary ML\melb_data.csv')

# cleaning data
melbourne_data = melbourne_data.dropna(axis=0)

# indicating prediction target
y = melbourne_data["Price"]

# indicating features
melbourne_features = ['Rooms', 'Bathroom', 'Landsize', 'Lattitude', 'Longtitude']
X = melbourne_data[melbourne_features]

# splitting validation data from train data
train_X, val_X, train_y, val_y = train_test_split(X, y, random_state=0)
model=DecisionTreeRegressor()

# comparing MAE scores from different values for max_leaf_nodes
def get_mae(max_leaf_nodes, train_X, val_X, train_y, val_y):
    model = DecisionTreeRegressor(max_leaf_nodes=max_leaf_nodes, random_state=0)
    model.fit(train_X, train_y)
    preds_val = model.predict(val_X)
    mae = mean_absolute_error(val_y, preds_val)
    return(mae)

for max_leaf_nodes in [5, 50, 500, 5000]:
    my_mae = get_mae(max_leaf_nodes, train_X, val_X, train_y, val_y)
    print("Max leaf nodes:  %d  \t   Mean absolute error:  %d" % (max_leaf_nodes, my_mae))
