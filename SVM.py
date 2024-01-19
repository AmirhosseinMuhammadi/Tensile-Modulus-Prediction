from sklearn.svm import SVR
from sklearn.metrics import r2_score
from sklearn.inspection import permutation_importance
import pandas as pd

df = pd.read_csv("Tensile modulus.csv")
features = ["direction" , "temperature" , "Humidity"]
X = df[features]
y = df['Tensile modulus']

reg = SVR(kernel="rbf" , C=1 , epsilon=0.1)
reg.fit(X.values , y.values)

test = [[0 , 20 , 50]]
predictedValue = reg.predict(test)
print(f"The predicted value: {predictedValue[0]}")

r2 = r2_score(y , reg.predict(X.values))
print(f"The coefficient of determination: {r2}")


importance = permutation_importance(reg , X.values , y.values)
for i in range(3):
    print(f"{features[i]} importance: {importance['importances_mean'][i] / sum(importance['importances_mean'])}")