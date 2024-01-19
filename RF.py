from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score
import pandas as pd

df = pd.read_csv("Tensile modulus.csv")
features = ["direction" , "temperature" , "Humidity"]
X = df[features]
y = df['Tensile modulus']

reg = RandomForestRegressor(random_state=0 , n_estimators=100)
reg.fit(X.values , y.values)
test = [[0 , 20 , 50]]
predictedValue = reg.predict(test)
print(f"The predicted value: {predictedValue[0]}")

r2 = r2_score(y , reg.predict(X.values))
print(f"The coefficient of determination: {r2}")


importance = reg.feature_importances_
for i in range(3):
    print(f"{features[i]} importance: {importance[i]}")