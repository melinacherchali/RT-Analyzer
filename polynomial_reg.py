regression_data = regression_data_generator(n = 50)

from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression

m3 = PolynomialFeatures(degree=3, include_bias=False)
poly_features = m3.fit_transform(regression_data.drop(['y'], axis=1).values)

poly_reg_model = LinearRegression()
poly_reg_model.fit(poly_features, regression_data["y"])

y_predicted = poly_reg_model.predict(poly_features)
