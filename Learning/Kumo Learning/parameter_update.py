x = 2.0
y_true = 10.0

# Initial Parameters
w = 1.5
b = 0.5
learning_rate = 0.1

y_pred = (x * w) + b
print(f"Old Prediction: {y_pred:.2f} | Old Loss: {(y_pred - y_true)**2:.2f}")

dL_dy_pred = 2 * (y_pred - y_true)
dL_dw = dL_dy_pred * x
dL_db = dL_dy_pred * 1.0

w = w - (learning_rate * dL_dw)
b = b - (learning_rate * dL_db)

y_pred_new = (x * w) + b
print(f"New Prediction: {y_pred_new:.2f} | New Loss: {(y_pred_new - y_true)**2:.2f}")