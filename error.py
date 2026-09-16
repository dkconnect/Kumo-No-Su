x = 2.0
y_true = 10.0
w = 1.5
b = 0.5

y_pred = (x * w) + b 

# Mean Squared Error
loss = (y_pred - y_true) ** 2

print(f"Prediction: {y_pred}")
print(f"Target: {y_true}")
print(f"Loss: {loss}")