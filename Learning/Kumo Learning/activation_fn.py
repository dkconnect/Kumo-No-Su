import math

x = 0.5
y_true = 1.0 

w = 0.8
b = -0.2
alpha = 0.5

z = (x * w) + b 

y_pred = 1 / (1 + math.exp(-z))  

loss = (y_pred - y_true) ** 2

dL_dy_pred = 2 * (y_pred - y_true)
dy_pred_dz = y_pred * (1 - y_pred)  
dz_dw = x

dL_dw = dL_dy_pred * dy_pred_dz * dz_dw

w_new = w - (alpha * dL_dw)

print(f"z: {z:.4f}")
print(f"Prediction (Sigmoid): {y_pred:.4f}")
print(f"dL/dw Gradient: {dL_dw:.4f}")
print(f"Updated Weight: {w_new:.4f}")