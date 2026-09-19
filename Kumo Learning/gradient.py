x = 2.0
y_true = 10.0
w = 1.5
b = 0.5

y_pred = (x * w) + b  # 3.5

loss = (y_pred - y_true) ** 2  # 42.25

dL_dy_pred = 2 * (y_pred - y_true)  
dy_pred_dw = x                    

dL_dw = dL_dy_pred * dy_pred_dw     

print(f"Loss Gradient w.r.t Weight (dL/dw): {dL_dw}")