class KumoLayer:
    def __init__(self, in_features, out_features, degree=3):
        self.in_features = in_features
        self.out_features = out_features
        self.degree = degree
        self.D = degree + 1

        self.C = np.random.randn(in_features, out_features, self.D) * 0.1
        self.b = np.zeros((1, out_features))

    def backward(self, dL_dY, lr):
        batch_size = dL_dY.shape[0]
        
        dL_dC = np.einsum('nid,no->iod', self.B, dL_dY) / batch_size
        dL_db = np.sum(dL_dY, axis=0, keepdims=True) / batch_size
        dL_dB = np.einsum('no,iod->nid', dL_dY, self.C)
        
        dB_dX = np.zeros_like(self.B)
        for d in range(1, self.D):
            dB_dX[:, :, d] = d * (self.X_in ** (d - 1))
            
        dL_dX = np.sum(dL_dB * dB_dX, axis=2)
        
        self.C -= lr * dL_dC
        self.b -= lr * dL_db
        
        return dL_dX 