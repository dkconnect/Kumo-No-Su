```Kumo test accuracy = 91.50```

**the 10000 image test set was not used to train Kumo or choose the checkpoint**

The progression is now quite clear:

| Experiment | Training data | Best accuracy |
|---|---:|---:|
| MSE prototype | 2,000 | 57.2% |
| Softmax + CE | 2,000 | 74.0% validation |
| Softmax + CE | 10,000 | 89.4% validation |
| Full experiment | 50,000 | **91.84% validation** |
| Untouched test set | 10,000 | **91.50% test** |

Also the validation/test gap is small: ```91.84 - 91.50 = 0.34```

There were 2 amazing observations in the output. Training loss was **still decreasing at epoch 10** and epoch 10 was also the best validation checkpoint:

```
Epoch 9  | Loss 0.284314 | 91.64%
Epoch 10 | Loss 0.275793 | 91.84%
```

one more interesting case is:

```
True: 5 | Predicted: 6 | Confidence: 97.62%
```

## Currently Kumo at

```784 pixels --> 16 hidden Kumo nodes --> 10 digit outputs```

with degree 2 functions on its edges.

The Parameter count is: ```784(16)(3) + 16(10)(3) = 37,632 + 480 = ``` ```38,112 learned coefficients```

And also those 38k polynomial coefficients are enough for **91.5% MNIST test accuracy** in my current implementation.

### Next phase: To make Kumo kinda usable
I am done with
- TRAIN KUMO
- SAVE MODEL                 
- LOAD MODEL                 

Now moving towards
- DRAW DIGIT ON CANVAS      
- convert drawing to 28×28 and 784 normalized pixels
- Kumo.forward()
- softmax
- prediction + confidence
- visualize Kumo's web