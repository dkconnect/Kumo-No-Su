# Kumo-No-Su

In a standard neural network:

* Connections carry simple numbers called scalar weights ($W$).
* Nodes apply fixed, unchangeable activation curves (Sigmoid, ReLU).

I tried to go a different route, so in Kumo No Su:

* Connections carry dynamic, learnable mathematical curves ($\phi(x)$).
* Nodes perform simple addition. 
* Each edge uses a linear combination of polynomial basis functions to shape its own response curve:

$$\phi(x) = c_0 + c_1 x + c_2 x^2 + c_3 x^3 + \dots + c_k x^k$$