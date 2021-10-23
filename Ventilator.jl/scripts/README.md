# Overview of models

## Data format

The data is in a format of a sequence. Given a sequence of 80 datapoints for each breath, the task is to predict ventilator pressure at each time step. These are the variables used for prediction:
- **time_step**: the time of measurement
- **u_in**: the control input for the inspiratory solenoid valve (ranges from 0 to 100)
- **u_out**: indicator variable, 0 for breathe-in, 1 for breathe-out
- **R**: lung attribute indicating how restricted the airway is (in cmH2O/L/S). Physically, this is the change in pressure per change in flow (air volume per time). Intuitively, one can imagine blowing up a balloon through a straw. We can change R by changing the diameter of the straw, with higher R being harder to blow
- **C**: lung attribute indicating how compliant the lung is (in mL/cmH2O). Physically, this is the change in volume per change in pressure. Intuitively, one can imagine the same balloon example. We can change C by changing the thickness of the balloon’s latex, with higher C having thinner latex and easier to blow

Have in mind that R and C are the same for each time-point in each set (for a single breath).

*Important note: The final score is taken only from the variables that correspond with breathe-in.*

## Models

### single_value_predictor

The most basic approach. If we completely abandon the grouping based on `breath_id`, we can approach the data as single datapoints where we want to predict `pressure` from a vector of `time_step, u_in, u_out, R, C`.

We can create a simple neural network such as
```
model = Chain(Dense(5, hdim, relu), Dense(hdim, hdim, relu), Dense(hdim, 1))
```
and use it to predict single values of pressure. The loss function will simply be MSE or MAE `loss(x, p) = mae(model(x),p)`, where $x$ is the given vector and $p$ is the pressure to be predicted.

>  Approximate best score: $\approx$ **2.6**.

---

### 3layer_net

This model take `breath_id` into account and uses a single breath as a datapoint.

The easiest way to model the pressure from input data is to simply create a vector of all observed variables. Vectors $t, u_{in}$ are vertically concatenated with variables $R, C$ to give a 1D vector of dimension 162. This input vector is then used to predict the pressure vector $P$ with a simple neural network.

The architecture of the 3layer_net is fairly simple, just a stact of 3 Dense layers with a `relu` activation function as
```
model = Chain(Dense(168, hdim, relu), Dense(hdim, hdim, relu), Dense(hdim, 80))
```

The loss function is either MAE or MSE loss
```
loss(X, P) = mae(model(X), P) # X is input vector, P is the pressure to be predicted
```

Since the score is calculated only as MAE between predicted and true pressure for **breathe-in** (`u_out = 0`), it makes sense to give the prediction of pressure for `u_out = 0` higher weight than for `u_out = 1`. The modification of the loss function can be as follows
```
function loss(X, P, b; β=10)
    W = model(X)
    Win, Wout = W[b .== 0], W[b .== 1]
    Pin, Pout = P[b .== 0], P[b .== 1]
    β*mae(Win, Pin) + mae(Wout,Pout)
end
```
where $\beta$ is a scaling parameter which gives more importance to `u_out = 0` and `b` is the `u_out` indicator variable.

*Note: The MAE loss can be optionally replaced with MSE in training.*

>  Approximate best score: $\approx$ **1.0**.