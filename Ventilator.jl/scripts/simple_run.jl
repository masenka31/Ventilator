# packages
using DrWatson
@quickactivate
using Ventilator
using Flux
using Base.Iterators: repeated
using Flux: throttle, @epochs, mae, mse
using StatsBase
using Ventilator

using Plots
ENV["GKSwstype"] = "100"

# load data and unpack it
seed = rand(1:1000)
data = load_data_vectors(;seed=seed);
X_train, P_train = data[1];
X_val, P_val = data[2];
X_test, P_test = data[3];
B_train, B_val, B_test = data[4]
@info "Data loaded."

"""
    sample_params()

Samples parameters - hdim, activation function, MAE vs MSE,
divided or not, scaling term etc.
"""
function sample_params()
    hdim = sample(2 .^ [5,6,7,8,9])
    act = sample(["swish", "relu", "tanh", "sigmoid"])
    err = sample(["mae", "mse"])
    div = sample([true, false])
    β = sample(2:2:50)
    return hdim, act, err, div, β
end

idim = length(X_train[1])
odim = length(P_train[1])

hdim, act, err, div, β = sample_params()
err_fun = eval(:($(Symbol(err))))
activation = eval(:($(Symbol(act))))

model = Chain(Dense(idim, hdim, activation),Dense(hdim,hdim,activation),Dense(hdim,odim))

lossfun(X, P; err=mae) = err(model(X), P)
function lossfun_div(X, P, b; err=mae,β=10)
    W = model(X)
    Win, Wout = W[b .== 0], W[b .== 1]
    Pin, Pout = P[b .== 0], P[b .== 1]
    β*err(Win, Pin) + err(Wout,Pout)
end

if div
    loss(X, P, b) = lossfun_div(X, P, b; err=err_fun, β=β)
else
    loss(X, P, b) = lossfun(X, P; err=err_fun)
end

score(X, P) = Flux.mae(model(X), P)
score(X, P, b) = Flux.mae(model(X)[b .== 0], P[b .== 0])
best_score(X, P, b) = Flux.mae(best_model(X)[b .== 0], P[b .== 0])
opt = ADAM()

best_val_score = Inf
best_model = deepcopy(model)

start_time = time()
max_train_time = 60*60*3.5
k = 1

@info "Started training model with $hdim number of hidden neurons, $act activation function, $err loss (divided=$div)."
while true
    if div
        batch = RandomBatch(X_train, P_train, B_train, batchsize=64)
    else
        batch = RandomBatch(X_train, P_train, batchsize=64)
    end
    Flux.train!(loss, Flux.params(model), zip(batch...),opt)

    l = mean(score.(X_val, P_val, B_val))
    if l < best_val_score
        global best_model = deepcopy(model)
        global best_val_score = l
        @info "Epoch $k: score = $(round(l, digits=3))"
    end
    global k += 1

    # stop early if time is running out
    if (time() - start_time > max_train_time)
        @info "Stopped training, time limit exceeded."
        break
    end
end

train_sc = mean(score.(X_train, P_train, B_train))
val_sc = mean(score.(X_val, P_val, B_val))
test_sc = mean(score.(X_test, P_test, B_test))

# save best model
using BSON
d = Dict(
    :model => best_model,
    :seed => seed,
    :hdim => hdim,
    :activation => act,
    :loss => err,
    :divide => div,
    :beta => β,
    :train_score => train_sc,
    :val_score => val_sc,
    :test_score => test_sc
)
name = savename("model", d, "bson")
safesave(datadir("models", name), d)