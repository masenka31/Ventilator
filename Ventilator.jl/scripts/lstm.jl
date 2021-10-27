# packages
using DrWatson
@quickactivate
using Ventilator
using Flux
using Base.Iterators: repeated
using Flux: throttle, @epochs, mae, mse
using StatsBase

using Plots
# ENV["GKSwstype"] = "100"

# load data and unpack it
# seed = rand(1:1000)
seed = 1
data = load_data_bags(;seed=seed, rnn=true);
X_train, P_train = data[1];
X_val, P_val = data[2];
X_test, P_test = data[3];
B_train, B_val, B_test = data[4];
x, p = X_train[1], P_train[1]
@info "Data loaded."

function loss(x, y)
    Flux.reset!(model)
    # TO DO
    # b = hcat(x...)[3,:]
    # mean(Flux.mae(model(xi), yi) for (xi, yi) in zip(x[b .== 0], y[b .== 0]))
    mean(Flux.mae(model(xi), yi) for (xi, yi) in zip(x, y))
end
function sumloss(x, y)
    Flux.reset!(model)
    sum(Flux.mae(model(xi), yi) for (xi, yi) in zip(x, y))
end

opt = ADAM()
model = Chain(LSTM(5,32), LSTM(32,32), Dense(32,1))
best_val_score = Inf
best_model = deepcopy(model)
max_train_time = 60*60*9
k = 1

# start training
start_time = time()
while true
    # create a minibatch and train the model on a minibatch
    batch = RandomBatch(X_train, P_train, batchsize=128)
    Flux.train!(loss, Flux.params(model), repeated((batch[1], batch[2]), 5), opt)

    # only save a best model which has the best score on validation data
    # sc = loss(RandomBatch(X_val, P_val, batchsize=7500)...)
    sc = mean(score.(X_val, P_val))
    if sc < best_val_score
        global best_model = deepcopy(model)
        global best_val_score = sc
        @info "Epoch $k: validation score = $(round(sc, digits=3))"
    end
    global k += 1

    # stop when training time is exceeded
    if (time() - start_time > max_train_time)
        @info "Stopped training, time limit exceeded."
        break
    end
end

# calculate the score given the best model after training is finished
train_sc = loss(RandomBatch(X_train, P_train, batchsize=60000)...)
val_sc = loss(RandomBatch(X_val, P_val, batchsize=7500)...)
test_sc = loss(RandomBatch(X_test, P_test, batchsize=7500)...)

x = X_train[1]
y = P_train[1]
loss(x, y)

prediction = vcat([model(xi) for xi in x]...)
pressure = vcat(p...)
Flux.mae(prediction, pressure)

function predict(model, x)
    Flux.reset!(model)
    vcat([model(xi) for xi in x]...)
end
function score(x::Vector{Vector{Float32}}, p::Vector{Vector{Float32}})
    b = hcat(x...)[3,:]
    Flux.mae(predict(model, x)[b .== 0], vcat(p...)[b .== 0])
end
function best_score(x, p)
    b = hcat(x...)[3,:]
    Flux.mae(predict(best_model, x)[b .== 0], vcat(p...)[b .== 0])
end
score(x, p)
best_score(x, p)

mean(score.(X_val, P_val))
mean(best_score.(X_val, P_val))
mean(best_score.(X_test, P_test))

i = rand(1:7500)
plot(vcat(P_val[i]...), marker=:circle, label="ground truth", color=:green);
plot!(predict(best_model, X_val[i]), marker=:square, label="prediction", title="MAE: $(round(best_score(X_val[i], P_val[i]), digits=3))")