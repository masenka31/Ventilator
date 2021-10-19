using Ventilator
using Flux
using Base.Iterators: repeated
using Flux: throttle, @epochs
using StatsBase
using Ventilator

using Plots
ENV["GKSwstype"] = "100"

data = load_data_vectors(;seed=1);
X_train, P_train = data[1];
X_val, P_val = data[2];
X_test, P_test = data[3];
B_train, B_val, B_test = data[4]
x = X_val[1]
p = P_val[1]
b = B_val[1]

function sample_params()
    hdim = sample(2 .^ [5,6,7,8,9])
    act = sample(["swish", "relu", "tanh", "sigmoid"])
    return hdim, act
end

idim = length(X_train[1])
odim = length(P_train[1])

hdim, act = sample_params()
activation = eval(:($(Symbol(act))))
model = Chain(Dense(idim, hdim, activation),Dense(hdim,hdim,activation),Dense(hdim,odim))

loss(X, P) = Flux.mse(model(X), P)
function loss(X, P, b)
    W = model(X)
    Win, Wout = W[b .== 0], W[b .== 1]
    Pin, Pout = P[b .== 0], P[b .== 1]
    10*Flux.mae(Win, Pin) + Flux.mae(Wout,Pout)
end

score(X, P) = Flux.mae(model(X), P)
score(X, P, b) = Flux.mae(model(X)[b .== 0], P[b .== 0])
best_score(X, P, b) = Flux.mae(best_model(X)[b .== 0], P[b .== 0])
opt = ADAM()

best_val_score = Inf
best_model = deepcopy(model)

for k in 1:1000
    #batch = RandomBatch(X_train, P_train, batchsize=64)
    #Flux.train!(loss, Flux.params(model), zip(batch...),opt)
    batch = RandomBatch(X_train, P_train, B_train, batchsize=64)
    Flux.train!(loss, Flux.params(model), zip(batch...),opt)
    l = mean(score.(X_val, P_val, B_val))
    if l < best_val_score
        best_model = deepcopy(model)
        best_val_score = l
        @info "Epoch $k: score = $(round(l, digits=3))"
    end
end

# save best model
using JLD2
save(datadir("models", "simplest.jld2"), Dict(:model => model, :seed => 1))

# plot results
w = best_model(x)
p
plot(w, marker=:square,label="predicted")
p = plot!(p, marker=:circle,label="true")
wsave(plotsdir("simplest_model_x.png"),p)

mean(best_score.(X_test, P_test, B_test))

# get test data and create predictions
test_data = CSV.read(datadir("test.csv"), DataFrame)
test_groups = groupby(test_data, :breath_id)

X_sub = Array{Float32,1}[]
@time for g in test_groups
    x = reshape(g[!, [:time_step, :u_in]] |> Array, 80*2)
    R, C = g[1,:R], g[1,:C]
    push!(X_sub, vcat(x, R, C))
end

prd = best_model.(X_sub)
predictions = vcat(prd...)

submission = DataFrame(hcat(test_data[:,:id], predictions), [:id, :pressure])
submission[!, :id] = convert.(Int, submission[:, :id])
safesave(datadir("submissions", "simplest.csv"), submission)
