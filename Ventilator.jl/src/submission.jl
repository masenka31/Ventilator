# test data
# predictions

using DrWatson
@quickactivate
using CSV, DataFrames, Flux
using Ventilator

data = Ventilator.load_test_data(;rnn=true);
X, B = data;

df = collect_results(datadir("RNN"))
sort!(df, :val_score)
model = df[1,:model]

function predict(model, x)
    Flux.reset!(model)
    vcat([model(xi) for xi in x]...)
end

predict(model, X[1])

d = CSV.read(datadir("test.csv"), DataFrame)
id = d[!,:id]

predictions = map(x -> predict(model, x), X)
predictions = vcat(predictions...)

res = DataFrame(:id => id, :pressure => predictions)
wsave(datadir("submissions", "rnn_predictions.csv"), res)

################
### 9 models ###
################

df = collect_results(datadir("9models"))
sort!(df, :val_score)

# check if some individual models are better maybe?

val_score = df[:, :val]
val_dfs = DataFrame.(val_score)
val_df = vcat(val_dfs...)

val_arr = val_df |> Array
map(x -> findmin(x), eachcol(val_arr))
# if all indexes are 1, then there is nothing else to do

best_models = df[1,:models]

function load_test_RC(;vars::Vector{Symbol}=[:time_step, :u_in, :u_out, :R, :C], rnn = true)
    data = CSV.read(datadir("test.csv"), DataFrame)
    RC_groups = groupby(data, [:R, :C])

    dataset = Dict()
    for g in RC_groups
        df = g |> DataFrame
        X, id = preprocess_test(df; vars = vars, rnn = rnn)
        k = Symbol("R=$(g[1,:R]), C=$(g[1,:C])")
        d = Dict(
            :X => X,
            :id => id
        )
        push!(dataset, k => d)
    end

    return dataset
end

function preprocess_test(df::DataFrame; vars::Vector{Symbol}=[:time_step, :u_in, :u_out, :R, :C], rnn=false)
    # group by breath_id
    groups = groupby(df, :breath_id)

    # allocate vectors
    if rnn
        X_data, ids = Vector{Vector{Float32}}[], Array{Int64}[]
    else
        X_data, ids = Array{Float32,2}[], Array{Int64}[]
    end

    for g in groups
        # the feature engineering part
        _x = collect(g[!, vars] |> Array |> transpose)
        xf0, x0 = pool_feature_add(_x; u = 0)
        xf1, x1 = pool_feature_add(_x; u = 1)
        xfull = vcat(hcat(x0,x1), xf0, xf1)

        # push to allocated vectors
        if rnn
            push!(X_data, [xi for xi in eachcol(xfull)])
        else
            push!(X_data, xfull)
        end

        b = g[:, :id]
        push!(ids, b)
    end

    return X_data, ids
end

dataset = load_test_RC();

# function to predict pressure
function predict(model, x)
    Flux.reset!(model)
    vcat([model(xi) for xi in x]...)
end

function get_predictions(dataset, models, ks)
    predictions = Dict([(k, []) for k in keys(dataset)])

    Threads.@threads for i in 1:9
        # model based on key
        m = models[ks[i]]

        # score
        X = dataset[ks[i]][:X]
        pred = map(x -> predict(m, x), X)
        
        predictions[ks[i]] = pred
    end
    
    return predictions
end

ks = Dict([(i,k) for (i,k) in enumerate(keys(data))])

predictions = get_predictions(dataset, best_models, ks)

# now to get it all together

vec = []
for i in 1:9
    ids = dataset[ks[i]][:id]
    pred = predictions[ks[i]]
    v = (vcat(ids...), vcat(pred...))
    push!(vec, v)
end
id = vcat(map(x -> x[1], vec)...)
pressure = vcat(map(x -> x[2], vec)...)
sub = DataFrame(:id => id, :pressure => pressure)

wsave(datadir("submissions", "9models.csv"), sub)