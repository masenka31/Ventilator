using DrWatson
@quickactivate
using DataFrames, CSV
using Random
using StatsBase

function load_data_bags(;seed=nothing, vars::Vector{Symbol}=[:time_step, :u_in, :u_out, :R, :C], rnn=false)

    data = CSV.read(datadir("train.csv"), DataFrame)
    groups = groupby(data, :breath_id)

    if rnn
        X_data, P_data, B_data = Vector{Vector{Float32}}[], Vector{Vector{Float32}}[], Array{Float32,2}[] #Array{Float32,2}[], Array{Float32,2}[], Array{Float32,2}[]
    else
        X_data, P_data, B_data = Array{Float32,2}[], Array{Float32,2}[], Array{Float32,2}[]
    end
    @time for g in groups
        x = collect(g[!, vars] |> Array |> transpose)
        p = collect(g[:, :pressure]')

        if rnn
            push!(X_data, [xi for xi in eachcol(x)])
            push!(P_data, [pi for pi in eachcol(p)])
        else
            push!(X_data, x)
            push!(P_data, p)
        end

        b = collect(g[:, :u_out]')
        push!(B_data, b)
    end

    # set seed
    (seed == nothing) ? nothing : Random.seed!(seed)

    idx = sample(1:75000, 75000, replace=false)
    tix, val_ix, test_ix = idx[1:60000], idx[60001:67500], idx[67501:end]
    X_train, P_train, B_train = X_data[tix], P_data[tix], B_data[tix]
    X_val, P_val, B_val = X_data[val_ix], P_data[val_ix], B_data[val_ix]
    X_test, P_test, B_test = X_data[test_ix], P_data[test_ix], B_data[test_ix]

    # reset seed
	(seed !== nothing) ? Random.seed!() : nothing

    return (X_train, P_train), (X_val, P_val), (X_test, P_test), (B_train, B_val, B_test)
end

function load_data_vectors(;seed=nothing)

    data = CSV.read(datadir("train.csv"), DataFrame)
    groups = groupby(data, :breath_id)

    X_data, P_data, B_data = Array{Float32,1}[], Array{Float32,1}[], Array{Float32,1}[]
    @time for g in groups
        x = reshape(g[!, [:time_step, :u_in]] |> Array, 80*2)
        R, C = g[1,:R], g[1,:C]
        push!(X_data, vcat(x, R, C))
        p = g[:, :pressure]
        push!(P_data, p)
        b = g[:, :u_out]
        push!(B_data, b)
    end

    # set seed
    (seed == nothing) ? nothing : Random.seed!(seed)

    idx = sample(1:75000, 75000, replace=false)
    tix, val_ix, test_ix = idx[1:60000], idx[60001:67500], idx[67501:end]
    X_train, P_train, B_train = X_data[tix], P_data[tix], B_data[tix]
    X_val, P_val, B_val = X_data[val_ix], P_data[val_ix], B_data[val_ix]
    X_test, P_test, B_test = X_data[test_ix], P_data[test_ix], B_data[test_ix]

	# reset seed
	(seed !== nothing) ? Random.seed!() : nothing

    return (X_train, P_train), (X_val, P_val), (X_test, P_test), (B_train, B_val, B_test)
end

function load_data_single(;seed=nothing)

    data = CSV.read(datadir("train.csv"), DataFrame)
    groups = groupby(data, :breath_id)

    _X_data = data[:, [:time_step, :u_in, :u_out, :R, :C]] |> Array |> transpose |> collect
    X_data = [_X_data[:,i] for i in 1:size(_X_data,2)]
    P_data = data[:, :pressure]

    # set seed
    (seed == nothing) ? nothing : Random.seed!(seed)

    l = length(P_data)
    a = Int(l*0.8)
    b = (l - a) / 2 |> Int
    idx = sample(1:l, l, replace=false)
    tix, val_ix, test_ix = idx[1:a], idx[a+1:a+b], idx[a+b+1:end]
    X_train, P_train = X_data[tix], P_data[tix]
    X_val, P_val = X_data[val_ix], P_data[val_ix]
    X_test, P_test = X_data[test_ix], P_data[test_ix]

	# reset seed
	(seed !== nothing) ? Random.seed!() : nothing

    return (X_train, P_train), (X_val, P_val), (X_test, P_test)
end

# data standartization
# https://juliastats.org/StatsBase.jl/stable/transformations/

"""
dtx = hcat(X_train...)
dtp = hcat(P_train...)

x_tf = fit(ZScoreTransform, dtx, dims=2)
p_tf = fit(ZScoreTransform, dtp, dims=2)

StatsBase.transform(x_tf, dtx)
StatsBase.transform(p_tf, dtp)
"""

function RandomBatch(Xd::Vector{Vector{Vector{Float32}}}, Yd::Vector{Vector{Vector{Float32}}}; batchsize=64)
    l = length(Xd)
    idx = sample(1:l, batchsize, replace=false)
    X = Xd[idx]
    Y = Yd[idx]
    x = [hcat([X[i][j] for i in 1:batchsize]...) for j in 1:80]
    y = [hcat([Y[i][j] for i in 1:batchsize]...) for j in 1:80]
    (x, y)
end