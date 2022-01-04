# transformations
function maxabs_scaler(dataset)
    # maxabsscaler
    # we have dataset -> lets get pressure first
    pt = map(x -> x[:train][2], values(dataset))
    pt2 = vcat(pt...)
    pt3 = vcat(vcat(pt2...)...)
    maxabs_pressure = maximum(pt3)

    # now for the data
    X = map(x -> x[:train][1], values(dataset))
    X2 = vcat(X...)
    X3 = map(x -> hcat(x...), X2)
    X4 = hcat(X3...)

    maxabs_x = maximum(abs.(X4), dims=2)
    return maxabs_pressure, maxabs_x
end

function scale_data(scp, scx, dataset, ks)
    dataset_copy = deepcopy(dataset)
    for i in 1:9
        Xtr, Ytr = dataset[ks[i]][:train]
        xtr = map(x -> map(y -> y ./ scx, x), Xtr)
        ytr = map(x -> map(y -> y ./ scp, x), Ytr)

        Xv, Yv = dataset[ks[i]][:val]
        xv = map(x -> map(y -> y ./ scx, x), Xv)
        yv = map(x -> map(y -> y ./ scp, x), Yv)

        Xt, Yt = dataset[ks[i]][:test]
        xt = map(x -> map(y -> y ./ scx, x), Xt)
        yt = map(x -> map(y -> y ./ scp, x), Yt)

        dataset_copy[ks[i]][:train] = xtr, ytr
        dataset_copy[ks[i]][:val] = xv, yv
        dataset_copy[ks[i]][:test] = xt, yt
    end
    return dataset_copy
end