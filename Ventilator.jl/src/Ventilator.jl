module Ventilator

using DrWatson
using Plots
using StatsBase
using Random

include("plotting.jl")
include("data.jl")
include("utils.jl")

export scatter2, scatter2!
export load_data_vectors, load_data_bags, load_data_single
export RandomBatch
export rnn_constructor

end #module