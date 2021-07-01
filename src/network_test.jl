using Test

# File containing the functions we are testing
include("network.jl")

# Does the constructor work?
t_network = NN(rand(8, 8), rand(8))

# Test feed_foreward
feed_foreward(t_network)
@test sum(t_network.input) > 0 ? t_network != zeros(size(t_network.output)) : true

# Does dReLU work as intended?
@test dReLU(8) == 1