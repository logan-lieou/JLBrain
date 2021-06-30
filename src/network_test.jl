using Test

# File containing the functions we are testing
include("network.jl")

# Does the constructor work? ~> NO!
@test NN(randn(8, 8), randn(8))

@test 