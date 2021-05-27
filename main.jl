using LinearAlgebra: dot

# mutable neural network struct
mutable struct NN{T <: AbstractArray}
    input::T
    weights::T
    output::T
    y::T
end

# y[1] == M[2]
# default constructor
NN(input, y) = NN(input, randn(size(input)), zeros(size(y)), y)

function dReLU(x)
    if (x > 0)
        return 1
    end
    return 0
end

function default_loss(x)
    sum(x .^ 2)
end

function feed_foreward(network::NN)
    # single node nn, layers = irrelevant
    network.output .= max(0, dot(network.weights, network.input))
end

function back_propagate(network::NN)
    # update the weights
    d_weights = dot(transpose(network.output), (2 .* (network.y - network.output) .* (network.output)))
    network.weights = network.weights .+ d_weights
end

function getResult(network::NN)
    println(network.output)
end

#function possible_training(network::NN, η, ε)
#    for i = 1:ε
#        grads = calc_grad(network.weights, network.y)
#        updateWeights(network, network.weights + (-η * grads))
#    end
#end

#temp = NN([3.24 2.30; 2.33 4.41], randn(2, 2), randn(2, 2), randn(4, 2))
#updateWeights(temp, randn(2, 2))

# huge bug somewhere in the code
temp = NN(randn(8, 8), randn(8, 1))
feed_foreward(temp)
getResult(temp)
back_propagate(temp)
getResult(temp)
