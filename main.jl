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

function updateWeights(network::NN, newWeights)
    println("before ", network.weights)
    network.weights = newWeights
    println("after ", network.weights)
end

function default_loss(x)
    sum(x .^ 2)
end

function feed_foreward(network::NN)
    return max(0, dot(network.weights, network.input))
    # network.output = max(0, dot(layer1, weights2))
end

function backprop()
end

#function ghetto_training(network::NN, η, ε)
#    for i = 1:ε
#        grads = calc_grad(network.weights, network.y)
#        updateWeights(network, network.weights + (-η * grads))
#    end
#end

#temp = NN([3.24 2.30; 2.33 4.41], randn(2, 2), randn(2, 2), randn(4, 2))
#updateWeights(temp, randn(2, 2))

temp = NN(randn(8, 8), randn(8, 1))
println(feed_foreward(temp))
