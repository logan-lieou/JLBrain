using LinearAlgebra: dot

# Neural network struct is mutable
mutable struct NN{T <: AbstractArray}
    input::T
    weights::T
    output::T
    y::T
end

# Constructor
NN(input, y) = NN(input, randn(size(input)), zeros(size(y)), y)

# Function is diffrential of ReLU
function dReLU(x)
    if (x > 0)
        return 1
    end
    return 0
end

# Loss function
function default_loss(x)
    sum(x .^ 2)
end

# Neural network is a single node
function feed_foreward(network::NN)
    network.output .= max(0, dot(network.weights, network.input))
end

# Update weight values
function back_propagate(network::NN)
    d_weights = dot(transpose(network.output), (2 .* (network.y - network.output) .* (network.output)))
    network.weights = network.weights .+ d_weights
end

# Get the output from the network
function getResult(network::NN)
    return network.output
end