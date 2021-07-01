using LinearAlgebra: dot

# Neural network struct is mutable
mutable struct NN
    input
    weights
    output
    y
end

# Constructor
NN(input, y) = 
		NN(input, randn(size(input)), zeros(size(y)), y)

# Function is diff of ReLU
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

# Training function
function trainNetwork!(network::NN, epochs::Int)
	for i = 1:epochs
		feed_foreward(network)
		back_propagate(network)
	end
end