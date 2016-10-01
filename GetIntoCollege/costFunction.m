function [J, grad] = costFunction(theta, X, y)

m  = length(y);
cost_neg = (1-y)' *log(1-sigmoid(X*theta));
cost_pos = y' * log(sigmoid(X*theta));
J = 1/m * (-cost_pos - cost_neg);
grad = 1/m * X' * (sigmoid(X*theta) - y);

end