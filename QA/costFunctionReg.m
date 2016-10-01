function [J, grad] = costFunctionReg(theta, X, y, lambda)
m = length(y);
pos = y' * log(sigmoid(X * theta));
neg = (1-y)' * log(1-sigmoid(X * theta));
J = 1/m*(-pos-neg) +lambda/ (2*m) * sum( (theta(2:end)).^2 );
grad = 1/m * X' * (sigmoid(X*theta)-y) + [0; lambda/m*theta(2:end)];
end