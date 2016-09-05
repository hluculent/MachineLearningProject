function [theta, J_history] = gradientDescent(X, y, theta, alpha, num_iter)
m = length(y);
J_history = zeros(num_iter, 1);
for iter = 1: num_iter
    theta = theta - alpha * 1/m * X' * (X * theta -y);
    J_history (iter) = computeCost(X, y, theta);
end
