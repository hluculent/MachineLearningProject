%% Linear Regression with one variable

% Predict profits for a food truck.
% Suppose you are the CEO of a  restaurant franchise and are considering
% different cities for opening a new outlet. The chain already has trucks in
% various cities and you have data for profits and populations from the
% cities.
 

clear; close all; clc

%% plot data
fprintf('Plotting Data ...\n')
data = load('ProfitData.txt');
X = data(:,1); y = data(:,2);
m = length(y); % number of training examples

plotData(X,y);

fprintf('Progress paused.\n')
pause;

%% compute cost using gradient descent
X = [ones(m, 1), data(:,1)]; % add a column of ones for theta_0
theta = zeros(2, 1); % initialize fitting parameters

% gradient descent settings
iterations = 1500;
alpha = 0.01;

theta = gradientDescent(X, y , theta, alpha, iterations);

fprintf('Theta computed by gradient descent:');
fprintf('%f %f \n', theta(1), theta(2));

% plot the linear fit
hold on;
plot(X(:,2), X*theta, '-')
legend('Training data', 'Linear regression')
hold off;

% predict
predict1 = [1, 3.5] *theta;
fprintf('For population = 35,000, we predict a profit of %f\n',...
    predict1*10000);
predict2 = [1, 7] * theta;
fprintf('For population = 70,000, we predict a profit of %f\n',...
    predict2*10000);

fprintf('Progress paused.\n')
pause;

%% Visualizing cost function J(theta)
fprintf('Visualizing J(theta_0, theta_1)...\n')

theta0_vals = linspace(-10, 10, 100);
theta1_vals = linspace(-1, 4, 100);
J_vals = zeros(length(theta0_vals), length(theta1_vals));

% fill out J_vals
for i = 1: length(theta0_vals)
    for j = 1: length(theta1_vals)
        t = [theta0_vals(i); theta1_vals(j)];
        J_vals(i,j) = computeCost(X, y, t);
    end
end

% Because of the way meshgrids work in the surf command, we need to 
% transpose J_vals before calling surf, or else the axes will be flipped
J_vals = J_vals';

figure;
% surface plot
surf(theta0_vals, theta1_vals, J_vals)
xlabel('\theta_0');
ylabel('\theta_1');
%contour plot
figure;
contour(theta0_vals, theta1_vals, J_vals, logspace(-2,3,20))
xlabel('\theta_0');
ylabel('\theta_1');
hold on;
plot (theta(1), theta(2), 'rx', 'MarkerSize', 10, 'LineWidth',2);
hold off;
        



