%% Linear regression with multiple variables
% In this part, you will implement linear regression with multiple variables to
% predict the prices of houses. Suppose you are selling your house and you
% want to know what a good market price would be. 
% In training data HousePrice.txt,
%The first column is the size of the house (in square feet), the
% second column is the number of bedrooms, and the third column is the price
% of the house.

%% Feature Normalization
clear; close all; clc;

fprintf('Loading data... \n')
data = load('HousePrice.txt');
X = data( : , 1:2);
y = data( : , 3);
m = length(y);

% print out some data points
fprintf('First 10 examples from the dataset: \n')
% fetch from first column to second, transpose the sample matrix
fprintf(' x = [%.0f %.0f], y = %.0f \n', [X(1:10,:) y(1:10,:)]') 

fprintf('Program pause...\n')
pause;

fprintf('Scale features and set them to zero means.\n')

[X mu sigma] = featureNormalize(X);
X = [ones(m, 1) X];

%% Gradient descent
fprintf('Running gradient descent...\n')

iteration = 50;
alpha = 0.1;

% initialize theta
theta = zeros(3, 1);

[theta, J_history] = gradientDescent(X, y, theta, alpha, iteration);

% Plot the convergence graph
figure;
plot(1:numel(J_history), J_history, '-b', 'LineWidth', 2);
xlabel('Number of interations');
ylabel('Cost J');

% theta is a vector, if you put in %f, the sentence repeats three times
fprintf('Theta compute from gradient descent is:  \n')
fprintf('%f \n',theta)

%% Predict price
price = ([1 1650 3] - [0 mu]) ./ [1 sigma] * theta;
% if you try to use multiple lines for a cmd, please [] it for print
fprintf(['Predict price of a 1650 sq-ft, 3 br house'...
            '(using gradient descent):\n'])
fprintf('$ %f\n', price)

fprintf('Program pause...\n')
pause;

%% Normal equations
fprintf('Calculating theta using normal equations...\n')
data  = load('HousePrice.txt');
%data = csvread('HousePrice.txt');
X = data(:,1:2);
y = data(:, 3);
m = length(y);

X = [ones(m,1) X];

theta = normalEquation(X, y);

fprintf('Theta computed from the normal Equation:\n')
fprintf('%f \n', theta);

price = [1 1650 3] * theta;
fprintf(['Predict price of a 1650 sq-ft, 3 br house'...
            '(using gradient descent):\n'])
fprintf('$ %f\n', price)