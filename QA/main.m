%% regularized logistic regression

% predict whether microchips from a fabrication plant passes quality assurance (QA). 
%During QA, each microchip goes through various tests to ensure it is functioning correctly

clear; close all; clc

%% load data
data = load('microchipTest.txt');
X = data(:,[1 2]);
y = data(:,3);
m = size(X,1);
n = size(X,2);

plotData(X,y);
hold on;
xlabel('Microchip Test1');
ylabel('Microchip Test2');
legend('y=1','y=0');
hold off;

%% regularized logistic regression
% since the figure is not linear sperated, we should add some polynominal
% features
X = mapFeature(X(:,1), X(:,2));

initial_theta = zeros(size(X, 2), 1);

%regurlarization parameter
lambda = 1;
[cost, grad] = costFunctionReg(initial_theta, X, y, lambda);

fprintf('Cost at initial theta (zeros): %f\n', cost);

fprintf('\nProgram paused. Press enter to continue.\n');
pause;

%% regularization and accuracies

initial_theta = zeros(size(X,2), 1);
lambda = 1;
options = optimset('GradObj','on','MaxIter',400);

%optimize
[theta, J, exit_flag] = ...
    fminunc(@(t)(costFunctionReg(t, X, y, lambda)), initial_theta,  options);

%plot
plotDecisionBoundary(theta, X, y);
hold on;
title(sprintf('lambda = %g', lambda))
xlabel('Microchip Test 1')
ylabel('Microchip Test 2')
legend('y = 1', 'y = 0', 'Decision boundary')
hold off;

p = predict(theta, X);

fprintf('Train Accuracy: %f\n', mean(double(p == y)) * 100);