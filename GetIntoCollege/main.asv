%% Logistic Regression
%Suppose that you are the administrator of a university department and you
%want to determine each applicant's chance of admission based on their 
%results on two exams. You have historical data from previous applicants 
%that you can use as a training set for logistic regression. For each training 
%example, you have the applicant's scores on two exams and the admissions 
%decision. 

clear;  close all; clc

%% Load data
data = load('ExamScore.txt');
X = data(:, [1,2]); y = data(:,3);

%Plot data
fprintf(['Plotting data with + indicating (y=1) examples and'...
    ' o indicating (y=0) examples.\n']);
plotData(X, y);
hold on;
xlabel('Exam 1 score');
ylabel('Exam 2 score');
legend('Admitted', 'Not admitted');
hold off;

fprintf('\nProgram paused.\n');
pause;

%% compute cost and gradient
[m, n] =size(X);
X = [ones(m, 1) X];

%Initialize fitting parameters
initial_theta = zeros(n+1, 1);

[cost, grad] = costFunction(initial_theta, X, y);

fprintf('Cost at initial theta (zeros): %f\n', cost);
fprintf('Gradient at inital theta (zeros): \n');
fprintf('    %f  \n', grad);

fprintf('Program paused.\n');
pause;

%% Optimizing using fminuc

options = optimset('GradObj', 'on', 'MaxIter', 400);
[theta, cost] = ...
    fminunc(@(t) (costFunction(t, X, y)), initial_theta, options);
fprintf('Cost at theta found by fminunc: %f\n', cost);
fprintf('theta: \n');
fprintf('    %f \n', theta);

%plot decision boundary
plotDecisionBoundary(theta, X, y);
hold on;
xlabel('Exam 1 score');
ylabel('Exam 2 score');
legend('Admitted', 'Not admitted');
hold off;

fprintf('Program paused.\n');
pause;

%% predict and Accuracies

prob = sigmoid([1 45 85]*theta);
fprintf(['For a student with scores 45 and 85, we predict an admission '...
    'probability of %f\n\n'], prob);

% Compute accuracy on our training set
p = predict(theta, X);
fprintf('Train Accuracy: %f \n',  mean(double(p==y))*100);
%fprintf('Train Accuracy: %f \n', mean(sum(p==y))*100);