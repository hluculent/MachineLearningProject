function theta = normalEquation(X, y)
%theta = (X'*X)^-1 * X' * y
theta = pinv(X' * X) * X' * y;
end