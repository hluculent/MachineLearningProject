function [ X_norm mu sigma] = featureNormalize(X)
X_norm = X;
mu = mean(X);
sigma = std(X);
% is there a one-hot solution?
for i = 1: size(X, 1)
    X_norm(i, :) = (X(i,:) - mu) ./ sigma;
end

end
