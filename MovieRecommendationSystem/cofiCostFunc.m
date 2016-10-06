function [J, grad] = cofiCostFunc(params, Y, R, num_users, num_movies, ...
                                  num_features, lambda)
                              
% Unfold the U and W matrices from params
X = reshape(params(1:num_movies*num_features), num_movies, num_features);
Theta = reshape(params(num_movies*num_features+1:end), ...
                num_users, num_features);            
% initialization
J = 0;
X_grad = zeros(size(X));
Theta_grad = zeros(size(Theta));

J = 1/2 * sum(sum((((X * Theta'  - Y).^2).*R)));

for i = 1:num_movies
  idx = find(R(i,:)==1);
  Theta_tmp = Theta(idx,:);
  Y_tmp = Y(i,idx);
  X_grad(i,:) = (X(i,:) * Theta_tmp' - Y_tmp) * Theta_tmp + lambda* X(i,:);
end

for j = 1: num_users
  idx = find(R(:,j) == 1);
  X_tmp = X(idx,:);
  Y_tmp = Y(idx,j);
  Theta_grad(j,:) = (X_tmp * Theta(j,:)' - Y_tmp)' * X_tmp + lambda * Theta(j,:);
end

J = J + lambda/2 * sum(sum(Theta.^2)) + lambda/2 * sum(sum(X.^2));


grad = [X_grad(:); Theta_grad(:)];

end
