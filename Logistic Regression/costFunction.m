function [J, grad] = costFunction(theta, X, y)
m = length(y);
grad = zeros(size(theta));
h = sigmoid(X*theta);
J = (sum(((-1*y).*log(h))-((1-y).*log(1-h))))/(m);
for i = 1:size(X,2)
  grad(i) = (1/m)*(sum(((h)-y).*X(:,i)));
end
end
