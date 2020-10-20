function p = predict(Theta1, Theta2, X)
%PREDICT Predict the label of an input given a trained neural network
% Useful values
m = size(X, 1);
num_labels = size(Theta2, 1);

% You need to return the following variables correctly 
p = zeros(size(X, 1), 1);
X = [ones(m, 1) X]; 
a_2 = sigmoid(X*Theta1');
a_2 = [ones(m, 1) a_2]; 
h = sigmoid(a_2*Theta2');
for i = 1:m
  h(h==max(h(i,:))) = 1;
end
h(h<1)=0;
numbers = [1:num_labels];
p = h*numbers';
end