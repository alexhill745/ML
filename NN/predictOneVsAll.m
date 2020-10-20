function p = predictOneVsAll(all_theta, X)
m = size(X, 1);
num_labels = size(all_theta, 1);
p = zeros(size(X, 1), 1);
X = [ones(m, 1) X]; 
h = sigmoid(X*all_theta');
for i = 1:m
  h(h==max(h(i,:))) = 1;
end
h(h<1)=0;
numbers = [1:num_labels];
p = h*numbers';
end