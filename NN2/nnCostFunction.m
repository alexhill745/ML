function [J grad] = nnCostFunction(nn_params, ...
                                   input_layer_size, ...
                                   hidden_layer_size, ...
                                   num_labels, ...
                                   X, y, lambda)
%NNCOSTFUNCTION Implements the neural network cost function for a two layer
%neural network which performs classification 
Theta1 = reshape(nn_params(1:hidden_layer_size * (input_layer_size + 1)), ...
                 hidden_layer_size, (input_layer_size + 1));

Theta2 = reshape(nn_params((1 + (hidden_layer_size * (input_layer_size + 1))):end), ...
                 num_labels, (hidden_layer_size + 1));

m = size(X, 1);
J = 0;
Theta1_grad = zeros(size(Theta1));
Theta2_grad = zeros(size(Theta2));
new_y = zeros(size(y,1),num_labels);
empty = zeros(num_labels,1);
for i = 1:size(y, 1)
  ind = y(i);
  empty(ind) = 1;
  new_y(i,:) = empty;
  empty = zeros(num_labels,1);
endfor
X = [ones(m, 1) X];
a_2 = sigmoid(X*Theta1');
a_2 = [ones(m, 1) a_2]; 
h = sigmoid(a_2*Theta2');
J = ((sum((sum(((-1*new_y).*log(h))-((1-new_y).*log(1-h))))))/m)+(((lambda)/(2*m))*((sum(sum(Theta1(:,2:end).^2)))+(sum(sum(Theta2(:,2:end).^2)))));
d_3 = h-new_y;
z_2 = X*Theta1';
d_2 = (d_3*Theta2(:,2:end)).*sigmoidGradient(z_2);
del_1 = d_2'*X;
del_2 = d_3'*a_2;
Theta1_grad = del_1/m;
Theta2_grad = del_2/m;
Theta1(:,1) = 0;
Theta2(:,1) = 0;
Theta1_reg = Theta1*(lambda/m);
Theta2_reg = Theta2*(lambda/m);
Theta1_grad = Theta1_grad + Theta1_reg;
Theta2_grad = Theta2_grad + Theta2_reg;
grad = [Theta1_grad(:) ; Theta2_grad(:)];
end