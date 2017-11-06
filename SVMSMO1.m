clc;
clear all;
close all;


load fisheriris.mat

y = zeros(length(species),1);

%Standardize input data
m = length(meas);
muX = mean(meas);
stdX = std(meas);

repstd = repmat(stdX,m,1);

repmu = repmat(muX,m,1);

meas = (meas-repmu)./repstd;

disp('Standardization Result');
[mean(meas); std(meas)]

%meas(:,1:end)=zscore(meas(:,1:end));

for i = 1:length(species)
  %We are clumping versicolor and virginica together.
  if strcmp(species{i,1},'setosa')
    s = 1;
  else 
    s = -1;   
  end
  y(i) = s;
end
iris = [meas y];
iris = iris(randperm(size(iris,1)),:); % Shuffle data around

% Randomly pick out ~50% of the data for training.
randVar = rand(length(iris),1);
Index = randVar >= 0.5; 

%Index the data
iris_train = iris(Index,:);
iris_test = iris(~Index,:);

%Store sizes of each set of data.
[N_train,~] = size(iris_train);
[N_test,~] = size(iris_test);


x_train = iris_train(:,1:4);
y_train = iris_train(:,5);

x_test = iris_test(:,1:4);
y_test = iris_test(:,5);

%
%%% Plot of Input
%figure
%hold on
%scatter(x(y==1,1),x(y==1,2),'*b')
%scatter(x(y==-1,1),x(y==-1,2),'*r')
%axis ([-2 2 -2 2])
%title('Linearly seperable data')
%xlabel('{x_1}'),ylabel('{x_2}')
%legend('setosa','virginica and versicolor')
%hold off




%Intialize params:
b = 0;
epsilon = 1e-5;
%lagrange subject to that condition. I guess all zeros works, so I'm going to do
%it for now.

%Alpha is constrained by condition that their sum = 0
alpha = rand(1,length(y_train));
alpha * y_train % not equal 0
alpha(1) = -(alpha(2:end)*y_train(2:end) )/ y_train(1);
alpha* y_train  %Confirm that it is ~zero

classified = false; %Force while loop to be true
  
while ~classified
  %calculate w
  for i = 1:N_train
    temp(i,:) = alpha(i).*y_train(i).*x_train(i,:);
  end
  w = sum(temp)
  
  %Measure classification accuracy
  guesses = tanh(w * x_test' + b);
  results = guesses == y_test';
  disp(sum(results))
%  pause
  if sum(results) == N_test
    classified = true;
    
  end
  

  %calculate KKT conditions
  for i = 1:N_train
    KKT(i) = alpha(i) * (y_train(i) * (w * x_train(i,:)' + b) - 1);
  end
%  
%  disp('KKT Cond.');
%  fprintf('%0.2f\n', KKT);
%  

  %Now pick x1 with the highest KKT condition
  [KKT_max, i1] = max(KKT);
  x1 = x_train(i1,:);
  
%  %re-sort x1 to correct position
%  tempx = x(1,:);
%  x(1,:) = x(i1,:);
%  x(i1,:) = tempx;
  
  %simple kernel
  %Kx = x_train * x_train';
  
  %now select the 2nd data point
  for i = 1:N_train
    for j = 1:N_train
      tempE(i,j) = alpha(j)*y_train(j)*K(x_train(i,:),x_train(j,:)); 
    end
  end
  
  %calculate hypothesis
  h = sum(tempE')+b;
  %calculate E
  E = h-y_train';
  

  %eval |Ei-Ej| and find the x location of the max one. I think it is okay to do
  %1:n for E because E1-E1 = 0 and shouldn't be a max.
  [val i2] = max(abs(E(i1)-E(:)));
  x2 = x_train(i2,:);
  
%  %re-sort x2 to the correct position
%  tempx = x(2,:);
%  x(2,:) = x(i2,:);
%  x(i2,:) = tempx;
%  

  % take one w/ highest abs value
  %2nd point is X3

  k = K(x_train(i1,:),x_train(i1,:)) + K(x_train(i2,:),x_train(i2,:)) -...
      2* K(x_train(i1,:),x_train(i2,:));

  alpha_new = alpha;

  alpha_new(i2) = alpha(i2) + (y_train(i2) * E(i2) / k);
  alpha_new(i1) = alpha(i1) + (y_train(i1) * y_train(i2) * (alpha(i2) - alpha_new(i2)));

  disp('alpha condition:');
%  disp(alpha_new * y );% Confirm that the sum to 0 condition still holds after update.
%  pause
  alpha = alpha_new; %Switch back to alpha for next loop.
  
  %Step 7
  for i = 1:N_train
    if alpha(i) < epsilon
      alpha(i) = 0;
    end
  end
  
  ind = alpha > 0;   
  b = mean( (1/y_train(ind))' + x_train(ind,:)*w');
  
  %Remaining x's are "support vectors"
  x_train = x_train(ind,:);
  y_train = y_train(ind); 
  [N_train,~] = size(y_train);

end

disp("Classified!");
