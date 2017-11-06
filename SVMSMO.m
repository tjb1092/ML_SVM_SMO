%#####################################################################
% Group : Shahapurkar_Vudatha_Poliseti_Bailey_ML_HW5
% Students names : Aparna Shahapurkar, Akhila Vudatha , 
% Tony Bailey, Devi Sai Priyanka Polisetti
% M# : M12505807 , M12508856, M05727970, M12510921  
%##################################################################### 
%#####################################################################
%##                        Machine Learning HW5                     ##
%##                Support Vector Machine SMO Algorithm             ##
%#####################################################################
%#####################################################################

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

% Randomly pick out ~50% of the data for training.
randVar = rand(length(iris),1);
Index = randVar >= 0.5; 

%Index the data
iris_train = iris(Index,:);
iris_test = iris(~Index,:);

%Create training and test arrays
x_train = iris_train(:,1:4);
y_train = iris_train(:,5);
x_test = iris_test(:,1:4);
y_test = iris_test(:,5);

%Store sizes of each set of data.
[N_train,~] = size(iris_train);
[N_test,~] = size(iris_test);

%Intialize params:
b = 0;
epsilon = 1e-3;

%Alpha is constrained by condition that their sum = 0
alpha = rand(1,length(y_train));
disp('Random alpha assignment')
disp(alpha * y_train) % not equal 0
alpha(1) = -(alpha(2:end)*y_train(2:end) )/ y_train(1);
disp('Enforce Zero Sum condition')
disp(alpha * y_train)  %Confirm that it is ~zero

classified = false; %Force while loop to be true
counter = 1;  
max_num_runs = 1000; %So it doesn't hang forever.

while ~classified
    
    %Step 2
    %calculate w
    clear temp; %cleared to accommodate for changing training size.
    for i = 1:N_train
    temp(i,:) = alpha(i).*y_train(i).*x_train(i,:);
    end
    w = sum(temp);
    
    %Effectively Step 10,
    %Measure training classification accuracy based on w changing.
    guesses = sign(w * x_train' + b); %sigmoid func. to classify
    result = sum(guesses == y_train');
    results(counter) = sum(guesses == y_train')/length(y_train);
    fprintf('Iteration %i Accuracy:\n%i / %i\n',counter, result,length(y_train));
    
    %if we perfectly classify the training set, we are done!
    if sum(result) == N_train || counter == max_num_runs 
        classified = true;
        break
    end
    
    %Step 3
    %calculate KKT conditions
    clear KKT; %cleared to accommodate for changing training size.
    for i = 1:N_train
    KKT(i) = alpha(i) * (y_train(i) * (w * x_train(i,:)' + b) - 1);
    end
    
    %Step 4, pick x1 & x2.
    
    %Pick x1 with the highest KKT condition
    [KKT_max, i1] = max(KKT);
    x1 = x_train(i1,:);

    %now select the 2nd data point
    clear tempE; %cleared to accommodate for changing training size.
    for i = 1:N_train
    for j = 1:N_train
      tempE(i,j) = alpha(j)*y_train(j)*K(x_train(i,:),x_train(j,:)); 
    end
    end

    %calculate hypothesis
    h = sum(tempE')+b;
    %calculate E vector
    E = h-y_train';

    %Find the max of |Ei-Ej| and find the x location of the max term. 
    [val, i2] = max(abs(E(i1)-E(:)));
    x2 = x_train(i2,:);

    %Update Alpha1 and Alpha2 (steps 5 & 6)
    k = K(x_train(i1,:),x_train(i1,:)) + K(x_train(i2,:),x_train(i2,:)) -...
      2* K(x_train(i1,:),x_train(i2,:));

    alpha_new = alpha;

    alpha_new(i2) = alpha(i2) + (y_train(i2) * (E(i1)- E(i2))) / k;
    alpha_new(i1) = alpha(i1) + (y_train(i1) * y_train(i2) * (alpha(i2) - alpha_new(i2)));

    alpha = alpha_new; %Switch back to alpha for next loop.

    %Step 7. Remove alphas < 0
    for i = 1:N_train
        if alpha(i) < epsilon
          alpha(i) = 0;
        end
    end
    
    %Step 8. Calculate b w/ support vectors
    ind = alpha > 0;   
    b = mean( (1/y_train(ind))' + x_train(ind,:)*w');

    %Remove unneeded points. Remaining points are "support vectors".
    alpha = alpha(ind);
    x_train = x_train(ind,:);
    y_train = y_train(ind); 
    [N_train,~] = size(y_train);
    counter = counter + 1;
end

%Plot Training Accuracy over iteration.
plot(1:counter, results*100);
xlabel('Iteration #');
ylabel('Training Accuracy (%)');
title('Training Accuracy Over Iteration');
grid on;

%Display final parameters
fprintf('Final w:\n');
disp(w);
fprintf('Final b: %0.4f\n', b);
fprintf('Number of support Vectors : %i\n',N_train)

%Measure classification accuracy on test set
guesses = sign(w * x_test' + b);
testresults = sum(guesses == y_test');
fprintf('Test Accuracy:\n %0.2f\n',100*sum(testresults)/length(y_test));

%Plotting the iris data and Support Vectors
Xsupport=x_train;Ysupport=y_train;
figure
hold on
subplot(4,4,1)
plot(1,1,'r*',1,1,'b*',1,1,'ro',1,1,'bo');
legend('Test Positive class','Test Negative class','Positive Support Vectors', 'Negative Support Vectors')

PlotSVM(x_test,y_test,2,Xsupport,Ysupport,1,2);
PlotSVM(x_test,y_test,3,Xsupport,Ysupport,1,3);
PlotSVM(x_test,y_test,4,Xsupport,Ysupport,1,4);

PlotSVM(x_test,y_test,5,Xsupport,Ysupport,2,1);
PlotSVM(x_test,y_test,7,Xsupport,Ysupport,2,3);
PlotSVM(x_test,y_test,8,Xsupport,Ysupport,2,4);

PlotSVM(x_test,y_test,9,Xsupport,Ysupport,3,1);
PlotSVM(x_test,y_test,10,Xsupport,Ysupport,3,2);
PlotSVM(x_test,y_test,12,Xsupport,Ysupport,3,4);

PlotSVM(x_test,y_test,13,Xsupport,Ysupport,4,1);
PlotSVM(x_test,y_test,14,Xsupport,Ysupport,4,2);
PlotSVM(x_test,y_test,15,Xsupport,Ysupport,4,3);
hold off


