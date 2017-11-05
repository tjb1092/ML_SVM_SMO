clc;
clear all;
close all;

load fisheriris.mat

y = zeros(length(species),1);
for i = 1:length(species)
  %We are clumping versicolor and virginica together.
  if strcmp(species{i,1},'setosa')
    s = 1;
  else 
    s = 0;   
  end
  y(i) = s;
end
iris = [meas y];

for runs = 1:10

  % Randomly pick out ~50% of the data for training.
  randVar = rand(length(iris),1);
  Index = randVar >= 0.5; 

  %Index the data
  iris_train = iris(Index,:);
  iris_test = iris(~Index,:);

  for bins = 5:5:20

    [m_train, ~] = size(iris_train);
    [m_pos, ~] = size(iris_train(iris_train(:,5) == 1,:));
    [m_neg, ~] = size(iris_train(iris_train(:,5) ~= 1,:));
        
    p_yes = m_pos/m_train;
    p_no = m_neg/m_train;
         
    %Search for best attribute.
    for i = 1:4
      [m_train, ~] = size(iris_train);
      [n, x] = hist(iris_train(:,i),bins);
      thresholds = (x(1:end-1) + x(2:end)) / 2;  
      %For each attribute, get the arrays of values associated with each bin.
      for j = 1:(length(thresholds)+1)
        if j == 1
          children{j} = iris_train((iris_train(:, i) < thresholds(j)),:);
        elseif j == (length(thresholds)+1)
          children{j} = iris_train((iris_train(:, i) >= thresholds(j-1)),:);
        else
          children{j} = iris_train(((iris_train(:, i) < thresholds(j)) & (iris_train(:, i) >= thresholds(j-1))),:);
        end
      end
      
      %Build up probability table
      for k = 1:length(children)
        temp = children{k};
        [m_child(k), ~] = size(temp);
        [m_pos, ~] = size(temp(temp(:,5) == 1,:));
        [m_neg, ~] = size(temp(temp(:,5) ~= 1,:));
        
        p_pos(k) = m_pos/m_child(k);
        p_neg(k) = m_neg/m_child(k);
      end
      
      
      T(i).child = children;
      T(i).thresholds = thresholds;
      T(i).p_y_Bn = p_pos;
      T(i).p_n_Bn = p_neg;
    end
      
    
    
    %Get test performance metrics
    runningTotal = 0;
    for s = 1:length(iris_test)
      p_1 = p_yes;
      p_0 = p_no;
      for attribute = 1:4
        %Find the bin for each observation
        for t = 1:(length(T(attribute).thresholds)+1)
          if t == 1            
            is_right = iris_test(s, attribute) < T(attribute).thresholds(t);
            
          elseif t == (length(T(attribute).thresholds)+1)          
            is_right = iris_test(s, attribute) >= T(attribute).thresholds(t-1);
            
          else         
            is_right = (iris_test(s, attribute) < T(attribute).thresholds(t)) & (iris_test(s, attribute) >= T(attribute).thresholds(t-1));
          
          end
          
          if(is_right)
            bin = t;
            break;
          end
          
        end 
        %Update marginal with likelihoood of the bin.
        p_1 = p_1 * (p_yes * T(attribute).p_y_Bn(bin));
        p_0 = p_0 * (p_no * T(attribute).p_n_Bn(bin));
      end
      
      %argmax of two options.
      if(p_1 >= p_0)
        guess = 1;
      else
        guess = 0;
      end
      
      %If correct, increment total "good" picks.
      if(guess == iris_test(s,5))
        runningTotal = runningTotal + 1;
      end

    end
   
    accuracy(runs,(bins/5)) = runningTotal/length(iris_test);
  end
  
end

%Plot Results
bins = 5:5:20;
max_accuracy = max(accuracy);
min_accuracy = min(accuracy);
mean_accuracy = mean(accuracy);
figure();
plot(bins,min_accuracy,'b-*',bins,mean_accuracy,'r-*',bins,max_accuracy,'g-*');
axis([4,21]);
grid on;
xlabel('# of bins');
ylabel('Accuracy = (# classified correctly)/(total # of samples)');
title('Naive Bayes Classifier of Iris Dataset');
