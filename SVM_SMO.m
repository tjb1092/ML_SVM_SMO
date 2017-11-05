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
    s = -1;   
  end
  y(i) = s;
end
iris = [meas y];
iris = iris(randperm(size(iris,1)),:); % Shuffle data around
[N,~] = size(iris);

x = iris(:,1:4);
y = iris(:,5);

%Intialize params:
b = 0;
%lagrange subject to that condition. I guess all zeros works, so I'm going to do
%it for now.
alpha = zeros(N,1);
w = zeros(N,4);

classified = false;

while ~classified
  for i = 1:N
    w(i,:) = alpha(i) * y(i) * x(i,:);    
  end
  
  
end





for runs = 1:10

  % Randomly pick out ~50% of the data for training.
  randVar = rand(length(iris),1);
  Index = randVar >= 0.5; 

  %Index the data
  iris_train = iris(Index,:);
  iris_test = iris(~Index,:);

  for bins = 5:5:20
    
    is_done_training = false;
    while(~is_done_training)
    
      [m_train, ~] = size(iris_train);
      [m_pos, ~] = size(iris_train(iris_train(:,5) == 1,:));
      [m_neg, ~] = size(iris_train(iris_train(:,5) ~= 1,:));
          
      p_pos = m_pos/m_train;
      p_neg = m_neg/m_train;
      
      Entropy_a = nansum(-p_pos .* log2(p_pos) - p_neg .* log2(p_neg));
      
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
        
        for k = 1:length(children)
          temp = children{k};
          [m_child(k), ~] = size(temp);
          [m_pos, ~] = size(temp(temp(:,5) == 1,:));
          [m_neg, ~] = size(temp(temp(:,5) ~= 1,:));
          
          p_pos(k) = m_pos/m_child(k);
          p_neg(k) = m_neg/m_child(k);
        end
        T(i).n = children;
        T(i).thresholds = thresholds;
        
        Entropy(i) = nansum(((m_child)./m_train) .*( -p_pos .* log2(p_pos) - p_neg .* log2(p_neg)));        
      end
      
      InformationGain = Entropy_a - Entropy;
      %Pick Best Attribute
      [m_IG, attribute] = max(InformationGain);  
      
      
      T = T(attribute);
      if(m_IG == Entropy_a)
        %Perfectly Classified Training Set
        is_done_training = true;
        for m = 1:length(T.n)
          if(~isempty(T.n{m}))
            labels(m) =  T.n{m}(1,5); %Assume all labels are the same b/c perfectly classified. pick out one.        
          end
        end
        
      else
        %Tree needs more work
        %From testing, each set can be linearly classified by looking @ 
        %petal length regardless of bin size.
        disp('HELP MAKE MY LIFE HELL...');
      end
    end
    
    %Get test performance metrics
    runningTotal = 0;
    for t = 1:(length(T.thresholds)+1)
      if t == 1
        n = iris_test((iris_test(:, attribute) < T.thresholds(t)),:);
        
      elseif t == (length(T.thresholds)+1)
        n = iris_test((iris_test(:, attribute) >= T.thresholds(t-1)),:);
      else
        n = iris_test(((iris_test(:, attribute) < T.thresholds(t)) & (iris_test(:, attribute) >= T.thresholds(t-1))),:);
      end
      [correct, ~] = size(n(n(:,5) == labels(t),5));
      runningTotal = runningTotal + correct;
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
title('ID3 Classifier of Iris Dataset');



