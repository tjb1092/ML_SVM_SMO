%diary on
X=[1,0;2,0;3,2];



y = [-1 1 -1];


alpha = rand(1, 3);


alpha(1:2) * y(1:2)';

alpha(3) = -(alpha(1:2)*y(1:2)') / y(3);

%simple kernel
K = X * X';

n = length(X(:,1));

% n = 3

for i = 1:n
	temp(i,:) = alpha(i)*y(i)*X(i,:);
end
w = sum(temp);
b=0;

%KKT Conditions	
for i = 1:n
	KKT(i) = alpha(i)*(y(i)*(w*X(i,:)'+b)-1);
end

%choose arbitrary value for b
b=1;

%Solve for KKT components
KKT1 = eval(KKT);

xx(1,:) = X(2,:);

%now select the 2nd data point
for i = 1:n
  for j = 1:n
    tempE(i,j) = alpha(j)*y(j)*K(i,j); 
  end
end

tempE

syms b
h = sum(tempE')+b;

E = h-y;

% recall y;
y

%eval |Ei-Ej|

E(1) - E(3)

E(2) - E(1)

E(2) - E(3)

% take one w/ highest abs value
%2nd point is X3

A = K(1,1) + K(3,3) - 2* K(1,3)

alpha_new = alpha;

alpha_new(3) = alpha_old(3) - y(3) * E(1) - E(3) / A;
alpha_new(1) = alpha_old(1) + y(3) * y(1) * (alpha_old(3) - alpha_new(3));

alpha_new * y'; %~0

%now loop starting from KKT cond.
%each time you update alpha, need to update decision surface.
%if it classifies, you good. else, continue

