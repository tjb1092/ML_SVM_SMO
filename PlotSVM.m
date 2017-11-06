function PlotSVM(x, y, p, xSupport, ySupport, f1, f2)
%Plots a subplot of iris data set
subplot(4,4,p);
hold on
labels = {'Sepal Length', 'Sepal Width', 'Petal Length', 'Petal Width'};
plot(x(y==1,f1),x(y==1,f2),'*r')
plot(x(y==-1,f1),x(y==-1,f2),'*b')
plot(xSupport(ySupport==1,f1),xSupport(ySupport==1,f2),'ro')
plot(xSupport(ySupport==-1,f1),xSupport(ySupport==-1,f2),'bo')
title(strcat(labels{f2}, ' vs.  ', labels{f1}))
xlabel(labels{f1}),ylabel(labels{f2})
hold off;
end

