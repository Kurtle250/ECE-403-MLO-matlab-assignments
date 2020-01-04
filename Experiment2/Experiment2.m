clear all; close all; clc;

LOAD = 1;

if LOAD == 1
    load('D_iris_tr');
    load('D_iris_te');
end

%% Training and Testing datasets: 
% Data sets xtr1, xtr2, and xtr3 are train data from Setosa, Versicolor, and Virginica,
% respectively, each contains 40 samples. Data sets xte1, xte2, and xte3 are test data from
% Setosa, Versicolor, and Virginica, respectively, each contains 10 samples
%D_iris_tr(4,:) = 1;
%D_iris_te(4,:) = 1;
%D_iris_tr(4,:) = 1;
%D_iris_te(4,:) = 1;

xtr1 = D_iris_tr(:,(1:40));
xtr2 = D_iris_tr(:,(41:80));
xtr3 = D_iris_tr(:,(81:end));

xte1 = D_iris_te(:,(1:10));
xte2 = D_iris_te(:,(11:20));
xte3 = D_iris_te(:,(21:end));

xte = [xte1 xte2 xte3];
% Creating 3 binary classifications to produce linear models

X = [ xtr1 xtr2 xtr3 ];

Y = [ones(40,1);-ones(80,1)];

for k = 1:3   
    switch k
        case 1
            P1 = X(:,1:40);
            N1 = X(:,41:end);
            X1 = [P1 N1];
            Xh1 = [X1 ;ones(1,120)];
        case 2
            P2 = X(:,41:80);
            N2 = [X(:,(1:40)) X(:,(81:end))];
            X2 = [P2 N2];
            Xh2 = [X2;ones(1,120)];
        case 3
            P3 = X(:,81:end);
            N3 = X(:,(1:80));
            X3 = [P3 N3];
            Xh3 = [X3;ones(1,120)];
    end
end

% Training
xy1 = Xh1*Y;
wh1 = (Xh1*Xh1')\xy1;
xy2 = Xh2*Y;
wh2 = (Xh2*Xh2')\xy2;
xy3 = Xh3*Y;
wh3 = (Xh3*Xh3')\xy3;

wh = [wh1';wh2';wh3'];

ws = wh(:,1:4);

bs = wh(:,5);

% Testing
mis_class = 0; 
TestM = [ones(1,10) 2*ones(1,10) 3*ones(1,10)];

e1 = [1 0 0]'; e2 = [0 1 0]'; e3 = [0 0 1]';
E = zeros(3,30);
for i = 1:30
    Ytest = ws*xte(:,i) + bs;
    [~, Idx] = max(Ytest);
    E(Idx,i) = 1;
    if Idx ~= TestM(i)
        mis_class = mis_class + 1;
    end
end
figure
    subplot(3,1,1)
    stem(E(1,:),'LineStyle','-.',...
     'MarkerFaceColor','red',...
     'MarkerEdgeColor','blue')
    title('setosa'); 
    subplot(3,1,2)
    stem(E(2,:),'LineStyle','-.',...
     'MarkerFaceColor','black',...
     'MarkerEdgeColor','green')
    title('versicolor'); 
    subplot(3,1,3) 
    stem(E(3,:),'LineStyle','-.',...
     'MarkerFaceColor','red',...
     'MarkerEdgeColor','yellow')
     title('virginica');

disp('testing setosa:')
E1 = E(:,1:10); c1 = sum(E1')'
[E1 c1];
disp('testing versicolor:')
E2 = E(:,11:20); c2 = sum(E2')'
[E2 c2];
disp('testing virginica:')
E3 = E(:,21:30); c3 = sum(E3')'
[E3 c3];
disp('Confusion matrix:')

C = [c1 c2 c3]

mis_class

figure()
set(gca,'fontsize',14,'fontname','times')
plot3(Ytest(1,1:10),Ytest(2,1:10),Ytest(3,1:10),'bo','linew',1.5)
hold on
plot3(Ytest(1,11:20),Ytest(2,11:20),Ytest(3,11:20),'rx','linew',1.5)
plot3(Ytest(1,21:30),Ytest(2,21:30),Ytest(3,21:30),'k.','linew',1.5)
hold off

    
    
    
    


