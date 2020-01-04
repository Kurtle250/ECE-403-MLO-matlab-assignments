clear all, clc;

% Load Datasets into workspace
load('D_build_tr'); 
load('D_build_te');

%% Format dataset into X and Y for training and testing
Xtr = D_build_tr(1:8,:);
Ytr = D_build_tr(9:10,:);

Xte = D_build_te(1:8,:);
Yte = D_build_te(9:10,:);

%% Training the model
% Paramaters:
Lamda = 0.001;

% Make matrix with transposed Xtr and add ones Column 
Xhat = [Xtr' ones(640,1)];
% make matrix with transposed Ytr 
Yhat = [Ytr'];
% wb is the HAT matrix multiplied by Yhat
wb = ((Xhat'*Xhat+Lamda*eye())^-1)*Xhat'*Yhat;
% h and c indicates heating and cooling load elements
wh = wb(1:8,1);
wc = wb(1:8,2);
bh = wb(9,1);
bc = wb(9,2);

%% Testing the model with given test files.
% h and c indicates heating and cooling load elements
Yh = wh'*Xte +bh;
Yc = wc'*Xte +bc;
Y = [Yh;Yc];

DeltaY = Yte - Y;
% Calculate the overall relative prediction error
Eps = norm(DeltaY,'fro')/norm(Yte,'fro')

%% Plotting
figure(1)
subplot(3,1,1)
plot(Yte(1,:),'r-');
hold on;
plot(Yh,'b-');
xlabel('Samples');
ylabel('Value');
title('Heating');
legend('True','Prediction');


subplot(3,1,2)
plot(Yte(2,:),'r-');
hold on;
plot(Yc,'-b');
xlabel('Samples');
ylabel('value');
title('Cooling');
legend('True','Prediction');

subplot(3,1,3)
plot(Yh,'r-');
hold on;
plot(Yc,'-b');
xlabel('Samples');
ylabel('Value');
title('Heating Vs Cooling');
legend('Heating','Cooling');
