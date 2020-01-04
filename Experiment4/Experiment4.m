%% Step 1
% load datasets into matlab
load D_bc_te
load D_bc_tr

%% Step 2
% format testing and training datasets and normalization dataset

NumFeatures = 30;
K = 75;

Xtrain = zeros(30,480);
for i = 1:30
 xi = D_bc_tr(i,:);
 mi = mean(xi);
 vi = sqrt(var(xi));
 Xtrain(i,:) = (xi - mi)/vi;
end
Xtest = zeros(30,89);
for i = 1:30
 xi = D_bc_te(i,:);
 mi = mean(xi);
 vi = sqrt(var(xi));
 Xtest(i,:) = (xi - mi)/vi;
end

ytrain = D_bc_tr(31,:);
ytest = D_bc_te(31,:);

Dtrain = [Xtrain;ytrain];
Dtest = [Xtest;ytest];

%% Step 3
% Objective function and gradient

%% Step 4
% Minimize f(wh)
eps = 10^-9;
w = zeros(NumFeatures+1,1);
f = zeros(1,K);
k = 1;

% step 2: find d
gk = g_wcdc(w,Dtrain);
dk = -gk;

% step 3: find alpha
alpha = bt_lsearch(w,dk,'f_wcdc','g_wcdc',Dtrain);

%% Training
while ((norm(alpha*dk) >= eps) && (k<+K))

    w = w + alpha*dk;

    % step 2: find d
    gk = g_wcdc(w,Dtrain);
    dk = -gk;

    % step 3: find alpha
    alpha = bt_lsearch(w,dk,'f_wcdc','g_wcdc',Dtrain);

    %check if f is decreasing.
    f(k) = f_wcdc(w,Dtrain);

    k = k + 1;
end


%% Testing
Dt = [Xtest; ones(1,89)];
Result = w'*Dt;
TestLabel = zeros(1,length(Result));
FalsePos = 0;
FalseNeg = 0;

if PLOT == 1
    AfterTr = Result;
    Z = zeros(1,length(AfterTr));

    figure, plot(AfterTr,'b*');
    hold on
    plot(Z,'r-');
    xlabel('Samples');
    ylabel('Value');
end

for ii = 1:length(Result)
    if Result(ii) < 0
        TestLabel(ii) = -1;
        if ytest(ii) > 0
            FalseNeg = FalseNeg + 1;
        end
    else
        TestLabel(ii) = 1;
        if ytest(ii) > 0
            FalsePos = FalsePos + 1;
        end
    end
end


disp('No. of false +ve ')
FalsePos
disp('No. of False -ve')
FalseNeg
disp('Misclassification ratio')
(FalsePos+FalseNeg)/100
