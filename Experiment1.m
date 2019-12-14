%% Load Datasets & Verify Dataset
clc
clear
clear all

PLOT = 0;
PLOT1 = 0;
LOAD = 1;

if LOAD == 1
    load X1600;
    load Te28;
    load Lte28;
end

len = length(Lte28);
% Plot dataset to verifiy images
if PLOT == 1
    for i = 1:10
        figure(1)
        title('Numbers 0-9')
        subplot(2,5,i) % Plots 1st 10 {0,1,2,3,4,5,6,7,8,9} classes(Label)
        num = X1600(:,1600*(i-1)+1);
        num = reshape(num,28,28);
        imshow(num);
        title(i-1);
        subplot(2,5,10)
        num9 = X1600(:,14400+1);
        num9 = reshape(num9,28,28);
        imshow(num9);
        title('9');
        
    end
end

%% PCA - principal component analysis
num_feature = 784; % 28x28
nj = 1600;         % # of samples
ni = 10 ;          % # of classes
q = 29;            % # of rank approximation
mu = zeros(num_feature, ni);
U = zeros(num_feature, q*ni);

for i = 1:ni
    % STEP-1 Calculate mean-j & covariance-j matrices
    Ai = X1600(:,(i-1)*nj+1:i*nj);
    mu_j = mean(Ai,2);
    Ah = Ai - mu_j*ones(1,nj);
    Cov = (Ah*Ah');
    
    % STEP-2 Calculate the q eigenvectors of covariance-j matrices
    [Uq,~] = eigs(Cov,q);
    U(:,(i-1)*q+1:i*q) = Uq;
    mu(:,i) = mu_j;
end

%% Testing
t0 = cputime;
Predicted_Label = zeros(len,1);

for j = 1:len
    At = Te28(:,j);
    e = zeros(1,ni);
    for i = 1:ni
        % STEP-3 Calculate principal components 
        Cov2 = At - mu(:,i);
        fj = U(:,(i-1)*q+1:i*q).'*Cov2;
        
        % STEP-4 Calculate PCA approximation  
        Aj = U(:,(i-1)*q+1:i*q)*fj + mu(:,i);
        
        % STEP-5 Calculate error
        e(i) = norm(At-Aj);
    end
    % STEP-6  Identify the target class Aj for data point x. 
    [~,MinIdx] = min(e);
    Predicted_Label(j) = MinIdx - 1;
end

cpt = cputime - 1;
E = (Lte28 ~= Predicted_Label);

if PLOT1 == 1
    figure(3)
    bar(e);
    set(gca,'xticklabel',{'0','1','2','3','4','5','6','7','8','9'});
    xlabel('Numbers 0-9');
    ylabel('% Error');
    title('Mislabeled Samples');
end

disp('Number of errors')
n_mis = sum(E)
disp(' % error:')
Enorm = sum(E)/10000*100
disp('CPU time(s):')
cpt
