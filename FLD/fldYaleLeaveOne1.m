clc;
tic;
%%
n = [15,11]; % 2 classes, max 15 per class

res = [231,195];
path = '../centered/';
T = zeros(n(1)*n(2),res(1)*res(2));
%
train_imgs = n(2)*n(1)-1;
%Ir = zeros(train_imgs,res(1)*res(2));
classes1=ones(1,n(1))*n(2);
train_labels1=repelem(1:n(1), 1, n(2));
%
test_imgs = 1;
%testIr = zeros(test_imgs,res(1)*res(2));
test_labels=ones(1,1);

%% input 
files =dir(fullfile(path,'*pgm')); %dir(fullfile(path,'*.glasses.pgm')); %
sz = size(files);
for i=1:sz
    img=im2double(imread(strcat(path,files(i).name))); 
    T(i,:) = reshape(img,[res(1)*res(2),1]);
end
%% testing
ctr=0;
for i=1:n(1)*n(2)
    
    Ir=T;
    Ir(i,:)=[];
    testIr=T(i,:);
    train_labels=train_labels1;
    train_labels(:,i)=[];
    test_labels(1)=train_labels1(i);
    classes=classes1;
    classes(floor((i-1)/n(2))+1)=classes(floor((i-1)/n(2))+1)-1;
    x = mean(Ir);
    X = (Ir - x)';
    [U,S,~] = svd(X,'econ');
    c=n(1); % classes
    dim=train_imgs-c;
    [maxval,kmax] = maxk(diag(S),dim+3);
    Wpca = U(:,kmax(1:dim)); %id*dim
    Xr=Wpca'*X; %%dim reduced img in cols dim*ni

    Wfld=myFLD(Xr,classes);
    Wopt=Wpca*Wfld;
  
    testX = (testIr - x)';
    ec=Wfld'*Xr;
    testEc = Wopt'*testX;

    for j = 1:test_imgs
        diff = ec - testEc(:,j);
        sqdiff = diag(diff'*diff);
        [M,In] = min(sqdiff);

        if  train_labels(In) == test_labels(j) %%train_labels(In)
            ctr = ctr + 1;
        end
    end
    disp(['iter ',num2str(i), ' ctr ', num2str(ctr)]);
end
rate=ctr/(n(1)*n(2));
disp(['rate ',num2str(rate)]);