clc;
tic;
%%
n = [2,15]; % 2 classes, max 15 per class

res = [231,195];
path = '../centered/';
%
train_imgs = 0;
Ir = zeros(n(1)*n(2),res(1)*res(2));
classes=zeros(1,n(1));
train_labels=zeros(1,n(1)*n(2));
%
test_imgs = 0;
testIr = zeros(n(1)*n(2),res(1)*res(2));
test_labels=zeros(1,n(1)*n(2));

%% input glases
files =dir(fullfile(path,'*.happy.pgm')); %dir(fullfile(path,'*.glasses.pgm')); %
sz = size(files);
for i=1:sz
    img=im2double(imread(strcat(path,files(i).name))); 
    %%figure;
    %%imshow(img);
    if i <= train 
        train_imgs=train_imgs+1;
        classes(1)=classes(1)+1;
        train_labels(train_imgs)=1;
        Ir(train_imgs,:) = reshape(img,[res(1)*res(2),1]);
    else
        test_imgs=test_imgs+1;
        test_labels(test_imgs)=1;
        testIr(test_imgs,:) = reshape(img,[res(1)*res(2),1]); 
    end
end
%% input no glases
files = dir(fullfile(path,'*.sad.pgm'));%dir(fullfile(path,'*.noglasses.pgm'));  %
sz = size(files);
for i=1:sz
    img=im2double(imread(strcat(path,files(i).name))); 
    if i<=train 
        train_imgs=train_imgs+1;
        classes(2)=classes(2)+1;
        train_labels(train_imgs)=2;
        Ir(train_imgs,:) = reshape(img,[res(1)*res(2),1]);
    else
        test_imgs=test_imgs+1;
        test_labels(test_imgs)=2;
        testIr(test_imgs,:) = reshape(img,[res(1)*res(2),1]); 
    end
end
Ir=Ir(1:train_imgs,:);
testIr=testIr(1:test_imgs,:);
train_labels=train_labels(1,1:train_imgs);
test_labels=test_labels(1,1:test_imgs);
toc;
disp('finished input');
%%
tic;
x = mean(Ir);
X = (Ir - x)';
testX = (testIr - x)';
[U,S,~] = svd(X,'econ');
dim=train_imgs;
[~,kmax] = maxk(diag(S),dim+3);
Wpca = U(:,kmax(1:dim)); %id*dim
ec=Wpca'*X; %%dim reduced img in cols dim*ni
toc;
disp('finished pca');

%% testing

testEc = Wpca'*testX;
ctr = 0;
for i = 1:test_imgs
    diff = ec - testEc(:,i);
    sqdiff = diag(diff'*diff);
    [M,In] = min(sqdiff);
    
    if  train_labels(In) == test_labels(i) %%train_labels(In)
        ctr = ctr + 1;
    end
end
rate = ctr/test_imgs;
disp(['rate ',num2str(rate)]);
toc;
