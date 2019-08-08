clc;
tic;
%% extrapolation method

n = [38,40];
res = [192,168];
path = '../CroppedYale';

train_imgs = 0;
Ir = zeros(n(1)*n(2),res(1)*res(2));
classes=zeros(1,n(1));
train_labels=zeros(1,n(1)*n(2));
%
test_imgs = 0;
testIr = zeros(n(1)*n(2),res(1)*res(2));
test_labels=zeros(1,n(1)*n(2));
for i = 1:n(1)
   if i < 10
       filepath = strcat(path,'/yaleB0',num2str(i,'%d'),'/');
   elseif i < 14 
       filepath = strcat(path,'/yaleB',num2str(i,'%d'),'/');
   else 
       filepath = strcat(path,'/yaleB',num2str(i+1,'%d'),'/');
   end
   files = dir(fullfile(filepath,'*pgm'));
   sz = size(files);
   %
   for j = 1:sz
       name=files(j).name;
       lat=str2double(name(14:16));
       lgt=str2double(name(19:20));
       %   
       if max(lat, lgt)<=10 || (max(lat, lgt)<=130 &&max(lat, lgt)>=110) %max(lat, lgt)<=85 &&max(lat, lgt)>=75
           train_imgs=train_imgs+1;
           classes(i)=classes(i)+1;
           train_labels(train_imgs)=i;
           img=imread(strcat(filepath,files(j).name));
           img=im2double(img);
           Ir(train_imgs,:) = reshape(img,[res(1)*res(2),1]);
       elseif (max(lat, lgt)<110 && max(lat, lgt)> 10)
           test_imgs=test_imgs+1;
           test_labels(test_imgs)=i;
           img=imread(strcat(filepath,files(j).name));
           img=im2double(img);
           testIr(test_imgs,:) = reshape(img,[res(1)*res(2),1]);
       end
   end
end
Ir=Ir(1:train_imgs,:);
testIr=testIr(1:test_imgs,:);
train_labels=train_labels(1,1:train_imgs);
test_labels=test_labels(1,1:test_imgs);
disp('finished input');
toc;
%%
x = mean(Ir);
X = (Ir - x)';
[U,S,~] = svd(X,'econ');
c=n(1); % classes
dim=train_imgs-c;
[maxval,kmax] = maxk(diag(S),dim+3);
Wpca = U(:,kmax(1:dim+3)); %id*dim
Xr=Wpca'*X; %%dim reduced img in cols dim*ni
toc;
disp('finished pca');
%%
tic;
Wfld=myFLD(Xr,classes);
Wopt=Wpca*Wfld;
disp('fisnished computing transformation matrix');
toc;
%% testing
testX = (testIr - x)';
ec=Wfld'*Xr;
testEc = Wopt'*testX;

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
