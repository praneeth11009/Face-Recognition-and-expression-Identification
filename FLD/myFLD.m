function Wfld = myFLD(Xr,classes)
%MYFLD 
%args: Xr, 
%classes : consecutive elem classes, # elem in each class 

%% Sb computation
xi=zeros(size(Xr));
c=size(classes,2);
for i=1:c
    st=sum(classes(1:i-1))+1;
    ed=sum(classes(1:i));
    xi1=mean( Xr(:,st:ed), 2); % mean of ith class
    xi(:,st:ed)=repelem(xi1, 1,classes(i));
end
Sb=xi*xi';
%% Sw computation
Irc=Xr-xi; 
Sw=Irc*Irc';
%%disp('determinant ', det(Sw));
%% Wfld
[Vf, ef]=eig(Sb, Sw);
[~,kmax] = maxk(diag(ef),c);
Wfld = Vf(:,kmax(1:c-1));
end

