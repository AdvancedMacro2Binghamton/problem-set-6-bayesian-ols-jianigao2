close all;
clear;

%%%%%%%%READ DATA AND DO OLS REGRESSION%%%%%
load matlab.mat;
Y=log(wage);
X=[ones(size(Y)),educ,exper,smsa,black,south];
[beta,Sigma,resid,CovB,logL]=mvregress(X,Y,'varformat','full');

%%%%%%%MH APPROACH%%%%%%%%%%%%%%%%%%%%%%%%%%

beta=repmat([beta',Sigma],[1 1 2]);
acc=zeros(2,1);
cov=transpose(diag(CovB));

%%%%%%%%flat prior%%%%%%%%%%%%%%%%%%%%%%%%%%
for j=2:5000
    l=L1(Y,X,beta(j-1,1:end-1,1),beta(j-1,end,1));
    betaNew=beta(1,:,1)+mvnrnd(zeros(size(beta(1,:,1))),cov);
    lNew=L1(Y,X,betaNew(1:end-1),betaNew(end));
    d=lNew-l;
    c=rand;
    
    if(d>=c)
        beta(j,:,1)=betaNew;
        acc(1,1)=acc(1,1)+1;%%%accept
    else
        beta(j,:,1)=beta(j-1,:,1);%%%%%reject
    end
end

%%%%%%%%%%%using prior for education%%%%%%%%%
for i=2:5000
     l=L2(Y,X,beta(i-1,1:end-1,2),beta(i-1,end,2),cov);
    betaNew=beta(i-1,:,2)+mvnrnd(zeros(size(beta(1,:,2))),cov);
    lNew=L2(Y,X,betaNew(1:end-1),betaNew(end),cov);
    d=lNew-l;
    c=rand;
    if(d>=c)
        beta(i,:,2)=betaNew;
        acc(2,1)=acc(2,1)+1;%%acc
    else
        beta(i,:,2)=beta(i-1,:,2);%%%rej
    end
end


%%%%%%%%%%%%plot%%%%%%%%%%%%%%%%%%%
str={'Constant','education','black','smsa','south','experience','Sigma'};
title={'2(a) flat prior','2(b) education'};
for i=1:2
figure('Name',title{i})
  for j=1:7
   subplot(4,2,j);
   histogram(beta(:,j,i));
   title(str{j});
  end
end
