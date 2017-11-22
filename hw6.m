close all;
clear;
clc;

%%%%%%%%READ DATA%%%%%
load matlab.mat;
Y=log(wage);
X=[ones(length(Y),1),educ,exper,smsa,black,south];
[n,k]=size(X);

%%%%%%%ESTIMATION%%%%
beta=inv(X'*X)*X'*Y;
residual=Y-X*beta;
sigma_sq=sum((residual-mean(residual)).^2)/(length(Y)-k);
covb=inv(X'*X)*sigma_sq;
varb=diag(covb);
varsig=2/(length(Y)-k)*sigma_sq^2;
var=[varb;varsig];
beta_mh=[beta;sigma_sq];
r=zeros(length(beta_mh),5000);

%%%%%%%part 1 flat prior%%%%%
l=normpdf(Y-X*beta,0,sqrt(sigma_sq));
l=sum(log(l));
acc=0;
for i= 2:5000
    betanew=beta_mh+mvnrnd(zeros(size(beta_mh)),diag(var))';
    lnew=normpdf(Y-X*betanew(1:6),0,sqrt(betanew(7)));
    lnew=sum(log(lnew));
    c=rand;
    if exp(lnew-l)>=c
        acc=acc+1;
        l=lnew;
        beta_mh=betanew;
    else
    end;
    r(:,i)=beta_mh;
end
str={'constant','educ','exper','smsa','black','south','sigma'};
figure
for j=1:7
    subplot(4,2,j)
    histogram(r(j,:))
    title(str{j})
end

%%%%using education prior%%%%
conf=[0.035,0.085];
a=norminv(0.975);
b=norminv(0.025);
edu=0.06;
s=(conf(2)-edu)/a;
beta_mh=[beta;sigma_sq];
r=zeros(length(beta_mh),5000);
l=normpdf(Y-X*beta,0,sqrt(sigma_sq));
l=sum(log(l))+log(normpdf(beta_mh(2)-edu,0,s));
acc=0;
for i= 2:5000
    betanew=beta_mh+mvnrnd(zeros(size(beta_mh)),diag(var))';
    lnew=normpdf(Y-X*betanew(1:6),0,sqrt(betanew(7)));
    lnew=sum(log(lnew))+log(normpdf(betanew(2)-edu,0,s));
    c=rand;
    if exp(lnew-l)>=c
        acc=acc+1;
        l=lnew;
        beta_mh=betanew;
    else
    end;
    r(:,i)=beta_mh;
end
str={'constant','educ','exper','smsa','black','south','sigma'};
figure
for j=1:7
    subplot(4,2,j)
    histogram(r(j,:))
    title(str{j})
end