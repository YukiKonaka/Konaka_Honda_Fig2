clear all 
close all
rng(1)
clc

%%Fig2 B-G

T=1000;
time=1:T;

%set parameters
a=zeros(1,T);%action
o=zeros(1,T);%reward
ml=zeros(1,T);%mean of gaussian distribution (left option)
mr=zeros(1,T);%mean of gaussian distribution (right option)
pl=zeros(1,T);%precision of gaussian distribution (left option)
pr=zeros(1,T);%precision of gaussian distribution (right option)
Gl=zeros(1,T);%expected utility(left)
Gr=zeros(1,T);%expected utility(right)
fl=zeros(1,T);
fr=zeros(1,T);

C=1*ones(1,T);%curiosity

tt=1:T;
Pl=(tt<=T/2)*0.8+(tt>T/2)*0.8; %reward probability(left)
Pr=(tt<=T/2)*0.2+(tt>T/2)*0.2; %reward probability(right)

Po=0.9;
vw=0.04;
am=0.1;

po=0.1;
mo=0;

%set initial values
a(1)=rand<0.5;

ml(1)=mo; 
pl(1)=po;

mr(1)=mo;
pr(1)=po;

dt=1;


%%%%%%%%%%%%%Simmulation%%%%%%%%%%%%%%


for t=2:T
  %%%%%%%%%%%%% Estimation of the reward probability%%%%%%%%%%%%%%  
    if t==2
        a(t)=rand<0.5;
    else
       al_prob=1/(1+(exp(-(Gr(t-1)-Gl(t-1)))));
       a(t)=rand<al_prob;
    end
    
    fl(t)=(1/(1+exp(-ml(t-1))));   
    fr(t)=(1/(1+exp(-mr(t-1)))); 
    
    if a(t)==1 
        o(t) = rand < Pl(t);
        
        
        dmldt=((1/pl(t-1))+vw)*(o(t)-fl(t));        
        ml(t)=ml(t-1)+am*dmldt*dt;
        Sl(t)=(1/(1+exp(-ml(t))));
        pl(t)=((1/vw*pl(t-1))/(pl(t-1)+1/vw))+(Sl(t)*(1-Sl(t)));
        
        
        mr(t)=mr(t-1);
        pr(t)=((1/vw*pr(t-1))/(pr(t-1)+1/vw));
    
    else
        o(t) = rand < Pr(t);
        
        
        dmrdt=((1/pr(t-1))+vw)*(o(t)-fr(t));
        mr(t)=mr(t-1)+am*dmrdt*dt;
        Sr(t)=(1/(1+exp(-mr(t))));
        pr(t)=((1/vw*pr(t-1))/(pr(t-1)+1/vw))+(Sr(t)*(1-Sr(t)));
        
        
        ml(t)=ml(t-1);
        pl(t)=((1/vw*pl(t-1))/(pl(t-1)+1/vw));
         
    end 
    lamdal(t)=1/(1+exp(-ml(t)));
    lamdar(t)=1/(1+exp(-mr(t)));
   
  %%%%%%%%%%%%% Calculate expected utility%%%%%%%%%%%%%%
  
POAl(t)=lamdal(t)+0.5*lamdal(t)*(1-lamdal(t))*(1-2*lamdal(t))*((1/pl(t))+vw);  
Al(t)= -lamdal(t)*log(lamdal(t))-(1-lamdal(t))*log(1-lamdal(t));
Bl(t)=-0.5*(lamdal(t)*(1-lamdal(t))*(1+(1-2*lamdal(t))*(log(lamdal(t))-log(1-lamdal(t)))))*((1/pl(t))+vw);
Cl(t)=(1-POAl(t))*log(1-POAl(t))+POAl(t)*log(POAl(t));
Dl(t)=-POAl(t)*log(Po/(1-Po))-(1-POAl(t))*0;


POAr(t)=lamdar(t)+0.5*lamdar(t)*(1-lamdar(t))*(1-2*lamdar(t))*((1/pr(t))+2*vw);
Ar(t)= -lamdar(t)*log(lamdar(t))-(1-lamdar(t))*log(1-lamdar(t));
Br(t)=-0.5*(lamdar(t)*(1-lamdar(t))*(1+(1-2*lamdar(t))*(log(lamdar(t))-log(1-lamdar(t)))))*((1/pr(t))+vw);
Cr(t)=(1-POAr(t))*log(1-POAr(t))+POAr(t)*log(POAr(t));
Dr(t)=-POAr(t)*log(Po/(1-Po))-(1-POAr(t))*0;


Gl(t)=C(t)*(Al(t)+Bl(t)+Cl(t))+Dl(t);
Gr(t)=C(t)*(Ar(t)+Br(t)+Cr(t))+Dr(t);

Precisionl(t)=pl(t-1)/(fl(t)^2)./((1-fl(t))^2);
Precisionr(t)=pr(t-1)/(fr(t)^2)./((1-fr(t))^2); 

Curiosityl(t)=(Al(t)+Bl(t)+Cl(t));
Curiosityr(t)=(Ar(t)+Br(t)+Cr(t));


end

N_step=length(a);
 width=100;
for i=1:N_step-width
    ppl(i+0.5*width)=mean(a(i:i+width));
    
end

ppl(1:0.5*width)=NaN;


f = figure;
subplot(4,3,1);
f.Position(3:4) = [1000 750];
plot(ppl)
hold on
plot(1-ppl)
legend("left","right")
ylim([0 1])
yticks([0 0.25 0.5 0.75 1])
box('off')
xticks([0 500 1000])


subplot(4,3,2)
plot(lamdal,'r-','LineWidth',5)
hold on
plot(lamdar,'g-','LineWidth',5)
hold on 
plot(Pl,'--','LineWidth',3)
hold on
plot(Pr,'--','LineWidth',3)
xlabel('Trials','FontSize',20,'FontWeight','bold')
ylabel({'Estimated';'Reward';'Probability'},'FontSize',20,'FontWeight','bold')
xlim([0,1000])
ylim([0,1])
xticks([0 500 1000])
yticks([0 0.25 0.5 0.75 1])
box('off')

subplot(4,3,3)
plot(Precisionl,'r-','LineWidth',5)
hold on
plot(Precisionr,'g-','LineWidth',5)
hold on
xlabel('Trials','FontSize',20,'FontWeight','bold')
ylabel('Precision','FontSize',20,'FontWeight','bold')
xlim([0,1000])
ylim([0,200])
xticks([0 500 1000])
yticks([0 50 100 150 200])
box('off')


subplot(4,3,4)
plot(-Curiosityl,'r-','LineWidth',5)
hold on
plot(-Curiosityr,'g-','LineWidth',5)
xlabel('Trials','FontSize',20,'FontWeight','bold')
ylabel({'Brief';'update'},'FontSize',25,'FontWeight','bold')
xlim([0,1000])
ylim([0,0.16])
xticks([0 500 1000])
yticks([0 0.04 0.08 0.12 0.16])
box('off')

subplot(4,3,5)
plot(-Dl,'r-','LineWidth',5)
hold on
plot(-Dr,'g-','LineWidth',5)
xlabel('Trials','FontSize',20,'FontWeight','bold')
ylabel({'Expected';'Reward'},'FontSize',25,'FontWeight','bold')
xlim([0,1000])
ylim([0,2])
xticks([0 500 1000])
yticks([0 0.5 1 1.5 2])
box('off')

subplot(4,3,6)
plot(-Gl,'r-','LineWidth',5)
hold on
plot(-Gr,'g-','LineWidth',5)
xlabel('Trials','FontSize',20,'FontWeight','bold')
ylabel({'Expected';'Utility'},'FontSize',25,'FontWeight','bold')
xlim([0,1000])
ylim([0,3.0])
xticks([0 500 1000])
yticks([0 0.75 1.5 2.25 3.0])
box('off')




 
Times_of_al=sum(a);
Times_of_ar=T-Times_of_al;

Rate_of_al=Times_of_al/(Times_of_al+Times_of_ar);
Rate_of_ar=Times_of_ar/(Times_of_al+Times_of_ar);







%%

%Fig2 B'-G'

T=1000;
time=1:T;

%set parameters
a=zeros(1,T);%action
o=zeros(1,T);%reward
ml=zeros(1,T);%mean of gaussian distribution (left option)
mr=zeros(1,T);%mean of gaussian distribution (right option)
pl=zeros(1,T);%precision of gaussian distribution (left option)
pr=zeros(1,T);%precision of gaussian distribution (right option)
Gl=zeros(1,T);%expected utility(left)
Gr=zeros(1,T);%expected utility(right)
fl=zeros(1,T);
fr=zeros(1,T);

C=1*ones(1,T);%curiosity


tt=1:T;
Pr=(tt<=T/2)*0.5+(tt>T/2)*0.5;
Pl=0.5*cos(2*pi*time/(T/1))+0.5;

Po=0.9;
vw=0.4;
am=0.05;
po=0.1;
mo=0;

%set initial balue
a(1)=rand<0.5;

ml(1)=mo; 
pl(1)=po;

mr(1)=mo;
pr(1)=po;

dt=1;
%%%%%%%%%%%%%Simmulation%%%%%%%%%%%%%%


for t=2:T
  %%%%%%%%%%%%% Estimation of the reward probability%%%%%%%%%%%%%%  
   
    if t==2
        a(t)=rand<0.5;
    else
       al_prob=1/(1+(exp(-(Gr(t-1)-Gl(t-1)))));
       a(t)=rand<al_prob;
    end
    
    fl(t)=(1/(1+exp(-ml(t-1))));   
    fr(t)=(1/(1+exp(-mr(t-1)))); 
    
    if a(t)==1 
        o(t) = rand < Pl(t);
        
        
        dmldt=((1/pl(t-1))+vw)*(o(t)-fl(t)-(1/(2*pl(t-1)))*fl(t)*(1-fl(t))*(1-2*fl(t)));
        ml(t)=ml(t-1)+am*dmldt*dt;
        Sl(t)=(1/(1+exp(-ml(t))));
        pl(t)=((1/vw*pl(t-1))/(pl(t-1)+1/vw))+(Sl(t)*(1-Sl(t)));
        
        
        mr(t)=mr(t-1);
        pr(t)=((1/vw*pr(t-1))/(pr(t-1)+1/vw));
    
    else
        o(t) = rand < Pr(t);
        
        
        dmrdt=((1/pr(t-1))+vw)*(o(t)-fr(t)-(1/(2*pr(t-1)))*fr(t)*(1-fr(t))*(1-2*fr(t)));
        mr(t)=mr(t-1)+am*dmrdt*dt;
        Sr(t)=(1/(1+exp(-mr(t))));
        pr(t)=((1/vw*pr(t-1))/(pr(t-1)+1/vw))+(Sr(t)*(1-Sr(t)));
        
        
        ml(t)=ml(t-1);
        pl(t)=((1/vw*pl(t-1))/(pl(t-1)+1/vw));
         
    end 
    lamdal(t)=1/(1+exp(-ml(t)));
    lamdar(t)=1/(1+exp(-mr(t)));
   
  %%%%%%%%%%%%%Calculate expected Utility%%%%%%%%%%%%%%
  
POAl(t)=lamdal(t)+0.5*lamdal(t)*(1-lamdal(t))*(1-2*lamdal(t))*((1/pl(t))+vw);  
Al(t)= -lamdal(t)*log(lamdal(t))-(1-lamdal(t))*log(1-lamdal(t));
Bl(t)=-0.5*(lamdal(t)*(1-lamdal(t))*(1+(1-2*lamdal(t))*(log(lamdal(t))-log(1-lamdal(t)))))*((1/pl(t))+vw);
Cl(t)=(1-POAl(t))*log(1-POAl(t))+POAl(t)*log(POAl(t));
Dl(t)=-POAl(t)*log(Po/(1-Po))-(1-POAl(t))*0;

POAr(t)=lamdar(t)+0.5*lamdar(t)*(1-lamdar(t))*(1-2*lamdar(t))*((1/pr(t))+2*vw);
Ar(t)= -lamdar(t)*log(lamdar(t))-(1-lamdar(t))*log(1-lamdar(t));
Br(t)=-0.5*(lamdar(t)*(1-lamdar(t))*(1+(1-2*lamdar(t))*(log(lamdar(t))-log(1-lamdar(t)))))*((1/pr(t))+vw);
Cr(t)=(1-POAr(t))*log(1-POAr(t))+POAr(t)*log(POAr(t));
Dr(t)=-POAr(t)*log(Po/(1-Po))-(1-POAr(t))*0;

Gl(t)=C(t)*(Al(t)+Bl(t)+Cl(t))+Dl(t);
Gr(t)=C(t)*(Ar(t)+Br(t)+Cr(t))+Dr(t);

Precisionl(t)=pl(t-1)/(fl(t)^2)./((1-fl(t))^2);
Precisionr(t)=pr(t-1)/(fr(t)^2)./((1-fr(t))^2); 

Curiosityl(t)=(Al(t)+Bl(t)+Cl(t));
Curiosityr(t)=(Ar(t)+Br(t)+Cr(t));


end


N_step=length(a);
 width=100;
for i=1:N_step-width
    ppl2(i+0.5*width)=mean(a(i:i+width));
    
end

ppl2(1:0.5*width)=NaN;

subplot(4,3,7);
f.Position(3:4) = [1000 750];
plot(ppl2)
hold on
plot(1-ppl2)
legend("left","right")
box('off')
ylim([0 1])
yticks([0 0.25 0.5 0.75 1])
xticks([0 500 1000])


subplot(4,3,8)
plot(lamdal,'r-','LineWidth',5)
hold on
plot(lamdar,'g-','LineWidth',5)
hold on 
plot(Pl,'--','LineWidth',3)
hold on
plot(Pr,'--','LineWidth',3)
xlabel('Trials','FontSize',20,'FontWeight','bold')
ylabel({'Estimated';'Reward';'Probability'},'FontSize',20,'FontWeight','bold')
xlim([0,1000])
ylim([0,1])
xticks([0 500 1000])
yticks([0 0.25 0.5 0.75 1])

box('off')


subplot(4,3,9)
plot(Precisionl,'r-','LineWidth',5)
hold on
plot(Precisionr,'g-','LineWidth',5)
hold on
xlabel('Trials','FontSize',20,'FontWeight','bold')
ylabel('Precision','FontSize',20,'FontWeight','bold')
xlim([0,1000])
ylim([0,250])
xticks([0 500 1000])
yticks([0 60 120 180 240])
box('off')



subplot(4,3,10)
plot(-Curiosityl,'r-','LineWidth',5)
hold on
plot(-Curiosityr,'g-','LineWidth',5)
xlabel('Trials','FontSize',20,'FontWeight','bold')
ylabel({'Brief';'update'},'FontSize',25,'FontWeight','bold')
xlim([0,1000])
ylim([0,1])
xticks([0 500 1000])
yticks([0 0.25 0.5 0.75 1])
box('off')



subplot(4,3,11)
plot(-Dl,'r-','LineWidth',5)
hold on
plot(-Dr,'g-','LineWidth',5)
xlabel('Trials','FontSize',20,'FontWeight','bold')
ylabel({'Expected';'Reward'},'FontSize',25,'FontWeight','bold')
xlim([0,1000])
ylim([0,2])
xticks([0 500 1000])
yticks([0 0.5 1 1.5 2])
box('off')


subplot(4,3,12)
plot(-Gl,'r-','LineWidth',5)
hold on
plot(-Gr,'g-','LineWidth',5)
xlabel('Trials','FontSize',20,'FontWeight','bold')
ylabel({'Expected';'Utility'},'FontSize',25,'FontWeight','bold')
xlim([0,1000])
ylim([0,3.0])
xticks([0 500 1000])
yticks([0 0.75 1.5 2.25 3.0])
box('off')



 
Times_of_al=sum(a);
Times_of_ar=T-Times_of_al;

Rate_of_al=Times_of_al/(Times_of_al+Times_of_ar);
Rate_of_ar=Times_of_ar/(Times_of_al+Times_of_ar);

