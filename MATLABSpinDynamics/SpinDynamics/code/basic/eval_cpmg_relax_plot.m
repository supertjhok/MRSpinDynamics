function [T_relax,rat]=eval_cpmg_relax_plot(expt,NE,T2)

%rat=linspace(1,20,20);
rat=linspace(1,1.5,10);

numpts=length(rat);
T_relax=zeros(1,numpts);
for j=1:numpts
    [T_relax(j),~,~,~,~]=eval_cpmg_relax(expt,NE,rat(j)*T2,T2,0);
    disp(j)
end

figure(3);
plot(rat,T_relax/T2,'bo-'); hold on;
xlabel('T_{1} / T_{2}');
ylabel('T_{2,eff} / T_{2}');