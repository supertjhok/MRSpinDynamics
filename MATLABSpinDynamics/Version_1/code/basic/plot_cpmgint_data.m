function [echo_int]=plot_cpmgint_data

close all;

path='cpmgint_van';
expts=1:7;

echo_int=zeros(length(expts),1);
i=sqrt(-1);

clr={'k.-','k*-','ks-','kd-','kv-','k^-','ko-'};

for j=1:length(expts)
    filname=[path '\' num2str(expts(j)) '\data.csv'];
    tmp=csvread(filname);
    tim=tmp(:,1);
    dat=tmp(:,2)+i*tmp(:,3);
    necho=length(tim); % Number of echoes
    
    % Running average filter
    %windowSize=5;
    %dat=filter(ones(1,windowSize)/windowSize,1,dat);
    
    % Ignore first 2 echoes
    echo_int(j)=trapz(tim(3:necho),abs(dat(3:necho)));
    %echo_int(j)=max(abs(dat(3:necho)));
    dat=abs(dat);
    semilogy(tim/1e3,dat*1e3,clr{j},'LineWidth',2); hold on;
end

% Use experiment #2 as a reference for normalization
echo_int=echo_int/echo_int(expts==2);

set(gca,'FontSize',14);
xlabel('Time (ms)');
ylabel('Echo integral (\mu V)');

    
    
