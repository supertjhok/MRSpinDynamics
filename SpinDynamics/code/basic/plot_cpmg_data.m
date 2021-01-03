function plot_cpmg_data(expts)

close all;

path='cpmg_van2';
te=1.8; % echo time, ms
ne=50; % number of echoes
dw=2.048; % dwell time, us

i=sqrt(-1);

%clr={'k.-','k*-','ks-','kd-','kv-','k^-','ko-'};
clr={'b-','r-','k-','m-','g','b--','r--'};

i=sqrt(-1);
for j=1:length(expts)
    filname=[path '\' num2str(expts(j)) '\data2d.csv'];
    tmp=csvread(filname);
    siztmp=size(tmp);
    necho=siztmp(2)/2; % number of points/echo
    tmp2=zeros(siztmp(1),necho);
    for k=1:necho
       tmp2(:,k)=tmp(:,2*k-1)+i*tmp(:,2*k);
    end
    techo=2*dw*necho/1e3; % time/echo, ms
    tvect=zeros(1,ne*necho);
    dat=zeros(1,ne*necho);
    for k=1:ne
        dat((k-1)*necho+1:k*necho)=tmp2(k,:);
        tvect((k-1)*necho+1:k*necho)=(k-1)*te+linspace(0,techo,necho);
    end
    save data2d.mat dat tvect
    
    figure(1);
    plot(tvect,abs(dat),clr{j}); hold on;
end

figure(1);
set(gca,'FontSize',14);
xlabel('Time (ms)');
ylabel('Echo (magnitude)  (\mu V)');