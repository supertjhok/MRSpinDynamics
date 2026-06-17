function [echo_rms,echo_max]=plot_cpmgadd2_data(run_num,expts,ref_expts,delt,plt,filt)

zf=8; % Zero-filling ratio
windowSize=5; % Running average LPF
close all;

path=['cpmgadd_van' num2str(run_num)];

echo_rms=zeros(length(expts),1); echo_max=echo_rms;
echo_rms_ref=zeros(length(ref_expts),1); echo_max_ref=echo_rms_ref;
i=sqrt(-1);

%clr={'k.-','k*-','ks-','kd-','kv-','k^-','ko-'};
clr={'b-','r-','k-','m-','g-','b--','r--','k--','m--','g--','b.-','r.-','k.-'};

for j=1:length(expts)
    filname=[path '\' num2str(expts(j)) '\data.csv'];
    tmp=csvread(filname);
    tim=tmp(:,1);
    dat=tmp(:,2)+i*tmp(:,3);
    
    % Echo spectrum - with zero-filling to get more resolution
    fs=1/(tim(2)-tim(1)); %kHz
    fvect=linspace(-fs/2,fs/2,length(dat)*zf);
    dat_zf=zeros(zf*length(dat),1);
    dat_zf((zf-1)*length(dat)/2:(zf+1)*length(dat)/2-1)=dat;
    spect=zf*abs(fftshift(fft(real(dat_zf))+fft(imag(dat_zf))))/length(dat_zf);
    
    % Running average filter
    if filt
        dat=filter(ones(1,windowSize)/windowSize,1,dat);
    end
    
    ind=abs(tim-max(tim)/2)<delt/2;
    echo_rms(j)=trapz(tim,abs(dat(ind)).^2);
    echo_max(j)=max(abs(dat));
    if plt
        figure(1);
        plot(tim*1e3,real(dat)*1e3,clr{j},'LineWidth',2); hold on;
        figure(2);
        plot(tim*1e3,imag(dat)*1e3,clr{j},'LineWidth',2); hold on;
        figure(3);
        plot(tim*1e3,abs(dat)*1e3,clr{j},'LineWidth',2); hold on;
        figure(4);
        plot(fvect,abs(spect),clr{j},'LineWidth',2); hold on;
    end
end

for j=1:length(ref_expts)
    filname=[path '\' num2str(ref_expts(j)) '\data.csv'];
    tmp=csvread(filname);
    tim=tmp(:,1);
    dat=tmp(:,2)+i*tmp(:,3);
    
    % Running average filter
    if filt
        dat=filter(ones(1,windowSize)/windowSize,1,dat);
    end
    
    ind=abs(tim-max(tim)/2)<delt/2;
    echo_rms_ref(j)=trapz(tim,abs(dat(ind)).^2);
    echo_max_ref(j)=max(abs(dat));
end

% Use reference experiments for normalization
echo_rms=echo_rms%./echo_rms_ref;
echo_max=echo_max%./echo_max_ref;

if plt
    figure(1);
    set(gca,'FontSize',14);
    xlabel('Time (\mus)');
    ylabel('Echo (real)  (\mu V)');
    
    figure(2);
    set(gca,'FontSize',14);
    xlabel('Time (\mus)');
    ylabel('Echo (imaginary)  (\mu V)');
    
    figure(3);
    set(gca,'FontSize',14);
    xlabel('Time (\mus)');
    ylabel('Echo (absolute value)  (\mu V)');
    
    figure(4);
    set(gca,'FontSize',14);
    xlabel('Frequency (kHz)');
    ylabel('Echo spectrum (absolute value)');
end