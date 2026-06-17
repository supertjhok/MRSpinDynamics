function [echo_int,T_E]=plot_varyte(exp_begin)

[data1,parameter1]=readbrukerfile('cpmg_oneshot_sp_tevar',exp_begin); % Rectangular
[data2,parameter2]=readbrukerfile('cpmg_oneshot_sp_tevar',exp_begin+1); % CP-M8
[data3,parameter3]=readbrukerfile('cpmg_oneshot_sp_tevar',exp_begin+2); % CP-M15

% Parameters
ne=32; % Number of echoes
dw=parameter1.dw;
delays=parameter1.delays;
pulses=parameter1.pulses;

delt=2*delays(26); % T_FP increment
T_180=pulses(3)/1e6;

siz=size(data1);
len=siz(1); % Total length
le=len/ne; % Samples per echo
te_vect=dw*linspace(0,le-1,le);
te_vect=te_vect-max(te_vect)/2;

numstep=siz(2);

T_FP=2*delays(21)+linspace(0,(numstep-1)*delt,numstep);
T_E=T_FP+T_180;

% Plot asymptotic echo shape
echo1=zeros(le,numstep); echo2=echo1; echo3=echo1;
for i=11:ne
    echo1=echo1+data1((i-1)*le+1:i*le,:);
    echo2=echo2+data2((i-1)*le+1:i*le,:);
    echo3=echo3+data3((i-1)*le+1:i*le,:);
    if i==11
        figure(1);
        plot(te_vect*1e6,real(data1((i-1)*le+1:i*le,1)),'b-'); hold on;
        plot(te_vect*1e6,real(data2((i-1)*le+1:i*le,1)),'r-');
        plot(te_vect*1e6,real(data3((i-1)*le+1:i*le,1)),'k-');
        plot(te_vect*1e6,imag(data1((i-1)*le+1:i*le,1)),'b--');
        plot(te_vect*1e6,imag(data2((i-1)*le+1:i*le,1)),'r--');
        plot(te_vect*1e6,imag(data3((i-1)*le+1:i*le,1)),'k--');
    end
    if i==ne
        figure(2);
        numavg = 3;
        data1((i-1)*le+1:i*le,numstep)=filter(ones(1,numavg)/numavg,1,data1((i-1)*le+1:i*le,numstep));
        data2((i-1)*le+1:i*le,numstep)=filter(ones(1,numavg)/numavg,1,data2((i-1)*le+1:i*le,numstep));
        data3((i-1)*le+1:i*le,numstep)=filter(ones(1,numavg)/numavg,1,data3((i-1)*le+1:i*le,numstep));
        plot(te_vect*1e6,real(data1((i-1)*le+1:i*le,numstep)),'b-'); hold on;
        plot(te_vect*1e6,real(data2((i-1)*le+1:i*le,numstep)),'r-');
        plot(te_vect*1e6,real(data3((i-1)*le+1:i*le,numstep)),'k-');
        plot(te_vect*1e6,imag(data1((i-1)*le+1:i*le,numstep)),'b--');
        plot(te_vect*1e6,imag(data2((i-1)*le+1:i*le,numstep)),'r--');
        plot(te_vect*1e6,imag(data3((i-1)*le+1:i*le,numstep)),'k--');
    end
end

% Running average filter definition
%numavg = 5;
%echo1=filter(ones(1,numavg)/numavg,1,echo1);
%echo2=filter(ones(1,numavg)/numavg,1,echo2);
%echo3=filter(ones(1,numavg)/numavg,1,echo3);

%close all;
figure(3);

plot(te_vect*1e6,abs(echo1(:,numstep)),'b-'); hold on;
plot(te_vect*1e6,abs(echo2(:,numstep)),'r-');
plot(te_vect*1e6,abs(echo3(:,numstep)),'k-');
xlabel('Time (\mus)');
legend('Rectangular','CP\_M8','CP\_M15');

echo_int=zeros(3,numstep);
echo_int(1,:)=trapz(te_vect,abs(echo1).^2);
echo_int(2,:)=trapz(te_vect,abs(echo2).^2);
echo_int(3,:)=trapz(te_vect,abs(echo3).^2);

% Normalize by rectangular case
for i=3:-1:1
    echo_int(i,:)=echo_int(i,:)./echo_int(1,:);
end

figure(4);
rat=T_FP/T_180;

plot(T_E*1e3,echo_int(2,:),'bo-'); hold on;
plot(T_E*1e3,echo_int(3,:),'kd-');
xlabel('T_{E} (ms)')
ylabel('Asymptotic echo amplitude (normalized)');
legend('CP\_M8','CP\_M15');