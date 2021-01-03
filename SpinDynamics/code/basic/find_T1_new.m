% Find T1 from 2D Bruker data (IR)
% tvect and T1 are in ms
% necho is the number of echoes (CPMG detection)
% ---------------------------------------------------
% Modified by Soumyajit Mandal on 03/15/11 to use the procedure 
% described in Hurlimann, JMR 2001, which is more accurate in inhomogeneous
% magnetic fields

function [T1]=find_T1_new(dirname,expno,tvect,necho)

nignore=5; % Ignore first few echoes to avoid envelope transient

close all;
[data,parameter]=readbrukerfile(dirname,expno);
sizdata=size(data);

pts_echo=sizdata(1)/necho; % Points/echo
dw=parameter.dw;
te_vect=dw*linspace(-pts_echo/2,pts_echo/2-1,pts_echo);

amp=zeros(pts_echo,sizdata(2));
amp_corr=zeros(pts_echo,sizdata(2)-1);
amp_int=zeros(1,sizdata(2)-1);

ntheta=300;
theta=linspace(0,pi,ntheta);
err=zeros(1,ntheta);

% Calculate asymptotic echoes and phase them correctly
figure(1);
for j=1:sizdata(2)
    for n=nignore:necho
        amp(:,j)=amp(:,j)+data(pts_echo*(n-1)+1:pts_echo*n,j);
    end
    
    for n=1:ntheta
        tmp=amp(:,j)*exp(1i*theta(n));
        err(n)=abs(sum(imag(tmp)));
    end
    [~,ind]=min(err);
    amp(:,j)=amp(:,j)*exp(1i*theta(ind));
    
    plot(te_vect*1e6,real(amp(:,j)),'r-'); hold on;
    plot(te_vect*1e6,imag(amp(:,j)),'b-');
    
    amp(:,j)=real(amp(:,j));
end
xlabel('Time (\mus)');

% Subtract each echo from last echo to limit dependence to T1
% Note: this procedure only works if the last value of tvect >> T1
figure(2);
for j=1:sizdata(2)-1
    amp_corr(:,j)=amp(:,sizdata(2))-amp(:,j);
    amp_int(j)=trapz(amp_corr(:,j));
    plot(amp_corr(:,j)); hold on;
end
xlabel('Time (\mus)');

amp_int=amp_int/amp_int(1);
tvect=tvect(1:sizdata(2)-1);

% Estimate T1 by fitting the decay of amp_int to exp(-t/T1)
p=polyfit(tvect,log(amp_int),1);
log_amp_est=polyval(p,tvect);
amp_est=exp(log_amp_est);

T1=-1/p(1);

figure(3);
semilogy(tvect*1e3,amp_int,'bo'); hold on;
semilogy(tvect*1e3,amp_est,'b--');
xlabel('Time (ms)');
title(['Estimated T_{1} = ' num2str(T1) ' sec']);