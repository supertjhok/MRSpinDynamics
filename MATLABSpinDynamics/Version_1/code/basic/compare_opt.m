function [fom]=compare_opt(name)

T_90=20; % us
T_FP=200; % us
NE=10; % Number of echoes to simulate
T1=100; T2=100; % ms

tmp=load('exc_timings.mat');
results=tmp.results;

tmp=size(results);
numres=tmp(1);
for i=1:numres
    if strcmpi(name,results{i,1})
        texc=T_90*results{i,2};
        pexc=results{i,3};
    end
end

if strcmpi(name,'rectangular')
    % Rectangular refocusing pulse
    tref=2*T_90;
    pref=pi/2;
    refname='rectangular';
else
    % RP2-1.0a refocusing pulse
    tref=2*T_90*[0.14,0.72,0.14];
    pref=pi*[3,1,3]/2;
    refname='RP2_1.0a';
end

[echo_opt,tvect_opt]=cpmg_van_spin_dynamics_opt(tref,pref,T_90,NE,T_FP,T1,T2);
[echo,tvect]=cpmg_van_spin_dynamics_echo(texc,tref,pexc,pref,T_90,NE,T_FP,T1,T2);

close all;

figure(1);
plot(tvect_opt*1e6,abs(echo_opt),'b-','LineWidth',2);
hold on;
plot(tvect*1e6,abs(echo),'r-','LineWidth',2);

legstr={};
legstr{1}=['Optimal, ' refname];
legstr{2}=[name ', ' refname];

set(gca,'FontSize',14)
xlabel('Time (\mus)')
ylabel('Echo amplitude')
legend(legstr);

% Figures of merit
fom=zeros(2,2);
fom(1,:)=[max(abs(echo)) max(abs(echo_opt))]; % Maximum
fom(2,:)=[trapz(tvect,abs(echo).^2) trapz(tvect_opt,abs(echo_opt).^2)]; % SNR with matched filter