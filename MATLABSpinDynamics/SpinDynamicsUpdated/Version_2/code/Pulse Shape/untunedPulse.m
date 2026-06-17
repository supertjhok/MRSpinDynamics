% Simulate a single rectangular pulse from an untuned probe

function untunedPulse

VBB = 62.5; % Transmitter supply voltage

[sp,pp] = set_params_untuned_JMR; % Set parameters

[tvect2,Icr2,tvect,Ic] = untuned_probe_lp(sp,pp); % Calculate normalized pulse

% close all;
% Scale pulse by VBB
Ic=VBB*Ic; Icr2=VBB*Icr2;

figure(11+sp.plt_tx);
% RF (imaginary component should be 0)
plot(tvect*1e6,real(Ic),'LineWidth',1);  hold on;
plot(tvect*1e6,imag(Ic),'r-','LineWidth',1);
xlabel('Time (\mus)');
ylabel('RF coil current (normalized)');
set(gca,'FontSize',15); set(gca,'FontWeight','bold');
legend({'Real','Imag'});

figure(11+sp.plt_tx+1);
% Rotating frame
plot(tvect2*1e6,real(Icr2),'LineWidth',1);  hold on;
plot(tvect2*1e6,imag(Icr2),'r-','LineWidth',1);
%plot(tvect2*1e6,abs(Icr2),'k-','LineWidth',1);
xlabel('Time (\mus)');
ylabel('Rotating frame current (normalized)');
set(gca,'FontSize',15); set(gca,'FontWeight','bold');
legend({'Real','Imag'});

% Plot receiver TF
macq=ones(1,length(sp.del_w));
[~,~,tf] = untuned_probe_rx(sp,pp,macq); % With input transformer
%[~,~,tf] = untuned_probe_rx2(sp,pp,macq); % Without input transformer

figure(1);
subplot(2,1,1); semilogy(sp.del_w,abs(tf),'LineWidth',1); hold on;
ylabel('mag(TF)');
set(gca,'FontSize',15); set(gca,'FontWeight','bold');

subplot(2,1,2); plot(sp.del_w,(180/pi)*angle(tf),'LineWidth',1); hold on;
ylabel('phase(TF)');
xlabel('\Delta\omega_{0}/\omega_{1,max}');
set(gca,'FontSize',15); set(gca,'FontWeight','bold');

