% Simulate a single rectangular pulse from a tuned-and-matched probe

function matchedPulse

[sp,pp] = set_params_matched_JMR; % Set parameters

% Design matching network
[C1,C2]=matching_network_design2(sp.L,sp.Q,sp.f0,sp.Rs,sp.plt_mn);
sp.C1=C1; sp.C2=C2; % Save matching capacitor values

% Create required fields in pp structure
pp.tp=pp.tref;
pp.phi=pp.pref;
pp.amp=pp.aref;

% Find coil current and receiver TFs
[tvec2,yr2,tvec,y,tf1,tf2] = find_coil_current_orig(sp,pp);

figure(15+sp.plt_tx);
% RF 
plot(tvec*1e6,real(y(:,1)),'LineWidth',1); hold on; 
plot(tvec*1e6,imag(y(:,1)),'LineWidth',1);
xlabel('Time (\mus)');
ylabel('RF coil current (normalized)'); 
set(gca,'FontSize',15); set(gca,'FontWeight','bold');
legend({'Real','Imag'});
    
figure(15+sp.plt_tx+1);
% Rotating frame
plot(tvec2*1e6,real(yr2),'LineWidth',1); hold on; 
plot(tvec2*1e6,imag(yr2),'LineWidth',1);
xlabel('Time (\mus)');
ylabel('Rotating frame current (normalized)');
set(gca,'FontSize',15); set(gca,'FontWeight','bold');
legend({'Real','Imag'});

% Plot receiver TF
figure(1);
subplot(2,1,1); semilogy(sp.del_w,abs(tf1),'LineWidth',1); hold on;
ylabel('mag(TF)');
set(gca,'FontSize',15); set(gca,'FontWeight','bold');

subplot(2,1,2); plot(sp.del_w,(180/pi)*angle(tf1),'LineWidth',1); hold on;
ylabel('phase(TF)');
xlabel('\Delta\omega_{0}/\omega_{1,max}');
set(gca,'FontSize',15); set(gca,'FontWeight','bold');

% Show other receiver properties
[~,~,~,~] = matched_probe_rx(sp,pp,ones(1,length(sp.del_w)),tf1,tf2);