% Optimize CPMG refocusing pulse, odd number of segments
% Optimize segment lengths, keep phases fixed
% Soumyajit Mandal, 02/27/13
% --------------------------------------------------------------
% Assume pulses with interal anti-symmetry
% --------------------------------------------------------------

function [out]=opt_ref_pulse_asymp_mag4(params)

tfp=params.tfp;
tref=params.tref; tref=tref';
pref=params.pref; pref=pref';
aref=params.aref; aref=aref';
tmin=params.tmin;
tmax=params.tmax;

del_w=params.del_w; numpts=length(del_w);
opt_window=params.opt_window; % window function for optimization
tacq=params.tacq;
window = sinc(del_w*tacq/(2*pi)); % window function for acquisition

nref=length(tref); lref=sum(tref);
start=tref;

% Excitation pulse definition
% Use nonlinear function minimization - all segment times must be positive
lb=tmin*ones(nref,1); % Lower bound
ub=tmax*ones(nref,1); % Upper bound
Aeq=ones(nref,nref); beq=lref*ones(nref,1); % Keep total pulse length fixed

% trust-region-reflective algorithm (fmincon default) will not work for
% this problem because of the constraints, so use interior-point, sqp, or
% active-set algorithms instead
options=optimset('Algorithm','interior-point','Display','iter','TolFun',1e-4,'MaxFunEvals',2e3);
%options=optimset('Algorithm','active-set','Display','iter','TolFun',1e-4,'MaxFunEvals',2e3);
%options=optimset('Algorithm','sqp','Display','iter','TolFun',1e-4,'MaxFunEvals',2e3);

tref=fmincon(@(params)fit_function(params,pref,aref,tfp,del_w,opt_window,window),start,[],[],Aeq,beq,lb,ub,[],options);
trefc=[tfp; tref; flipud(tref(1:nref-1)); tfp]; % Create refocusing cycle
prefc=[0; pref; -flipud(pref(1:nref-1)); 0];
arefc=[0; aref; aref(1:nref-1); 0];
[neff]=calc_rot_axis_arba2(trefc,prefc,arefc,del_w,0);
nx=dot(neff,neff).*neff(1,:); fy = conv(nx,window);
nx = fy(((numpts+1)/2:3*(numpts-1)/2+1))./sum(window);

out.tref=tref';
out.pref=pref';
out.aref=aref';
out.axis_rms=sqrt(trapz(del_w,abs(nx).^2));

function val=fit_function(params,pref,aref,tfp,del_w,opt_window,window)

numpts=length(del_w); nref=length(params);
trefc=[tfp; params; flipud(params(1:nref-1)); tfp]; % Create refocusing cycle
prefc=[0; pref; -flipud(pref(1:nref-1)); 0];
arefc=[0; aref; aref(1:nref-1); 0];
[neff]=calc_rot_axis_arba2(trefc,prefc,arefc,del_w,0);
nx=dot(neff,neff).*neff(1,:); fy = conv(abs(nx),window);
nx = fy(((numpts+1)/2:3*(numpts-1)/2+1)).*opt_window./sum(window);
%nx = fy(((numpts+1)/2:3*(numpts-1)/2+1))./sum(window);

% Optimize refocusing axis
%val=-sqrt(trapz(del_w,abs(nx).^2));

% Add penalty for asymmetric spectrum
val=-sqrt(trapz(del_w,abs(nx).^2))+0.5*sqrt(trapz(del_w,(abs(nx-fliplr(nx))).^2));