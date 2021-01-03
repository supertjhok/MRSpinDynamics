% Calculate equivalent rotation matrix of a composite pulse

function [Rtot] = calc_rotation_matrix(sp,pp)

% Load parameters
tp=pp.tp;
phi=pp.phi; 
amp=pp.amp;
del_w=sp.del_w; 
w_1=sp.w_1;

% Calculate rotation matrix
num_pulses=length(phi); 
for j=1:num_pulses
    w1=amp(j)*w_1;
    Omega=sqrt(w1.*w1+del_w.*del_w);
    mat=calc_matrix_elements(del_w,w1,Omega,tp(j),phi(j)); % Current rotation matrix
    
    if j==1
        Rtot=mat;
    else
        tmp=Rtot; % Estimate Rtot = mat * Rtot
        Rtot.R_00=mat.R_00.*tmp.R_00+mat.R_0m.*tmp.R_m0+mat.R_0p.*tmp.R_p0;
        Rtot.R_0m=mat.R_00.*tmp.R_0m+mat.R_0m.*tmp.R_mm+mat.R_0p.*tmp.R_pm;
        Rtot.R_0p=mat.R_00.*tmp.R_0p+mat.R_0m.*tmp.R_mp+mat.R_0p.*tmp.R_pp;
        Rtot.R_m0=mat.R_m0.*tmp.R_00+mat.R_mm.*tmp.R_m0+mat.R_mp.*tmp.R_p0;
        Rtot.R_mm=mat.R_m0.*tmp.R_0m+mat.R_mm.*tmp.R_mm+mat.R_mp.*tmp.R_pm;
        Rtot.R_mp=mat.R_m0.*tmp.R_0p+mat.R_mm.*tmp.R_mp+mat.R_mp.*tmp.R_pp;
        Rtot.R_p0=mat.R_p0.*tmp.R_00+mat.R_pm.*tmp.R_m0+mat.R_pp.*tmp.R_p0;
        Rtot.R_pm=mat.R_p0.*tmp.R_0m+mat.R_pm.*tmp.R_mm+mat.R_pp.*tmp.R_pm;
        Rtot.R_pp=mat.R_p0.*tmp.R_0p+mat.R_pm.*tmp.R_mp+mat.R_pp.*tmp.R_pp;
    end
end

% Calculate matrix elements for RF pulses, neglect relaxation
function R = calc_matrix_elements(del_w,w1,Omega,tp,phi)

dw=del_w./Omega; 
dw_2=dw.*dw;
w1n=w1./Omega;
w1n_2=w1n.*w1n;
ph=exp(1i*phi);
sn=sin(Omega*tp); 
cs=cos(Omega*tp);

R.R_00=dw_2+w1n_2.*cs;
R.R_0p=0.5*w1n.*(dw.*(1-cs)-1i*sn)*conj(ph); 
R.R_0m=conj(R.R_0p);
R.R_p0=w1n.*(dw.*(1-cs)-1i*sn)*ph; 
R.R_m0=conj(R.R_p0);
R.R_pp=0.5*(w1n_2+(1+dw_2).*cs)+1i*dw.*sn; 
R.R_mm=conj(R.R_pp);
R.R_pm=0.5*w1n_2.*(1-cs)*ph*ph; 
R.R_mp=conj(R.R_pm);