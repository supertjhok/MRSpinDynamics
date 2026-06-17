function [Lambda,Linit]=lambdas_new(texc,pexc,aexc,tref,pref,aref,del_w)

refmat=calc_spin_mat_arba_delw(tref,pref,aref,del_w);

if length(texc)==1
    % Apply timing correction
    excmat=calc_spin_mat_arba_delw...
        ([texc -1/aexc(1)],[pexc 0],[aexc 0],del_w);
else
    excmat=calc_spin_mat_arba_delw(texc,pexc,aexc,del_w);
end

% Find indices
c0=map_pi(0); cm=map_pi(-1); cp=map_pi(+1);

% Map from [0 - +] to [- 0 +] for compatibility with Martin's code
% Note: the terminology here follows the JMR paper (2001) and not the old
% code, which had the order of the coherence indices reversed

Lambda = zeros(9,length(del_w));
Linit  = zeros(2,length(del_w));
for k=1:length(del_w)
    Lambda(1,k) = refmat(cm,cm,k);  % -1, -1
    Lambda(2,k) = refmat(c0,cm,k);  %  0, -1
    Lambda(3,k) = refmat(cp,cm,k);  % +1, -1
    Lambda(4,k) = refmat(cm,c0,k);  % -1, 0
    Lambda(5,k) = refmat(c0,c0,k);  % 0, 0
    Lambda(6,k) = refmat(cp,c0,k);  % +1, 0
    Lambda(7,k) = refmat(cm,cp,k);  % -1, +1
    Lambda(8,k) = refmat(c0,cp,k);  % 0, +1
    Lambda(9,k) = refmat(cp,cp,k);  % +1, +1
    
    Linit(1,k)  = excmat(cm,c0,k);  % -1, 0
    Linit(2,k)  = excmat(cp,c0,k);  % +1, 0
end

%figure(3);
%plot(del_w,real(Linit(1,:)),'b-'); hold on;
%plot(del_w,imag(Linit(1,:)),'r-');

% Mapping from coherence to index
% [0,-1,+1] -> [1,2,3]
function [ind]=map_pi(coh)

if coh==0
    ind=1;
else if coh==-1
        ind=2;
    else
        ind=3;
    end
end