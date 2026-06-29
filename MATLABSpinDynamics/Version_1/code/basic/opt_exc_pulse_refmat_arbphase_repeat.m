% Repeated optimization runs, store current maximum
function opt_exc_pulse_refmat_arbphase_repeat(nseg,tref,pref,T_90,NE,T_FP,delt)

count=1;
refmat=calc_refocusing_mat(tref,pref,T_90,NE,T_FP);

results=[];
while(1)
    [texc,pexc,delph,echo_pk,echo_rms]=opt_exc_pulse_refmat_arbphase...
        (nseg,T_90,T_FP,refmat,delt,[]); % Fixed phase
    if count==1
        echo_rms_max=echo_rms;
        echo_pk_max=echo_pk;
        save exc_pulse_results_pk.mat texc pexc delph echo_pk echo_rms
        save exc_pulse_results_rms.mat texc pexc delph echo_pk echo_rms
    else
        if echo_rms>echo_rms_max
            echo_rms_max=echo_rms;
            save exc_pulse_results_rms.mat texc pexc delph echo_pk echo_rms
        end
        if echo_pk>echo_pk_max
            echo_pk_max=echo_pk;
            save exc_pulse_results_pk.mat texc pexc delph echo_pk echo_rms
        end
    end
    
    results(count,1:nseg)=texc;
    results(count,nseg+1:2*nseg)=pexc;
    results(count,2*nseg+1)=delph;
    results(count,2*nseg+2)=echo_pk;
    results(count,2*nseg+3)=echo_rms;
    save results.mat results
    
    disp(count)
    disp(echo_pk_max)
    disp(echo_rms_max)
    count=count+1;
end
