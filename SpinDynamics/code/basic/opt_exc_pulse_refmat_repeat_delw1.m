% Repeated optimization runs, store current maximum
function opt_exc_pulse_refmat_repeat_delw1(nseg,tref,pref,T_90,del_w1,NE,T_FP,delt)

count=1;
refmat=calc_refocusing_mat_delw1(tref,pref,T_90,del_w1,NE,T_FP);

results=[];
while(1)
    [texc,pexc,echo_pk,echo_rms]=opt_exc_pulse_refmat_delw1(nseg,T_90,del_w1,T_FP,refmat,delt,[]); % Fixed phases
    if count==1
        echo_rms_max=echo_rms;
        echo_pk_max=echo_pk;
        save exc_pulse_results_pk.mat texc pexc echo_pk echo_rms
        save exc_pulse_results_rms.mat texc pexc echo_pk echo_rms
    else
        if echo_rms>echo_rms_max
            echo_rms_max=echo_rms;
            save exc_pulse_results_rms.mat texc pexc echo_pk echo_rms
        end
        if echo_pk>echo_pk_max
            echo_pk_max=echo_pk;
            save exc_pulse_results_pk.mat texc pexc echo_pk echo_rms
        end
    end
    
    results(count,1:nseg)=texc;
    results(count,nseg+1:2*nseg)=pexc;
    results(count,2*nseg+1)=echo_pk;
    results(count,2*nseg+2)=echo_rms;
    save results_delw1.mat results
    
    disp(count)
    disp(echo_pk_max)
    disp(echo_rms_max)
    count=count+1;
end
