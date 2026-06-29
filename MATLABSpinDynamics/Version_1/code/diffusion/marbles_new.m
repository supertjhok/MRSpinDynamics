%/dsk/670/hurliman/pulsecraft/DIFFUSION/PATHWAYS/marbles.m

% this routine calculates all possible ways to put k identical marbles
% into N numbered depressions. (Only one marble per depression, N >= k).
% Output: row of array "pointer' gives number of depressions that contains a marble.

function [pointerplus,pointerminus,imax]=marbles_new(N,k)

index = 1:k;
pointerplus = index;
pointerminus = k+1:N;
istep = N-k;

imax = prod(N-k+1:N)/prod(1:k);

for ip = 2:imax
    if index(k) < N;
        index(k) = index(k) +1;
    else
        istep = max(sign((N-k+1:N)-index).*(1:k));
        index(istep:k) = index(istep)+(1:k-istep+1);
    end
    
    pointerplus(ip,:) = index;
    
    indexminus = (1:index(1)-1);
    for im = 1:k-1
        indexminus = [indexminus (index(im)+1:index(im+1)-1)];
    end
    indexminus = [indexminus (index(k)+1:N)];
    pointerminus(ip,:) = indexminus;
    
end