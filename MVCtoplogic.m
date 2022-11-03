% Multi-view Clustering on Topological Manifold
function [result, S, Tim, Obj] = MVCtoplogic(data,labels, alpha, beta, knn, normData)
% [result, mu, G, Z, S, lambda] = MVCtoplogic(data,labels, alpha, beta, knn, normData)
% data: cell array, view_num by 1, each array is num_samp by d_v
% num_clus: number of clusters
% num_view: number of views
% num_samp
% k: number of adaptive neighbours
% labels: groundtruth of the data, num by 1
%
% figure()
% [S1,~] = mapminmax(S, 0, 1)
% imshow(S1) 
% figure()
% [S1,~] = mapminmax(S, 0, 1)
% imshow(S1,'InitialMagnification','fit')
% colormap('jet');
%
if nargin < 3
    alpha = 1;
end
if nargin < 4
    beta = 1;
end
if nargin < 5
    knn = 20;
end
if nargin < 6
    normData = 1;
end
num_view = size(data,1);
num_samp = size(labels,1);
num_clus = length(unique(labels));
mu = 1/num_view*ones(1,num_view);
% lambda = randperm(30,1);
lambda = 1;
NITER = 30;
zr = 1e-10;

% =====================   Normalization =====================
if normData == 1
    for i = 1 :num_view
        for  j = 1:num_samp
            normItem = std(data{i}(j,:));
            if (0 == normItem)
                normItem = eps;
            end
            data{i}(j,:) = (data{i}(j,:) - mean(data{i}(j,:)))/normItem;
        end
    end
end

%  ====== Initialization =======
% claculate G_v for all the views
G = cell(num_view,1);
sumG = zeros(num_samp);
for v = 1:num_view
    Gv = constructW_PKN(data{v}',knn);
    G{v} = Gv;
    sumG = sumG + Gv;
    clear Gv
end 
tic; 
% initialize S
S = sumG/num_view;
% initialize F
S0 = S-diag(diag(S));
w0 = (S0+S0')/2;
D0 = diag(sum(w0));
L = D0 - w0;
[F0,~,~] = eig1(L,num_clus,0);
F = F0(:,1:num_clus);
I = eye(num_samp);
% update ...
for iter = 1:NITER
    % update Z_v
    Z = cell(num_view,1);
    for v = 1:num_view
        A = (alpha + beta*mu(v)^2)*I + L;
        for ni = 1:num_samp
            index = find(G{v}(ni,:)>0);
            Ii = I(ni,index);
            si = S(ni,index);
            b = 2*alpha*Ii + 2*beta*mu(v)*si;
            % solve z^T*A*z-z^T*b
            [zi, ~] = fun_alm(A(index,index),b);
            Z{v}(ni,index) = zi';
        end
    end
    
    % update S
    dist_h = L2_distance_1(F',F');
    S = zeros(num_samp);
    for ni = 1:num_samp
        sumZi = zeros(1,num_samp);
        for v = 1:num_view
            Zv = Z{v};
            sumZi = sumZi + mu(v)*Zv(ni,:);
        end
        index = find(sumZi>0); 
        ad = sumZi(index) - (lambda/(2*beta))*dist_h(ni,index);
        S(ni,index) = EProjSimplex_new(ad);
    end
    
    % update \mu_v
    sumTrSZZ = 0;
    sumZZ = 0;
    for v = 1:num_view
        SZv(v) = trace(S*Z{v}');
        ZvZv(v) = trace(Z{v}*Z{v}');
        sumTrSZZ = sumTrSZZ + SZv(v)/ZvZv(v);
        sumZZ = sumZZ + 1/(2*ZvZv(v));
    end
    gamma = (sumTrSZZ-1)/sumZZ;
    for v =1:num_view
        mu(v) = (2*SZv(v)-gamma)/(2*ZvZv(v));
    end
    mu = max(mu,eps);
%     if min(mu)<0
%        fprintf('the %d -th iteration ->\n',iter)
%        fprintf('the weight of each view is %f\n',mu)
%     end
    
    % update F
    S = (S+S')/2;                                                      
    D = diag(sum(S));
    L = D-S;
    F_old = F;
    [F, ~, ev]=eig1(L, num_clus, 0);
    
    % calculate obj
    obj = 0;
    Obj = [];
    for v = 1:num_view
        obj = obj + norm(Z{v}'*L*Z{v},'fro')^2; % + alpha*norm(Z{v}-I,'fro')^2 + beta*norm(S-mu(v)*Z{v},'fro')^2; 
    end
    Obj(iter) = obj;
    
    fn1 = sum(ev(1:num_clus));
    fn2 = sum(ev(1:num_clus+1));
    if fn1 > zr
        lambda = 2*lambda;
    elseif fn2 < zr
        lambda = lambda/2;  
        F = F_old;
    else
        fprintf('the %d -th iteration -> end ...\n',iter)
        break;
    end 
end
Tim = toc;
% =====================  result =====================
[clusternum, y]=graphconncomp(sparse(S)); 
y = y';
if clusternum ~= num_clus
    sprintf('Can not find the correct cluster number: %d', num_clus)
end
result = EvaluationMetrics(labels, y);
end


function [v, obj] = fun_alm(A,b)
if size(b,1) == 1
    b = b';
end

% initialize
rho = 1.5;
mu = 30;
n = size(A,1);
alpha = ones(n,1);
v = ones(n,1)/n;
% obj_old = v'*A*v-v'*b;

obj = [v'*A*v-v'*b];
iter = 0;
while iter < 10
    % update z
    z = v-A'*v/mu+alpha/mu;

    % update v
    c = A*z-b;
    d = alpha/mu-z;
    mm = d+c/mu;
    v = EProjSimplex_new(-mm);

    % update alpha and mu
    alpha = alpha+mu*(v-z);
    mu = rho*mu;
    iter = iter+1;
    obj = [obj;v'*A*v-v'*b];
end
end


function [x] = EProjSimplex_new(v, k)
%
% Problem
%
%  min  1/2 || x - v||^2
%  s.t. x>=0, 1'x=k
%
if nargin < 2
    k = 1;
end;

ft=1;
n = length(v);

v0 = v-mean(v) + k/n;
%vmax = max(v0);
vmin = min(v0);
if vmin < 0
    f = 1;
    lambda_m = 0;
    while abs(f) > 10^-10
        v1 = v0 - lambda_m;
        posidx = v1>0;
        npos = sum(posidx);
        g = -npos;
        f = sum(v1(posidx)) - k;
        lambda_m = lambda_m - f/g;
        ft=ft+1;
        if ft > 100
            x = max(v1,0);
            break;
        end
    end
    x = max(v1,0);

else
    x = v0;
end
end
