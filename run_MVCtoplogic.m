clear 
close all
para_alpha = [0.01, 0.05, 0.1, 0.5, 1, 2, 5, 10];
para_beta = [0.01, 0.05, 0.1, 0.5, 1, 2, 5, 10]; 
load('3Sources.mat'); 
% res = [acc, nmi, Pu, Fscore, Precision, Recall, ARI];
[result, S, Tim, Obj] = MVCtoplogic(data,labels, para_alpha(8), para_beta(7));

