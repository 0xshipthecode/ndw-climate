close all
load ../results/nao_correlation_components_01.mat

% connectivity
figure;
subplot(2,3,1);
imagesc(Dnao_minus);
title('NAO-');
subplot(2,3,2);
imagesc(Dnao_plus);
title('NAO+');
subplot(2,3,3);
imagesc(Dnao_plus - Dnao_minus);
title('Diff');
subplot(2,3,4);
imagesc(Cnao_minus.^2);
title('NAO-');
subplot(2,3,5);
imagesc(Cnao_plus.^2);
title('NAO+');
subplot(2,3,6);
imagesc(Cnao_plus.^2 - Cnao_minus.^2);
title('Diff');


% degree distribution
figure;
subplot(3, 1, 1);
bar(degrees_und(Dnao_minus));
title('NAO- DD');
subplot(3, 1, 2);
bar(degrees_und(Dnao_plus));
title('NAO+ DD');
subplot(3, 1, 3);
bar(degrees_und(Dnao_plus) - degrees_und(Dnao_minus));
title('DD diff');
mean(degrees_und(Dnao_plus) - degrees_und(Dnao_minus))

% clustering coefficient
figure;
subplot(3, 1, 1);
bar(clustering_coef_bu(Dnao_minus));
title('NAO- CC');
subplot(3, 1, 2);
bar(clustering_coef_bu(Dnao_plus));
title('NAO+ CC');
subplot(3, 1, 3);
bar(clustering_coef_bu(Dnao_plus) - clustering_coef_bu(Dnao_minus));
title('CC diff');
mean(clustering_coef_bu(Dnao_plus) - clustering_coef_bu(Dnao_minus))

% betweenness centrality
figure;
subplot(3, 1, 1);
bar(betweenness_bin(Dnao_minus));
title('NAO- BC');
subplot(3, 1, 2);
bar(betweenness_bin(Dnao_plus));
title('NAO+ BC');
subplot(3, 1, 3);
bar(betweenness_bin(Dnao_plus) - betweenness_bin(Dnao_minus));
title('BC diff');
mean(betweenness_bin(Dnao_plus) - betweenness_bin(Dnao_minus))

% avg path length
mask = triu(true(45));
dist_np = distance_bin(Dnao_plus);
dist_np(dist_np == inf) = nan;
dist_nm = distance_bin(Dnao_minus);
dist_nm(dist_nm == inf) = nan;
nanmean(dist_np(mask)) - nanmean(dist_nm(mask))

% char path
[plambda, peff, ~, prad, pdiam] = charpath(Dnao_plus)
[mlambda, meff, ~, mrad, mdiam] = charpath(Dnao_minus)
