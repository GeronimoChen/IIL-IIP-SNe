library(fpca)
library(parallel)
cl<-makeCluster(4)

getwd()
setwd('F:/Project LiuXuewen/Deep Learning Supernovae Spectra/Filtering/')
disintegratee<-read.csv('DisInt.csv')
disintegratee<-apply(disintegratee,2,as.numeric)

print('To notice, please load the workspace before running the following codes.')
disintbase<-fpca.score(disintegratee,grids.new,muest,evalest,eigenfest,sig2est,r)
disintpred<-fpca.pred(disintegratee,muest,eigenfest)

write.csv(disintbase,file='FPCSplus.csv',row.names=F,quote=F)
