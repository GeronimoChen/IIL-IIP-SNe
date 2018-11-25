library(fpca)
getwd()
setwd('Your_Directory/FPCA-Halpha/')
dating<-read.csv('Rdata.csv')
dating2<-read.csv('restdata.csv')
dating<-apply(dating,2,as.numeric)
dating2<-apply(dating2,2,as.numeric)

result<-fpca.mle(dating,55,30,grids=seq(0,1,0.002))
grids.new<-result$grid
r<-result$selected_model[2]
M<-result$selected_model[1]
evalest<-result$eigenvalues
sig2est<-result$error_var
eigenfest<-result$eigenfunctions
muest<-result$fitted_mean
fpcs<-fpca.score(dating,grids.new,muest,evalest,eigenfest,sig2est,r)
fpcs2<-fpca.score(dating2,grids.new,muest,evalest,eigenfest,sig2est,r)
pred<-fpca.pred(fpcs,muest,eigenfest)

write.csv(fpcs, file="FPCS.csv",row.names=F,quote=F)
write.csv(fpcs2,file="FPCS2.csv",row.names=F,quote=F)
write.csv(result$eigenfunctions,file="EigFuns.csv",row.names =F,quote=F)
write.csv(result$grid,file="GRID.csv",row.names=F)
write.csv(muest,file="muest.csv",row.names=F)



