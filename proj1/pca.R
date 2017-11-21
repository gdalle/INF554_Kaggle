setwd(dirname(rstudioapi::getActiveDocumentContext()$path))

library(FactoMineR)
library(readr)
data <- as.data.frame(read_csv("data/train.csv"))
rownames(data) <- data$id
data$id <- NULL

na_find <- function(v){
  v[which(v==-1)] <- NA
  return(v)
}

data <- as.data.frame(sapply(data,na_find))

bin_var <- which(grepl("^.*bin$",colnames(data)))
cat_var <- which(grepl("^.*cat$",colnames(data)))
quanti_var <- which(!(grepl("^.*bin$",colnames(data))) & !(grepl("^.*cat$",colnames(data))))
quanti_var <- quanti_var[c(2:length(quanti_var))]
data$target = as.factor(data$target)

N_0 = 1000
N_1 = 100

idx_0 = sample(which(data$target==0),N_0)
idx_1 = sample(which(data$target==1),N_1)


res.pca <- PCA(data[c(idx_0,idx_1),c(1,quanti_var)],scale.unit=TRUE,quali.sup=c(1),ncp=5,graph=FALSE)
plot.PCA(res.pca,habillage=1,label="none",invisible="quali",col.hab=c("black","red"))
plot(res.pca,choix="var",select="cos2 0.6")
