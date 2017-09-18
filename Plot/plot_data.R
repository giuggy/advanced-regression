setwd('C:/Users/Giulia/Uni/FDS')
df <- read.csv('C:/Users/Giulia/Uni/FDS/train.csv')
 
for(i in colnames(df)){
  if(i != 'Id' && i != 'SalePrice'){
    title = paste(i, '.jpg', sep = '')
    jpeg(title)
    boxplot(df$SalePrice ~df[[i]], main=paste(i))
    dev.off()
  }
}

#Elimination of columns

drop <-  c('Bedroom', 'BldgType', 'BsmtFinType1', 'BsmtFinType2', 'BsmtFullBath','HalfBath', 
            'LandSlope', 'LotConfig', 'MoSold', 'Utilities', 'YrSold')
data <- df[,!(names(df)) %in% drop]

write.csv(data, file = "TrainData.csv")
