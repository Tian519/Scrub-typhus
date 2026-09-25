rm(list=ls());
gc() 

library(stringr)
library(lubridate) # extract date (year,month ....)
library(tidyr) # unite function
library(dplyr)
library(mgcv)   #gam 
library(xgboost)
library(dismo)
library(gbm)

outDir <- "out_path"

#load model
load("RF_ST_all.RData")


#import future data
dataDir <- "Final_futuredata"
fileNames <- dir(dataDir) 
filePath <- sapply(fileNames, function(x){paste(dataDir,x,sep='\\')}) 
filenum <- length(fileNames)


predict_popchanged = function(i){
  data <- read.csv(filePath[i])
  data$pop_log <- log10(data$POP)
  #data$GDP <- ifelse(is.na(data$GDP), data$GDP2015, data$GDP)
  #data <- rename(data, 'GDP_SSP2' = 'GDP')
  
  data$deforestation <- ifelse(data$forest_change > 0, 0, data$forest_change)
  data$Grassland_degradation <- ifelse(data$grass_change > 0, 0, data$grass_change)
  data$deforestation_abs <- abs(data$deforestation)
  data$Grassland_degradation_abs <- abs(data$Grassland_degradation)
  data$Urban_development <- ifelse(data$Urban_change < 0, 0, data$Urban_change)
  data$Cultivation_reduction <- ifelse(data$Cropland_change > 0, 0, data$Cropland_change)
  data$Cultivation_reduction_abs <- abs(data$Cultivation_reduction)
  
  data$Ld <- as.factor(data$Ld)
  data$Ls <- as.factor(data$Ls)
  data$Month <- as.factor(data$month)
  
  data = data[order(data[,2],data[,4],data[,7]),] 
  
  pred_all <-data[,c('CODE','Year.x','Month')]
  
  for (j in (1:20)){
    print(j)
    rf_model = RF_list[[j]]
    #all data
    oob_all_0 <- predict(rf_model, newdata=data)
    predicted1 <- data.frame(oob_all_0$predicted)
    pred_all <- cbind(pred_all, predicted1)
  }
  
  pred_all$Cases <- apply(pred_all[,4:23], 1, mean)
  pred_all2 <- pred_all[,c('CODE', 'Year.x', 'Month','Cases')]
  
  outputfile = paste(outDir, "pop_changed02",fileNames[i], sep="\\")
  write.csv(pred_all2, outputfile)
}

for (filepath in 1:filenum){
  print(filepath)
  predict_popchanged(filepath)
}



