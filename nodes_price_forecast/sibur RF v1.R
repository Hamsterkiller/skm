# Draft

# Nodes
# SPOT_ATS101160_H	SPOT_ATS101161_H

# Station Координаты	lat 58.2436, lon 68.4521
# STUMENE7	Тобольская ТЭЦ	TOBLTETZ	ООО "ТТЭЦ"	ООО "ТТЭЦ"

# GTP
#GTUMENE7	Тобольская ТЭЦ	TOBLTETZ	ООО "ТТЭЦ"	ООО "ТТЭЦ"	FZURTU08:1	1	452	71	1
#GTUMEN57	Тобольская ТЭЦ ТГ-3,ТГ-5	TOBLTETZ	ООО "ТТЭЦ"	ООО "ТТЭЦ"	FZURTU08:1	1	213,3	71	1
#PTUMENE7	Тобольская ТЭЦ	TOBLTETZ	ООО "ТТЭЦ"	ООО "ТТЭЦ"	FZURTU08:1	1		71	1

# RGE
#1172	Тобольская ТЭЦ ТГ- 3	GTUMEN57	Тобольская ТЭЦ ТГ-3,ТГ-5	1	71	1	103,6
#1173	Тобольская ТЭЦ ТГ- 5	GTUMEN57	Тобольская ТЭЦ ТГ-3,ТГ-5	1	71	1	109,7

# Predictors
# Normal temperature Tobolsk meteostation 28275 TEMPAVG_28275_D
# Day off schedule HOLIDAYS_RU_D
# Day of the week
# Solar Day length
# Natural gas price - Not done yet

# Day light
# https://cran.r-project.org/web/packages/geosphere/geosphere.pdf

#### Prediction data set ####
dayStart <- as.Date("2018-05-01") # ENTER START DATE FOR PREDICTION HERE
dayEnd <- as.Date("2018-07-31") # ENTER END DATE FOR PREDICTION HERE
days <- as.numeric(dayEnd - dayStart + 1)

library("geosphere")
dayLength <- rep(NA,days)
dayWeek <- rep(NA,days)
date <- rep(NA,days)

for(i in 1:days){
        dayLength[i] <- daylength(lat = 58.2436, dayStart+i-1)
        dayWeek[i] <- as.POSIXlt(dayStart+i-1)$wday # numeric weekday (0-6 starting on Sunday)
        date[i] <- as.character(dayStart+i-1)
}

temperature <-read.csv(paste("https://exergy.skmenergy.com/api/data/?alt=csv&from=",
                           dayStart,"T00:00&to=",
                           dayEnd,
                           "T23:00&series=TEMPAVG_28275_D", sep=""))

dayOff <- read.csv(paste("https://exergy.skmenergy.com/api/data/?alt=csv&from=",
                         dayStart,"T00:00&to=",
                         dayEnd,
                         "T23:00&series=HOLIDAYS_RU_D", sep=""))    

nodePrice101160 <-rep(NA,24*days)
nodePrice101161 <-rep(NA,24*days)
date <-rep(date,each=24)
hour <-rep(1:24,days)
dayLength <-rep(dayLength,each=24)
dayWeek <- rep(dayWeek, each=24)
dayOff <- rep(dayOff[,2], each=24)
temperature <- rep(temperature[,2],each=24)

predDs <-data.frame(nodePrice101160,nodePrice101161,date,hour,dayLength,dayWeek,dayOff,temperature)

#### Study data set ####
dayStartStudy <- as.Date("2015-01-01") # ENTER START DATE OF STUDY DATA SET HERE
#dayEndStudy <- Sys.Date()-1
dayEndStudy <- as.Date("2018-05-21")
daysStudy <- as.numeric(dayEndStudy - dayStartStudy + 1)

library("geosphere")
dayLength <- rep(NA,daysStudy)
dayWeek <- rep(NA,daysStudy)
date <- rep(NA,daysStudy)

for(i in 1:daysStudy){
        dayLength[i] <- daylength(lat = 58.2436, dayStartStudy+i-1)
        dayWeek[i] <- as.POSIXlt(dayStartStudy+i-1)$wday # numeric weekday (0-6 starting on Sunday)
        date[i] <- as.character(dayStartStudy+i-1)
}

temperature <-read.csv(paste("https://exergy.skmenergy.com/api/data/?alt=csv&from=",
                             dayStartStudy,"T00:00&to=",
                             dayEndStudy,
                             "T23:00&series=TEMPMEAN_28275_D", sep=""))

dayOff <- read.csv(paste("https://exergy.skmenergy.com/api/data/?alt=csv&from=",
                         dayStartStudy,"T00:00&to=",
                         dayEndStudy,
                         "T23:00&series=HOLIDAYS_RU_D", sep=""))    


nodePrice101160 <-read.csv(paste("https://exergy.skmenergy.com/api/data/?alt=csv&from=",
                                 dayStartStudy,"T00:00&to=",
                                 dayEndStudy,
                                 "T23:00&series=SPOT_ATS101160_H", sep=""))[,2]
nodePrice101161 <-read.csv(paste("https://exergy.skmenergy.com/api/data/?alt=csv&from=",
                                 dayStartStudy,"T00:00&to=",
                                 dayEndStudy,
                                 "T23:00&series=SPOT_ATS101161_H", sep=""))[,2]

date <-rep(date,each=24)
hour <-rep(1:24,daysStudy)
dayLength <-rep(dayLength,each=24)
dayWeek <- rep(dayWeek, each=24)
dayOff <- rep(dayOff[,2], each=24)
temperature <- rep(temperature[,2],each=24)

predDsStudy <-data.frame(nodePrice101160,nodePrice101161,date,hour,dayLength,dayWeek,dayOff,temperature)

rm(date,dayLength,dayOff,dayWeek,hour,temperature,i,nodePrice101160,nodePrice101161)



#### Random forest####

plot(predDsStudy$nodePrice101160,type="l")
lines(predDsStudy$nodePrice101161, col=3)

library(randomForest)
#sum(is.na(predDsStudy))

# RF easy
rfModel101160par1 <- randomForest(nodePrice101160 ~ hour + dayLength + dayWeek + dayOff + temperature ,data=predDsStudy)
rfModel101160par1
pred101160par1 <-  predict(rfModel101160par1, predDs)
plot(pred101160par1, type="l", col="red")
rm(rfModel101160par1)

rfModel101161par1 <- randomForest(nodePrice101161 ~ hour + dayLength + dayWeek + dayOff + temperature ,data=predDsStudy)
rfModel101161par1
pred101161par1 <-  predict(rfModel101161par1, predDs)
lines(pred101161par1, col="red")
rm(rfModel101161par1)

# RF hard
rfModel101160par2 <- randomForest(nodePrice101160 ~ hour + dayLength + dayWeek + dayOff + temperature ,data=predDsStudy, ntree = 500,mtry = 3)
rfModel101160par2
pred101160par2 <-  predict(rfModel101160par2, predDs)
lines(pred101160par2, col="green")
rm(rfModel101160par2)

rfModel101161par2 <- randomForest(nodePrice101161 ~ hour + dayLength + dayWeek + dayOff + temperature ,data=predDsStudy, ntree = 500,mtry = 3)
rfModel101161par2
pred101161par2 <-  predict(rfModel101161par2, predDs)
lines(pred101161par2, col="green")
rm(rfModel101161par2)


# Save results
finalDs <- data.frame(predDs[,-c(5,6,7,8)])
finalDs$nodePrice101160 <- round(pred101160par1,2)
finalDs$nodePrice101161 <- round(pred101161par1,2)
write.csv2(finalDs,file = paste(dayStart,dayEnd,"forecast Sibur RF par1 EASY.csv"),row.names = FALSE)

# Save results
finalDs <- data.frame(predDs[,-c(5,6,7,8)])
finalDs$nodePrice101160 <- round(pred101160par2,2)
finalDs$nodePrice101161 <- round(pred101161par2,2)
write.csv2(finalDs,file = paste(dayStart,dayEnd,"forecast Sibur RF par2 HARD.csv"),row.names = FALSE)


