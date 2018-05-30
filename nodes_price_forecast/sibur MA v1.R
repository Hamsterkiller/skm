#### LOADING NODE PRICES DATA ####
rm(list = ls())
java_path = Sys.getenv("JAVA_HOME")
Sys.setenv(JAVA_HOME=java_path)

# Checking Java version
system("java -version")

# Loading libraries
library(rJava)
library(RJDBC)
library(tsauxfunc)
library(dplyr)
library(RPostgreSQL)
library(xts)
library(forecast)
library(TTR)

# Create connection driver and open connection
## ORACLE:
pg = dbDriver("PostgreSQL")
con = dbConnect(pg, user="zemskov", password="GrandPik9",
                host="skm-moscow-server", port=5432, dbname="skmrus")
dbDisconnect(con)



# prediction period dates
dayStart <- as.Date("2018-05-01") # ENTER START DATE FOR PREDICTION HERE
dayEnd <- as.Date("2018-07-31") # ENTER END DATE FOR PREDICTION HERE
days <- as.numeric(dayEnd - dayStart + 1)

# train set dates
dayStartStudy <- as.Date("2016-01-01")
dayEndStudy <- as.Date("2018-05-21")
daysStudy <- as.numeric(dayEndStudy - dayStartStudy + 1)

# load fact prices
nodePrice101160 <-read.csv(paste("https://exergy.skmenergy.com/api/data/?alt=csv&from=",
                                 dayStartStudy,"T00:00&to=",
                                 dayEndStudy,
                                 "T23:00&series=SPOT_ATS101160_H", sep=""))[,2]
nodePrice101161 <-read.csv(paste("https://exergy.skmenergy.com/api/data/?alt=csv&from=",
                                 dayStartStudy,"T00:00&to=",
                                 dayEndStudy,
                                 "T23:00&series=SPOT_ATS101161_H", sep=""))[,2]

# converting to xts
time_index <- seq(from = as.POSIXct("2016-01-01 00:00"), 
                  to = as.POSIXct("2018-05-21 23:00"), by = "hour")
nodePrice101160 <- xts(nodePrice101160, order.by = time_index, unique = TRUE)
nodePrice101161 <- xts(nodePrice101161, order.by = time_index, unique = TRUE)
attr(nodePrice101160, 'frequency') <- 168 # 24 - for dayly seasonality, 168 - for weekly, 8736 - yearly
attr(nodePrice101161, 'frequency') <- 168 # 24 - for dayly seasonality, 168 - for weekly, 8736 - yearly

# convert to ts objects
nodePrice101160_ts <- as.ts(nodePrice101160)
nodePrice101161_ts <- as.ts(nodePrice101161)

# Building models 
model_0 <- HoltWinters(nodePrice101160_ts, beta = FALSE, seasonal = "add") # Exponential smoothing
model_1 <- HoltWinters(nodePrice101161_ts, beta = FALSE, seasonal = "add") # Exponential smoothing

# forecasting prices for days * 24 hours
time_index_fcst <- seq(from = as.POSIXct("2018-05-01 00:00"), 
                       to = as.POSIXct("2018-07-31 23:00"), by = "hour")
nodePrice101160_fcst <- predict(model_0, days * 24)
nodePrice101161_fcst <- predict(model_1, days * 24)

nodePrice101160_fcst <- xts(as.vector(nodePrice101160_fcst), order.by = time_index_fcst, unique = TRUE)
nodePrice101161_fcst <- xts(as.vector(nodePrice101161_fcst), order.by = time_index_fcst, unique = TRUE)

# check fact vs forecasted values for may2018
fact_may <- tail(nodePrice101161, 504)
fcst_may <- head(nodePrice101161_fcst, 504)

plot(fact_may, type="l", col = "red")
lines(fcst_may, type="l", col ="blue")

write.csv2(nodePrice101160_fcst, file = "node_101160_fcst_hw.csv", row.names = FALSE)
write.csv2(nodePrice101161_fcst, file = "node_101161_fcst_hw.csv", row.names = FALSE)

