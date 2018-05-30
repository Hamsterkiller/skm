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
library(lubridate)

# prediction period dates
dayStart <- as.Date("2018-05-01") # ENTER START DATE FOR PREDICTION HERE
dayEnd <- as.Date("2018-07-31") # ENTER END DATE FOR PREDICTION HERE
days <- as.numeric(dayEnd - dayStart + 1)

# train set dates
dayStartStudy <- as.Date("2015-01-01")
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

# converting to msts
nodePrice101160 <- msts(nodePrice101160, seasonal.periods = c(24, 168, 8736), 
                        start = decimal_date(as.Date("2015-01-01")))
nodePrice101161 <- msts(nodePrice101161, seasonal.periods = c(24, 168, 8736))


# fitting tbats multiple seasonality model
model_0 <- tbats(nodePrice101160) 
model_1 <- tbats(nodePrice101161, seasonal.periods = c(24, 168, 8736))

# forecasting
fcst_0 <- forecast(model_0, h = days * 24)
fcst_0 <- forecast(model_0)
fcst_1 <- forecast(model_1, h = days * 24)

plot(head(fcst_0$mean, 504), type="l", col = "red")
plot(tail(nodePrice101160, 504), type="l", col="blue")

fcst_1 <- predict(model_1, days * 24)

nodePrice101160_fcst <- xts(as.vector(fcst_0$fitted), order.by = time_index_fcst, unique = TRUE)
nodePrice101161_fcst <- xts(as.vector(nodePrice101161_fcst), order.by = time_index_fcst, unique = TRUE)
