rm(list = ls())

# RNN model with LSTM unit

dayStart <- as.Date("2018-06-01") # ENTER START DATE FOR PREDICTION HERE
dayEnd <- as.Date("2018-08-31") # ENTER END DATE FOR PREDICTION HERE
days <- as.numeric(dayEnd - dayStart + 1)

# creating dummy vectors for the features to extract later
library("geosphere")  # for computing length of the day
dayLength <- rep(NA,days)
dayWeek <- rep(NA,days)
date <- rep(NA,days)


for(i in 1:days){
  dayLength[i] <- daylength(lat = 58.2436, dayStart+i-1)
  dayWeek[i] <- as.POSIXlt(dayStart+i-1)$wday # numeric weekday (0-6 starting on Sunday)
  date[i] <- as.character(dayStart+i-1)
}

# temperature from exergy series
temperature <-read.csv(paste("https://exergy.skmenergy.com/api/data/?alt=csv&from=",
                             dayStart,"T00:00&to=",
                             dayEnd,
                             "T23:00&series=TEMPAVG_28275_D", sep=""))
# holidays calendar
dayOff <- read.csv(paste("https://exergy.skmenergy.com/api/data/?alt=csv&from=",
                         dayStart,"T00:00&to=",
                         dayEnd,
                         "T23:00&series=HOLIDAYS_RU_D", sep=""))    

# dummy vectors for node prices by hours
nodePrice101160 <-rep(NA,24*days)
nodePrice101161 <-rep(NA,24*days)

# replicating for hour-data
date <-rep(date,each=24)
hour <-rep(1:24,days)

dayLength <-rep(dayLength,each=24)
dayWeek <- rep(dayWeek, each=24)
dayOff <- rep(dayOff[,2], each=24)

temperature <- rep(temperature[,2],each=24)

# creating testing set (for predictions)
predDs <-data.frame(nodePrice101160,nodePrice101161,date,hour,dayLength,dayWeek,dayOff,temperature)

#### Study data set ####
dayStartStudy <- as.Date("2015-01-01") # ENTER START DATE OF STUDY DATA SET HERE
#dayEndStudy <- Sys.Date()-1
dayEndStudy <- as.Date("2018-05-31")
daysStudy <- as.numeric(dayEndStudy - dayStartStudy + 1)

# replicate for hour-data for training set
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

# imputing missing data (with median values of the temperature in the corresponding month)
library("lubridate")

# create month variable containing 1st day of every month
temperature$date <- as.POSIXlt(temperature$date, format="%m/%d/%Y")
temperature$month <- floor_date(temperature$date, "month")

# create empty dataframe for later filling
result_df <- data.frame(matrix(ncol = ncol(temperature), nrow = 0))
colnames(result_df) <- colnames(temperature)
impute.var <- colnames(temperature)[2]
months <- unique(temperature$month)

# imputing loop
for (m in 1:length(months)) {
  df_m <- temperature[temperature$month == months[m], ]
  if (length(df_m[which(is.na(df_m[, impute.var]) == TRUE), impute.var]) > 0) {
    print("Imputing missing data...")
    df_m[which(is.na(df_m[, impute.var]) == TRUE), impute.var] = 
      median(df_m[which(is.na(df_m[, impute.var]) == FALSE), impute.var])
  }
  result_df <- rbind(result_df, df_m)
}

temperature <- result_df[, colnames(temperature)[1: 2]]

# loading holiday and price historical data from Exergy
dayOff <- read.csv(paste("https://exergy.skmenergy.com/api/data/?alt=csv&from=",
                         dayStartStudy,"T00:00&to=",
                         dayEndStudy,
                         "T23:00&series=HOLIDAYS_RU_D", sep=""))    
nodePrice101160 <- read.csv(paste("https://exergy.skmenergy.com/api/data/?alt=csv&from=",
                                 dayStartStudy,"T00:00&to=",
                                 dayEndStudy,
                                 "T23:00&series=SPOT_ATS101160_H", sep=""))[,2]
nodePrice101161 <- read.csv(paste("https://exergy.skmenergy.com/api/data/?alt=csv&from=",
                                 dayStartStudy,"T00:00&to=",
                                 dayEndStudy,
                                 "T23:00&series=SPOT_ATS101161_H", sep=""))[,2]

date <- rep(date,each=24)
hour <- rep(1:24,daysStudy)
dayLength <- rep(dayLength,each=24)
dayWeek <- rep(dayWeek, each=24)
dayOff <- rep(dayOff[,2], each=24)
temperature <- rep(temperature[,2],each=24)

predDsStudy <- data.frame(nodePrice101160,nodePrice101161,date,hour,dayLength,dayWeek,dayOff,temperature)

#rm(date, dayLength, dayOff, dayWeek, hour, temperature, i, nodePrice101160, nodePrice101161)

# adding lag of price variables
library(Hmisc)
# train + test set
fullDs <- rbind(predDsStudy, predDs)
fullDs$nodePrice101160_lag1 <- Lag(fullDs$nodePrice101160, shift = 1)
fullDs$nodePrice101161_lag1 <- Lag(fullDs$nodePrice101161, shift = 1)
fullDs$nodePrice101160_lag2 <- Lag(fullDs$nodePrice101160, shift = 2)
fullDs$nodePrice101161_lag2 <- Lag(fullDs$nodePrice101161, shift = 2)
fullDs$nodePrice101160_y0 <- Lag(fullDs$nodePrice101160, shift = 8736)
fullDs$nodePrice101161_y0 <- Lag(fullDs$nodePrice101161, shift = 8736)
fullDs$nodePrice101160_y1 <- Lag(fullDs$nodePrice101160, shift = 8737)
fullDs$nodePrice101161_y1 <- Lag(fullDs$nodePrice101161, shift = 8737)
fullDs$nodePrice101160_y2 <- Lag(fullDs$nodePrice101160, shift = 8738)
fullDs$nodePrice101161_y2 <- Lag(fullDs$nodePrice101161, shift = 8738)
fullDs <- fullDs[as.Date(fullDs$date) != as.Date("2016-02-29"), ] # delete 29.02.2016 data

fullDs <- subset(fullDs, select = -c(date))

# normalizing train data
normalize <- function(df.column) {
  scaled <- (df.column - mean(na.omit(df.column))) / (max(na.omit(df.column)) - min(na.omit(df.column)))
  return(scaled)
}

# for later invert transformation
max_price_0 <- max(na.omit(fullDs$nodePrice101160))
min_price_0 <- min(na.omit(fullDs$nodePrice101160))
avg_price_0 <- mean(na.omit(fullDs$nodePrice101160))
max_price_1 <- max(na.omit(fullDs$nodePrice101160))
min_price_1 <- min(na.omit(fullDs$nodePrice101160))
avg_price_1 <- mean(na.omit(fullDs$nodePrice101160))

# normalizing
fullDs <- sapply(fullDs, FUN=normalize)

# separate them back
predDsStudy <- fullDs[1:(dim(predDsStudy)[1] - 24), ]
predDs <- fullDs[(dim(predDsStudy)[1] + 1): dim(fullDs)[1], ]

# delete all rows with NAs
predDsStudy <- na.omit(predDsStudy)

X_0 <- subset(predDsStudy, select=-c(nodePrice101160, nodePrice101161, nodePrice101161_lag1, nodePrice101161_lag2, 
                                     nodePrice101161_y0, nodePrice101161_y1, nodePrice101161_y2))
X_1 <- subset(predDsStudy, select=-c(nodePrice101160, nodePrice101161, nodePrice101160_lag1, nodePrice101160_lag2, 
                                     nodePrice101160_y0, nodePrice101160_y1, nodePrice101160_y2))
y_0 <- subset(predDsStudy, select=c(nodePrice101160))
y_1 <- subset(predDsStudy, select=c(nodePrice101161))
target_0 <- "nodePrice101160"
target_1 <- "nodePrice101161"

features_0 <- colnames(X_0)
features_1 <- colnames(X_1)

# normalizing test data
X_test_0 <- subset(predDs, select=-c(nodePrice101160, nodePrice101161, nodePrice101161_lag1, nodePrice101161_lag2, 
                                     nodePrice101161_y0, nodePrice101161_y1, nodePrice101161_y2))
X_test_1 <- subset(predDs, select=-c(nodePrice101160, nodePrice101161, nodePrice101160_lag1, nodePrice101160_lag2, 
                                     nodePrice101160_y0, nodePrice101160_y1, nodePrice101160_y2))

# reshape X_i to 3 dimensions
dim(X_0) <- c(dim(X_0)[1], 1, dim(X_0)[2])
dim(X_1) <- c(dim(X_1)[1], 1, dim(X_1)[2])
# dim(X_test_0) <- c(dim(X_test_0)[1], 1, dim(X_test_0)[2])
# dim(X_test_1) <- c(dim(X_test_1)[1], 1, dim(X_test_1)[2])

# defining RNN model for node 101160
library(keras)
rnn_lstm_model_0 <- keras_model_sequential()
rnn_lstm_model_1 <- keras_model_sequential()
rnn_lstm_model_0 %>%
  layer_lstm(units = 1, input_shape = c(1, 10)) %>%
  layer_dense(1)
# summary(rnn_lstm_model)
rnn_lstm_model_1 %>%
  layer_lstm(units = 1, input_shape = c(1, 10)) %>%
  layer_dense(1)
# summary(rnn_lstm_model)

rnn_lstm_model_0 %>% compile(
  loss = 'mae',
  optimizer = 'adam'
)
set.seed(42)
rnn_lstm_model_1 %>% compile(
  loss = 'mae',
  optimizer = 'adam'
)
set.seed(42)

# for 101160 node
history_101160 <- rnn_lstm_model_0 %>% fit(
  X_0, y_0,
  epochs = 30, batch_size = 128,
  validation_split = 0.2
)

# for 101161 node
history_101161 <- rnn_lstm_model_1 %>% fit(
  X_1, y_1,
  epochs = 30, batch_size = 128,
  validation_split = 0.2
)

# make predictions
plot(history_101160)


# makes test row for one prediction iteration
predict_prices = function(df, model) {
  
  y <- c()
  
  for (i in 1:dim(df)[1]) {
    row <- df[i, ]
    if (i == 1) {
      dim(row) <- c(1, 1, 10)
      y[i] <- model %>% predict(row, batch_size=1) %>% .[,1]
    } else if (i == 2) {
      row[6] = y[i-1]
      dim(row) <- c(1, 1, 10)
      y[i] <- model %>% predict(row, batch_size=1) %>% .[,1]
    } else {
      row[6] = y[i-1]
      row[7] = y[i-2]
      dim(row) <- c(1, 1, 10)
      y[i] <- model %>% predict(row, batch_size=1) %>% .[,1]
    }
  }
  return (y)
}

prediction_0 <- predict_prices(X_test_0, rnn_lstm_model_0)

# invert transformation
prediction_0 <- prediction_0 * (max_price_0 - min_price_0) + avg_price_0

plot(head(prediction_0, 192), type="line")
