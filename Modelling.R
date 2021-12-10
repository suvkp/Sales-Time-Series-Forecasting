library(lubridate)
library(xgboost)
library(caret)
library(dplyr)

calendar <- read.csv("calendar_afcs2021.csv")
train <- read.csv("sales_train_validation_afcs2021.csv")
test <- read.csv("sales_test_validation_afcs2021.csv")
sale_price <- read.csv("sell_prices_afcs2021.csv")
sample <- read.csv("sample_submission_afcs2021.csv")

# Transposing
transpose <- function(df) {
  tdf <- t(df)
  col <- df[,1]
  tdf <- as.data.frame(tdf[-1,]) 
  colnames(tdf) <- col  
  tdf <- as.data.frame(tdf)
}

trainT <- transpose(train)
trainT <- tibble::rownames_to_column(trainT, "d")
testT <- transpose(test)
testT <- tibble::rownames_to_column(testT, "d")

# Typecasting
trcolnm <- colnames(trainT)
trcolnm <- trcolnm[-1]
for(cols in trcolnm){
  trainT[,cols] <- as.numeric(trainT[,cols])
}

testcolnm <- colnames(testT)
testcolnm <- testcolnm[-1]
for(cols in testcolnm){
  testT[,cols] <- as.numeric(testT[,cols])
}

# Merging with calendar data
train2 <- merge(calendar, trainT, by = "d")
test2 <- merge(calendar, testT, by = "d")

# Typecasting dates
train2$date <- mdy(train2$date)
test2$date <- mdy(test2$date)

# Dummy variable for events
train2 <- train2 %>%
  mutate(
    event = if_else((event_name_1 != "" | event_type_1 != "" | event_name_2 != "" | event_type_2 != ""), 1, 0)
  )

test2 <- test2 %>%
  mutate(
    event = if_else((event_name_1 != "" | event_type_1 != "" | event_name_2 != "" | event_type_2 != ""), 1, 0)
  )

# Decomposing date to get day column
train2 <- train2 %>% dplyr::mutate(
  day = day(date)
)

test2 <- test2 %>% dplyr::mutate(
  day = day(date)
)

# Keeping required columns
train3 <- train2 %>% dplyr::select(-c(d, wm_yr_wk, date, weekday, event_name_1, event_name_2, event_type_1, event_type_2))
test3 <- test2 %>% dplyr::select(-c(d, wm_yr_wk, date, weekday, event_name_1, event_name_2, event_type_1, event_type_2))

X <- train3 %>% dplyr::select(c(day, month, year, wday, event, snap_CA))
y <- train3 %>% dplyr::select(-c(day, month, year, wday, event, snap_CA))
tst <- test3 %>% dplyr::select(c(day, month, year, wday, event, snap_CA))

# Defining start time
start.time <- Sys.time()

for(c in colnames(y)) {
  
  # train features, train labels and test set
  X_train <- as.matrix(X)
  y_train <- y[[c]]
  X_test <- as.matrix(tst)
  
  # Defining CV parameters
  xgb_trcontrol <- caret::trainControl(
    method = "cv", 
    number = 5,
    allowParallel = TRUE, 
    verboseIter = FALSE, 
    returnData = FALSE
  )
  
  # Defining hyper parameters
  xgb_grid <- base::expand.grid(
    list(
      nrounds = c(100, 200),
      max_depth = c(10, 15, 20), # maximum depth of a tree
      colsample_bytree = seq(0.5), # subsample ratio of columns when construction each tree
      eta = 0.1, # learning rate
      gamma = 0, # minimum loss reduction
      min_child_weight = 1,  # minimum sum of instance weight (hessian) needed ina child
      subsample = 1 # subsample ratio of the training instances
    ))
  
  # Model trainings
  xgb_model <- caret::train(
    X_train, y_train,
    trControl = xgb_trcontrol,
    tuneGrid = xgb_grid,
    method = "xgbTree",
    nthread = 1
  )
  
}

# End time recording
end.time <- Sys.time()
time.taken <- round(end.time - start.time,2)
time.taken


# ******* Test for one product ********
X_train <- as.matrix(X)
y_train <- y[['FOODS_3_001_TX_3_validation']] # as.matrix(y)
X_test <- as.matrix(tst) # xgboost::xgb.DMatrix(as.matrix(tst))

xgb_trcontrol <- caret::trainControl(
  method = "cv", 
  number = 5,
  allowParallel = TRUE, 
  verboseIter = FALSE, 
  returnData = FALSE
)

xgb_grid <- base::expand.grid(
  list(
    nrounds = c(100, 200),
    max_depth = c(10, 15, 20), # maximum depth of a tree
    colsample_bytree = seq(0.5), # subsample ratio of columns when construction each tree
    eta = 0.1, # learning rate
    gamma = 0, # minimum loss reduction
    min_child_weight = 1,  # minimum sum of instance weight (hessian) needed ina child
    subsample = 1 # subsample ratio of the training instances
  ))

xgb_model <- caret::train(
  X_train, y_train,
  trControl = xgb_trcontrol,
  tuneGrid = xgb_grid,
  method = "xgbTree",
  nthread = 1
)

# Checking model parameters
xgb_model$bestTune

# Prediction
xgb_pred <- xgb_model %>% stats::predict(X_test)

# RMSE calculation
m <- test3 %>% dplyr::select(-c(day, month, year, wday, event, snap_CA))
RMSE(xgb_pred, m$FOODS_3_718_TX_3_validation)













