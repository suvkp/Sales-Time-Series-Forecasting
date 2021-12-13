library(lubridate)
library(xgboost)
library(caret)
library(dplyr)
library(reshape2)
library(data.table)

# --------- DATA LOAD -----------
calendar <- read.csv("calendar_afcs2021.csv")
train <- read.csv("sales_train_validation_afcs2021.csv")
test <- read.csv("sales_test_validation_afcs2021.csv")
sale_price <- read.csv("sell_prices_afcs2021.csv")
sample <- read.csv("sample_submission_afcs2021.csv")

# ----------- DATA PREPARATION --------------
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

# Melting the product id variables into records
trainT <- melt(trainT, id.vars = "d")
testT <- melt(testT, id.vars = "d")

# Generating unique numeric ID for each product
train1 <- trainT %>% mutate(ID = group_indices_(trainT, .dots="variable")) %>% select(-c(variable))
test1 <- testT %>% mutate(ID = group_indices_(testT, .dots="variable")) %>% select(-c(variable))

# Keeping separate dataset with original and new ID
oid <- trainT %>% mutate(ID = group_indices_(trainT, .dots="variable")) %>% select(-c(d, value)) %>% distinct()

# Merging with calendar data
train2 <- merge(calendar, train1, by = "d")
test2 <- merge(calendar, test1, by = "d")

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

# Type-casting to 'value' to numeric columns
train3$value <- as.numeric(train3$value)
test3$value <- as.numeric(test3$value)

# ------- MODELLING ---------

X_trn <- train3 %>% dplyr::arrange(ID) %>% dplyr::select(-c(value))
y_trn <- train3 %>% dplyr::arrange(ID) %>% dplyr::select(c(value))
X_tst <- test3 %>% dplyr::select(-c(value)) %>% dplyr::arrange(ID)
y_tst <- test3 %>% dplyr::arrange(ID) %>% dplyr::select(c(value))

X_train <- as.matrix(X_trn)
y_train <- y_trn$value
X_test <- as.matrix(X_tst)
y_test <- y_tst$value

# preparing matrix
dtrain <- xgb.DMatrix(data=X_train, label=y_train) 
dtest <- xgb.DMatrix(data=X_test, label=y_test)

# default parameters
params <- list(booster = "gbtree", objective = "reg:squarederror", eta=0.3, gamma=0, 
               max_depth=6, min_child_weight=1, subsample=1, colsample_bytree=1)

# cross-validation using above mentioned parameters
xgbcv <- xgb.cv(params = params, data = dtrain, nrounds = 200, nfold = 5, showsd = T, 
                stratified = T, print.every.n = 10, early.stop.round = 20, maximize = F)

# CV metrics
min(xgbcv$test.error.mean)
xgbcv$best_iteration
xgbcv$best_ntreelimit
xgbcv$evaluation_log 

# Training with default parameters
xgb1 <- xgb.train (params = params, data = dtrain, nrounds = 200, watchlist = list(val=dtest,train=dtrain), 
                   print.every.n = 10, early.stop.round = 10, maximize = F , eval_metric = "rmse")

# ---------- HYPERPARAMETER TUNING USING RANDOM SEARCH -----------
library(mlr)

# Task creation
traintask <- makeRegrTask(data = train3, target = "value")
testtask <- makeRegrTask(data = test3, target = "value")

#create learner
lrn <- makeLearner("regr.xgboost",predict.type = "response")
lrn$par.vals <- list( objective="reg:squarederror", eval_metric="rmse", nrounds=100L)

#set parameter space
params <- makeParamSet( makeDiscreteParam("booster", values = c("gbtree","gblinear")), 
                        makeIntegerParam("max_depth",lower = 3L,upper = 10L), 
                        makeNumericParam("min_child_weight",lower = 1L,upper = 10L), 
                        makeNumericParam("subsample",lower = 0.5,upper = 1), 
                        makeNumericParam("colsample_bytree",lower = 0.5,upper = 1),
                        makeNumericParam("eta",lower = .01,upper = 1L))


#set resampling strategy
rdesc <- makeResampleDesc("CV",stratify = F,iters=10L)

# Random search with 10 different XGB models
ctrl <- makeTuneControlRandom(maxit = 20L)

#set parallel backend
library(parallel)
library(parallelMap) 
parallelStartSocket(cpus = detectCores())

#parameter tuning
mytune <- tuneParams(learner = lrn, task = traintask, resampling = rdesc, measures = rmse, 
                     par.set = params, control = ctrl, show.info = T)

#set hyperparameters
lrn_tune <- setHyperPars(lrn, par.vals = mytune$x)

#train model
xgmodel <- train(learner = lrn_tune,task = traintask)