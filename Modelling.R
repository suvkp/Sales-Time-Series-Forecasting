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

X <- train3 %>% dplyr::arrange(ID) %>% dplyr::select(-c(value))
y <- train3 %>% dplyr::arrange(ID) %>% dplyr::select(c(value))
tst <- test3 %>% dplyr::select(-c(value)) %>% dplyr::arrange(ID)

X_train <- as.matrix(X)
y_train <- y$value
X_test <- as.matrix(tst)

set.seed(123)

# ------- TRAINING ---------
xgb_model <- xgboost(data = X_train, 
        label = y_train, 
        eta = 0.1,
        max_depth = 10, 
        nround=1000, 
        eval_metric = "rmse",
        objective = "reg:squarederror",
        nthread = 3
)

# -------- PREDICTION ----------
predval <- predict(xgb_model, newdata = X_test)
prod_pred <- cbind(tst$ID, predval)
prod_pred <- data.frame(prod_pred)
colnames(prod_pred) <- c('ID', 'predval')

preddf <- merge(prod_pred, oid, by = 'ID')

preddf <- preddf %>% group_by(variable) %>% dplyr::mutate(forecast = row_number())

output <- preddf %>% tidyr::spread(forecast, predval)
output <- output %>% select(-ID) 
colnames(output) <- c('id','F1','F2','F3','F4','F5','F6','F7','F8','F9','F10','F11','F12',
                      'F13','F14','F15','F16','F17','F18','F19','F20','F21','F22','F23','F24','F25','F26','F27','F28')

# --------- FINAL DATASET -----------

write.csv(output, 'output.csv')

# RMSE calculation
# m <- test3 %>% dplyr::select(-c(day, month, year, wday, event, snap_CA)) %>% dplyr::arrange(ID)
# RMSE(predval, m$value)