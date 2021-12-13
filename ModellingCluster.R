library(lubridate)
library(xgboost)
library(caret)
library(dplyr)
library(reshape2)
library(data.table)
library(dtwclust)

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

# ------------ CLUSTER DATA --------------------
# Transposing the train and test set to obtain data in the desired format
trainT <- transpose(train);
testT <- transpose(test)

# Make data numeric
nm <- colnames(trainT)
for(cols in nm){
  trainT[,cols] <- as.numeric(trainT[,cols])
}

nm2 <- colnames(testT)
for(cols in nm2){
  testT[,cols] <- as.numeric(testT[,cols])
}

#get all columns that are of type factor
factor_columns <- sapply(trainT, is.factor)
factor_columns2 <- sapply(testT, is.factor)

#change them from type factor to numeric so the numeric data can be plotted 
trainT[factor_columns] <- lapply(trainT[factor_columns], function(x) as.numeric(as.character(x)))
testT[factor_columns2] <- lapply(testT[factor_columns2], function(x) as.numeric(as.character(x)))

# add d to columns
trainT <- tibble::rownames_to_column(trainT, "d")
testT <- tibble::rownames_to_column(testT, "d")

# Merge with calendar
train_new <- merge(calendar, trainT, by = "d")
test_new <- merge(calendar, testT, by = "d")

# Only keep the date of the calendar dataframe into the new dataframe
trainNew <- subset(train_new, select = -c(wm_yr_wk,weekday, wday,month,year,event_name_1, event_type_1,event_name_2,event_type_2,snap_CA))
testNew <- subset(test_new, select = -c(wm_yr_wk,weekday, wday,month,year,event_name_1, event_type_1,event_name_2,event_type_2,snap_CA))

# Changing character date as date type
trainNew$date <- mdy(trainNew$date)
testNew$date <- mdy(testNew$date)

# Set d as index
rownames(trainNew) <- trainNew$d
trainNew <- subset(trainNew, select= -c(d))
trainNew <- na.omit(trainNew)

rownames(testNew) <- testNew$d
testNew <- subset(testNew, select= -c(d))
testNew <- na.omit(testNew)

#get all columns that are of type factor
factor_columns <- sapply(trainNew, is.factor)
factor_columns2 <- sapply(testNew, is.factor)

#change them from type factor to numeric so the numeric data can be plotted 
trainNew[factor_columns] <- lapply(trainNew[factor_columns], function(x) as.numeric(as.character(x))) 

#prepare the data to be clustered (removing the date column)
cluster_data <- t(trainNew[(- ncol(trainNew)),]); 
cluster_data <- subset(trainNew, select = -c(date));

cluster_data2 <- t(testNew[(- ncol(testNew)),]);
cluster_data2 <- subset(testNew, select = -c(date));

#save all the dates of the training data
train_dates <- trainNew[1];
test_dates <- testNew[1];

# transpose data so the products will be clusters instead of the dates
cluster_data <- tibble::rownames_to_column(cluster_data, "d");
cluster_data <- transpose(cluster_data); cluster_data

cluster_data2 <- tibble::rownames_to_column(cluster_data2, "d");
cluster_data2 <- transpose(cluster_data2); cluster_data2

# cluster data
clusters <- tsclust(cluster_data, type = "partitional", k = 2L, 
                    distance = "dtw_basic", centroid = "pam", 
                    seed = 3247L, trace = TRUE,
                    args = tsclust_args(dist = list(window.size = 2L)))

clusters2 <- tsclust(cluster_data2, type = "partitional", k = 2L, 
                     distance = "dtw_basic", centroid = "pam", 
                     seed = 3247L, trace = TRUE,
                     args = tsclust_args(dist = list(window.size = 2L)))

#saving the clusters 
cluster <- clusters@cluster
cluster2 <- clusters2@cluster

#add the corresponding cluster to the data 
train_cluster_set <- cbind(cluster_data, cluster)
test_cluster_set <- cbind(cluster_data2, cluster)

# Make the different clusters as df
cluster1 <- train_cluster_set %>% filter(cluster == 1)
cluster2 <- train_cluster_set %>% filter(cluster == 2)

cluster1Test <- test_cluster_set %>% filter(cluster == 1)
cluster2Test <- test_cluster_set %>% filter(cluster == 2)


# ------------ DATA PREPARATION CLUSTER 1 ---------
c1 <- tibble::rownames_to_column(cluster1, "Products")
nc1 <- transpose(c1)
trainT <- tibble::rownames_to_column(nc1, "d")

c1T <- tibble::rownames_to_column(cluster1Test, "Products")
nc1T <- transpose(c1T)
testT <- tibble::rownames_to_column(nc1T, "d")

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

output2 <- preddf %>% tidyr::spread(forecast, predval)
output2 <- output2 %>% select(-ID) 
colnames(output2) <- c('id','F1','F2','F3','F4','F5','F6','F7','F8','F9','F10','F11','F12',
                       'F13','F14','F15','F16','F17','F18','F19','F20','F21','F22','F23','F24','F25','F26','F27','F28')

caret::RMSE(prod_pred$predval, test3$value)

# --------- FINAL DATASET CLUSTER 1 -----------

write.csv(output2, 'output.csv')

# RMSE calculation
# m <- test3 %>% dplyr::select(-c(day, month, year, wday, event, snap_CA)) %>% dplyr::arrange(ID)
# RMSE(predval, m$value)


# --------- CLUSTER 2 -------------------

c2 <- tibble::rownames_to_column(cluster2, "Products")
nc2 <- transpose(c2)
trainT <- tibble::rownames_to_column(nc2, "d")

c2T <- tibble::rownames_to_column(cluster2Test, "Products")
nc2T <- transpose(c2T)
testT <- tibble::rownames_to_column(nc2T, "d")

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

caret::RMSE(prod_pred$predval, test3$value)

# --------- FINAL DATASET CLUSTER 2 -----------

write.csv(output, 'output2.csv')

# ---------- MERGE OUTPUT CLUSTERS ------------
dfMerge <- rbind(output, output2)
dfMerge[dfMerge < 0] <- 0
write.csv(dfMerge, 'outputCluster.csv')
