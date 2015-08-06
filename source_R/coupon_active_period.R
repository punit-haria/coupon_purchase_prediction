library(dplyr)
library(ggplot2)

load("data/base.RData")

details_train$I_DATE <- as.Date(details_train$I_DATE, format="%Y-%m-%d ")

activity <- details_train %>% group_by(COUPON_ID_hash) %>%
  summarise(PERIOD=max(I_DATE)-min(I_DATE))

count <- activity %>% group_by(PERIOD) %>%
  summarise(COUNT=n())

qplot(x=as.integer(PERIOD), y=COUNT, data=count)

#-------------------------------------------------------

coupons_train$DISPFROM <- as.Date(coupons_train$DISPFROM,
                                  format="%Y-%m-%d ")
coupons_train$DISPEND <- as.Date(coupons_train$DISPEND,
                                  format="%Y-%m-%d ")
coupons_train$VALIDFROM <- as.Date(coupons_train$VALIDFROM,
                                  format="%Y-%m-%d ")
coupons_train$VALIDEND <- as.Date(coupons_train$VALIDEND,
                                  format="%Y-%m-%d ")


coupons_test$DISPFROM <- as.Date(coupons_test$DISPFROM,
                                  format="%Y-%m-%d ")
coupons_test$DISPEND <- as.Date(coupons_test$DISPEND,
                                 format="%Y-%m-%d ")
coupons_test$VALIDFROM <- as.Date(coupons_test$VALIDFROM,
                                   format="%Y-%m-%d ")
coupons_test$VALIDEND <- as.Date(coupons_test$VALIDEND,
                                  format="%Y-%m-%d ")

#-------------------------------------------------------

# comparisons of Coupon Sales periods in training and test set

coupons_train %>% select(DISPFROM, DISPEND, VALIDFROM, VALIDEND) %>% summary()
coupons_test %>% select(DISPFROM, DISPEND, VALIDFROM, VALIDEND) %>% summary()
summary(details_train$I_DATE)






