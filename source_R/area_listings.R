library(dplyr)
library(ggplot2)

load("data/base.RData")

# dropping SMALL_AREA_NAME
area_train <- coupons_area_train %>% select(PREF_NAME, COUPON_ID_hash) %>% unique()

# joining with location data
area_train <- left_join(area_train, locations, by="PREF_NAME")

# joining with training coupons
area_train <- left_join(coupons_train, area_train, by="COUPON_ID_hash")
area_train$COUPON_ID_hash <- as.factor(area_train$COUPON_ID_hash)

# NOTE: 45 COUPONS HAVE NO AREA INFORMATION.

counts <- area_train %>% group_by(COUPON_ID_hash) %>% summarise(COUNT=n())
counts <- counts %>% group_by(COUNT) %>% summarise(NUM_IDS=n())


