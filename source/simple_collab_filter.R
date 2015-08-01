library(dplyr)

load("data/base.RData")

# idea: build a model
# X --> Y
# where X is item + user features and Y is user rating
# then, pick top 10 items for each user


raw_coupon_vars <- c("CAPSULE_TEXT", "GENRE_NAME", "PRICE_RATE", "CATALOG_PRICE",
                     "DISCOUNT_PRICE", "large_area_name", "ken_name", "small_area_name")

raw_user_vars <- c("SEX_ID", "AGE", "PREF_NAME")







