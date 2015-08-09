library(dplyr)
library(assertthat)

user_list <- read.delim("raw_data/user_list.csv", 
                        encoding="UTF-8", 
                        header=TRUE, sep=",", quote="")

assert_that(are_equal(nrow(user_list), 22873))
assert_that(are_equal(ncol(user_list), 6))

coupons_train <- read.delim("raw_data/coupon_list_train.csv", 
                        encoding="UTF-8", 
                        header=TRUE, sep=",", quote="")

coupons_test <- read.delim("raw_data/coupon_list_test.csv", 
                        encoding="UTF-8", 
                        header=TRUE, sep=",", quote="")

assert_that(are_equal(nrow(coupons_train), 19413))
assert_that(are_equal(ncol(coupons_train), 24))

assert_that(are_equal(nrow(coupons_test), 310))
assert_that(are_equal(ncol(coupons_test), 24))

visits_train <- read.delim("raw_data/coupon_visit_train.csv", 
                           encoding="UTF-8", 
                           header=TRUE, sep=",", quote="")

assert_that(are_equal(nrow(visits_train), 2833180))
assert_that(are_equal(ncol(visits_train), 8))

details_train <- read.delim("raw_data/coupon_detail_train.csv", 
                           encoding="UTF-8", 
                           header=TRUE, sep=",", quote="")

assert_that(are_equal(nrow(details_train), 168996))
assert_that(are_equal(ncol(details_train), 6))

coupons_area_train <- read.delim("raw_data/coupon_area_train.csv", 
                            encoding="UTF-8", 
                            header=TRUE, sep=",", quote="")

coupons_area_test <- read.delim("raw_data/coupon_area_test.csv", 
                            encoding="UTF-8", 
                            header=TRUE, sep=",", quote="")

assert_that(are_equal(nrow(coupons_area_train), 138185))
assert_that(are_equal(ncol(coupons_area_train), 3))

assert_that(are_equal(nrow(coupons_area_test), 2165))
assert_that(are_equal(ncol(coupons_area_test), 3))

locations <- read.delim("raw_data/prefecture_locations.csv",
                        encoding="UTF-8",
                        header=TRUE, sep=",", quote="")

assert_that(are_equal(nrow(locations), 47))
assert_that(are_equal(ncol(locations), 4))


save.image(file="data/base.RData")
































