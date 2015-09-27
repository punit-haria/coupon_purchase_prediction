library(dplyr)

load("data/base.RData")

print("Number of users: ")
print(dim(user_list)[1])

# find if user visits coupon after purchase

purchases = details_train %>% 
  select(USER_ID_hash, COUPON_ID_hash, I_DATE)

visits = visits_train %>% 
  select(USER_ID_hash, VIEW_COUPON_ID_hash, I_DATE)
names(visits) = c('USER_ID_hash', 'COUPON_ID_hash', 'VISIT_DATE')

intsct = inner_join(purchases, visits, by=c('USER_ID_hash', 'COUPON_ID_hash'))
intsct = intsct %>%
  group_by(USER_ID_hash, COUPON_ID_hash, I_DATE) %>%
  summarise(max(VISIT_DATE)) %>%
  ungroup()
names(intsct)[4] = 'VISIT_DATE'

visits_after = intsct %>% filter(VISIT_DATE > I_DATE)


# proportion of visited coupons that were purchased

visits = visits_train %>% 
  select(USER_ID_hash, VIEW_COUPON_ID_hash) %>%
  filter(VIEW_COUPON_ID_hash %in% coupons_train$COUPON_ID_hash) %>%
  unique()
names(visits) = c('USER_ID_hash', 'COUPON_ID_hash')
visits = mutate(visits, UC_CONCATE=paste(USER_ID_hash, COUPON_ID_hash))

purchases = details_train %>%
  select(USER_ID_hash, COUPON_ID_hash) %>%
  unique()
purchases = mutate(purchases, UC_CONCATE=paste(USER_ID_hash, COUPON_ID_hash))

total_visited = visits %>% 
  group_by(USER_ID_hash) %>%
  summarise(NUM_VISITED=n()) %>%
  ungroup()

purchased_from_visited = visits %>%
  filter(UC_CONCATE %in% purchases$UC_CONCATE) %>%
  group_by(USER_ID_hash) %>%
  summarise(NUM_PURCHASED=n()) %>%
  ungroup()

prob_purchase = left_join(total_visited, purchased_from_visited,
                          by=c('USER_ID_hash'))
prob_purchase = prob_purchase %>%
  mutate(NUM_PURCHASED = replace(NUM_PURCHASED, 
                                 is.na(NUM_PURCHASED), 0))
prob_purchase = prob_purchase %>% 
  mutate(PROB_PURCHASE=NUM_PURCHASED/NUM_VISITED) %>%
  as.data.frame()

write.table(prob_purchase, file="datalibfm/prob_purchase.txt",
            quote=FALSE, sep=",", row.names=FALSE)



