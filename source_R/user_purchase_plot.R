library(dplyr)
library(ggplot2)

load("data/base.RData")

details_train$I_DATE <- as.Date(details_train$I_DATE, format="%Y-%m-%d ")

# number of transactions by date
purchases <- details_train %>% group_by(I_DATE) %>% summarise(COUNT=n())

qplot(x=I_DATE, y=COUNT, data=purchases)

# number of items purchased by date
items <- details_train %>% group_by(I_DATE) %>% summarise(COUNT=sum(ITEM_COUNT))

qplot(x=I_DATE, y=COUNT, data=items)



