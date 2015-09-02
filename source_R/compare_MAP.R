library(dplyr)
library(ggplot2)

compare_map <- function(m1_fname, m2_fname){
  m1 <- read.delim(m1_fname, encoding="UTF-8", header=TRUE, sep=",")
  m2 <- read.delim(m2_fname, encoding="UTF-8", header=TRUE, sep=",")
  
  m1_name <- names(m1)[2]
  m2_name <- names(m2)[2]
  
  comb <- inner_join(m1, m2, by="users")
  
  comb["difference"] <- comb[m2_name] - comb[m1_name]
  
  comb <- comb %>% mutate(categories = difference)
  comb <- comb %>% mutate(categories = replace(categories, difference < 0, "diminished"))
  comb <- comb %>% mutate(categories = replace(categories, difference == 0, "unchanged"))
  comb <- comb %>% mutate(categories = replace(categories, difference > 0, "improved"))
  
  comb$categories <- as.factor(comb$categories)
  
  return(comb)
}

comb_compare_map <- function(comb1, comb2){
  
  comb <- inner_join(comb1, comb2, by="users")
  
  comb <- comb %>% mutate(combined_categories = paste(categories.x, categories.y))
  comb$combined_categories <- as.factor(comb$combined_categories)
  
  return(comb)
}


compare_map("scores/model_14.txt", "scores/model_13.txt") %>% select(categories) %>% summary()
compare_map("scores/model_13.txt", "scores/model_15.txt") %>% select(categories) %>% summary()
compare_map("scores/model_13.txt", "scores/model_16.txt") %>% select(categories) %>% summary()
compare_map("scores/model_15.txt", "scores/model_16.txt") %>% select(categories) %>% summary()

c1 <- compare_map("scores/model_14.txt", "scores/model_13.txt")
c2 <- compare_map("scores/model_13.txt", "scores/model_15.txt")

comb_compare_map(c1, c2) %>% select(combined_categories) %>% summary(100)