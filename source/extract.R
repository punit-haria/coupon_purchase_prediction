library(dplyr)

user_list <- read.delim("raw_data/user_list.csv", 
                        encoding="UTF-8", 
                        header=TRUE, sep=",", quote="")

