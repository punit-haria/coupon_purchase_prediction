library(dplyr)
library(ggplot2)
library(reshape2)
library(RColorBrewer)


hmap <- function(mvm){
  # generate missing value matrix
  mvm <- data.frame(lapply(mvm, as.character), stringsAsFactors=FALSE)
  mvm <- replace(mvm, !is.na(mvm), "1")
  mvm <- replace(mvm, is.na(mvm), "0")
  mvm <- data.frame(lapply(mvm, as.numeric))
  mvm <- as.matrix(mvm)
  mvm <- mvm[do.call(order, lapply(1:NCOL(mvm), function(i) mvm[, i])), ]
  mvm <- melt(mvm)
  names(mvm) <- c("Row", "Dimension", "value")
  mvm$value <- as.character(mvm$value)
  mvm <- mutate(mvm, value = replace(value, value == "0", "missing"))
  mvm <- mutate(mvm, value = replace(value, value == "1", "available"))
  
  # plot heatmap of missing values
  color_palette <- colorRampPalette(c("#FFFFFF", "#801515"))(2)
  plt <- ggplot(mvm, aes(Row, Dimension, fill=factor(value))) + geom_raster() + scale_fill_manual(values = color_palette)
  plt
}

