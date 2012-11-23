#!/usr/local/bin/Rscript

library(ggplot2)
X11()
pdf("tuvs.pdf", w=24, h=6)
# header = TRUE ignores the first line, check.names = FALSE allows '+' in 'C++'
  benchmark <- read.table("tuvs.dat", header = TRUE,
      row.names = "NumberOfSamples", check.names = FALSE)
numClasses = ncol(benchmark)
colors <- read.table("color_table.txt", header = FALSE)
colors <- rgb(colors, maxColorValue=255)
# 't()' is matrix tranposition, 'beside = TRUE' separates the benchmarks, 'heat'
# provides nice colors
barplot(t(as.matrix(benchmark)), beside = TRUE, main="vote-distribution for the
sample with the highest TUV" ,  xlab = "#labeled samples - label of chosen
training sample with the highest TUV", ylab
= "#votes per tree ", col = rev(colors[1:numClasses]), border=NA)
#heat.colors(ncol(benchmark)))
# 'cex' stands for 'character expansion', 'bty' for 'box type' (we don't want
# borders)
    legend("topleft", names(benchmark), cex = 0.9, bty = "n", fill =
rev(colors[1:numClasses]))
dev.off()
