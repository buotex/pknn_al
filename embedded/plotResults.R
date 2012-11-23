#!/usr/local/bin/Rscript

library(ggplot2)
library(colorspace)
X11()
pdf("accuracy.pdf", w=24, h=6)
#colors =  c("black", "red", "yellow", "green", "violet", "orange", "blue", "pink", "cyan") 
#colors =  c("red", "yellow", "green", "violet", "orange", "blue") 
colors <- read.table("color_table.txt", header = FALSE)
colors <- rgb(colors, maxColorValue=255)
# header = TRUE ignores the first line, check.names = FALSE allows '+' in 'C++'
  benchmark <- read.table("results.dat", header = TRUE,
      row.names = "NumberOfSamples", check.names = FALSE)
numClasses = ncol(benchmark)
# 't()' is matrix tranposition, 'beside = TRUE' separates the benchmarks, 'heat'
# provides nice colors
bp <- barplot(t(as.matrix(benchmark)), beside = TRUE, main="Accuracy of AL-approach
starting from zero", xlab = "#labeled samples - label of most recently added
training sample", ylab
= "Accuracy", col = rev(colors[1:numClasses]), border = NA)
x <- (bp[numClasses / 2,] + bp[numClasses / 2 + 1,])/2
lines(x, rowSums(as.matrix(benchmark) / numClasses))
#print(bp[numClasses / 2,])
#print(rowSums(as.matrix(benchmark)) / numClasses)
#heat.colors(ncol(benchmark)))
# 'cex' stands for 'character expansion', 'bty' for 'box type' (we don't want
# borders)
    legend("topleft", names(benchmark), cex = 0.9, bty = "n", fill =
rev(colors[1:numClasses]))
legend("topright", "mean accuracy", fill="black")


dev.off()
#heat.colors(ncol(benchmark)))
#message("Press Return To Continue")
#invisible(readLines("stdin", n=1))
