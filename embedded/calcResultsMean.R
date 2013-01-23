#!/usr/local/bin/Rscript

#benchmark <- read.table("results.dat", header = TRUE,
args=commandArgs(trailingOnly=TRUE)
benchmark <- read.table(args[1], header = TRUE,
row.names = "NumberOfSamples", check.names = FALSE)

numClasses = ncol(benchmark)

print(sum(rowSums(as.matrix(benchmark) / numClasses)) / nrow(benchmark))
