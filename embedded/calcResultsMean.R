#!/usr/local/bin/Rscript

benchmark <- read.table("results.dat", header = TRUE,
row.names = "NumberOfSamples", check.names = FALSE)

numClasses = ncol(benchmark)

print(sum(rowSums(as.matrix(benchmark) / numClasses)) / nrow(benchmark))
