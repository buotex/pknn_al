package.path =  package.path .. ";/home/bxu/code/toolbox/datatypes/?.lua"
package.path = package.path .. ";/home/bxu/code/toolbox/?.lua"
local ffi = require("ffi")
--local narray = require("ljarray.array")
local Matrix = require("ljmatrix.densematrix")
local RandomForest = require("randomforest.rf")
local dtools = require("debugtools.debug")

kernelfile = io.input("kernel")
dump = io.read("*line")
kernelM = io.read("*number")
kernelN = io.read("*number")
dump = io.read("*line")
kernelmatrix = io.read("*all")
io.close()
labelsfile = io.input("labels")
dump = io.read("*line")
labelsM = io.read("*number")
labelsN = io.read("*number")
dump = io.read("*line")
labelsmatrix = io.read("*all")
kernelp =  ffi.cast("double*", kernelmatrix)
labelsp =  ffi.cast("double*", labelsmatrix)

kernel = Matrix.mat(kernelM, kernelN, kernelp)
labels = Matrix.mat(labelsM, labelsN, labelsp)


--dtools.printarray(kernel)
--dtools.printarray(labels)

local rf = RandomForest.create(5)

rf:learn(kernel, labels)







