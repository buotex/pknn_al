package.path =  package.path .. ";/home/bxu/code/toolbox/datatypes/?.lua"
package.path = package.path .. ";/home/bxu/code/toolbox/?.lua"
local os  = require("os")
local ffi = require("ffi")
--local narray = require("ljarray.array")
local Matrix = require("ljmatrix.densematrix")
local RandomForest = require("randomforest.rf")
local Dtools = require("debugtools.debug")

RandomForest.Debug = false

local function loadArmadilloMatrix(filename)
  local file = io.input(filename)
  local dump = io.read("*line")

  local N = io.read("*number")
  local M = io.read("*number")
  local dump = io.read("*line")
  local matrix = io.read("*all")
  io.close()
  local pointer = ffi.cast("double*", matrix)
  return Matrix.mat(M, N, pointer):transposed()
end

local trainingKernel = loadArmadilloMatrix("data/gb2.matrix"):transposed()
local trainingClasses = loadArmadilloMatrix("data/trainingClasses.matrix")
trainingClasses:reshapeTo(trainingClasses.n_cols, trainingClasses.n_rows)

trainSet = {kernel = trainingKernel, labels = trainingClasses}

local testKernel = loadArmadilloMatrix("data/test-gb2.matrix"):transposed()
local testClasses = loadArmadilloMatrix("data/testClasses.matrix")
testClasses:reshapeTo(testClasses.n_cols, testClasses.n_rows)

--testSet = {kernel = testKernel, labels = testClasses}

--trainingKernel:params()
--trainingClasses:params()
--testKernel:params()
--testClasses:params()

local function zoomIndices(labels, classes)

  local indexarrays = {}
  local i = 0
  for label in labels:all() do
    indexarrays[label] = indexarrays[label] or {}
    local t = indexarrays[label]
    t[#t + 1] = i
    i = i + 1
  end
  return indexarrays

end

local classes = {10,20,15,30,56,100}
local classes = Matrix.range(10,30)
local indexarrays = zoomIndices(trainSet.labels, classes)

local samplesPerClass = 15
local trainingIndices = {}
local queryIndices = {}


--create trainingIndices

for _,i in pairs(classes) do
  for j = 1, samplesPerClass do
    local index = math.random(1,#indexarrays[i])
    trainingIndices[#trainingIndices + 1] = indexarrays[i][index]
    indexarrays[i][index] = nil
  end
end

for _,i in pairs(classes) do
  for k,v in pairs(indexarrays[i]) do
    queryIndices[#queryIndices + 1] = v
  end
end
--Dtools.printtable(trainingIndices, "trainingIndices")
--Dtools.printtable(queryIndices, "queryIndices")


local rf = RandomForest.create(100)
local defaultBagSize = #trainingIndices
local defaultIndices = trainingIndices
--local defaultIndices = Matrix.range(0,defaultBadSize)
--local shift = 40

--local featureIndices = "all"
local featureIndices = defaultIndices


local defaultData = trainSet.kernel(defaultIndices, featureIndices)
local defaultLabels = trainSet.labels(defaultIndices, "all")

--local bagSize = defaultBagSize * 2 --number of combined training and unlabeled samples
local bagSize = math.floor(defaultBagSize *1) --number of combined training and unlabeled samples

local options = {bootstrapper = 
RandomForest.policies.queryBootstrapper(
defaultData, defaultLabels, bagSize),
--splitPolicy = RandomForest.policies.oblique(RandomForest.heuristics.gini),
--stopPolicy = RandomForest.policies.stopBinary()
}
--Dtools.printtable(options)
--defaultLabels:print("labels")
rf:learn(trainSet.kernel(queryIndices, featureIndices), trainSet.labels(queryIndices, "all"), options)

local leafSizeCounts = rf:visitAllNodes(RandomForest.accumulators.leafSizeCounter)
local classSizeCounts = rf:visitAllNodes(RandomForest.accumulators.classSizeCounter):finalize()
local featureIdCounts = rf:visitAllNodes(RandomForest.accumulators.featureIdCounter)

--Dtools.printtable(classSizeCounts.numSamples)


Dtools.printtable(leafSizeCounts.results, "Number of leaf nodes with size #")
--Dtools.printtable(classSizeCounts.results, "Number of Elements per node per class")
--Dtools.printtable(featureIdCounts.results, "Ids used for splitting")
--------------------------------------------------------------------------------

print("finished learning")


local testSet1 = {

  kernel = trainingKernel(queryIndices, featureIndices),
  labels = trainingClasses(queryIndices, "all")

}
local testSet2 = (function()
  local indexarrays = zoomIndices(testClasses, classes)
  local testIndices = {}
  for _,i in pairs(classes) do
    for k,v in pairs(indexarrays[i]) do
      testIndices[#testIndices + 1] = v
    end
  end
  local testSet = {

    kernel = testKernel(testIndices, featureIndices),
    labels = testClasses(testIndices, "all")

  }
  return testSet
end)()


local testSet = testSet1

local accumulator = rf:predict(testSet.kernel, {accumulator = RandomForest.accumulators.singleVoter})

local accuracy = {}
local accuracy_counts = {}
for _,i in pairs(classes) do
  accuracy_counts[i] = {correct = 0, count = 0}
end
for i = 0, #accumulator.results do
  --print(accumulator.results[i].label)
  --print(labels:view(queryIndices, "all"):get(i,0))
  local correctLabel = testSet.labels:get(i,0)
  accuracy[i] = (accumulator.results[i].label ==  correctLabel)
  if accuracy[i] then accuracy_counts[correctLabel].correct = accuracy_counts[correctLabel].correct + 1 end
  accuracy_counts[correctLabel].count = accuracy_counts[correctLabel].count + 1 

end
--some accuracy statistics:

local acc_summary = {correct = 0, count = 0}
for _,i in pairs(classes) do
  print("Label: ", i)
  print("ratio: ", accuracy_counts[i].correct / accuracy_counts[i].count)
  acc_summary.correct = acc_summary.correct + accuracy_counts[i].correct
  acc_summary.count = acc_summary.count + accuracy_counts[i].count
end
print("global ratio: ", acc_summary.correct / acc_summary.count)
