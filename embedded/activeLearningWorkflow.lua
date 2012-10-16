#!/usr/local/bin/luajit
package.path =  package.path .. ";/home/bxu/code/toolbox/datatypes/?.lua"
package.path = package.path .. ";/home/bxu/code/toolbox/?.lua"
local os  = require("os")
local ffi = require("ffi")
--local narray = require("ljarray.array")
local Matrix = require("ljmatrix.densematrix")
local RandomForest = require("randomforest.rf")
local Dtools = require("debugtools.debug")
local TUV = require("getTUV")

RandomForest.Debug = false

--Keywords: STATISTICS, OPTION, DEBUG
local SHOW_ACCURACY= true

--START LoadingPhase
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
--END LoadingPhase
--The data is now loaded in c-order, NxD.

--testSet = {kernel = testKernel, labels = testClasses}

--trainingKernel:params()
--trainingClasses:params()
--testKernel:params()
--testClasses:params()

--This function extracts the indices of the items that belong to the classes
--given as the second parameter.
local function zoomIndices(labels, classes)

  local indexarrays = {}
  local candidateIndices = {}
  local i = 0
  for label in labels:all() do
    indexarrays[label] = indexarrays[label] or {}
    local t = indexarrays[label]
    t[#t + 1] = i
    i = i + 1
  end

  for k,i in pairs(classes) do
    for j = 1, #indexarrays[i] do
      candidateIndices[#candidateIndices + 1] = indexarrays[i][j]
    end
  end


  return candidateIndices
end

--filter the table t, remove all values for which the function returns true
local function filterTable(t, func)
  for k,v in pairs(t) do
    if func(v) then t[k] = nil end
  end
  return t
end

--search if element is in table, return true if it is
local function findTable(t, elem)
  for k, v in pairs(t) do
    if v == elem then return true end
  end
  return false
end


--OPTION
local classes = {1,5,10,15,20,30}
--local classes = Matrix.range(10,20) 
local candidateIndices = zoomIndices(trainSet.labels, classes)

--Initialize trainingData
--Going with the idea of picking the highest density
--

local densities = trainingKernel:view(candidateIndices, candidateIndices):sum("cols")
local maxMatrix, indexMatrix = densities:max()
--have to shift by 1, as candidateIndices is 1-based
--We are using the top-two indices, sorted by the density (or evidence) in the
--zoomed kernel matrix.

--local trainingIndices = {candidateIndices[indexMatrix.data[0]+1], candidateIndices[indexMatrix.data[1] + 1]}
local trainingIndices = {candidateIndices[indexMatrix.data[indexMatrix.n_elem - 1]+1] }
trainSet.labels(trainingIndices, "all"):print("Starting labels")
--create a mapping from the classes, active in classification to 0,1,2,3, etc.



local function indexClasses(classes, training, allLabels) 
  --local activeClasses, activeClassCounter = {}, 0
  local classMapping ={}
  local counter = 0
  for k = 0, #classes do
    local class = classes[k]
    if class then 
      classMapping[class] = counter
      counter = counter + 1
    end
  end
  local labels, classesFound = allLabels(training, "all"):count()
  
  return classMapping, counter, classesFound 
end
local numRuns = 30
local run = 0
while run < numRuns do

  run = run + 1
  --remove trainingIndices from the candidates
  filterTable(candidateIndices, 
  function(v) return findTable(trainingIndices, v) end
  )
  --copy candidateIndices to queryIndices
  local queryIndices = {}
  for k,v in pairs(candidateIndices) do
    queryIndices[#queryIndices + 1] = v
  end

  --use the kernel-dimensions implied by the trainingIndices as features /
  --variables.
  --local featureIndices = "all"
  local featureIndices = trainingIndices

  local classMapping, numClasses, classesFound  = indexClasses(classes, trainingIndices, trainSet.labels)

  --Dtools.printtable(trainingIndices, "trainingIndices")
  --Dtools.printtable(queryIndices, "queryIndices")
  Dtools.printtable(trainSet.labels(trainingIndices, "all"):count(), "TrainingLabels")


  local numTrees = 200
  local defaultBagSize = #trainingIndices
  --OPTION
  local defaultIndices = trainingIndices
  local rf = RandomForest.create(numTrees)
  --local defaultIndices = Matrix.range(0,defaultBadSize)
  --local shift = 40

  --local featureIndices = "all"


  local defaultData = trainSet.kernel(defaultIndices, featureIndices)
  local defaultLabels = trainSet.labels(defaultIndices, "all")

  --local bagSize = defaultBagSize * 2 --number of combined training and unlabeled samples
  local bagSize = math.floor(defaultBagSize *1.5) --number of combined training and unlabeled samples
  --OPTION
  local options = {bootstrapper = 
  RandomForest.policies.queryBootstrapper(
  defaultData, defaultLabels, bagSize),
  --  splitPolicy = RandomForest.policies.oblique(RandomForest.heuristics.gini),
  --stopPolicy = RandomForest.policies.stopBinary() 
}
--Dtools.printtable(options)
--defaultLabels:print("labels")
--Dtools.printtable(queryIndices)
--Dtools.printtable(featureIndices)
--trainSet.kernel:params()
--trainSet.labels:params()
rf:learn(trainSet.kernel(queryIndices, featureIndices), trainSet.labels(queryIndices, "all"), options)

--STATISTICS
--local leafSizeCounts = rf:visitAllNodes(RandomForest.accumulators.leafSizeCounter)
--local classSizeCounts = rf:visitAllNodes(RandomForest.accumulators.classSizeCounter):finalize()
--local featureIdCounts = rf:visitAllNodes(RandomForest.accumulators.featureIdCounter)
local idMatrix = rf:visitAllNodes(RandomForest.accumulators.classAffinity)
--[[
local affinityMatrix2 = Matrix.mat(numClasses, numClasses)
for k,t in pairs(idMatrix.results) do
  --convert indices to labels
  --Dtools.printtable(t, "blub")
  for i = 1, #t do
    t[i] = trainSet.labels:at(featureIndices[t[i]+1 ])
  end
  for i = 1, #t do
    for j = 1, #t do
      affinityMatrix2.data[affinityMatrix2:index(classMapping[t[i] ], classMapping[t[j] ]) ] = affinityMatrix2:get(classMapping[t[i] ], classMapping[t[j] ]) + 1 
    end
  end
end
--affinityMatrix2:print("affinityMatrix2", "%d")
]]--
--Dtools.printtable(idMatrix.results, "used featureIds")

--Dtools.printtable(classSizeCounts.numSamples)

--STATISTICS
--Dtools.printtable(leafSizeCounts.results, "Number of leaf nodes with size #")
--Dtools.printtable(classSizeCounts.results, "Number of Elements per node per class")
--Dtools.printtable(featureIdCounts.results, "Ids used for splitting")
--------------------------------------------------------------------------------

print("finished learning")


local testSet1 = {

  kernel = trainingKernel(queryIndices, featureIndices),
  labels = trainingClasses(queryIndices, "all")

}
local testSet2 = (function()
  local testIndices = zoomIndices(testClasses, classes)
  local testSet = {

    kernel = testKernel(testIndices, featureIndices),
    labels = testClasses(testIndices, "all")

  }
  return testSet
end)()


local testSet = testSet2

local accumulator = rf:predict(testSet.kernel, {accumulator = RandomForest.accumulators.singleVoter})
--OPTIONAL_ACCURACY_START

if SHOW_ACCURACY  then
  local accuracy = {}
  local accuracy_counts = {}
  for _,i in pairs(classes) do
    accuracy_counts[i] = {correct = 0, count = 0}
  end


  local affinity_matrix = Matrix.mat(numClasses, numClasses)
  affinity_matrix:view("all", "all"):set(0)

  for i = 0, #accumulator.results do
    --print(accumulator.results[i].label)
    --print(labels:view(queryIndices, "all"):get(i,0))
    local correctLabel = testSet.labels:get(i,0)
    local predictedLabel = accumulator.results[i].label
    affinity_matrix.data[
    affinity_matrix:index(classMapping[predictedLabel], classMapping[correctLabel])] = 

    affinity_matrix.data[affinity_matrix:index(classMapping[predictedLabel], classMapping[correctLabel])] + 1

    accuracy[i] = (predictedLabel ==  correctLabel)
    if accuracy[i] then accuracy_counts[correctLabel].correct = accuracy_counts[correctLabel].correct + 1 end
    accuracy_counts[correctLabel].count = accuracy_counts[correctLabel].count + 1 

  end
  affinity_matrix:print("affinity_matrix","%d")
  --some accuracy statistics:

  local acc_summary = {correct = 0, count = 0}
  for _,i in pairs(classes) do
    print("Label: ", i)
    print("ratio: ", accuracy_counts[i].correct / accuracy_counts[i].count)
    acc_summary.correct = acc_summary.correct + accuracy_counts[i].correct
    acc_summary.count = acc_summary.count + accuracy_counts[i].count
  end
  print("global ratio: ", acc_summary.correct / acc_summary.count)


end

--create counts Matrix:
--OPTION
--ATTENTION: this will probably hurt me down the road: let's create the counts-matrix in
--C-order 

local accumulator = rf:predict(testSet1.kernel, {accumulator = RandomForest.accumulators.singleVoter})
local counts = Matrix.mat(#accumulator.results + 1,numClasses)
for i = 0, #accumulator.results do
  for k,v in pairs(accumulator.results[i]) do
    if type(k) == "number" and k > 0 then
      local column = classMapping[k]
      counts.data[counts:index(i,column)] = v/numTrees
    end
  end
end
--counts:print()

--OPTION
--create marginalProbs
local marginalProbs = trainSet.kernel(queryIndices, "all"):sum("cols")
marginalProbs:view("all", "all"):set(1)

--calculate TUV, add new label
local lambda = {1,1}

local tuvIndices, tuv = TUV.getBestTuv(counts, marginalProbs, lambda)
--tuv:view(testIndices, "all"):print("testIndices")
counts:view(tuvIndices, "all"):view({0,1,2,3,4,5},"all"):print("counts")
--Dtools.printtable(testIndices)

trainingIndices[#trainingIndices + 1] = queryIndices[tuvIndices[1] + 1 ]
--trainSet.kernel(trainingIndices, trainingIndices):print("TrainingKernel")

--Dtools.printtable(activeClasses)


end
