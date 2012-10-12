package.path =  package.path .. ";/home/bxu/code/toolbox/datatypes/?.lua"
package.path = package.path .. ";/home/bxu/code/toolbox/?.lua"
local os  = require("os")
local ffi = require("ffi")
--local narray = require("ljarray.array")
local Matrix = require("ljmatrix.densematrix")
local RandomForest = require("randomforest.rf")
local Dtools = require("debugtools.debug")
local GetTUV = require("getTUV")

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
--local classes = {10,20,15,30,56,100}
local classes = Matrix.range(1,6) 
local candidateIndices = zoomIndices(trainSet.labels, classes)

--Initialize trainingData
--Going with the idea of picking the highest density
--

local densities = trainingKernel:view(candidateIndices, candidateIndices):sum("cols")
local maxMatrix, indexMatrix = densities:max()
--have to shift by 1, as candidateIndices is 1-based
--We are using the top-two indices, sorted by the density (or evidence) in the
--zoomed kernel matrix.

local trainingIndices = {candidateIndices[indexMatrix.data[0]+1], candidateIndices[indexMatrix.data[1] + 1]}
--create a mapping from the classes, active in classification to 0,1,2,3, etc.
local function indexClasses(indices, labels) 
  local activeClasses, activeClassCounter = {}, 0
  for k, index in ipairs(indices) do
    activeClasses[labels:at(index)] = activeClassCounter
    activeClassCounter = activeClassCounter + 1
  end
  return activeClasses, activeClassCounter
end
local numRuns = 16
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
  local featureIndices = trainingIndices


  Dtools.printtable(trainingIndices, "trainingIndices")
  --Dtools.printtable(queryIndices, "queryIndices")
  Dtools.printtable(trainSet.labels(trainingIndices, "all"):count(), "TrainingLabels")


  local rf = RandomForest.create(400)
  local defaultBagSize = #trainingIndices
  --OPTION
  local defaultIndices = trainingIndices
  --local defaultIndices = Matrix.range(0,defaultBadSize)
  --local shift = 40

  --local featureIndices = "all"


  local defaultData = trainSet.kernel(defaultIndices, featureIndices)
  local defaultLabels = trainSet.labels(defaultIndices, "all")

  --local bagSize = defaultBagSize * 2 --number of combined training and unlabeled samples
  local bagSize = math.floor(defaultBagSize *2) --number of combined training and unlabeled samples

  local options = {bootstrapper = 
  RandomForest.policies.queryBootstrapper(
  defaultData, defaultLabels, bagSize),
  --splitPolicy = RandomForest.policies.oblique(RandomForest.heuristics.gini),
  stopPolicy = RandomForest.policies.stopBinary() 
}
--Dtools.printtable(options)
--defaultLabels:print("labels")
--Dtools.printtable(queryIndices)
--Dtools.printtable(featureIndices)
--trainSet.kernel:params()
--trainSet.labels:params()
rf:learn(trainSet.kernel(queryIndices, featureIndices), trainSet.labels(queryIndices, "all"), options)

local leafSizeCounts = rf:visitAllNodes(RandomForest.accumulators.leafSizeCounter)
local classSizeCounts = rf:visitAllNodes(RandomForest.accumulators.classSizeCounter):finalize()
local featureIdCounts = rf:visitAllNodes(RandomForest.accumulators.featureIdCounter)

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
end

--create counts Matrix:
--OPTION
--ATTENTION: this will probably hurt me down the road: let's create the counts-matrix in
--C-order 

local accumulator = rf:predict(testSet1.kernel, {accumulator = RandomForest.accumulators.singleVoter})
local activeClasses, activeClassCounter = indexClasses(trainingIndices, trainSet.labels)
local counts = Matrix.mat(#accumulator.results + 1,activeClassCounter)
for i = 0, #accumulator.results do
  for k,v in pairs(accumulator.results[i]) do
    if type(k) == "number" and k > 0 then
      local column = activeClasses[k]
      counts.data[counts:index(i,column)] = v
    end
  end
end
--counts:print()

--OPTION
--create marginalProbs
local marginalProbs = trainSet.kernel(queryIndices, "all"):sum("cols")

--calculate TUV, add new label

local tuvIndices, tuv = GetTUV(counts, marginalProbs)
--tuv:view(testIndices, "all"):print("testIndices")
Dtools.printtable(testIndices)

trainingIndices[#trainingIndices + 1] = queryIndices[tuvIndices[1] + 1 ]

--Dtools.printtable(activeClasses)


end
