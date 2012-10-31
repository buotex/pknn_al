#!/usr/local/bin/luajit
package.path =  package.path .. ";/home/bxu/code/toolbox/datatypes/?.lua"
package.path =  package.path .. ";/home/bxu/code/toolbox/algorithms/?.lua"
package.path = package.path .. ";/home/bxu/code/toolbox/?.lua"
local os  = require("os")
local ffi = require("ffi")
--local narray = require("ljarray.array")
local Matrix = require("ljmatrix.densematrix")
local RandomForest = require("randomforest.rf")
local Dtools = require("debugtools.debug")
local TUV = require("getTUV")
local Workflow = require("activelearning.workflow")

math.randomseed(0)

local mapping = io.input("classMapping.txt")
local classNameMapping = {}
local _counter = 1
while true do
  local className = io.read("*line")
  if not className then break end
  classNameMapping[className] = _counter
  _counter = _counter + 1
end
--[[
for k,v in pairs(classNameMapping) do 
  print(k,v) 
end
]]--

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

--filter the table t, remove all values for which the function returns true


--OPTION
local classnames = {"BACKGROUND_Google", "octopus", "hawksbill", "pigeon", "watch", "umbrella", "electric_guitar", "garfield", "gerenuk", "inline_skate", "lobster", "lotus", "mandolin", "okapi"}
--local classnames = {"octopus", "hawksbill", "pigeon", "watch", "umbrella"}
--local classnames = {"octopus", "hawksbill"}
local classes = {}
for i = 1, #classnames do
  classes[i] = classNameMapping[classnames[i] ]
end
--Dtools.printtable(classes,"classes")
--Dtools.waitInput()
--local classes = {1,5,10,20,25,30}

--local classes = Matrix.range(10,20) 

--Initialize trainingData
--Going with the idea of picking the highest density
--

--have to shift by 1, as candidateIndices is 1-based
--We are using the top-two indices, sorted by the density (or evidence) in the
--zoomed kernel matrix.

--local trainingIndices = {candidateIndices[indexMatrix.data[0]+1], candidateIndices[indexMatrix.data[1] + 1]}
--local trainingIndices = {candidateIndices[indexMatrix.data[indexMatrix.n_elem - 1]+1] }
--trainSet.labels(trainingIndices, "all"):print("Starting labels")
--create a mapping from the classes, active in classification to 0,1,2,3, etc.



local function indexClasses(classes) 
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

  return classMapping, counter
end


local numRuns = 1
local run = 0 

local classMapping, numClasses = indexClasses(classes)
local allResults = {}
local maxSamples = 12
while run < numRuns do
  run = run + 1

  local candidateIndices = Workflow.filterClasses(trainSet.labels, classes)
  local densities = trainSet.kernel:view(candidateIndices, candidateIndices):sum("cols")
  --densities:print("densities")
  local maxMatrix, indexMatrix = densities:max()
    --  indexMatrix:print("indexMatrix")
  
  local trainingIndices = {[0] = candidateIndices[indexMatrix.data[0] ] }

  local numLabeledSamples = 0

  local results = Matrix.mat(maxSamples, numClasses)
  local pickedLabels = Matrix.mat(maxSamples, 1)
  local bestTUV      = Matrix.mat(maxSamples, numClasses)
  while numLabeledSamples < maxSamples do

    numLabeledSamples = numLabeledSamples + 1

    local queryIndices = Workflow.updateQueryIndices(candidateIndices, trainingIndices) 
    --trainSet.labels(queryIndices, "all"):print("query-labels")
    --Dtools.waitInput()

    --use the kernel-dimensions implied by the trainingIndices as features /
    --variables.
    --local featureIndices = "all"
    local featureIndices = trainingIndices

    local labels, classesFound = trainSet.labels(trainingIndices, "all"):count()

    --Dtools.printtable(trainingIndices, "trainingIndices")
    --Dtools.printtable(queryIndices, "queryIndices")
    --Dtools.printtable(trainSet.labels(trainingIndices, "all"):count(), "TrainingLabels")

    local rf = Workflow.buildForest(
    trainSet.kernel, trainSet.labels, trainingIndices, queryIndices, featureIndices
    ) 
    ----STATISTICS
    ----local leafSizeCounts = rf:visitAllNodes(RandomForest.accumulators.leafSizeCounter)
    ----local classSizeCounts = rf:visitAllNodes(RandomForest.accumulators.classSizeCounter):finalize()
    ----local featureIdCounts = rf:visitAllNodes(RandomForest.accumulators.featureIdCounter)
    --local idMatrix = rf:visitAllNodes(RandomForest.accumulators.classAffinity)
    --local affinityMatrix2 = Matrix.mat(numClasses, numClasses)
    --for k,t in pairs(idMatrix.results) do
    ----convert indices to labels
    ----Dtools.printtable(t, "blub")
    --for i = 1, #t do
    --t[i] = trainSet.labels:at(featureIndices[t[i]+1 ])
    --end
    --for i = 1, #t do
    --for j = 1, #t do
    --affinityMatrix2.data[affinityMatrix2:index(classMapping[t[i] ], classMapping[t[j] ]) ] = affinityMatrix2:get(classMapping[t[i] ], classMapping[t[j] ]) + 1 
    --end
    --end
    --end
    --affinityMatrix2:print("affinityMatrix2", "%d")
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
      local testIndices = Workflow.filterClasses(testClasses, classes)
      local testSet = {

        kernel = testKernel(testIndices, featureIndices),
        labels = testClasses(testIndices, "all")

      }
      return testSet
    end)()


    local testSet = testSet1

    --OPTIONAL_ACCURACY_START

    if SHOW_ACCURACY  then

      --local bagSize = math.floor(classesFound *2) --number of combined training and unlabeled samples
      --local bagSize = math.floor(defaultBagsize *2) --number of combined training and unlabeled samples

      --local rf_mod = Workflow.buildForest(trainSet.kernel, trainSet.labels, trainingIndices, queryIndices, featureIndices, options) 

      local acc_summary, acc_counts = Workflow.testAccuracy(rf, testSet.kernel, testSet.labels, classes)
      for class, t in pairs(acc_counts) do
        local ratio = t.correct / t.count
        io.write(string.format("class: %d %.02f\n", class, ratio)) 
        results.data[results:index(numLabeledSamples - 1, classMapping[class])] = ratio

      end
      print("global ratio: ", acc_summary.correct / acc_summary.count)


      --local accumulator = rf_mod:predict(testSet.kernel, {accumulator = RandomForest.accumulators.treeCounter(numClasses, 6, classMapping, numTrees)})
      --if run == numRuns and numLabeledSamples == maxSamples then
      --  for i = 0, 10 do
      --    accumulator.results[i]:print("distribution")
      --  end
      --end
    end
    --create counts Matrix:
    --OPTION
    --ATTENTION: this will probably hurt me down the road: let's create the counts-matrix in
    --C-order 

    local accumulator = rf:predict(
    trainSet.kernel(queryIndices, featureIndices), 
    {accumulator = RandomForest.accumulators.singleVoter}
    )
    local counts = Matrix.mat(#accumulator.results + 1,numClasses)
    for i = 0, #accumulator.results do
      for k,v in pairs(accumulator.results[i]) do
        if type(k) == "number" and k > 0 then
          local column = classMapping[k]
          counts.data[counts:index(i,column)] = v/Workflow.numTrees
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
    --ANALYSIS
    pickedLabels.data[numLabeledSamples - 1] = classMapping[trainSet.labels(trainingIndices[#trainingIndices], "all").data[0] ]
    bestTUV:view(numLabeledSamples - 1, "all"):set(counts:view(tuvIndices, "all"):view(0,"all"):all())

    --END ANALYSIS
    --tuv:view(testIndices, "all"):print("testIndices")
    --counts:view(tuvIndices, "all"):view({0,1,2,3,4,5},"all"):print("counts")
    --Dtools.printtable(testIndices)
    --1-based tuvIndices, for sorting reasons
    trainingIndices[#trainingIndices + 1] = queryIndices[tuvIndices[1]]
    --trainSet.kernel(trainingIndices, trainingIndices):print("TrainingKernel")

    --Dtools.printtable(activeClasses)


  end

  allResults[run - 1] = {labels = pickedLabels, results = results, tuvs = bestTUV}


end

--calculate mean + errorbars

local mean_aggregate     = Matrix.mat(maxSamples, numClasses)
local variance_aggregate = Matrix.mat(maxSamples, numClasses)

for i = 0, maxSamples - 1 do
  for j = 0, numClasses - 1 do
    local mean = 0
    local variance = 0
    for k = 0, numRuns - 1 do
      mean = mean + allResults[k].results:get(i,j)          
    end
    mean = mean / numRuns 
    for k = 0, numRuns - 1 do
      variance = variance + math.pow((allResults[k].results:get(i,j) - mean), 2)
    end
    variance = variance / (numRuns)

    mean_aggregate.data[mean_aggregate:index(i, j)] = mean
    variance_aggregate.data[variance_aggregate:index(i, j)] = variance
  end
end



local tuv_mean_aggregate     = Matrix.mat(maxSamples, numClasses)
local tuv_variance_aggregate = Matrix.mat(maxSamples, numClasses)

for i = 0, maxSamples - 1 do
  for j = 0, numClasses - 1 do
    local mean = 0
    local variance = 0
    for k = 0, numRuns - 1 do
      mean = mean + allResults[k].tuvs:get(i,j)          
    end
    mean = mean / numRuns 
    for k = 0, numRuns - 1 do
      variance = variance + math.pow((allResults[k].tuvs:get(i,j) - mean), 2)
    end
    variance = variance / (numRuns)

    tuv_mean_aggregate.data[mean_aggregate:index(i, j)] = mean
    tuv_variance_aggregate.data[variance_aggregate:index(i, j)] = variance
  end
end



--[[
for k,v  in pairs(allResults) do
v.labels:print("pickedLabels", "%d")
v.results:print("results")
end
]]--
mean_aggregate:print("mean")
variance_aggregate:print("variance")

local resultfile = io.open("results.dat", "w+")
io.output(resultfile)
io.write("NumberOfSamples\t")
for j = 1, #classnames do
  io.write(string.format("%s\t", classnames[j]))
end
io.write("\n")
for i = 0, maxSamples - 1 do
  io.write(string.format("%d-%d",i + 1, allResults[0].labels:at(i)+1))
  for j = 0, mean_aggregate.n_cols -1 do
    io.write(string.format("\t%.04f", mean_aggregate:get(i,j)))
  end
  io.write("\n")
end

io.close()


local resultfile = io.open("tuvs.dat", "w+")
io.output(resultfile)
io.write("NumberOfSamples\t")
for j = 1, #classnames do
  io.write(string.format("%s\t", classnames[j]))
end
io.write("\n")
for i = 0, maxSamples - 2 do
  --io.write(i + 1)
  io.write(string.format("%d-%d",i + 1, allResults[0].labels:at(i+1)+1))
  for j = 0, tuv_mean_aggregate.n_cols -1 do
    io.write(string.format("\t%.04f", tuv_mean_aggregate:get(i,j)))
  end
  io.write("\n")
end

io.close()



--Dtools.printtable(classnames)
--Dtools.printtable(classes)
