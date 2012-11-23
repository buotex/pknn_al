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
local Density = require("activelearning.getDensity")

math.randomseed(0)


local zmq = require("zmq")
local ctx = zmq.init()
local s = ctx:socket(zmq.REQ)
s:connect("ipc:///tmp/zmq-test")
s:send("toy_example")


local data = Matrix.fromZMQ(s)
local labels = Matrix.fromZMQ(s)
s:recv()
data:send(s)
labels:send(s)

local function parseLabels(labels)
  local classes = {}
  for i = 0, labels.n_elem - 1 do
    classes[labels.data[i]] = labels.data[i]  
  end
  local classnames = {}
  for k,v in ipairs(classes) do
    classnames[k] = tostring(v)
  end
  local numClasses = #classes
  local classMapping = Matrix.range(-1, numClasses)
  return classnames, classes, numClasses, classMapping
end

local classnames, classes, numClasses, classMapping = parseLabels(labels)

--Dtools.printtable(classes)
RandomForest.Debug = false

--Keywords: STATISTICS, OPTION, DEBUG
local SHOW_ACCURACY = true
local PLOT = true

local densities = Density.query(data, 4)
for i = 0, densities.n_elem - 1 do
  densities.data[i] = math.exp(densities.data[i])
end
densities:print()
densities:view("all", "all"):set(1)
--local zeroes = Matrix.mat(densities.n_rows, densities.n_cols)
--local densities = zeroes - densities

local numRuns = 1
local run = 0 

local allResults = {}
local maxSamples = 45
while run < numRuns do
  run = run + 1

  local candidateIndices = Workflow.filterClasses(labels, classes)


  --local densities = trainSet.kernel:view(candidateIndices, candidateIndices):sum("cols")
  --densities:print("densities")
  --local maxMatrix, indexMatrix = densities:max()
  --  indexMatrix:print("indexMatrix")

  --local trainingIndices = {[0] = candidateIndices[indexMatrix.data[0] ] }

  local trainingIndices = {[0] = 0}
  local numLabeledSamples = 0

  local results = Matrix.mat(maxSamples, numClasses)
  local pickedLabels = Matrix.mat(maxSamples, 1)
  local bestTUV      = Matrix.mat(maxSamples, numClasses)
  while numLabeledSamples < maxSamples do

    numLabeledSamples = numLabeledSamples + 1
    print("samples: ", numLabeledSamples)
    local queryIndices = Workflow.updateQueryIndices(candidateIndices, trainingIndices) 
    --trainSet.labels(queryIndices, "all"):print("query-labels")
    --Dtools.waitInput()

    --use the kernel-dimensions implied by the trainingIndices as features /
    --variables.
    --local featureIndices = "all"
    local featureIndices = "all"

    --local labels, classesFound = trainSet.labels(trainingIndices, "all"):count() --TODO

    --Dtools.printtable(trainingIndices, "trainingIndices")
    --Dtools.printtable(queryIndices, "queryIndices")
    --Dtools.printtable(trainSet.labels(trainingIndices, "all"):count(), "TrainingLabels")

    local rf = Workflow.buildForest(
    data, labels, trainingIndices, queryIndices, featureIndices
    ) 
    print("built")
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

    --local testSet = testSet1

    --OPTIONAL_ACCURACY_START

    if SHOW_ACCURACY  then

      --local bagSize = math.floor(classesFound *2) --number of combined training and unlabeled samples
      --local bagSize = math.floor(defaultBagsize *2) --number of combined training and unlabeled samples

      --local rf_mod = Workflow.buildForest(trainSet.kernel, trainSet.labels, trainingIndices, queryIndices, featureIndices, options) 

      local acc_summary, acc_counts, predictions = Workflow.testAccuracy(rf, data(queryIndices, "all"), labels(queryIndices, "all"), classes)
      for class, t in pairs(acc_counts) do
        local ratio = t.correct / t.count
        io.write(string.format("class: %d %.02f\n", class, ratio)) 
        results.data[results:index(numLabeledSamples - 1, classMapping[class])] = ratio

      end
      print("global ratio: ", acc_summary.correct / acc_summary.count)
      Matrix.Mat(trainingIndices):send(s)
      Matrix.Mat(queryIndices):send(s)
      predictions:send(s)

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
    data(queryIndices, featureIndices), 
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
    --local marginalProbs = trainSet.kernel(queryIndices, "all"):sum("cols")
    --local marginalProbs = Matrix.mat(#t, 1)
    --marginalProbs:view("all", "all"):set(1)

    local marginalProbs = densities(queryIndices, "all")
    --calculate TUV, add new label
    local lambda = {1,1}

    local tuvIndices, tuv = TUV.getBestTuv(counts, marginalProbs, lambda)
    tuv:send(s)
    counts:send(s)
    --ANALYSIS
    pickedLabels.data[numLabeledSamples - 1] = classMapping[labels(trainingIndices[#trainingIndices], "all").data[0] ]
    bestTUV:view(numLabeledSamples - 1, "all"):set(counts:view(tuvIndices, "all"):view(0,"all"):all())

    --END ANALYSIS
    --tuv:view(testIndices, "all"):print("testIndices")
    --counts:view(tuvIndices, "all"):view({0,1,2,3,4,5},"all"):print("counts")
    --Dtools.printtable(testIndices)
    --1-based tuvIndices, for sorting reasons
    trainingIndices[#trainingIndices + 1] = queryIndices[tuvIndices[1] ]
    Dtools.printtable(labels(trainingIndices, "all"):count())
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


s:close()
ctx:term()

--mean_aggregate:print("mean")
--variance_aggregate:print("variance")
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
--]]
