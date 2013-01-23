#!/usr/local/bin/luajit

package.path =  package.path .. ";/home/bxu/code/toolbox/datatypes/?.lua"
package.path =  package.path .. ";/home/bxu/code/toolbox/algorithms/?.lua"
package.path = package.path .. ";/home/bxu/code/toolbox/?.lua"
package.path = package.path .. ";/home/bxu/code/pknn_al/embedded/?.lua"
local os  = require("os")
local ffi = require("ffi")
--local narray = require("ljarray.array")
local Matrix = require("ljmatrix.densematrix")
local RandomForest = require("randomforest.rf")
local Dtools = require("debugtools.debug")
local TUV = require("getTUV")
local Workflow = require("activelearning.workflow")
--local Density = require("activelearning.getDensity")
local RangeFilter = require("rangefilter")
local Selector = require("selector")
local Dataset = require("dataset_selector")

local ENABLE_ZMQ=false
math.randomseed(0)

if not arg[1] then
  print("Please select a dataset yeast/page")
  error()
end
local data,labels,numClasses,classes,classMapping,classnames = Dataset[arg[1]]()

local betafactor = nil

if not arg[2] then
  print("betafactor set to default 1.5")
  betafactor = 1.5
else
  betafactor = arg[2]
end
local s,ctx = nil, nil
--Dtools.printtable(classes)
RandomForest.Debug = false

--Keywords: STATISTICS, OPTION, DEBUG
local SHOW_ACCURACY = true
local PLOT = true

local numRuns = 3
local run = 0 

local allResults = {}
local maxSamples = 150
while run < numRuns do
  run = run + 1

  Dtools.printtable(classes)

  local candidateIndices = Workflow.filterClasses(labels, classes)


  --local densities = trainSet.kernel:view(candidateIndices, candidateIndices):sum("cols")
  --densities:print("densities")
  --local maxMatrix, indexMatrix = densities:max()
  --  indexMatrix:print("indexMatrix")

  --local trainingIndices = {[0] = candidateIndices[indexMatrix.data[0] ] }

  local trainingIndices = {[0] = 0}
  local numLabeledSamples = 0

  local results = Matrix.mat(maxSamples, numClasses)
  local detectionRate = Matrix.mat(maxSamples, 1)
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


    local minData, minIndices = data(trainingIndices, featureIndices):min("rows")
    local maxData, maxIndices = data(trainingIndices, featureIndices):max("rows")
    --minData:print()
    --maxData:print()
    local min = minData:toTable()
    local max = maxData:toTable()
    --Dtools.waitInput() 

    --local RangeFilter = require("rangefilter")
    --points in rFilteredIndices are OUTSIDE the range spanned by min/max
    local queryData = data:view(queryIndices, featureIndices)
    local rFilteredIndicesView = RangeFilter.inverseFilterRanges(queryData, min, max)
    local rFilteredIndices = queryData:view(rFilteredIndicesView, "all").row_indices
   
    --points in filteredIndices are IN the range spanned by min/max
    local filteredIndicesView = RangeFilter.filterRanges(queryData, min, max)
    local filteredIndices = queryData:view(filteredIndicesView, "all").row_indices

    --data:view(filteredIndices, featureIndices):print("filteredData")
    local buildingIndices
    --local expanding = true
    
    --if numLabeledSamples % 2 == 1 or #filteredIndices == 0 then
    --[[
    if #rFilteredIndices >= 0.1 * #queryIndices then
      buildingIndices = rFilteredIndices
    else
      buildingIndices = filteredIndices
      expanding = false
    end
    ]]--
    buildingIndices=queryIndices
    --if #rFilteredIndices > #trainingIndices then
    --[[
    if 0 > 1 then
      buildingIndices = rFilteredIndices
    else
      buildingIndices = queryIndices
      expanding = false
    end
    ]]--

    print("number of query Indices: ", #buildingIndices)
    local rf = Workflow.buildForest(
    data, labels, trainingIndices, buildingIndices, featureIndices, nil, betafactor
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

    --local testSet = testSet1

    --OPTIONAL_ACCURACY_START

    if SHOW_ACCURACY  then

      --local bagSize = math.floor(classesFound *2) --number of combined training and unlabeled samples
      --local bagSize = math.floor(defaultBagsize *2) --number of combined training and unlabeled samples

      --local rf_mod = Workflow.buildForest(trainSet.kernel, trainSet.labels, trainingIndices, queryIndices, featureIndices, options) 
      local rf_mod = RandomForest.create(200)
      rf_mod:learn(data(trainingIndices, "all"), labels(trainingIndices, "all"))
      local acc_summary, acc_counts, predictions = Workflow.testAccuracy(rf_mod, data(queryIndices, "all"), labels(queryIndices, "all"), classes)
      for class, t in pairs(acc_counts) do
        local ratio = t.correct / t.count
        if ratio ~= ratio then ratio = 1 end
        io.write(string.format("class: %d %.02f\n", class, ratio)) 
        results.data[results:index(numLabeledSamples - 1, classMapping[class])] = ratio

      end
      print("global ratio: ", acc_summary.correct / acc_summary.count)
      if ENABLE_ZMQ then
        Matrix.Mat(trainingIndices):send(s)
        Matrix.Mat(queryIndices):send(s)
        predictions:send(s)
      end
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

    ----[[ this is the old version for predicting/voting
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
    --    ]]--
    --[[
    local accumulator = rf:predict(
    data(queryIndices, featureIndices), 
    {accumulator = RandomForest.accumulators.treeCounter(numClasses, 6, classMapping, Workflow.numTrees)}
    )

    local counts = accumulator.distributions
    ]]--

    --counts:print()

    --OPTION
    --create marginalProbs
    --local marginalProbs = trainSet.kernel(queryIndices, "all"):sum("cols")
    --local marginalProbs = Matrix.mat(#t, 1)
    --marginalProbs:view("all", "all"):set(1)
    if ENABLE_ZMQ then
      s:send(tostring(trainingIndices[#trainingIndices]))
    end
    if ENABLE_ZMQ then
      local densities_experimental = Matrix.fromZMQ(s)
    end
    --densities_experimental:params()
    --Dtools.waitInput()
    --local meta = s:recv()
    --local dat = s:recv()
    --print(meta)
    --s:send("OK")
    local marginalProbs = nil
    if ENABLE_ZMQ then
      s:recv()
      --marginalProbs = densities_experimental(queryIndices, "all")
    end
    marginalProbs = Matrix.mat(#queryIndices, #featureIndices)


    --marginalProbs:print()
    marginalProbs:view("all", "all"):set(1)
    --marginalProbs:print()
    --calculate TUV, add new label
    local lambda = {0,1}

    local tuvIndices, tuv = TUV.getBestTuv(counts, marginalProbs, lambda)
     
    


    --ANALYSIS

    local trainingIndicesHistogram, detectionCount = labels(trainingIndices, "all"):count()
    Dtools.printtable(trainingIndicesHistogram)

    detectionRate.data[numLabeledSamples - 1] = detectionCount / (#classes + 1)
    
    pickedLabels.data[numLabeledSamples - 1] = classMapping[labels(trainingIndices[#trainingIndices], "all").data[0] ]
    bestTUV:view(numLabeledSamples - 1, "all"):set(counts:view(tuvIndices, "all"):view(0,"all"):all())

    --END ANALYSIS
    --tuv:view(testIndices, "all"):print("testIndices")
    --counts:view(tuvIndices, "all"):view({0,1,2,3,4,5},"all"):print("counts")
    --Dtools.printtable(testIndices)
    --1-based tuvIndices, for sorting reasons

    --use best tuv as new trainingIndex
    --trainingIndices[#trainingIndices + 1] = queryIndices[tuvIndices[1] ]
    --Pick random sample
    --trainingIndices[#trainingIndices + 1] = queryIndices[math.random(0,#queryIndices)]
--[[
    if expanding then
      --trainingIndices[#trainingIndices + 1] = queryIndices[tuvIndices[1] ]
      trainingIndices[#trainingIndices + 1] = queryIndices[goodIndex]
    else
      --trainingIndices[#trainingIndices + 1] = queryIndices[tuvIndices[1] ]
      trainingIndices[#trainingIndices + 1] = queryIndices[goodIndex]
    end
    ]]--
    trainingIndices[#trainingIndices + 1] = queryIndices[tuvIndices[1]]
    --for random selection, use the next line instead:
    --trainingIndices[#trainingIndices +1] =
    --queryIndices[tuvIndices[math.random(1, #tuvIndices)]]
    --
    --just use the TUV based on density
    --trainingIndices[#trainingIndices + 1] = queryIndices[tuvIndices[1] ]

    --trainSet.kernel(trainingIndices, trainingIndices):print("TrainingKernel")

    --Dtools.printtable(activeClasses)


  end

  allResults[run - 1] = {labels = pickedLabels, results = results, tuvs = bestTUV, detection = detectionRate}


end

--calculate mean + errorbars

local detectionRate_aggregate     = Matrix.mat(maxSamples, 1)
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
  local detection = 0
  for k = 0, numRuns - 1 do
    detection = detection + allResults[k].detection:at(i)
  end
  detectionRate_aggregate.data[i] = detection / numRuns

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

if ENABLE_ZMQ then
  s:close()
  ctx:term()
end

--mean_aggregate:print("mean")
--variance_aggregate:print("variance")

local resultfile = io.open("results/" .. arg[1] .. "_constant_" .. tostring(betafactor) .. ".dat", "w+")
io.output(resultfile)
io.write("NumberOfSamples\t")
for j = 1, #classnames do
  io.write(string.format("%s\t", classnames[j]))
end
io.write("\n")
for i = 0, maxSamples - 1 do
  io.write(string.format("%d-%f",i + 1, detectionRate_aggregate.data[i]))
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
