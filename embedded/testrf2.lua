package.path =  package.path .. ";/home/bxu/code/toolbox/datatypes/?.lua"
package.path = package.path .. ";/home/bxu/code/toolbox/?.lua"
local os  = require("os")
local ffi = require("ffi")
--local narray = require("ljarray.array")
local Matrix = require("ljmatrix.densematrix")
local RandomForest = require("randomforest.rf")
local Dtools = require("debugtools.debug")

RandomForest.Debug = false

local kernelfile = io.input("kernel")
local dump = io.read("*line")
local kernelM = io.read("*number")
local kernelN = io.read("*number")
local dump = io.read("*line")
local kernelmatrix = io.read("*all")
io.close()
local labelsfile = io.input("labels")
local dump = io.read("*line")
local labelsM = io.read("*number")
local labelsN = io.read("*number")
local dump = io.read("*line")
local labelsmatrix = io.read("*all")
io.close()
local kernelp =  ffi.cast("double*", kernelmatrix)
local labelsp =  ffi.cast("double*", labelsmatrix)
local kernel = Matrix.mat(kernelM, kernelN, kernelp)
local labels = Matrix.mat(labelsM, labelsN, labelsp)


--Dtools.printarray(kernel)
--Dtools.printarray(labels)

local indexarrays = {}
local i = 0
for label in labels:all() do
  indexarrays[label] = indexarrays[label] or {}
  local t = indexarrays[label]
  t[#t + 1] = i
  i = i + 1
end

local samplesPerClass = 1
local candidateIndices = {}
local trainingIndices = {}
local queryIndices = {}
local classes = Matrix.range(1,7)

for k,i in pairs(classes) do
  for j = 1, #indexarrays[i] do
    candidateIndices[#candidateIndices + 1] = indexarrays[i][j]
  end
end

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


local rf = RandomForest.create(300)
local defaultBagSize = #trainingIndices
local defaultIndices = trainingIndices
--local defaultIndices = Matrix.range(0,defaultBadSize)
--local shift = 40
--
local defaultData = kernel(defaultIndices, defaultIndices)
local defaultLabels = labels(defaultIndices, "all")

--local bagSize = defaultBagSize * 2 --number of combined training and unlabeled samples
local bagSize = math.floor(defaultBagSize *(2)) --number of combined training and unlabeled samples
local dimensions = defaultIndices

local options = {bootstrapper = 
RandomForest.policies.queryBootstrapper(
defaultData, defaultLabels, bagSize),
splitPolicy = RandomForest.policies.oblique(RandomForest.heuristics.gini),
stopPolicy = RandomForest.policies.stopBinary()
}
--Dtools.printtable(options)
--defaultLabels:print("labels")
rf:learn(kernel(queryIndices, defaultIndices), labels(queryIndices, "all"), options)

local leafSizeCounts = rf:visitAllNodes(RandomForest.accumulators.leafSizeCounter)
local classSizeCounts = rf:visitAllNodes(RandomForest.accumulators.classSizeCounter):finalize()
local featureIdCounts = rf:visitAllNodes(RandomForest.accumulators.featureIdCounter)


--Dtools.printtable(classSizeCounts.numSamples)


Dtools.printtable(leafSizeCounts.results, "Number of leaf nodes with size #")
--Dtools.printtable(classSizeCounts.results, "Number of Elements per node per class")
--Dtools.printtable(featureIdCounts.results, "Ids used for splitting")
--]]--


print("finished learning")

local accumulator = rf:predict(kernel(queryIndices, defaultIndices), {accumulator = RandomForest.accumulators.singleVoter})
local accuracy = {}
local accuracy_counts = {}
for _,i in pairs(classes) do
  accuracy_counts[i] = {correct = 0, count = 0}
end
for i = 0, #accumulator.results do
  --print(accumulator.results[i].label)
  --print(labels:view(queryIndices, "all"):get(i,0))
  local correctLabel = labels:view(queryIndices, "all"):get(i,0)
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
--Dtools.printtable(accuracy)

--[[
print("elapsed time: ", os.time() - startTime)
--rf.trees[10]:print()

--Dtools.waitInput()
--[[

--defaultLabels:print()
--rf.trees[2]:print()
--Dtools.printtable(accumulator.results)
--Dtools.printtable(accumulator.results[0])

local votes = 0
for i = 0, #accumulator.results do
print("votecount", accumulator.results[i].n_votes)
print("non_zero_count", accumulator.results[i].non_zero_votes)
votes = votes + accumulator.results[i].non_zero_votes
end
print(votes)

print(rf.trees[1]:traceLabel(defaultLabels:at(15)))
local votes = 0
local testLabels = rf:predict(defaultData)
print("elapsed time: ", os.time() - startTime)
for i = 0, #testLabels.results do
-- print("real label: ", defaultLabels:at(i))
-- Dtools.printtable(testLabels.results[i])
local votecount = testLabels.results[i].n_votes
local non_zero_count = testLabels.results[i].non_zero_votes
print ("votecount", votecount)
print("non_zero_count", non_zero_count)
if votecount > 0 then 
print(i); 
print("label: ", defaultLabels:at(i))
local path, featureVals, featureIds = rf.trees[1]:traceLabel(defaultLabels:at(i))
Dtools.printtable(path)
Dtools.printtable(featureVals)
Dtools.printtable(featureIds)
if featureIds then  defaultData(i, featureIds):print("data") end
print("taken path:", testLabels.results[i].path)
--    error("") 
end
end
]]--
