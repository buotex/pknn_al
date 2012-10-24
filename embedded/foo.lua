package.path = package.path .. ";/home/bxu/code/toolbox/datatypes/?.lua"
package.path = package.path .. ";/home/bxu/code/toolbox/?.lua"
local ffi = require("ffi")
local Dtools = require("debugtools.debug")
local Matrix = require("ljmatrix.densematrix")
local RandomForest = require("randomforest.rf")
RandomForest.Debug = false


--Documentation: 
--This lua function is passed all the arguments from its C parent function (with
--a bit of parsing):
--3 Blocks of double-typed data (because matlab uses double types for indices,
--sad as it is), which consist of a (quadratic) Kernel-matrix of size kernelN x
--kernelN, the indices of the training data and the indices of the possible
--query data (both sized trnN and queN, respectively, because it's easier to
--handle the splitting into C-structs in C, as the necessary matlab includes are
--kinda... wonky to do in lua-ffi.
--

--Arguments:
--numTrees
local getCounts = function(kernel, trainingIndices, queryIndices, labels, numClasses, numTrees, returnData, featureIdCounts, testData, testVotes)
  
  local rf = RandomForest.create(numTrees)

  local defaultData = kernel(trainingIndices, trainingIndices)
  local defaultLabels = labels(trainingIndices, "all")
  local bagSize = math.floor(defaultData.n_rows * 2)
  local rfOptions = { bootstrapper = 
  RandomForest.policies.queryBootstrapper(defaultData, defaultLabels, bagSize),

  --stopPolicy = RandomForest.policies.stopBinary(),
  --splitPolicy = RandomForest.policies.oblique(RandomForest.heuristics.gini)
}
  print("options finished")

  rf:learn(kernel("all", trainingIndices), labels("all", "all"), rfOptions) 
--count ids used for splitting
  print("learning finished")
  --Dtools.printtable(trainingIndices)
  --Dtools.waitInput()
  local accumulator = rf:predict(kernel(queryIndices, trainingIndices))
  print("prediction finished")
  --what are we even doing here? Count how often features are used, to get a
  --ranking
  --Shouldn't that totally break with different featureIDs?
  
  --Doesn't do too much-------------------------------------------------------
  --Start  Block FeatureIdCounts
  --[[
  local fIdCounts = rf:visitAllNodes(RandomForest.accumulators.featureIdCounter)

  local index = 0
  --k,v :
  for k,v in pairs(fIdCounts.results) do
    featureIdCounts.data[index] = trainingIndices[k]
    featureIdCounts.data[featureIdCounts.n_cols + index] = v
    index = index + 1
  end
  ]]--
  --End Block FeatureIdCounts
  --Dummy:
  featureIdCounts:view("all", "all"):set(1)
  ------------------------------------------------------------------------------
  --Dtools.printtable(accumulator.results[3])
  --Dtools.waitInput()


  returnData:view("all", "all"):set(0)



  for i = 0, #accumulator.results do
    for k,v in pairs(accumulator.results[i]) do
      if type(k) == "number" and k > 0 then
        returnData.data[returnData:index(k-1, i)] = v
      end
    end
  end

  if testData and testVotes then
    local accumulator = rf:predict(testData("all", trainingIndices), {accumulator = RandomForest.accumulators.singleVoter})
    print("this worked out")
    for i= 0, #accumulator.results do
      testVotes.data[i] = accumulator.results[i].label
    end
  end

  return 
end


wrapper = function(
  vkernelData, kernelN,
  vtrnIndData, trnN,
  vqueIndData, queN,
  vlabelData, labelN,
  numClasses,
  numTrees,
  vretData,
  vfeatureIdCounts,
  vtestData,
  testN,
  vvotesData
  )
  math.randomseed(os.time())
  local dummy = math.random()


  --safety: Fill up return with 1, so we don't crash terribly

  local kernelData = ffi.new("double *", vkernelData)
  local trnIndData = ffi.new("double *", vtrnIndData)
  local queIndData = ffi.new("double *", vqueIndData)
  local labelData = ffi.new("double *", vlabelData)
  local retData = ffi.new("double *", vretData)
  local featureIdCountsData = ffi.new("double *", vfeatureIdCounts)
  local testDataData = ffi.new("double* ", vtestData)
  local votesData    = ffi.new("double* ", vvotesData)
  local testData, testVotes
  if vtestData then
    testData = Matrix.mat(testN, trnN, testDataData)
    testVotes = Matrix.mat(1, testN, votesData) --transpose because of matlab
  end

  local kernel = Matrix.mat(kernelN, kernelN, kernelData)
  local function shiftM1(x) return x - 1 end
  local test = Matrix.mat(trnN, 1, trnIndData)
  local training = Matrix.mat(trnN, 1, trnIndData):toTable(shiftM1)
  local query    = Matrix.mat(queN, 1, queIndData):toTable(shiftM1)
  local labels   = Matrix.mat(labelN, 1, labelData)
  local ret      = Matrix.mat(numClasses, queN, retData)
  local featureIdCounts = Matrix.mat(2, trnN, featureIdCountsData) --transpose because of matlab
  print("initialization finished")
  --local numTrees = 1
  getCounts(kernel, training, query, labels, numClasses, numTrees, ret, featureIdCounts, testData, testVotes)

  return 
end

--f = function(Kernel, trainingIndices, queryIndices, numAl)


