package.path = package.path .. ";/home/bxu/code/toolbox/datatypes/?.lua"
package.path = package.path .. ";/home/bxu/code/toolbox/?.lua"
local ffi = require("ffi")
local narray = require("ljarray.array")
local dtools = require("debugtools.debug")

blub = 5
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
local getIndices = function(kernel, training, query, labels, numClasses, ret, numRuns)
  

  local numTrees = 10
  local forests = {}
  for i = 1, numRuns do
    --createRandomForest()
    --pick _all_ trainingData
    --pack data together


    forests[i] = tree.create({n_trees = numTrees, n_classes = numClasses})
    local numelem = training.n * 2
    local numdim  = training.n

    --define data and labels
    local testdata = narray.create({numelem, numdim}, narray.float64)
    local testlabels   = narray.create({numelem}, narray.int8)
    --fill up testdata and labels
    for j = 0, training.n-1 do
      local index = training.data[j]
      for k = 0, numdim - 1 do
        --not caching, just for readability
        local singleIndex = index * kernel.n + k
        local val = kernel.data[singleIndex]
        testdata:set2(j,k,val)

      end

      local randomIndex = query.data[math.random(query.n-1)]
      for k = 0, numdim - 1 do
        local singleIndex = randomIndex * kernel.n + k
        local val = kernel.data[singleIndex]
        testdata:set2(j+training.n, k, val)
      end

      testlabels:set1(j, labels.data[index])
      testlabels:set1(j + training.n, 0)
    end
    --dtools.printarray(testdata)
    --dtools.printarray(testlabels)




    forests[i]:learn( testdata, testlabels )



  end 
  --pick random samples



end



wrapper = function(
  vkernelData, kernelN,
  vtrnIndData, trnN,
  vqueIndData, queN,
  vlabelData, labelN,
  numClasses,
  numAl,
  vretIndices
  )
  math.randomseed(os.time())
  local dummy = math.random()


  --safety: Fill up return with 1, so we don't crash terribly

  print("blub")
  local kernelData = ffi.new("double *", vkernelData)
  local trnIndData = ffi.new("double *", vtrnIndData)
  local queIndData = ffi.new("double *", vqueIndData)
  local labelData = ffi.new("double *", vlabelData)
  local retIndices = ffi.new("uint64_t *", vretIndices)
  print(kernelN, trnN, queN, numAl)
  local kernel = { data = kernelData, n = kernelN }
  local training = { data = trnIndData, n = trnN }
  local query = { data = queIndData, n = queN }
  local labels = { data = labelData, n = labelN}
  local ret = { data = retIndices, n = numAl }

  for i = 0, numAl - 1 do
    ret.data[i] = 1
  end
  local numTrees = 1
  getIndices(kernel, training, query, labels, numClasses, ret, numTrees)

  return 
end

--f = function(Kernel, trainingIndices, queryIndices, numAl)


