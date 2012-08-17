local ffi = require("ffi")


wrapper = function(
  vkernelData, kernelN,
  vtrnIndData, trnN,
  vqueIndData, queN,
  numAl,
  vretIndices
  )

  print("blub")
  local kernelData = ffi.new("double *", vkernelData)
  local trnIndData = ffi.new("double *", vtrnIndData)
  local queIndData = ffi.new("double *", vqueIndData)
  local retIndices = ffi.new("uint64_t *", vretIndices)
  print(kernelN, trnN, queN, numAl)
  
  for i = 0, numAl - 1 do
    retIndices[i] = math.random(0,queN - 1)
  end
  return 
end

--f = function(Kernel, trainingIndices, queryIndices, numAl)


