local Matrix = require("ljmatrix.densematrix")
Tuv = require("getTUV")

local numPoints = 100
--local results = Matrix.mat(numPoints, numPoints)
--counts:print("print")
--tuv:print()


--tuv:print()
--tuv:reshapeTo(numPoints, numPoints)

local numClasses = 13
local prior = 0
local scaling = 20 / (numPoints)
local numSamples = 1000
local function writeValues(outputfile, lambda)
  io.output(outputfile)

  for j = 0, numPoints - 1 do
    for i = j, numPoints -1 do
      local first = i * scaling + prior
      local second = j * scaling + prior
      local third = second
      local dummy = 1
      local t = {first, dummy, second, dummy, third}
      for k = 6, numClasses do t[k] = dummy end
      t[numClasses] = 0.5
      io.write(string.format("%e %e %e \n", first, second,
      Tuv.tableWrapper(
      t, 
      lambda, --lambda
      numSamples   --numSamples}
      )))
    end
  end

  io.close()

end


local resultfile = io.open("plotTUV.txt", "w+")
writeValues(resultfile, {1,1})
local resultfile = io.open("plotTUV_Uncertainty.txt", "w+")
writeValues(resultfile, {1,0})

local resultfile = io.open("plotTUV_Exploration.txt", "w+")
writeValues(resultfile, {0,1})
