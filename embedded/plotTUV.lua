local Matrix = require("ljmatrix.densematrix")
Tuv = require("getTUV")

local numPoints = 100
--local results = Matrix.mat(numPoints, numPoints)
--counts:print("print")
--tuv:print()


--tuv:print()
--tuv:reshapeTo(numPoints, numPoints)

local numClasses = 6
local prior = 1/numClasses
local scaling = 3 / (numPoints)
local resultfile = io.open("plotTUV.txt", "w+")
local numSamples = 1000
io.output(resultfile)
for j = 0, numPoints - 1 do
  for i = j, numPoints -1 do
    local first = i * scaling + prior
    local second = j * scaling + prior
    local dummy = prior
    local t = {first, second}
    for k = 3, numClasses do t[k] = dummy end
    io.write(string.format("%e %e %e \n", first, second,
    Tuv.tableWrapper(
    t, 
    {1,1}, --lambda
    numSamples   --numSamples}
    )))
  end
end
io.close()
local resultfile = io.open("plotTUV_Uncertainty.txt", "w+")
io.output(resultfile)
for j = 0, numPoints - 1 do
  for i = j, numPoints -1 do
    local first = i * scaling + prior
    local second = j * scaling + prior
    local dummy = prior
    local t = {first, second}
    for k = 3, numClasses do t[k] = dummy end
    io.write(string.format("%e %e %e \n", first, second,
    Tuv.tableWrapper(
    t, 
    {1,0}, --lambda
    numSamples   --numSamples}
    )))
  end
end
local resultfile = io.open("plotTUV_Exploration.txt", "w+")
io.output(resultfile)
for i = 0, numPoints -1 do
  for j = i, numPoints -1 do
    local first = i * scaling + prior
    local second = j * scaling + prior
    local dummy = prior
    local t = {first, second}
    for k = 3, numClasses do t[k] = dummy end
    io.write(string.format("%e %e %e \n", first, second,
    Tuv.tableWrapper(
    t, 
    {0,1}, --lambda
    numSamples --numSamples}
    )))
  end
end
