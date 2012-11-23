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
Workflow.numTrees = 200
math.randomseed(0)

local file = io.input("/home/bxu/data/uci/yeast.data")
local t = {}
while true do
  local line = io.read("*line")
  if not line then break end
  local linetable = {}
  for w in string.gmatch(line,"([^%s]+)") do 
--    print(w) 
    linetable[#linetable+1] = w
  end
  t[#t + 1] = linetable 
end
local data   = Matrix.mat(#t, #t[1] - 2)
local labels = Matrix.mat(#t, 1)
--data:view(0, "all"):print()
local classnames = {"CYT", "ERL", "EXC", "ME1", "ME2", "ME3", "MIT", "NUC",
"POX", "VAC" }
local classes = Matrix.range(1,#classnames + 1)
local numClasses = #classnames
local classMapping = Matrix.range(-1, numClasses)
local translationTable = {}
for i,v in ipairs(classnames) do
  translationTable[v] = i
end
--Dtools.printtable(classMapping)
--local classMapping = Matrix.range(-1,#classnames)
--local classnames = {"A","B","C","D","E","F","G","H","I","J","K","L","M","N","O","P","Q","R","S","T","U","V","W","X","Y","Z"}
--local classnames = {"text", "horiz.line", "graphic", "vert.line", "picture"}


for i = 1, #t do
  labels.data[labels:index(i - 1, 0)] = translationTable[t[i][#t[i] ] ]
  for j = 2, #t[i] - 1 do
    data.data[data:index(i - 1,j - 2)] = tonumber(t[i][j])
  end
end

--labels:print()

local data = data:view("all", {0,1,2,3,4,6,7}):materialize()


--Dtools.printtable(classes)
RandomForest.Debug = false

--Keywords: STATISTICS, OPTION, DEBUG
local SHOW_ACCURACY= true

local featureIndices = "all"
--data:print()
--[[
local densities = Density.query(data, 10)
local zeroes = Matrix.mat(densities.n_rows, densities.n_cols)
local densities = zeroes - densities

--[[
do
  local zmq = require("zmq")
  local ctx = zmq.init()
  local s = ctx:socket(zmq.REQ)
  s:connect("ipc:///tmp/zmq-test")
  data:send(s)
  labels:send(s)
  s:close()
  ctx:term()
end
]]--

local accuracy_counts = Workflow.leaveOneOut(data, labels, classes, candidateIndices, featureIndices)

for k,v in pairs(accuracy_counts) do
io.write(string.format("Class %d: %.02f\n", k, v.correct/v.count))
end

--[[
local rf = RandomForest.create(200)
rf:learn(data(trainingIndices, "all"), labels(trainingIndices, "all"))
]]--
