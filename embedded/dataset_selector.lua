local datasets = {}

package.path =  package.path .. ";/home/bxu/code/toolbox/datatypes/?.lua"
package.path =  package.path .. ";/home/bxu/code/toolbox/algorithms/?.lua"
package.path = package.path .. ";/home/bxu/code/toolbox/?.lua"
package.path = package.path .. ";/home/bxu/code/pknn_al/embedded/?.lua"
--local os  = require("os")
--local ffi = require("ffi")
--local narray = require("ljarray.array")
local Matrix = require("ljmatrix.densematrix")
--local RandomForest = require("randomforest.rf")
--local Dtools = require("debugtools.debug")
--local Density = require("activelearning.getDensity")
--local RangeFilter = require("rangefilter")
--local Dataset = require("dataset_selector")



local function getYeast() 

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
  --local classes = Matrix.range(2,4)
  local numClasses = #classnames
  local classMapping = Matrix.range(-1, numClasses)
  local translationTable = {}
  for i,v in ipairs(classnames) do
    translationTable[v] = i
  end
  for i = 1, #t do
    labels.data[labels:index(i - 1, 0)] = translationTable[t[i][#t[i] ] ]
    for j = 2, #t[i] - 1 do
      data.data[data:index(i - 1,j - 2)] = tonumber(t[i][j])
    end
  end
  return data, labels,numClasses, classes, classMapping
end

local function getPage()

  local file = io.input("/home/bxu/data/uci/page-blocks.data")
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
  local data   = Matrix.mat(#t, #t[1] - 1)
  local labels = Matrix.mat(#t, 1)

  for i = 1, #t do
    labels.data[labels:index(i - 1,labels.n_cols - 1)] = tonumber(t[i][#t[i] ])
    for j = 1, #t[i] - 1 do
      data.data[data:index(i - 1,j - 1)] = tonumber(t[i][j])
    end
  end
  local classes = Matrix.range(1,6)
  local numClasses = 5
  local classMapping = Matrix.range(-1,5)
  local classnames = {"1", "2", "3", "4", "5"}

  return data, labels,numClasses, classes, classMapping, classnames

end
datasets["yeast"] = getYeast
datasets["page"] = getPage
return datasets
