


local os  = require("os")
local ffi = require("ffi")
local Matrix = require("ljmatrix.densematrix")
local Dtools = require("debugtools.debug")

local modname = ...
local M = {}
_G[modname] = M
package.loaded[modname] = M
M.debug = true

local function inRange(min, max)
  return function(vector)
    for i = 0, #vector do
      if vector[i] < min[i] or vector[i] > max[i] then
        --print("beep", vector[i], min[i], max[i])
        return false
      end
    end
    return true
  end
end

function M.inverseFilterPoints(data, predicate)
 
  local indices = {} 

  for i = 0, data.n_rows - 1 do
    if not predicate(data:rows(i):materialize():toTable(nil, 0)) then 
      indices[#indices + 1 ] = i
    end
  end

  return indices
end


function M.inverseFilterRanges(data, min, max)

return M.inverseFilterPoints(data, inRange(min, max))
end


function M.filterPoints(data, predicate)
 
  local indices = {} 

  for i = 0, data.n_rows - 1 do
    if predicate(data:rows(i):materialize():toTable(nil, 0)) then 
      indices[#indices + 1 ] = i
    end
  end

  return indices
end


function M.filterRanges(data, min, max)

return M.filterPoints(data, inRange(min, max))
end
