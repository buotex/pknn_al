--package.path = package.path .. ";/home/bxu/code/toolbox/datatypes/?.lua"
--package.path = package.path .. ";/home/bxu/code/toolbox/?.lua"

local ffi = require("ffi")
local Dtools = require("debugtools.debug")
local Matrix = require("ljmatrix.densematrix")
local unc = ffi.load("uncertainty")

ffi.cdef[[
typedef struct {
double p1;
double p2;
} p_struct;

p_struct createDirich(double * alpha, size_t length, size_t numSamples); 
p_struct createOther(double * alpha, size_t length, size_t numSamples); 
int createDirichMatrix(double * alpha, size_t length, size_t numSamples, size_t numObjects, double * results);
int createUncertaintyMatrix(double * alpha, size_t length, size_t numSamples, size_t numObjects, double * results);

]]
local TUV = {}
TUV.numSamples = 100
function TUV.convert(counts, numClasses) 
  local prior = Matrix.mat(1, numClasses)
  prior:view("all", "all"):set(1/numClasses)
  local multiplier = 1
  --counts:params()
  local alpha = prior + counts:mult(multiplier)
  return alpha
end


function TUV.tuvCalcNew(counts, marginalProbs, lambda)

  local numQueries = counts.n_rows 
  local tuv        = Matrix.mat(numQueries, 1)

  local p_matrix = Matrix.mat(numQueries, 2)
  local alpha = counts
  local errorcode = unc.createUncertaintyMatrix(alpha.data, alpha.n_cols, TUV.numSamples, alpha.n_rows, p_matrix.data)
  for i = 0, numQueries-1 do
    --tuv.data[i] = marginalProbs.data[i] * (lambda[1] * p_matrix:get(i,0) - lambda[2] * p_matrix:get(i,1))
    tuv.data[i] = lambda[1] * marginalProbs.data[i] + lambda[2] * (p_matrix:get(i,0) - p_matrix:get(i,1))
  end
  
  return tuv
end


function TUV.tuvCalc(counts, marginalProbs, lambda)

  local numQueries = counts.n_rows 
  local tuv        = Matrix.mat(numQueries, 1)

  local p_matrix = Matrix.mat(numQueries, 2)
  local alpha    = Matrix.mat(numQueries, counts.n_cols)
  for i = 0, numQueries-1 do
    alpha:view(i, "all"):set(TUV.convert(counts:rows(i), counts.n_cols):all())
  end
  local errorcode = unc.createDirichMatrix(alpha.data, alpha.n_cols, TUV.numSamples, alpha.n_rows, p_matrix.data)
  for i = 0, numQueries-1 do
    --tuv.data[i] = marginalProbs.data[i] * (lambda[1] * p_matrix:get(i,0) - lambda[2] * p_matrix:get(i,1))
    tuv.data[i] = lambda[1] * marginalProbs.data[i] + lambda[2] * (p_matrix:get(i,0) - p_matrix:get(i,1))
  end
  
  --[[
  for i = 0, numQueries-1 do
    local alpha = TUV.convert(counts:rows(i), counts.n_cols)
    local p = unc.createDirich(alpha.data, alpha.n_elem, TUV.numSamples)
    tuv.data[i] = marginalProbs.data[i] * (lambda[1] * p.p1 - lambda[2] * p.p2)
  end
  ]]--
  return tuv
end
function TUV.tableWrapper(t, lambda, numSamples)
  local matrix = Matrix.Mat(t)
  local p = unc.createDirich(matrix.data, matrix.n_elem, numSamples)
  --print(p.p1, p.p2)
  return lambda[1] * p.p1 - lambda[2] * p.p2
  --return p.p2
end

function TUV.getBestTuv(counts, marginalProbs, lambda)
  local l = lambda or {1,1}
  local tuv = TUV.tuvCalc(counts, marginalProbs, l)
  return tuv:sortIndices('descend'), tuv
end

function TUV.getBestNewTuv(counts, marginalProbs, lambda)
  local l = lambda or {1,1}
  local tuv = TUV.tuvCalcNew(counts, marginalProbs, l)
  return tuv:sortIndices('descend'), tuv
end

function TUV.tableWrapper(t, lambda, numSamples)
  local counts = Matrix.Mat(t)
  counts:reshapeTo(1, counts.n_elem)
  local numQueries = counts.n_rows 
  local tuv        = Matrix.mat(numQueries, 1)
  local p_matrix = Matrix.mat(numQueries, 2)
  local alpha = counts
  local numSamples = numSamples or TUV.numSamples
  local errorcode = unc.createUncertaintyMatrix(alpha.data, alpha.n_cols, numSamples, alpha.n_rows, p_matrix.data)
  for i = 0, numQueries - 1 do
    tuv.data[i] = lambda[1] * p_matrix:get(i,0) - lambda[2] * p_matrix:get(i,1)
  end
  return tuv.data[0]
  --return p.p2
end

local function testTUV()
  local counts = 
  Matrix.Mat{3,0.2,4,0.5,2,1,3,0.5,0.8,
  --10,0.5,0,0,0,0,0,0,0.9,
  --10,2,20,2,0,0,0,0,0.2
  } 
  counts:reshapeTo(1,9)
  counts:print()
  local marginalProbs = Matrix.Mat{0}
  local indices, tuv = TUV.getBestNewTuv(counts, marginalProbs) 
  tuv:print()
  counts(indices, "all"):print()
  tuv(indices, "all"):print()

end

--testTUV()
setmetatable(TUV, {__call= function() return TUV.getBestTuv end})
return TUV
