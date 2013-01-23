local ffi = require("ffi")
local Matrix = require("ljmatrix.densematrix")
local Selector = ffi.load("selector")

ffi.cdef[[
  int getSampleIndex(double * data, int n_elem, double alpha);
]]


return function(tuvs)

--local index = Selector.getSampleIndex(testdata.data, testdata.n_elem, 2/3)
local index = Selector.getSampleIndex(tuvs.data, tuvs.n_elem, 1/3)
return index
end
--local testdata = Matrix.Mat{0.4,0.6,2,0.7,0.3,0.2}
--local index = Selector.getSampleIndex(testdata.data, testdata.n_elem, 1.0)
--print(testdata:at(index))
