local Matrix = require("ljmatrix.densematrix")
local TUV = require("getTUV")

local function calcTUVS(accumulator, marginalProbs, numClasses, classMapping, numTrees)
  local counts = Matrix.mat(#accumulator.results + 1,numClasses)
  for i = 0, #accumulator.results do
    for k,v in pairs(accumulator.results[i]) do
      if type(k) == "number" and k > 0 then
        local column = classMapping[k]
        counts.data[counts:index(i,column)] = v/numTrees
      end
    end
  end
  --counts:print()

  --OPTION
  --create marginalProbs

  --calculate TUV, add new label
  local lambda = {1,1}

  local tuvIndices, tuv = TUV.getBestTuv(counts, marginalProbs, lambda)
  --ANALYSIS

  --END ANALYSIS
 return tuvIndices[1] + 1,1, counts:view(tuvIndices, "all"):view(0,"all"):all()
end
return calcTUVS
