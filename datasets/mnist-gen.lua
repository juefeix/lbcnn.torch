-- mnist-gen.lua

local M = {}

local function convertToTensor(files)
   local data, labels

   for _, file in ipairs(files) do      
      local f = torch.load(file, 'ascii')      
      data = f.data:type(torch.getdefaulttensortype())
      labels = f.labels
   end

   return {
      data = data:contiguous():view(-1, 1, 32, 32),
      labels = labels,
   }
end

function M.exec(opt, cacheFile)   
   print(" | combining dataset into a single file")
   local trainData = convertToTensor({
      opt.data .. '/mnist/train_32x32.t7',
   })
   local testData = convertToTensor({
      opt.data .. '/mnist/test_32x32.t7',
   })

   print(" | saving MNIST dataset to " .. cacheFile)
   torch.save(cacheFile, {
      train = trainData,
      val = testData,
   })
end

return M