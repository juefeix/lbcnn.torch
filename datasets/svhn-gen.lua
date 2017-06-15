-- svhn.lua

local M = {}

local function convertToTensor(files)
   local data, labels

   for _, file in ipairs(files) do
      local m = torch.load(file, 'ascii')
      data = m.X:transpose(3,4)
      labels = m.y[1]
   end
   data = data:type(torch.getdefaulttensortype())

   return {
      data = data:contiguous():view(-1, 3, 32, 32),
      labels = labels,
   }
end

function M.exec(opt, cacheFile)   
   print(" | combining dataset into a single file")
   local trainData = convertToTensor({
      opt.data .. '/svhn/train_32x32.t7',
   })
   local testData = convertToTensor({
      opt.data .. '/svhn/test_32x32.t7',
   })

   print(" | saving SVHN dataset to " .. cacheFile)
   torch.save(cacheFile, {
      train = trainData,
      val = testData,
   })
end

return M
