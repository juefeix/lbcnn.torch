--
--  Copyright (c) 2016, Facebook, Inc.
--  All rights reserved.
--
--  This source code is licensed under the BSD-style license found in the
--  LICENSE file in the root directory of this source tree. An additional grant
--  of patent rights can be found in the PATENTS file in the same directory.
--
--  Script to compute list of ImageNet filenames and classes
--

local M = {}

local function convertToTensor(files)
   local data, labels

   for _, file in ipairs(files) do
      local m = torch.load(file, 'ascii')
      if not data then
         data = m.data:t()
         labels = m.labels:squeeze()
      else
         data = torch.cat(data, m.data:t(), 1)
         labels = torch.cat(labels, m.labels:squeeze())
      end
   end

   -- This is *very* important. The downloaded files have labels 0-9, which do
   -- not work with CrossEntropyCriterion
   labels:add(1)
   data = data:type(torch.getdefaulttensortype())

   return {
      data = data:contiguous():view(-1, 3, 32, 32),
      labels = labels,
   }
end

function M.exec(opt, cacheFile)   
   print(" | combining dataset into a single file")
   local trainData = convertToTensor({
      opt.data .. '/cifar-10-batches-t7/data_batch_1.t7',
      opt.data .. '/cifar-10-batches-t7/data_batch_2.t7',
      opt.data .. '/cifar-10-batches-t7/data_batch_3.t7',
      opt.data .. '/cifar-10-batches-t7/data_batch_4.t7',
      opt.data .. '/cifar-10-batches-t7/data_batch_5.t7',
   })
   local testData = convertToTensor({
      opt.data .. '/cifar-10-batches-t7/test_batch.t7',
   })

   if opt.subset then
      testData.data = testData.data[{ {1,opt.tstsize} }]
      testData.labels = testData.labels[{ {1,opt.tstsize} }]
      trainData.data = trainData.data[{ {1,opt.trsize} }]
      trainData.labels = trainData.labels[{ {1,opt.trsize} }]
   end

   print(" | saving CIFAR-10 dataset to " .. cacheFile)
   torch.save(cacheFile, {
      train = trainData,
      val = testData,
   })
end

return M
