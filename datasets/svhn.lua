-- mnist.lua

local t = require 'datasets/transforms'

local M = {}
local SVHNDataset = torch.class('resnet.SVHNDataset', M)

function SVHNDataset:__init(imageInfo, opt, split)
	assert(imageInfo[split], split)
	self.imageInfo = imageInfo[split]
	self.split = split	
end

function SVHNDataset:get(i)
   local image = self.imageInfo.data[i]:float()
   local label = self.imageInfo.labels[i]

   return {
      input = image,
      target = label,
   }
end

function SVHNDataset:size()
   return self.imageInfo.data:size(1)
end

local meanstd = {
	   mean = {111.6, 113.2, 120.6},
	   std  = {30.6,  31.4,  26.8},
}

-- Computed from entire MNIST training set
function SVHNDataset:preprocess()
   if self.split == 'train' then
   	  return t.Compose{
         t.ColorNormalize(meanstd),
         t.HorizontalFlip(0.5),
         t.RandomCrop(32, 4),
      }
   elseif self.split == 'val' then
      return t.ColorNormalize(meanstd)
   else
      error('invalid split: ' .. self.split)
   end
end

return M.SVHNDataset