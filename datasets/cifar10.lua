-- cifar10.lua

local t = require 'datasets/transforms'

local M = {}
local CIFAR10Dataset = torch.class('resnet.CIFAR10Dataset', M)

function CIFAR10Dataset:__init(imageInfo, opt, split)
	assert(imageInfo[split], split)
	self.imageInfo = imageInfo[split]
	self.split = split	
end

function CIFAR10Dataset:get(i)
   local image = self.imageInfo.data[i]:float()
   local label = self.imageInfo.labels[i]

   return {
      input = image,
      target = label,
   }
end

function CIFAR10Dataset:size()
   return self.imageInfo.data:size(1)
end

local meanstd = {
      mean = {125.3, 123.0, 113.9},
      std  = {63.0,  62.1,  66.7},
}

-- Computed from entire MNIST training set
function CIFAR10Dataset:preprocess()
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

return M.CIFAR10Dataset
