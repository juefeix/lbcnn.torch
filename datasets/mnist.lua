-- mnist.lua

local t = require 'datasets/transforms'

local M = {}
local MNISTDataset = torch.class('resnet.MNISTDataset', M)

function MNISTDataset:__init(imageInfo, opt, split)
	assert(imageInfo[split], split)
	self.imageInfo = imageInfo[split]
	self.split = split

	local meanstd
	local meanstdCache = opt.save .. '/meanCache.t7'
	print(meanstdCache)
	if paths.filep(meanstdCache) then
	   meanstd = torch.load(meanstdCache)
	   print('Loaded mean and std from cache.')
	else
	   local tm = torch.Timer()
	   local nSamples = math.max(10000,self.imageInfo.data:size(1))
	   print('Estimating the mean (per-channel, shared for all pixels) over ' .. nSamples .. ' randomly sampled training images')
	   local meanEstimate
	   if self.imageInfo.data:size(2) == 3 then
	   	meanEstimate = {0,0,0}
	   else
	   	meanEstimate = {0}
	   end
	   for i=1,nSamples do
	      local img = self.imageInfo.data[i]	      
	      for j=1,self.imageInfo.data:size(2) do
	         meanEstimate[j] = meanEstimate[j] + img[j]:mean()
	      end
	   end
	   for j=1,self.imageInfo.data:size(2) do
	      meanEstimate[j] = meanEstimate[j] / nSamples
	   end
	   mean = meanEstimate	   

	   print('Estimating the std (per-channel, shared for all pixels) over ' .. nSamples .. ' randomly sampled training images')
	   local stdEstimate
	   if self.imageInfo.data:size(2) == 3 then
	   	stdEstimate = {0,0,0}
	   else
	   	stdEstimate = {0}
	   end
	   for i=1,nSamples do
	      local img = self.imageInfo.data[i]
	      for j=1,self.imageInfo.data:size(2) do
	         stdEstimate[j] = stdEstimate[j] + img[j]:std()
	      end
	   end
	   for j=1,self.imageInfo.data:size(2) do
	      stdEstimate[j] = stdEstimate[j] / nSamples
	   end
	   std = stdEstimate

	   meanstd = {}
	   meanstd.mean = mean
	   meanstd.std = std
	   torch.save(meanstdCache, meanstd)
	   print('Time to estimate:', tm:time().real)
	end

	self.imageInfo.meanstd = meanstd	
end

function MNISTDataset:get(i)
   local image = self.imageInfo.data[i]:float()
   local label = self.imageInfo.labels[i]

   return {
      input = image,
      target = label,
   }
end

function MNISTDataset:size()
   return self.imageInfo.data:size(1)
end

-- Computed from entire MNIST training set
function MNISTDataset:preprocess()
   if self.split == 'train' then
      return t.Compose{
         t.GrayNormalize(self.imageInfo.meanstd),
         t.RandomCrop(32, 4),
      }
   elseif self.split == 'val' then
      return t.GrayNormalize(self.imageInfo.meanstd)
   else
      error('invalid split: ' .. self.split)
   end
end

return M.MNISTDataset