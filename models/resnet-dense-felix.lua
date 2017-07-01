-- resnet.lua

local nn = require 'nn'
require 'math'
require 'cunn'
require 'cutorch'

local Convolution = cudnn.SpatialConvolution
local Avg = cudnn.SpatialAveragePooling
local ReLU = cudnn.ReLU
local Sigmoid = cudnn.Sigmoid
local Max = nn.SpatialMaxPooling
local SBatchNorm = nn.SpatialBatchNormalization

local function createModel(opt)
   local function basicblock(nChIn,nChOut,sz)
      local s = nn.Sequential()
      local shareConv = Convolution(nChIn,nChOut,sz,sz,1,1,(sz-1)/2,(sz-1)/2)
      s:add(SBatchNorm(nChIn))      
      s:add(shareConv)
      s:add(ReLU())
      s:add(Convolution(nChOut,nChIn,1,1))
      local identity = nn.Identity()
      local output = nn.Sequential():add(nn.ConcatTable():add(s):add(identity)):add(nn.CAddTable(true))
      return output
   end

   local sz = opt.convSize
   local nInputPlane = opt.nInputPlane
   local nChIn = opt.numChannels
   local nChOut = opt.numWeights

   -- define model to train
   model = nn.Sequential()
   model:add(Convolution(nInputPlane,nChIn,sz,sz,1,1,1,1))
   model:add(SBatchNorm(nChIn))
   model:add(ReLU(true))

   for stages = 1,opt.depth do
      model:add(basicblock(nChIn,nChOut,sz))
      -- model:add(Max(3,3,2,2,1,1))
   end
   model:add(Avg(5,5,5,5))

   -- stage 3 : standard 2-layer neural network
   model:add(nn.Reshape(nChIn*opt.view))
   model:add(nn.Dropout(0.5))
   model:add(nn.Linear(nChIn*opt.view, math.max(opt.nClasses,opt.full)))
   model:add(cudnn.ReLU())
   model:add(nn.Dropout(0.5))
   model:add(nn.Linear(math.max(opt.full,opt.nClasses),opt.nClasses))
   model:cuda()

   return model
end

return createModel
