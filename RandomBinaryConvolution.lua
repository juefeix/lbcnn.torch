-- RandomBinaryConvolution.lua

local THNN = require 'nn.THNN'
local RandomBinaryConvolution, parent = torch.class('cudnn.RandomBinaryConvolution', 'cudnn.SpatialConvolution')

function RandomBinaryConvolution:__init(nInputPlane, nOutputPlane, kW, kH, dW, dH, padW, padH)
   parent.__init(self, nInputPlane, nOutputPlane, kW, kH, dW, dH, padW, padH)
   self:reset()
end

function RandomBinaryConvolution:reset()
	local numElements = self.nInputPlane*self.nOutputPlane*self.kW*self.kH
	self.weight = torch.CudaTensor(self.nOutputPlane,self.nInputPlane,self.kW,self.kH):fill(0)
	self.weight = torch.reshape(self.weight,numElements)
	local index = torch.Tensor(torch.floor(kSparsity*numElements)):random(numElements)
	for i = 1,index:numel() do
		self.weight[index[i]] = torch.bernoulli(0.5)*2-1
	end
	self.weight = torch.reshape(self.weight,self.nOutputPlane,self.nInputPlane,self.kW,self.kH)

	self.bias = nil
	self.gradBias = nil	
	self.gradWeight = torch.CudaTensor(self.nOutputPlane, self.nInputPlane, self.kH, self.kW):fill(0) 	
end

function RandomBinaryConvolution:accGradParameters(input, gradOutput, scale)
end

function RandomBinaryConvolution:updateParameters(learningRate)
end
