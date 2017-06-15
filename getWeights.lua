-- getWeights.lua

local nn = require 'nn'
require 'math'
require 'cunn'
require 'cudnn'
require 'cutorch'
require 'RandomBinaryConvolution'
local matio = require 'matio'

local opts = require 'opts'
local DataLoader = require 'dataloader'

opt = opts.parse(arg)
torch.manualSeed(opt.manualSeed)
cutorch.manualSeedAll(opt.manualSeed)
opt.dataset = 'cifar10'
opt.save = '/home/vboddeti/'
trainLoader, valLoader = DataLoader.create(opt)

local GPU = 1
cutorch.setDevice(GPU)
name = '/home/vboddeti/Downloads/LBCNN-model.net'
model = torch.load(name)

for n,sample in trainLoader:run() do
	local input = sample.input:cuda()
	output = model:forward(input)
	print(output:size())
	abcd = abcd + 1
end

-- for i = 1,5 do
-- 	local weight = model:get(i):get(1):get(1):get(2).weight
-- 	weight = weight:view(512,27)
-- 	matio.save('../plot/LBCNN-Weights-' .. tostring(i) .. '.mat',weight)
-- end