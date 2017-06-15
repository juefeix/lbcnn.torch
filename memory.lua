-- main.lua

require 'torch'
require 'paths'
require 'optim'
require 'nn'
require 'math'

local models = require 'models/init'
local opts = require 'memoryopts'
local checkpoints = require 'checkpoints'

torch.setdefaulttensortype('torch.FloatTensor')
torch.setnumthreads(1)

local opt = opts.parse(arg)
torch.manualSeed(opt.manualSeed)

-- Load previous checkpoint, if it exists
local checkpoint, optimState = checkpoints.latest(opt)
local optimState = checkpoint and torch.load(checkpoint.optimFile) or nil

-- Create model
model, criterion = models.setup(opt, checkpoint)