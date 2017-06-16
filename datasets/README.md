## Datasets

Each dataset consist of two files: `dataset-gen.lua` and `dataset.lua`. The `dataset-gen.lua` is responsible for one-time setup, while
the `dataset.lua` handles the actual data loading.

### `dataset-gen.lua`

The `dataset-gen.lua` performs any necessary one-time setup. For example, the [`cifar10-gen.lua`](cifar10-gen.lua) file downloads the CIFAR-10 dataset.

The module should have a single function `exec(opt, cacheFile)`.
- `opt`: the command line options
- `cacheFile`: path to output 

```lua
local M = {}
function M.exec(opt, cacheFile)
  local imageInfo = {}
  -- preprocess dataset, store results in imageInfo, save to cacheFile
  torch.save(cacheFile, imageInfo)
end
return M
```

### `dataset.lua`

The `dataset.lua` should return a class that implements three functions:
- `get(i)`: returns a table containing two entries, `input` and `target`
  - `input`: the training or validation image as a Torch tensor
  - `target`: the image category as a number 1-N
- `size()`: returns the number of entries in the dataset
- `preprocess()`: returns a function that transforms the `input` for data augmentation or input normalization
