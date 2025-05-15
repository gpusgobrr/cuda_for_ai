#pragma once

#include <torch/torch.h>

torch::Tensor shmem_softmax(const torch::Tensor &input);