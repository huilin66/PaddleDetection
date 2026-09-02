import paddle
import paddle.nn as nn

paddle.seed(1)
x1 = paddle.randn(shape=[4, 3])
x2 = paddle.randn(shape=[2, 3])

result = paddle.nn.functional.cosine_similarity(x1, x2, axis=1)
print(result)
# [0.97689527,  0.99996042, -0.55138415]

# import math
# from torch import nn
#
# class DotProductSimilarity(nn.Module):
#
#     def __init__(self, scale_output=False):
#         super(DotProductSimilarity, self).__init__()
#         self.scale_output = scale_output
#
#     def forward(self, tensor_1, tensor_2):
#         result = (tensor_1 * tensor_2).sum(dim=-1)
#         if self.scale_output:
#             # TODO why allennlp do multiplication at here ?
#             result /= math.sqrt(tensor_1.size(-1))
#         return result
#
# class MultiHeadedSimilarity(nn.Module):
#
#     def __init__(self,
#                  num_heads=4,
#                  tensor_1_dim=768,
#                  tensor_1_projected_dim=None,
#                  tensor_2_dim=None,
#                  tensor_2_projected_dim=None,
#                  internal_similarity=DotProductSimilarity()):
#         super(MultiHeadedSimilarity, self).__init__()
#         self.num_heads = num_heads
#         self.internal_similarity = internal_similarity
#         tensor_1_projected_dim = tensor_1_projected_dim or tensor_1_dim
#         tensor_2_dim = tensor_2_dim or tensor_1_dim
#         tensor_2_projected_dim = tensor_2_projected_dim or tensor_2_dim
#         if tensor_1_projected_dim % num_heads != 0:
#             raise ValueError("Projected dimension not divisible by number of heads: %d, %d"
#                              % (tensor_1_projected_dim, num_heads))
#         if tensor_2_projected_dim % num_heads != 0:
#             raise ValueError("Projected dimension not divisible by number of heads: %d, %d"
#                              % (tensor_2_projected_dim, num_heads))
#         self.tensor_1_projection = nn.Parameter(torch.Tensor(tensor_1_dim, tensor_1_projected_dim))
#         self.tensor_2_projection = nn.Parameter(torch.Tensor(tensor_2_dim, tensor_2_projected_dim))
#         self.reset_parameters()
#
#     def reset_parameters(self):
#         torch.nn.init.xavier_uniform_(self.tensor_1_projection)
#         torch.nn.init.xavier_uniform_(self.tensor_2_projection)
#
#     def forward(self, tensor_1, tensor_2):
#         projected_tensor_1 = torch.matmul(tensor_1, self.tensor_1_projection)
#         projected_tensor_2 = torch.matmul(tensor_2, self.tensor_2_projection)
#
#         # Here we split the last dimension of the tensors from (..., projected_dim) to
#         # (..., num_heads, projected_dim / num_heads), using tensor.view().
#         last_dim_size = projected_tensor_1.size(-1) // self.num_heads
#         new_shape = list(projected_tensor_1.size())[:-1] + [self.num_heads, last_dim_size]
#         split_tensor_1 = projected_tensor_1.view(*new_shape)
#         last_dim_size = projected_tensor_2.size(-1) // self.num_heads
#         new_shape = list(projected_tensor_2.size())[:-1] + [self.num_heads, last_dim_size]
#         split_tensor_2 = projected_tensor_2.view(*new_shape)
#
#         # And then we pass this off to our internal similarity function. Because the similarity
#         # functions don't care what dimension their input has, and only look at the last dimension,
#         # we don't need to do anything special here. It will just compute similarity on the
#         # projection dimension for each head, returning a tensor of shape (..., num_heads).
#         return self.internal_similarity(split_tensor_1, split_tensor_2)
#
# import torch
# import torch.nn.functional as F
#
# # 假设你有以下两个特征
# feature1 = torch.randn([2, 768])  # 2*768的特征
# feature2 = torch.randn([100, 768])  # 100*768的特征
#
# ms = MultiHeadedSimilarity()
# sim = ms(feature1, feature2)

# # 为了计算余弦相似度，我们需要将feature1复制100次以匹配feature2的第一维度
# feature1_expanded = feature1.expand(100, -1, -1)
#
# # 计算余弦相似度
# cos_sim = F.cosine_similarity(feature1_expanded, feature2, dim=2)

