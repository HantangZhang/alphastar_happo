# if self.is_cnn:
#     # 原alphastar是为了把auto地数据reshape了和图像数据拼在一起，但是空战目前没有图像数据，所以只能假定图像数据地宽高是8， 8 *8 =64
#     reshpe_channel = int(self.autoregressive_embedding_size / 64)
#     x = autoregressive_embedding.reshape(batch_size, -1, reshpe_channel, reshpe_channel)
#
#     x = F.relu(self.ds_1(F.relu(x)))
#
#     # if not self.use_improved_one:
#     #
#     x = self.film_net(x, gate=autoregressive_embedding)
#
#     # else:
#     #     x = self.film_net_mapskip(x, gate=autoregressive_embedding,
#     #                               map_skip=map_skip)
#     #     x = F.relu(x)