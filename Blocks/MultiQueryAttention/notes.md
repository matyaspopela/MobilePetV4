grads : equation

w_q : query = torch.matmul(tokenized, w_q).reshape(batch, height * width, h, d).transpose(1, 2)
w_kv : kv_fused = torch.matmul(downsampled_tokenized, w_kv).reshape(batch, 1, (height * width) / 4, d+v)
w_p : tokens = torch.matmul(projections_modified, w_p)
input : eqs in q, w_kv 