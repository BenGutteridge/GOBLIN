import torch
import triton
import triton.language as tl


@triton.jit
def relconv_aggregate_mean_kernel_forward(
        out_h, adj_rowptr, adj_indices, h_src,
        IN_CHAN: tl.constexpr, WG_SIZE: tl.constexpr,
):
    node_index_i = tl.program_id(0)
    feat_offsets = tl.arange(0, WG_SIZE) + tl.program_id(1) * WG_SIZE
    feat_valid_mask = feat_offsets < IN_CHAN
    feat_zeros = tl.zeros((WG_SIZE,), dtype=tl.float32)
    col_start = tl.load(adj_rowptr + node_index_i)
    col_end = tl.load(adj_rowptr + node_index_i + 1)
    col_count = col_end - col_start
    aggr_sum = feat_zeros
    for index in range(col_count):
        node_index_j = tl.load(adj_indices + col_start + index)
        neighbor_feat_j = tl.load(h_src + node_index_j * IN_CHAN + feat_offsets, feat_valid_mask, feat_zeros)
        aggr_sum += neighbor_feat_j
    col_count = tl.where(col_count == 0, 1, col_count)
    aggr_mean = aggr_sum / col_count
    tl.store(out_h + node_index_i * IN_CHAN + feat_offsets, aggr_mean, feat_valid_mask)


@triton.jit
def relconv_aggregate_mean_kernel_backward(
        dh_src, adj_rowptr, adj_indices, h_in, dh_out,
        IN_CHAN: tl.constexpr, WG_SIZE: tl.constexpr,
):
    node_index_i = tl.program_id(0)
    feat_offsets = tl.arange(0, WG_SIZE) + tl.program_id(1) * WG_SIZE
    feat_valid_mask = feat_offsets < IN_CHAN
    feat_zeros = tl.zeros((WG_SIZE,), dtype=tl.float32)
    col_start = tl.load(adj_rowptr + node_index_i)
    col_end = tl.load(adj_rowptr + node_index_i + 1)
    col_count = col_end - col_start
    h_out_grad = tl.load(dh_out + node_index_i * IN_CHAN + feat_offsets, feat_valid_mask, feat_zeros)
    h_out_grad = h_out_grad / tl.where(col_count == 0, 1, col_count)
    for index in range(col_count):
        node_index_j = tl.load(adj_indices + col_start + index)
        tl.atomic_add(dh_src + node_index_j * IN_CHAN + feat_offsets, h_out_grad, feat_valid_mask)


class ConvMeanAggr(torch.autograd.Function):
    @staticmethod
    def forward(ctx, h_in, rowptr, indices, out_node_count, work_group_size=None):
        torch.cuda.set_device(h_in.device)
        num_features_per_node = h_in.shape[1]
        if not work_group_size:
            work_group_size = 32
            while work_group_size < num_features_per_node:
                work_group_size *= 2
        num_work_groups = (num_features_per_node + work_group_size - 1) // work_group_size
        num_nodes = out_node_count
        h_out = torch.empty((out_node_count, h_in.shape[1]), dtype=h_in.dtype,
                            layout=h_in.layout, device=h_in.device)
        relconv_aggregate_mean_kernel_forward[(num_nodes, num_work_groups)](
            h_out, rowptr, indices, h_in,
            num_features_per_node, work_group_size, num_warps=32)
        h_in_grad = torch.zeros_like(h_in, requires_grad=False)
        work_group_size_shaped_dummy = torch.empty(work_group_size, dtype=torch.int8)
        ctx.save_for_backward(rowptr, indices, h_in_grad, h_in, work_group_size_shaped_dummy)
        return h_out

    @staticmethod
    def backward(ctx, h_out_grad):
        rowptr, indices, h_in_grad, h_in, work_group_size_shape_dummy = ctx.saved_tensors
        work_group_size = work_group_size_shape_dummy.shape[0]
        num_features_per_node = h_out_grad.shape[1]
        num_work_groups = (num_features_per_node + work_group_size - 1) // work_group_size
        num_nodes = h_out_grad.shape[0]
        relconv_aggregate_mean_kernel_backward[(num_nodes, num_work_groups)](
            h_in_grad, rowptr, indices, h_in, h_out_grad,
            num_features_per_node, work_group_size, num_warps=32)
        return h_in_grad, None, None, None, None
