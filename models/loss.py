import torch
import torch.nn as nn
import torch as t
import torch.nn.functional as F

def _loss_fn(x_target, x_pred):
    return t.mean(t.abs(x_pred - x_target)) 
    
class Motion_Loss(nn.Module):
    def __init__(self,):
        super().__init__()
        self.acc = 1
        self.vel = 1

    def forward(self, x_out, x_target):
        recons_loss = t.zeros(()).to(x_target.device)
        regularization = t.zeros(()).to(x_target.device)
        velocity_loss = t.zeros(()).to(x_target.device)
        acceleration_loss = t.zeros(()).to(x_target.device)

        recons_loss += _loss_fn(x_target, x_out)
        velocity_loss +=  _loss_fn( x_out[:, 1:] - x_out[:, :-1], x_target[:, 1:] - x_target[:, :-1])
        acceleration_loss +=  _loss_fn(x_out[:, 2:] + x_out[:, :-2] - 2 * x_out[:, 1:-1], x_target[:, 2:] + x_target[:, :-2] - 2 * x_target[:, 1:-1])

        loss = recons_loss + self.vel * velocity_loss + self.acc * acceleration_loss
        return loss


class Commit_Loss(nn.Module):
    def __init__(self,):
        super().__init__()
        self.commit = 0.02

    def forward(self, commit_losses):
        loss = self.commit * sum(commit_losses)
        return loss


class CLIPLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.logit_scale=torch.FloatTensor([4.6052]).cuda().exp()
        self.topk = 5

    def forward(self, sims):
        """
        Inputs: cosine similarities
            sims: n x n (text is dim-0)
            logit_scale: 1 x 1
        """
        logits = sims * self.logit_scale
        
        t2v_log_sm = F.log_softmax(logits, dim=1)
        t2v_neg_ce = torch.diag(t2v_log_sm)
        t2v_loss = -t2v_neg_ce.mean()

        v2t_log_sm = F.log_softmax(logits, dim=0)
        v2t_neg_ce = torch.diag(v2t_log_sm)
        v2t_loss = -v2t_neg_ce.mean()

        return (t2v_loss + v2t_loss) / 2.0


# torch.autograd.set_detect_anomaly(True)
# class CLIPLoss(nn.Module):
#     def __init__(self):
#         super().__init__()
#         self.logit_scale = torch.tensor([4.6052], device='cuda').exp()
#         self.top_k = 4

#     def forward(self, sims):
#         """
#         Inputs: cosine similarities
#             sims: n x n (text is dim-0)
#             logit_scale: 1 x 1
#         """

#         bottom_k = max(1, sims.shape[0]-self.top_k)
#         logits = sims * self.logit_scale
        
#         sims_mat = mask_extremes_no_loop(logits, bottom_k)
#         t2v_log_sm = F.log_softmax(sims_mat, dim=1)
#         t2v_neg_ce = torch.diag(t2v_log_sm)
#         t2v_loss = -t2v_neg_ce.mean()

#         sims_mat = mask_extremes_no_loop(logits.t(), bottom_k)
#         v2t_log_sm = F.log_softmax(sims_mat, dim=1)
#         v2t_neg_ce = torch.diag(v2t_log_sm)
#         v2t_loss = -v2t_neg_ce.mean()

#         return (t2v_loss + v2t_loss) / 2.0


# def mask_extremes_no_loop(matrix, bottom_k=5):

#     matrix = matrix.t()
#     _, bottom_indices = torch.topk(matrix, bottom_k, dim=0, largest=False)
    
#     masked_matrix = torch.full_like(matrix, float('-inf'))
#     col_indices = torch.arange(matrix.shape[1], device=matrix.device).unsqueeze(0).expand_as(bottom_indices)
#     diag_indices = torch.arange(min(matrix.shape[0], matrix.shape[1]), device=matrix.device)
    
#     masked_matrix.index_put_((bottom_indices, col_indices), matrix[bottom_indices, col_indices], accumulate=False)
#     masked_matrix.index_put_((diag_indices, diag_indices), matrix[diag_indices, diag_indices], accumulate=False)
    
#     return masked_matrix.t()


if __name__ == '__main__':
    
    print('hello')
    # Example usage:
    # m, n = 8, 8  # smaller dimensions for easier verification
    # matrix = torch.randn(m, n)  # create a random m x n matrix
    # print("Original matrix:\n", matrix)

    # masked_matrix = mask_extremes_no_loop(matrix)
    # print("Masked matrix:\n", masked_matrix)