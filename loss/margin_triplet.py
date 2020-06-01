import torch
import numpy as np
import random

class MarginTripletLoss(torch.nn.Module):
    def __init__(self, device, batch_size, temperature, use_cosine_similarity,semihard):
        super(MarginTripletLoss, self).__init__()
        self.batch_size = batch_size
        self.temperature = temperature
        self.device = device
        self.softmax = torch.nn.Softmax(dim=-1)
        self.mask_samples_from_same_repr = self._get_correlated_mask().type(torch.bool)
        self.similarity_function = self._get_similarity_function(use_cosine_similarity)
        self.criterion = torch.nn.CrossEntropyLoss(reduction="sum")
        self.semihard = semihard
    def _get_similarity_function(self, use_cosine_similarity):
        if use_cosine_similarity:
            self._cosine_similarity = torch.nn.CosineSimilarity(dim=-1)
            return self._cosine_simililarity
        else:
            return self._dot_simililarity
    def _get_correlated_mask(self):
        diag = np.eye(2 * self.batch_size)
        l1 = np.eye((2 * self.batch_size), 2 * self.batch_size, k=-self.batch_size)
        l2 = np.eye((2 * self.batch_size), 2 * self.batch_size, k=self.batch_size)
        mask = torch.from_numpy((diag + l1 + l2))
        mask = (1 - mask).type(torch.bool)
        return mask.to(self.device)

    def _get_semi_hard_mask(self,similarity_matrix,margin,d_a_p):
        zero = torch.zeros_like(similarity_matrix)
        one = torch.ones_like(similarity_matrix)
        mask = torch.where(similarity_matrix - d_a_p + margin > 0,one,zero)
        mask = torch.where(similarity_matrix < d_a_p, mask,zero)
        diag = np.eye(2 * self.batch_size)
        l1 = np.eye((2 * self.batch_size), 2 * self.batch_size, k=-self.batch_size)
        l2 = np.eye((2 * self.batch_size), 2 * self.batch_size, k=self.batch_size)
        m = torch.from_numpy((diag + l1 + l2))
        mask = torch.sub(mask,m)
        zero = torch.zeros_like(mask)
        mask = torch.where(mask < 0.0,zero,mask)
        mask = mask.type(torch.bool)
        return mask.to(self.device)
        

    @staticmethod
    def _dot_simililarity(x, y):
        v = torch.tensordot(x.unsqueeze(1), y.T.unsqueeze(0), dims=2)
        # x shape: (N, 1, C)
        # y shape: (1, C, 2N)
        # v shape: (N, 2N)
        return v

    def _cosine_simililarity(self, x, y):
        # x shape: (N, 1, C)
        # y shape: (1, 2N, C)
        # v shape: (N, 2N)
        # print(x.shape)
        # print(y.shape)
        # v = self._cosine_similarity(x.unsqueeze(1), y.unsqueeze(0))
        v = torch.cosine_similarity(x.unsqueeze(1),y.unsqueeze(0),dim=2)
        return v

    
    def forward(self, zis, zjs):
        representations = torch.cat([zjs, zis], dim=0)

        similarity_matrix = self.similarity_function(representations, representations)
        margin = torch.tensor(0.05)
        # filter out the scores from the positive samples
        l_pos = torch.diag(similarity_matrix, self.batch_size)
        r_pos = torch.diag(similarity_matrix, -self.batch_size)
        d_a_p = torch.cat([l_pos, r_pos]).view(2 * self.batch_size, 1)
        negatives = similarity_matrix[self.mask_samples_from_same_repr].view(2 * self.batch_size, -1)
        #semi-hard negtive
        if self.semihard=='Yes':
            semis = similarity_matrix[self._get_semi_hard_mask(similarity_matrix,margin,d_a_p)].view(-1, 1)
            i = np.arange(batch_size)
            x = np.random.choice(semis.shape[0],semis.shape[0])
            d_a_n = semis[i,x]
        #hard negtive
        else:
            d_a_n, indices = torch.max(negatives,1)
        losses = torch.sub(torch.add(d_a_n,margin),d_a_p)

        zero = torch.zeros_like(losses)
        losses = torch.where(losses < 0.0, zero, losses)
        losses = torch.sum(losses)
        return losses / (2 * self.batch_size)
    
    def top_eval(self,zis,zjs):
        representations = torch.cat([zjs, zis], dim=0)

        similarity_matrix = self.similarity_function(representations, representations)

        # filter out the scores from the positive samples
        l_pos = torch.diag(similarity_matrix, self.batch_size)
        r_pos = torch.diag(similarity_matrix, -self.batch_size)
        positives = torch.cat([l_pos, r_pos]).view(2 * self.batch_size, 1)

        negatives = similarity_matrix[self.mask_samples_from_same_repr].view(2 * self.batch_size, -1)

        labels = torch.zeros(2 * self.batch_size).to(self.device).long()
        logits = torch.cat((positives, negatives), dim=1)
        predicted1 = torch.argmax(logits, dim=1)
        _,predicted5 = logits.topk(5,1,True,True)
        _,predicted10 = logits.topk(10,1,True,True)
        _,predicted20 = logits.topk(20,1,True,True)
        _,predicted50 = logits.topk(50,1,True,True)
        _,predicted100 = logits.topk(100,1,True,True)
        correct1 = (predicted1 == labels).sum().item()
        labels = labels.view(-1,1)
        correct5 = torch.eq(predicted5, labels).sum().float().item()
        correct10 = torch.eq(predicted10, labels).sum().float().item()
        correct20 = torch.eq(predicted20, labels).sum().float().item()
        correct50 = torch.eq(predicted50, labels).sum().float().item()
        correct100 = torch.eq(predicted100, labels).sum().float().item()

        return correct1,correct5,correct10,correct20,correct50,correct100