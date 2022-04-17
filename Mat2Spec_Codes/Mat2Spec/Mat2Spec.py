import torch, numpy as np
import torch.optim as optim
from   torch.optim import lr_scheduler 
from   torch.nn import Linear, Dropout, Parameter
import torch.nn.functional as F 
import torch.nn as nn

from torch_geometric.nn.conv  import MessagePassing
from torch_geometric.utils    import softmax
from torch_geometric.nn       import global_add_pool, global_mean_pool
from torch_geometric.nn       import GATConv
from torch_scatter            import scatter_add
from torch_geometric.nn.inits import glorot, zeros

from random import sample
from copy import copy, deepcopy
from Mat2Spec.utils import *
from Mat2Spec.SinkhornDistance import SinkhornDistance
from Mat2Spec.pytorch_stats_loss import torch_wasserstein_loss

device = set_device()
torch.cuda.empty_cache()
kl_loss_fn = torch.nn.KLDivLoss()
sinkhorn = SinkhornDistance(eps=0.1, max_iter=50, reduction='mean').to(device)


# Note: the part of GNN implementation is modified from https://github.com/superlouis/GATGNN/

class COMPOSITION_Attention(torch.nn.Module):
    def __init__(self,neurons):
        super(COMPOSITION_Attention, self).__init__()
        self.node_layer1    = Linear(neurons+103,32)
        self.atten_layer    = Linear(32,1)

    def forward(self,x,batch,global_feat):
        #torch.set_printoptions(threshold=10_000)
        # global_feat, [bs*103], rach row is an atom composition vector
        # x: [num_atom * atom_emb_len]

        counts      = torch.unique(batch,return_counts=True)[-1]   # return the number of atoms per crystal
        # batch includes all of the atoms from the Batch of crystals, each atom indexed by its Batch index.

        graph_embed = global_feat
        graph_embed = torch.repeat_interleave(graph_embed, counts, dim=0)  # repeat rows according to counts
        chunk       = torch.cat([x,graph_embed],dim=-1)
        x           = F.softplus(self.node_layer1(chunk))  # [num_atom * 32]
        x           = self.atten_layer(x) # [num_atom * 1]
        weights     = softmax(x,batch) # [num_atom * 1]
        return weights


class GAT_Crystal(MessagePassing):
    def __init__(self, in_features, out_features, edge_dim, heads, concat=False,
                 dropout=0.0, bias=True, has_edge_attr=True, **kwargs):
        super(GAT_Crystal, self).__init__(aggr='add',flow='target_to_source', **kwargs)
        self.in_features       = in_features
        self.out_features      = out_features
        self.heads             = heads
        self.concat            = concat
        #self.dropout          = dropout
        self.dropout           = nn.Dropout(p=dropout)
        self.neg_slope         = 0.2
        self.prelu             = nn.PReLU()
        self.bn1               = nn.BatchNorm1d(heads)
        if has_edge_attr:
            self.W             = Parameter(torch.Tensor(in_features+edge_dim,heads*out_features))
        else:
            self.W = Parameter(torch.Tensor(in_features, heads * out_features))
        self.att               = Parameter(torch.Tensor(1,heads,2*out_features))

        if bias and concat       : self.bias = Parameter(torch.Tensor(heads * out_features))
        elif bias and not concat : self.bias = Parameter(torch.Tensor(out_features))
        else                     : self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        glorot(self.W)
        glorot(self.att)
        zeros(self.bias)

    def forward(self, x, edge_index, edge_attr=None):
        # x: [num_node, emb_len]
        # edge_index: [2, num_edge]
        # edge_attr: [num_edge, emb_len]
        return self.propagate(edge_index, x=x, edge_attr=edge_attr)

    def message(self, edge_index_i, x_i, x_j, size_i, edge_attr):
        # edge_index_i: [num_edge]
        # x_i: [num_edge, emb_len]
        # x_j: [num_edge, emb_len]
        # size_i: num_node
        # edge_attr: [num_edge, emb_len]
        if edge_attr is not None:
            x_i   = torch.cat([x_i,edge_attr],dim=-1)
            x_j   = torch.cat([x_j,edge_attr],dim=-1)

        x_i   = F.softplus(torch.matmul(x_i,self.W))
        x_j   = F.softplus(torch.matmul(x_j,self.W))

        x_i   = x_i.view(-1, self.heads, self.out_features) # [num_edge, num_head, emb_len]
        x_j   = x_j.view(-1, self.heads, self.out_features) # [num_edge, num_head, emb_len]

        alpha = F.softplus((torch.cat([x_i, x_j], dim=-1)*self.att).sum(dim=-1))  # [num_edge, num_head]

        # self.att: (1,heads,2*out_features)

        alpha = F.softplus(self.bn1(alpha))
        alpha = softmax(alpha, edge_index_i, size_i) # [num_edge, num_head]
        #alpha = softmax(alpha, edge_index_i) # [num_edge, num_head]
        alpha = self.dropout(alpha)

        return x_j * alpha.view(-1, self.heads, 1) # [num_edge, num_head, emb_len]

    def update(self, aggr_out):
        # aggr_out: [num_node, num_head, emb_len]
        if self.concat is True:    aggr_out = aggr_out.view(-1, self.heads * self.out_features)
        else:                      aggr_out = aggr_out.mean(dim=1)
        if self.bias is not None:  aggr_out = aggr_out + self.bias
        return aggr_out # [num_node, emb_len]

class FractionalEncoder(nn.Module):
    """
    Encoding element fractional amount using a "fractional encoding" inspired
    by the positional encoder discussed by Vaswani.
    https://arxiv.org/abs/1706.03762
    """
    def __init__(self,
                 d_model,
                 resolution=100,
                 log10=False,
                 compute_device=None):
        super().__init__()
        self.d_model = d_model//2
        self.resolution = resolution
        self.log10 = log10
        self.compute_device = compute_device

        x = torch.linspace(0, self.resolution - 1,
                           self.resolution,
                           requires_grad=False) \
            .view(self.resolution, 1)   # (resolution, 1)
        fraction = torch.linspace(0, self.d_model - 1,
                                  self.d_model,
                                  requires_grad=False) \
            .view(1, self.d_model).repeat(self.resolution, 1) # (resolution, d_model)

        pe = torch.zeros(self.resolution, self.d_model) # (resolution, d_model)
        pe[:, 0::2] = torch.sin(x /torch.pow(50, 2 * fraction[:, 0::2] / self.d_model))
        pe[:, 1::2] = torch.cos(x / torch.pow(50, 2 * fraction[:, 1::2] / self.d_model))
        pe = self.register_buffer('pe', pe) # (resolution, d_model)

    def forward(self, x):
        x = x.clone()
        if self.log10:
            x = 0.0025 * (torch.log2(x))**2
            x[x > 1] = 1
            # x = 1 - x  # for sinusoidal encoding at x=0
        x[x < 1/self.resolution] = 1/self.resolution
        frac_idx = torch.round(x * (self.resolution)).to(dtype=torch.long) - 1 # (bs, n_elem)
        out = self.pe[frac_idx] # (bs, n_elem, d_model)
        return out

class GNN(torch.nn.Module):
    def __init__(self,heads,neurons=64,nl=3,concat_comp=False):
        super(GNN, self).__init__()

        self.n_heads        = heads
        self.number_layers  = nl
        self.concat_comp    = concat_comp

        n_h, n_hX2          = neurons, neurons*2
        self.neurons        = neurons
        self.neg_slope      = 0.2  

        self.embed_n        = Linear(92,n_h)
        self.embed_e        = Linear(41,n_h)
        self.embed_comp     = Linear(103,n_h)
 
        self.node_att       = nn.ModuleList([GAT_Crystal(n_h,n_h,n_h,self.n_heads) for i in range(nl)])
        self.batch_norm     = nn.ModuleList([nn.BatchNorm1d(n_h) for i in range(nl)])

        self.comp_atten     = COMPOSITION_Attention(n_h)

        self.emb_scaler = nn.parameter.Parameter(torch.tensor([1.]))
        self.pos_scaler = nn.parameter.Parameter(torch.tensor([1.]))
        self.pos_scaler_log = nn.parameter.Parameter(torch.tensor([1.]))
        self.pe = FractionalEncoder(n_h, resolution=5000, log10=False)
        self.ple = FractionalEncoder(n_h, resolution=5000, log10=True)
        self.pe_linear = nn.Linear(103, 1)
        self.ple_linear = nn.Linear(103, 1)

        if self.concat_comp : reg_h   = n_hX2
        else                : reg_h   = n_h

        self.linear1    = nn.Linear(reg_h,reg_h)
        self.linear2    = nn.Linear(reg_h,reg_h)

    def forward(self,data):
        x, edge_index, edge_attr   = data.x, data.edge_index, data.edge_attr

        batch, global_feat, cluster = data.batch, data.global_feature, data.cluster

        x           = self.embed_n(x) # [num_atom, emb_len]

        edge_attr   = F.leaky_relu(self.embed_e(edge_attr),self.neg_slope) # [num_edges, emb_len]

        for a_idx in range(len(self.node_att)):
            x     = self.node_att[a_idx](x,edge_index,edge_attr) # [num_atom, emb_len]
            x     = self.batch_norm[a_idx](x)
            x     = F.softplus(x)

        ag        = self.comp_atten(x,batch,global_feat) # [num_atom * 1]
        x         = (x)*ag  # [num_atom, emb_len]
        
        # CRYSTAL FEATURE-AGGREGATION 
        y         = global_mean_pool(x,batch)#*2**self.emb_scaler#.unsqueeze(1).squeeze() # [bs, emb_len]
        #y         = F.relu(self.linear1(y))  # [bs, emb_len]
        #y         = F.relu(self.linear2(y))  # [bs, emb_len]

        if self.concat_comp:
            pe = torch.zeros([global_feat.shape[0], global_feat.shape[1], y.shape[1]]).to(device)
            ple = torch.zeros([global_feat.shape[0], global_feat.shape[1], y.shape[1]]).to(device)
            pe_scaler = 2 ** (1 - self.pos_scaler) ** 2
            ple_scaler = 2 ** (1 - self.pos_scaler_log) ** 2
            pe[:, :, :y.shape[1] // 2] = self.pe(global_feat)# * pe_scaler
            ple[:, :, y.shape[1] // 2:] = self.ple(global_feat)# * ple_scaler
            pe = self.pe_linear(torch.transpose(pe, 1,2)).squeeze()* pe_scaler
            ple = self.ple_linear(torch.transpose(ple, 1,2)).squeeze()* ple_scaler
            y = y + pe + ple
            #y = torch.cat([y, pe+ple], dim=-1)
            #y     = torch.cat([y, F.leaky_relu(self.embed_comp(global_feat), self.neg_slope)], dim=-1)

        return y

class Mat2Spec(nn.Module):
    def __init__(self, args, NORMALIZER):
        super(Mat2Spec, self).__init__()
        n_heads = args.num_heads
        number_neurons = args.num_neurons
        number_layers = args.num_layers
        concat_comp = args.concat_comp
        self.graph_encoder = GNN(n_heads, neurons=number_neurons, nl=number_layers, concat_comp=concat_comp)

        self.loss_type = args.Mat2Spec_loss_type
        self.NORMALIZER = NORMALIZER
        self.input_dim = args.Mat2Spec_input_dim
        self.latent_dim = args.Mat2Spec_latent_dim
        self.emb_size = args.Mat2Spec_emb_size
        self.label_dim = args.Mat2Spec_label_dim
        self.scale_coeff = args.Mat2Spec_scale_coeff
        self.keep_prob = args.Mat2Spec_keep_prob
        self.K = args.Mat2Spec_K
        self.args = args

        self.fx1 = nn.Linear(self.input_dim, 256)
        self.fx2 = nn.Linear(256, 512)
        self.fx3 = nn.Linear(512, 256)
        self.fx_mu = nn.Linear(256, self.latent_dim*self.K)
        self.fx_logvar = nn.Linear(256, self.latent_dim*self.K)
        self.fx_mix_coeff = nn.Linear(256, self.K)

        self.fe_mix_coeff = nn.Sequential(
            nn.Linear(self.label_dim, 128),
            nn.ReLU(),
            nn.Linear(128, self.label_dim)
        )

        self.fd_x1 = nn.Linear(self.input_dim + self.latent_dim, 512)
        self.fd_x2 = torch.nn.Sequential(
            nn.Linear(512, self.emb_size)
        )
        self.feat_mp_mu = nn.Linear(self.emb_size, self.label_dim)

        # label layers
        self.fe0 = nn.Linear(self.label_dim, self.emb_size)
        self.fe1 = nn.Linear(self.label_dim, 512)
        self.fe2 = nn.Linear(512, 256)
        self.fe_mu = nn.Linear(256, self.latent_dim)
        self.fe_logvar = nn.Linear(256, self.latent_dim)

        self.fd1 = self.fd_x1
        self.fd2 = self.fd_x2
        #self.fd = self.fd_x
        self.label_mp_mu = self.feat_mp_mu

        self.bias = nn.Parameter(torch.zeros(self.label_dim))

        assert id(self.fd_x1) == id(self.fd1)
        assert id(self.fd_x2) == id(self.fd2)

        self.dropout = nn.Dropout(p=self.keep_prob)
        self.emb_proj = nn.Linear(args.Mat2Spec_emb_size, 1024)
        self.W = nn.Linear(args.Mat2Spec_label_dim, args.Mat2Spec_emb_size) # linear transformation for label

    def label_encode(self, x):
        #h0 = self.dropout(F.relu(self.fe0(x)))  # [label_dim, emb_size]
        h1 = self.dropout(F.relu(self.fe1(x)))  # [label_dim, 512]
        h2 = self.dropout(F.relu(self.fe2(h1)))  # [label_dim, 256]
        mu = self.fe_mu(h2) * self.scale_coeff  # [label_dim, latent_dim]
        logvar = self.fe_logvar(h2) * self.scale_coeff  # [label_dim, latent_dim]

        fe_output = {
            'fe_mu': mu,
            'fe_logvar': logvar
        }
        return fe_output

    def feat_encode(self, x):
        h1 = self.dropout(F.relu(self.fx1(x)))
        h2 = self.dropout(F.relu(self.fx2(h1)))
        h3 = self.dropout(F.relu(self.fx3(h2)))
        mu = self.fx_mu(h3) * self.scale_coeff  # [bs, latent_dim]
        logvar = self.fx_logvar(h3) * self.scale_coeff
        mix_coeff = self.fx_mix_coeff(h3)  # [bs, K]

        if self.K > 1:
            mu = mu.view(x.shape[0], self.K, self.args.Mat2Spec_latent_dim) # [bs, K, latent_dim]
            logvar = logvar.view(x.shape[0], self.K, self.args.Mat2Spec_latent_dim) # [bs, K, latent_dim]

        fx_output = {
            'fx_mu': mu,
            'fx_logvar': logvar,
            'fx_mix_coeff': mix_coeff
        }
        return fx_output

    def label_reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def feat_reparameterize(self, mu, logvar, coeff=1.0):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def label_decode(self, z):
        d1 = F.relu(self.fd1(z))
        d2 = F.leaky_relu(self.fd2(d1))
        return d2

    def feat_decode(self, z):
        d1 = F.relu(self.fd_x1(z))
        d2 = F.leaky_relu(self.fd_x2(d1))
        return d2

    def label_forward(self, x, feat):  # x is label
        n_label = x.shape[1]  # label_dim
        all_labels = torch.eye(n_label).to(x.device)  # [label_dim, label_dim]
        fe_output = self.label_encode(all_labels)  # map each label to a Gaussian mixture.
        mu = fe_output['fe_mu']
        logvar = fe_output['fe_logvar']
        fe_output['fe_mix_coeff'] = self.fe_mix_coeff(x)
        mix_coeff = F.softmax(fe_output['fe_mix_coeff'], dim=-1)

        if self.args.train:
            z = self.label_reparameterize(mu, logvar) # [label_dim, latent_dim]
        else:
            z = mu
        z = torch.matmul(mix_coeff, z)

        label_emb = self.label_decode(torch.cat((feat, z), 1))
        fe_output['label_emb'] = label_emb
        return fe_output

    def feat_forward(self, x):
        fx_output = self.feat_encode(x)
        mu = fx_output['fx_mu']  # [bs, latent_dim]
        logvar = fx_output['fx_logvar']  # [bs, latent_dim]

        if self.args.train:
            z = self.feat_reparameterize(mu, logvar)
        else:
            z = mu
        if self.K > 1:
            mix_coeff = fx_output['fx_mix_coeff']  # [bs, K]
            mix_coeff = F.softmax(mix_coeff, dim=-1)
            mix_coeff = mix_coeff.unsqueeze(-1).expand_as(z)
            z = z * mix_coeff
            z = torch.sum(z, dim=1)  # [bs, latent_dim]

        feat_emb = self.feat_decode(torch.cat((x, z), 1))  # [bs, emb_size]
        fx_output['feat_emb'] = feat_emb
        return fx_output

    def forward(self, data):
        label = data.y
        feature = self.graph_encoder(data)

        fe_output = self.label_forward(label, feature)
        label_emb = fe_output['label_emb'] # [bs, emb_size]
        fx_output = self.feat_forward(feature)
        feat_emb  = fx_output['feat_emb'] # [bs, emb_size]
        W = self.W.weight # [emb_size, label_dim]
        label_out = torch.matmul(label_emb, W)  # [bs, emb_size] * [emb_size, label_dim] = [bs, label_dim]
        feat_out = torch.matmul(feat_emb, W)  # [bs, label_dim]

        label_proj = self.emb_proj(label_emb)
        feat_proj = self.emb_proj(feat_emb)
        fe_output.update(fx_output)
        output = fe_output

        if self.args.label_scaling == 'normalized_max':
            label_out = F.relu(label_out)
            feat_out = F.relu(feat_out)
            maxima, _ = torch.max(label_out, dim=1)
            label_out = label_out.div(maxima.unsqueeze(1)+1e-8)
            maxima, _ = torch.max(feat_out, dim=1)
            feat_out = feat_out.div(maxima.unsqueeze(1)+1e-8)

        output['label_out'] = label_out
        output['feat_out'] = feat_out
        output['label_proj'] = label_proj
        output['feat_proj'] = feat_proj
        return output

def kl(fx_mu, fe_mu, fx_logvar, fe_logvar):
    kl_loss = 0.5 * torch.sum(
        (fx_logvar - fe_logvar) - 1 + torch.exp(fe_logvar - fx_logvar) + (fx_mu - fe_mu)**2 / (
                torch.exp(fx_logvar) + 1e-8), dim=-1)
    return kl_loss

def compute_c_loss(BX, BY, tau=1):
    BX = F.normalize(BX, dim=1)
    BY = F.normalize(BY, dim=1)
    b = torch.matmul(BX, torch.transpose(BY, 0, 1)) # [bs, bs]
    b = torch.exp(b/tau)
    b_diag = torch.diagonal(b, 0).unsqueeze(1) # [bs, 1]
    b_sum = torch.sum(b, dim=-1, keepdim=True) # [bs, 1]
    c = b_diag/(b_sum-b_diag)
    c_loss = -torch.mean(torch.log(c))
    return c_loss

def compute_loss(input_label, output, NORMALIZER, args):
    fe_out, fe_mu, fe_logvar, label_emb, label_proj = output['label_out'], output['fe_mu'], output['fe_logvar'], output['label_emb'], output['label_proj']
    fx_out, fx_mu, fx_logvar, feat_emb, feat_proj = output['feat_out'], output['fx_mu'], output['fx_logvar'], output['feat_emb'], output['feat_proj']

    fx_mix_coeff = output['fx_mix_coeff']  # [bs, K]
    fe_mix_coeff = output['fe_mix_coeff']
    fx_mix_coeff = F.softmax(fx_mix_coeff, dim=-1)
    fe_mix_coeff = F.softmax(fe_mix_coeff, dim=-1)
    fe_mix_coeff = fe_mix_coeff.repeat(1, args.Mat2Spec_K)
    fx_mix_coeff = fx_mix_coeff.repeat(1, args.Mat2Spec_label_dim)
    mix_coeff = fe_mix_coeff * fx_mix_coeff
    fx_mu = fx_mu.repeat(1, args.Mat2Spec_label_dim, 1)
    fx_logvar = fx_logvar.repeat(1, args.Mat2Spec_label_dim, 1)
    fe_mu = fe_mu.squeeze(0).expand(fx_mu.shape[0], fe_mu.shape[0], fe_mu.shape[1])
    fe_logvar = fe_logvar.squeeze(0).expand(fx_mu.shape[0], fe_logvar.shape[0], fe_logvar.shape[1])
    fe_mu = fe_mu.repeat(1, args.Mat2Spec_K, 1)
    fe_logvar = fe_logvar.repeat(1, args.Mat2Spec_K, 1)
    kl_all = kl(fx_mu, fe_mu, fx_logvar, fe_logvar)
    kl_all_inv = kl(fe_mu, fx_mu, fe_logvar, fx_logvar)
    kl_loss = torch.mean(torch.sum(mix_coeff * (0.5*kl_all + 0.5*kl_all_inv), dim=-1))
    #c_loss = torch.mean(-1 * F.cosine_similarity(label_proj, feat_proj))
    c_loss = compute_c_loss(label_proj, feat_proj)

    if args.label_scaling == 'normalized_sum':
        assert args.Mat2Spec_loss_type == 'KL' or args.Mat2Spec_loss_type == 'WD'
        #input_label_normalize = F.softmax(torch.log(input_label+1e-6), dim=1)
        input_label_normalize = input_label / (torch.sum(input_label, dim=1, keepdim=True)+1e-8)
        pred_e = F.softmax(fe_out, dim=1)
        pred_x = F.softmax(fx_out, dim=1)
        #nll_loss = kl_loss_fn(torch.log(pred_e+1e-8), input_label_normalize)
        #nll_loss_x = kl_loss_fn(torch.log(pred_x+1e-8), input_label_normalize)
        P = input_label_normalize
        Q_e = pred_e
        Q_x = pred_x
        c1, c2, c3 = 1, 1.1, 0.1
        if args.ablation_LE:
            c2 = 0.0
        if args.ablation_CL:
            c3 = 0.0

        if args.Mat2Spec_loss_type == 'KL':
            nll_loss = torch.mean(torch.sum(P*(torch.log(P+1e-8)-torch.log(Q_e+1e-8)),dim=1)) \
                #+ torch.mean(torch.sum(Q_e*(torch.log(Q_e+1e-8)-torch.log(P+1e-8)),dim=1))
            nll_loss_x = torch.mean(torch.sum(P*(torch.log(P+1e-8)-torch.log(Q_x+1e-8)),dim=1)) \
                #+ torch.mean(torch.sum(Q_x*(torch.log(Q_x+1e-8)-torch.log(P+1e-8)),dim=1))
        elif args.Mat2Spec_loss_type == 'WD':
            #nll_loss, _, _ = sinkhorn(Q_e, P)
            #nll_loss_x, _, _ = sinkhorn(Q_x, P)
            nll_loss = torch_wasserstein_loss(Q_e, P)
            nll_loss_x = torch_wasserstein_loss(Q_x, P)
        total_loss = (nll_loss + nll_loss_x) * c1 + kl_loss * c2 + c_loss * c3

        return total_loss, nll_loss, nll_loss_x, kl_loss, c_loss, pred_e, pred_x

    else: # standardized or normalized_max
        assert args.Mat2Spec_loss_type == 'MAE' or args.Mat2Spec_loss_type == 'MSE'
        pred_e = fe_out
        pred_x = fx_out
        c1, c2, c3 = 1, 1.1, 0.1
        if args.ablation_LE:
            c2 = 0.0
        if args.ablation_CL:
            c3 = 0.0

        if args.Mat2Spec_loss_type == 'MAE':
            nll_loss = torch.mean(torch.abs(pred_e-input_label))
            nll_loss_x = torch.mean(torch.abs(pred_x-input_label))
        elif args.Mat2Spec_loss_type == 'MSE':
            nll_loss = torch.mean((pred_e-input_label)**2)
            nll_loss_x = torch.mean((pred_x-input_label)**2)
        total_loss = (nll_loss + nll_loss_x) * c1 + kl_loss * c2 + c_loss * c3

        return total_loss, nll_loss, nll_loss_x, kl_loss, c_loss, pred_e, pred_x







