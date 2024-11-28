from typing import Dict, Optional

import torch
import torch.nn as nn
from torch.functional import Tensor
import pointnet2._ext as _ext
from model.utils.position_embedding import PositionEmbeddingCoordsSine

from model.utils.helpers import (ACTIVATION_DICT, NORM_DICT, WEIGHT_INIT_DICT, GenericMLP, get_clones)
from utils.grasp_init import init_grasps


class GraspQDecoder(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.init_hand_cfg = cfg.init_hand
        self.encoder_to_decoder_projection = GenericMLP(
            input_dim=cfg.encoder_out_dim,
            hidden_dims=cfg.encoder_to_decoder.hidden_dims,
            output_dim=cfg.dim,
            norm_fn_name="bn1d",
            activation="relu",
            use_conv=True,
            output_use_activation=True,
            output_use_norm=True,
            output_use_bias=False,
        )
        decoder_layer = TransformerDecoderLayer(
            d_model=cfg.dim,
            nhead=cfg.num_heads,
            dim_feedforward=cfg.ffn_dim,
            dropout=cfg.dropout_prob,
            activation="relu",
            normalize_before=True,
            norm_fn_name=cfg.norm_func,
        )
        self.decoder = TransformerDecoder(
            decoder_layer,
            num_layers=cfg.num_layers,
            return_intermediate=True,
        )

        if cfg.query_embedding_name == "grasp":
            self.query_embedding = GenericMLP(
                input_dim=22 + 3 + cfg.init_hand.embed_mlp.rotation_dim,
                hidden_dims=cfg.init_hand.embed_mlp.hidden_dims,
                output_dim=cfg.dim,
                use_conv=True,
                norm_fn_name="bn1d",
                output_use_activation=True,
                output_use_norm=True,
            )
        elif cfg.query_embedding_name == "xyz":
            # self.pos_embedding = PositionEmbeddingCoordsSine(
            #     d_pos=cfg.dim,
            #     pos_type=cfg.position_embedding,
            #     normalize=True,
            # )
            self.query_embedding = GenericMLP(
                input_dim=3,
                hidden_dims=[cfg.dim],
                output_dim=cfg.dim,
                use_conv=True,
                output_use_activation=True,
                hidden_use_bias=True,
            )
        elif cfg.query_embedding_name == "learning":
            self.query_embedding = nn.Embedding(cfg.num_queries, cfg.dim)

        elif cfg.query_embedding_name == "xyz_learning":
            self.pos_embedding = PositionEmbeddingCoordsSine(
                d_pos=cfg.dim,
                pos_type=cfg.position_embedding,
                normalize=True,
            )
            self.query_embedding = GenericMLP(
                input_dim=3,
                hidden_dims=[cfg.dim],
                output_dim=cfg.dim,
                use_conv=True,
                output_use_activation=True,
                hidden_use_bias=True,
            )
            self.query_learn_embedding = nn.Embedding(cfg.num_queries, cfg.dim)
            self.query_fusion_mlp = GenericMLP(
                input_dim=cfg.dim*2,
                hidden_dims=[cfg.dim*2],
                output_dim=cfg.dim,
                use_conv=True,
                output_use_activation=True,
                hidden_use_bias=True,
            )
            self.query_fusion_attention = nn.MultiheadAttention(cfg.dim, 4, dropout=0.1)
            self.query_fusion_attention_dropout = nn.Dropout(0.1, inplace=False)
            self.query_fusion_attention_norm = NORM_DICT["bn"](cfg.dim)
        else:
            raise Exception("Not Vaild Query Embedding Setting")
    
        if cfg.enc_embedding_name == "grasp":
            self.enc_embedding = GenericMLP(
                input_dim=22 + 3 + cfg.init_hand.embed_mlp.rotation_dim,
                hidden_dims=cfg.init_hand.embed_mlp.hidden_dims,
                output_dim=cfg.dim,
                use_conv=True,
                norm_fn_name="bn1d",
                output_use_activation=True,
                output_use_norm=True,
            )
        elif cfg.enc_embedding_name == "xyz":
            self.enc_embedding = PositionEmbeddingCoordsSine(
                d_pos=cfg.dim,
                pos_type=cfg.position_embedding,
                normalize=True,
            )
        elif cfg.enc_embedding_name == "learning":
            self.enc_embedding = GenericMLP(
                input_dim=3,
                hidden_dims=cfg.init_hand.embed_mlp.hidden_dims,
                output_dim=cfg.dim,
                use_conv=True,
                norm_fn_name="bn1d",
                output_use_activation=True,
                output_use_norm=True,
            )
        else:
            raise Exception("Not Vaild Encoder Embedding Setting")

        self.enc_embedding_name = cfg.enc_embedding_name
        self.query_embedding_name = cfg.query_embedding_name
        self.num_queries = cfg.num_queries
        self.dim = cfg.dim

    def forward(self, input_dict: Dict[str, Tensor]):
        enc_xyz, enc_features = input_dict["enc_xyz"], input_dict["enc_feature"]
        enc_features = self.encoder_to_decoder_projection(enc_features).permute(2, 0, 1).contiguous()

        if self.query_embedding_name == "grasp":
            init_hand_poses, query_embed = self.get_query_anchor_embeddings(input_dict["obj_pc"], enc_xyz, input_dict["convex_hull"])
        elif self.query_embedding_name == "xyz":
            # obj_pc_dims = [
            #     input_dict["point_cloud_dims_min"],
            #     input_dict["point_cloud_dims_max"],
            # ]
            init_hand_poses, query_embed = self.get_query_xyz_embeddings(enc_xyz)
            # b,c,q -> q,b,c
            query_embed = query_embed.permute(2, 0, 1).contiguous()
        elif self.query_embedding_name == "learning":
            query_embed = self.query_embedding.weight.unsqueeze(1).repeat(1, enc_xyz.size(0), 1)  # (N, B, C)
            init_hand_poses = None 
    
        elif self.query_embedding_name == "xyz_learning":
            tpye_methods = 3
            if tpye_methods == 1:
                # obj_pc_dims = [
                #     input_dict["point_cloud_dims_min"],
                #     input_dict["point_cloud_dims_max"],
                # ]
                init_hand_poses, query_embed = self.get_query_xyz_embeddings(enc_xyz)
                query_learn_embed = self.query_learn_embedding.weight.unsqueeze(-1).repeat(query_embed.size(0), 1, query_embed.size(-1))  # (b, c, q)
                query_embed = self.query_fusion_mlp(torch.cat([query_embed, query_learn_embed], dim=1)).permute(2, 0, 1).contiguous() #q, b, c
            elif tpye_methods == 2:
                # (b, c, q)
                init_hand_poses, query_embed = self.get_query_xyz_embeddings(enc_xyz)
                query_learn_embed = self.query_learn_embedding.weight.unsqueeze(-1).repeat(query_embed.size(0), 1, 1)  # (b, c, 1)
                # b, q, c
                qkv = torch.cat([query_embed, query_learn_embed], dim=-1).permute(2, 0, 1).contiguous()
                query_embed = self.query_fusion_attention_dropout(self.query_fusion_attention(qkv, qkv, qkv)[0]) + qkv
                query_embed = self.query_fusion_attention_norm(query_embed).permute(1, 0, 2).contiguous()[:self.num_queries]
            elif tpye_methods == 3:
                init_hand_poses, query_embed = self.get_query_xyz_embeddings(enc_xyz)
                # print(query_embed.shape, self.query_learn_embedding.weight.shape)
                # 32, 256, 16(b,c,q); 16,256(q,c)
                query_learn_embed = self.query_learn_embedding.weight.unsqueeze(0).repeat(query_embed.size(0), 1, 1)  # (b, q, c)
                qkv = torch.cat([query_embed.transpose(1, 2).contiguous(), query_learn_embed], dim=1).transpose(0, 1).contiguous() # (2q, b, c)
                atten_mask = torch.eye(self.num_queries*2).to(qkv.device)
                atten_mask[-self.num_queries] = 1
                atten_mask[:, -self.num_queries] = 1
                query_embed = self.query_fusion_attention_dropout(self.query_fusion_attention(qkv, qkv, qkv, attn_mask=atten_mask)[0]) + qkv
                query_embed = self.query_fusion_attention_norm(query_embed)[:self.num_queries]               
            elif tpye_methods == 4:
                # q,b,c
                init_hand_poses, query_embed = self.get_query_xyz_embeddings(enc_xyz)
                # print(query_embed.shape, self.query_learn_embedding.weight.shape)
                # 32, 256, 16(b,c,q); 16,256(q,c)
                query_learn_embed = self.query_learn_embedding.weight.unsqueeze(1).repeat(1, query_embed.size(0), 1)  # (q, b, c)
                qkv = torch.cat([query_embed.permute(2, 0 ,1).contiguous(), query_learn_embed], dim=0)# (2q, b, c)
                query_embed = self.query_fusion_attention_dropout(self.query_fusion_attention(qkv, qkv, qkv)[0]) + qkv
                query_embed = self.query_fusion_attention_norm(query_embed)[:self.num_queries]
                


        if self.enc_embedding_name == "grasp":
            enc_pos = self.get_encoder_embedding(input_dict["obj_pc"], enc_xyz, input_dict["convex_hull"])  # (N, B, C)
        elif self.enc_embedding_name == "xyz":
            obj_pc_dims = [
                input_dict["point_cloud_dims_min"],
                input_dict["point_cloud_dims_max"],
            ]
            enc_pos = self.enc_embedding(enc_xyz, self.dim, input_range=obj_pc_dims)
            enc_pos = enc_pos.permute(2, 0, 1).contiguous()
        elif self.enc_embedding_name == "learning":
            enc_pos = self.enc_embedding(enc_xyz.transpose(1, 2).contiguous()).permute(2, 0, 1).contiguous()

        tgt = torch.zeros_like(query_embed)
        # torch.Size([15, 32, 256]) torch.Size([128, 32, 256]) torch.Size([15, 32, 256]) torch.Size([128, 32, 256])
        # print(tgt.shape, enc_features.shape, query_embed.shape, enc_pos.shape)
        rt_features = self.decoder(
            tgt, enc_features, query_pos=query_embed, pos=enc_pos, 
        )[0]
        return rt_features, init_hand_poses
    
    def get_query_anchor_embeddings(
        self,
        obj_point_cloud: Tensor,
        enc_xyz: Tensor,
        convex_hull: Tensor,
    ):
        B, num_encoder_tokens = enc_xyz.shape[:2]
        assert self.num_queries < num_encoder_tokens, f"num queries={self.num_queries}; {num_encoder_tokens}"

        query_inds = _ext.furthest_point_sampling(enc_xyz, self.num_queries).long()
        query_xyz = [torch.gather(enc_xyz[..., x], 1, query_inds) for x in range(3)]
        query_xyz = torch.stack(query_xyz).permute(1, 2, 0).contiguous()  # (B, num_queries, 3)

        init_hand_poses = init_grasps(
            obj_point_cloud,
            query_xyz,
            convex_hull,
            self.init_hand_cfg,
        )  # (B, num_points, translation(3) + rotation + joints(22))
        init_hand_poses = init_hand_poses.transpose(1, 2).contiguous()  # (B, 29, num_points)
        enc_embedding = self.query_embedding(init_hand_poses)  # (B, C, num_points)
        return init_hand_poses.transpose(1, 2).contiguous(), enc_embedding.permute(2, 0, 1).contiguous()
    
    def get_query_xyz_embeddings(self, encoder_xyz):
        B, num_encoder_tokens = encoder_xyz.shape[:2]
        assert self.num_queries < num_encoder_tokens, f"num queries={self.num_queries}; {num_encoder_tokens}"
        query_inds = _ext.furthest_point_sampling(encoder_xyz, self.num_queries).long()
        # (3, b, q)
        query_xyz = [torch.gather(encoder_xyz[..., x], 1, query_inds) for x in range(3)]
        # 3, b, q --> b, 3, q
        query_xyz = torch.stack(query_xyz).permute(1, 0, 2).contiguous()  
        # pos_embed = self.pos_embedding(query_xyz, input_range=point_cloud_dims)
        query_embed = self.query_embedding(query_xyz)  # (b, c, q)
        return query_xyz.permute(0, 2, 1).contiguous()  , query_embed
    
    def get_encoder_embedding(
        self,
        obj_point_cloud: Tensor,
        enc_xyz: Tensor,
        convex_hull: Tensor,
    ):
        """
        Params:
            obj_point_cloud: A Tensor of size (B, N, 3)
            enc_xyz: 3D coordinates of points (B, M, 3)
            convex_hull: Convex hull point cloud of objects (B, K, 3)
        Returns:
            enc_embedding: A Tensor of encoder embedding (num_queries, B, C)
        """
        init_hand_poses = init_grasps(
            obj_point_cloud,
            enc_xyz,
            convex_hull,
            self.init_hand_cfg,
        )  # (B, num_points, translation(3) + rotation + joints(22))
        init_hand_poses = init_hand_poses.transpose(1, 2).contiguous()  # (B, 29, num_points)
        enc_embedding = self.enc_embedding(init_hand_poses)  # (B, C, num_points)
        return enc_embedding.permute(2, 0, 1).contiguous()


class TransformerDecoder(nn.Module):
    def __init__(
        self, decoder_layer, num_layers,
        norm_fn_name="ln",
        return_intermediate=False,
        weight_init_name="xavier_uniform"
    ):
        super().__init__()
        self.layers = get_clones(decoder_layer, num_layers)
        self.num_layers = num_layers
        self.norm = None
        if norm_fn_name is not None:
            self.norm = NORM_DICT[norm_fn_name](self.layers[0].linear2.out_features)
        self.return_intermediate = return_intermediate
        self._reset_parameters(weight_init_name)

    def _reset_parameters(self, weight_init_name):
        func = WEIGHT_INIT_DICT[weight_init_name]
        for p in self.parameters():
            if p.dim() > 1:
                func(p)

    def forward(
        self, tgt, memory,
        tgt_mask: Optional[Tensor] = None,
        memory_mask: Optional[Tensor] = None,
        tgt_key_padding_mask: Optional[Tensor] = None,
        memory_key_padding_mask: Optional[Tensor] = None,
        pos: Optional[Tensor] = None,
        query_pos: Optional[Tensor] = None,
        transpose_swap: Optional[bool] = False,
        return_attn_weights: Optional[bool] = False,
    ):
        if transpose_swap:
            bs, c, h, w = memory.shape
            memory = memory.flatten(2).permute(2, 0, 1)  # memory: bs, c, t -> t, b, c
            if pos is not None:
                pos = pos.flatten(2).permute(2, 0, 1)
        output = tgt

        intermediate = []
        attns = []

        for layer in self.layers:
            output, attn = layer(
                output, memory, tgt_mask=tgt_mask,
                memory_mask=memory_mask,
                tgt_key_padding_mask=tgt_key_padding_mask,
                memory_key_padding_mask=memory_key_padding_mask,
                pos=pos, query_pos=query_pos,
                return_attn_weights=return_attn_weights
            )
            if self.return_intermediate:
                intermediate.append(self.norm(output))
            if return_attn_weights:
                attns.append(attn)

        if self.norm is not None:
            output = self.norm(output)
            if self.return_intermediate:
                intermediate.pop()
                intermediate.append(output)

        if return_attn_weights:
            attns = torch.stack(attns)

        if self.return_intermediate:
            return torch.stack(intermediate), attns

        return output, attns


class TransformerDecoderLayer(nn.Module):

    def __init__(self, d_model, nhead=4, dim_feedforward=256,
                 dropout=0.1, dropout_attn=None,
                 activation="relu", normalize_before=True,
                 norm_fn_name="ln"):
        super().__init__()
        if dropout_attn is None:
            dropout_attn = dropout
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        self.multihead_attn = nn.MultiheadAttention(
            d_model, nhead, dropout=dropout)

        self.norm1 = NORM_DICT[norm_fn_name](d_model)
        self.norm2 = NORM_DICT[norm_fn_name](d_model)

        self.norm3 = NORM_DICT[norm_fn_name](d_model)
        self.dropout1 = nn.Dropout(dropout, inplace=False)
        self.dropout2 = nn.Dropout(dropout, inplace=False)
        self.dropout3 = nn.Dropout(dropout, inplace=False)

        # Implementation of Feedforward model
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout, inplace=False)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.activation = ACTIVATION_DICT[activation]()
        self.normalize_before = normalize_before

    def with_pos_embed(self, tensor, pos: Optional[Tensor]):
        return tensor if pos is None else tensor + pos

    def forward_post(self, tgt, memory,
                     tgt_mask: Optional[Tensor] = None,
                     memory_mask: Optional[Tensor] = None,
                     tgt_key_padding_mask: Optional[Tensor] = None,
                     memory_key_padding_mask: Optional[Tensor] = None,
                     pos: Optional[Tensor] = None,
                     query_pos: Optional[Tensor] = None,
                     return_attn_weights: Optional[bool] = False):
        q = k = self.with_pos_embed(tgt, query_pos)
        tgt2 = self.self_attn(q, k, value=tgt, attn_mask=tgt_mask,
                              key_padding_mask=tgt_key_padding_mask)[0]
        tgt = tgt + self.dropout1(tgt2)
        tgt = self.norm1(tgt)
        tgt2, attn = self.multihead_attn(query=self.with_pos_embed(tgt, query_pos),
                                         key=self.with_pos_embed(memory, pos),
                                         value=memory, attn_mask=memory_mask,
                                         key_padding_mask=memory_key_padding_mask)
        tgt = tgt + self.dropout2(tgt2)
        tgt = self.norm2(tgt)
        tgt2 = self.linear2(self.dropout(self.activation(self.linear1(tgt))))
        tgt = tgt + self.dropout3(tgt2)
        tgt = self.norm3(tgt)
        if return_attn_weights:
            return tgt, attn
        return tgt, None

    def forward_pre(self, tgt, memory,
                    tgt_mask: Optional[Tensor] = None,
                    memory_mask: Optional[Tensor] = None,
                    tgt_key_padding_mask: Optional[Tensor] = None,
                    memory_key_padding_mask: Optional[Tensor] = None,
                    pos: Optional[Tensor] = None,
                    query_pos: Optional[Tensor] = None,
                    return_attn_weights: Optional[bool] = False):
        tgt2 = self.norm1(tgt)
        q = k = self.with_pos_embed(tgt2, query_pos)
        tgt2 = self.self_attn(q, k, value=tgt2, attn_mask=tgt_mask,
                              key_padding_mask=tgt_key_padding_mask)[0]
        tgt = tgt + self.dropout1(tgt2)
        tgt2 = self.norm2(tgt)
        tgt2, attn = self.multihead_attn(query=self.with_pos_embed(tgt2, query_pos),
                                         key=self.with_pos_embed(memory, pos),
                                         value=memory, attn_mask=memory_mask,
                                         key_padding_mask=memory_key_padding_mask)
        tgt = tgt + self.dropout2(tgt2)
        tgt2 = self.norm3(tgt)
        tgt2 = self.linear2(self.dropout(self.activation(self.linear1(tgt2))))
        tgt = tgt + self.dropout3(tgt2)
        if return_attn_weights:
            return tgt, attn
        return tgt, None

    def forward(self, tgt, memory,
                tgt_mask: Optional[Tensor] = None,
                memory_mask: Optional[Tensor] = None,
                tgt_key_padding_mask: Optional[Tensor] = None,
                memory_key_padding_mask: Optional[Tensor] = None,
                pos: Optional[Tensor] = None,
                query_pos: Optional[Tensor] = None,
                return_attn_weights: Optional[bool] = False):
        if self.normalize_before:
            return self.forward_pre(tgt, memory, tgt_mask, memory_mask,
                                    tgt_key_padding_mask, memory_key_padding_mask, pos, query_pos, return_attn_weights)
        return self.forward_post(tgt, memory, tgt_mask, memory_mask,
                                 tgt_key_padding_mask, memory_key_padding_mask, pos, query_pos, return_attn_weights)
