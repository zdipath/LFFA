import torch
import math
import torch.nn.functional as F
from torch.nn.functional import multi_head_attention_forward








class FMAdapter:
    def __init__(self,  agg_model, device, **kwargs):
        self.agg_model = agg_model.to(device)

        self.device = device
        self.kwargs = kwargs



    def wsi_embed_from_patch_features(self, features, indexs_or_coords, patch_size_lv0):
        raise NotImplementedError

    def attn_from_features(self, features, indexs_or_coords, patch_size_lv0):
        raise NotImplementedError
    
class ChiefAdapter(FMAdapter):
    def wsi_embed_from_patch_features(self, features, indexs_or_coords, patch_size_lv0):
        anatomical = int(self.kwargs.get('anatomical', 13))
        result = self.agg_model(features, torch.tensor([anatomical], device=features.device))
        return result['WSI_feature']

    def attn_from_features(self, features, indexs_or_coords, patch_size_lv0):
        """Return attention vector [N] for Chief; averaged over queries; CLS removed."""
        try:
            anatomical = int(self.kwargs.get('anatomical', 13))
        except Exception:
            anatomical = 13
        result = self.agg_model(features, torch.tensor([anatomical], device=features.device))
        attn_raw = result.get('attention_raw', None) if isinstance(result, dict) else None
        if attn_raw is None:
            return None
        a = attn_raw

        while a.dim() > 2:
            a = a.squeeze(0)
        N = features.size(0)

        if a.size(-1) == N + 1:
            a = a[:, 1:]
        elif a.size(-1) != N:
            a = a[..., :N]

        a = a.mean(dim=0)
        return a

class TitanAdapter(FMAdapter):
    def wsi_embed_from_patch_features(self, features, indexs_or_coords, patch_size_lv0):
        coords = indexs_or_coords
        diffs = torch.norm(coords[1:].float() - coords[:-1].float(), dim=1)
        count_512 = torch.sum(diffs == 512)
        count_1024 = torch.sum(diffs == 1024)
        count_256  = torch.sum(diffs == 256)
        count_128  = torch.sum(diffs == 128)
        counts = {
                256: count_256,
                128: count_128,
                1024: count_1024,
                512: count_512,
                    }
        patch_size_lv0 = max(counts, key=counts.get)
        return self.agg_model.encode_slide_from_patch_features(features, coords, patch_size_lv0)

    def attn_from_features(self, features, indexs_or_coords, patch_size_lv0):
        """Return attention vector [N] for Titan via transient forward hooks."""
        cache = {}
        try:
            attn_module = self.agg_model.vision_encoder.attn_pool_contrastive.attn
        except Exception:
            return None

        def pre_hook(module, inputs):
            cache['qkv'] = inputs

        def fwd_hook(module, inputs, output):
            query, key, value = cache.get('qkv', (None, None, None))
            if query is None:
                return
            attn_weights = multi_head_attention_forward(
                query=query, key=key, value=value,
                embed_dim_to_check=module.embed_dim, num_heads=module.num_heads,
                in_proj_weight=module.in_proj_weight, in_proj_bias=module.in_proj_bias,
                bias_k=module.bias_k, bias_v=module.bias_v,
                add_zero_attn=module.add_zero_attn, dropout_p=module.dropout,
                out_proj_weight=module.out_proj.weight, out_proj_bias=module.out_proj.bias,
                training=module.training, key_padding_mask=None, need_weights=True,
                attn_mask=None, use_separate_proj_weight=False, q_proj_weight=None,
                k_proj_weight=None, v_proj_weight=None, static_k=None, static_v=None,
                average_attn_weights=False,
            )[1]
            cache['weights'] = attn_weights

        h1 = attn_module.register_forward_pre_hook(pre_hook)
        h2 = attn_module.register_forward_hook(fwd_hook)
        try:

            _ = self.agg_model.encode_slide_from_patch_features(features, indexs_or_coords, patch_size_lv0)
        finally:
            h1.remove()
            h2.remove()

        attn = cache.get('weights', None)
        if attn is None:
            return None


        attn = attn[:, :, :, 1:]
        attn = attn.mean(dim=1).squeeze(0).squeeze(0)
        return attn

class GigaPathAdapter(FMAdapter):
    def wsi_embed_from_patch_features(self, features, indexs_or_coords, patch_size_lv0):

        tile_embeddings = features.unsqueeze(0)
        with torch.cuda.amp.autocast(dtype=torch.float16):
            coords = indexs_or_coords
            reprs = self.agg_model(tile_embeddings, coords, all_layer_embed=True)[-1]
        return reprs

    def attn_from_features(self, features, indexs_or_coords, patch_size_lv0, layer_idx: int = -1):

        encoder_layers = self.agg_model.encoder.layers
        if layer_idx < 0:
            layer_idx = len(encoder_layers) + layer_idx
        attn_module = encoder_layers[layer_idx].self_attn

        cache = {}

        def hook_q(m, inp, out): cache["q"] = out
        def hook_k(m, inp, out): cache["k"] = out
        def hook_v(m, inp, out): cache["v"] = out

        hq = attn_module.q_proj.register_forward_hook(hook_q)
        hk = attn_module.k_proj.register_forward_hook(hook_k)
        hv = attn_module.v_proj.register_forward_hook(hook_v)

        try:
            with torch.cuda.amp.autocast(dtype=torch.float16):
                _ = self.agg_model(features.unsqueeze(0).half(), indexs_or_coords)
        finally:
            hq.remove()
            hk.remove()
            hv.remove()

        q, k, v = cache["q"], cache["k"], cache["v"]

        B, L, _ = q.shape
        H = attn_module.num_heads
        d = attn_module.head_dim


        q = q.view(B, L, H, d).transpose(1, 2)
        k = k.view(B, L, H, d).transpose(1, 2)


        attn_scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(d)
        attn_probs = F.softmax(attn_scores, dim=-1)


        attn_cls = attn_probs[:, :, 0, 1:]
        attn_cls = attn_cls.mean(dim=1)     

        return attn_cls.squeeze(0)  

class PrismAdapter(FMAdapter):
    

    def wsi_embed_from_patch_features(self, features, indexs_or_coords, patch_size_lv0):
        tile_embeddings = features.unsqueeze(0)
        reprs = self.agg_model.slide_representations(tile_embeddings)
        return reprs['image_embedding']

    def attn_from_features(self, features, indexs_or_coords, patch_size_lv0):
        """Return attention vector [N] for Prism by reading xattn attn weights."""
        cache = {}
        xattn_module = self.agg_model.image_resampler.perceiver.layers[-1]['xattn']['xattn']
        xattn_module.return_attn = True

        def hook(module, inp, out):
            cache['weights'] = out[2]

        handle = xattn_module.register_forward_hook(hook)
        try:
            tile_mask = torch.ones((1, features.shape[0]), dtype=torch.bool, device=features.device)
            _ = self.agg_model.slide_representations(features.unsqueeze(0), tile_mask=tile_mask)
        finally:
            handle.remove()
        xattn_module.return_attn = False

        attn = cache.get('weights', None)
        attn = attn[:, :, 0, :].squeeze()
        return attn
    
class FeatherAdapter(FMAdapter):
    def wsi_embed_from_patch_features(self, features, indexs_or_coords, patch_size_lv0):
        features = features.unsqueeze(0)
        _, log_dict = self.agg_model(
            features,
            loss_fn=torch.nn.CrossEntropyLoss(),
            label=torch.LongTensor([1]).to(features.device),
            return_attention=False,
            return_slide_feats=True
        )
        return log_dict['slide_feats']
    
    def attn_from_features(self, features, indexs_or_coords, patch_size_lv0):
        features = features.unsqueeze(0)
        _, log_dict = self.agg_model(
            features,
            loss_fn=torch.nn.CrossEntropyLoss(),
            label=torch.LongTensor([1]).to(features.device),
            return_attention=True,
            return_slide_feats=True
        )
        return log_dict['attention']

class TangleAdapter(FMAdapter):
    def wsi_embed_from_patch_features(self, features, indexs_or_coords, patch_size_lv0):
        tile_embeddings = features.unsqueeze(0)
        return self.agg_model.get_features(tile_embeddings)
    def attn_from_features(self, features, indexs_or_coords=None, patch_size_lv0=None, return_raw: bool=False):
        """
        Capture per-head ABMIL attention via forward hooks, then average heads.
        returns:
            attn_mean: [N] (averaged over heads)
            (optional) raw_mean: [N] raw attention averaged over heads
        """
        if features.dim() == 3:
            tile_embeddings = features
        else:
            tile_embeddings = features.unsqueeze(0)
        model = self.agg_model.wsi_embedder
        assert hasattr(model, "attn"), "agg_model must have .attn (ModuleList of BatchedABMIL heads)."

        H = model.n_heads
        cache_attn = [None] * H
        cache_raw  = [None] * H

        def make_hook(idx):
            def hook(module, inputs, outputs):


                if isinstance(outputs, (tuple, list)) and len(outputs) >= 1:
                    cache_attn[idx] = outputs[0]
                    if len(outputs) >= 2:
                        cache_raw[idx] = outputs[1]
                else:

                    cache_attn[idx] = outputs
            return hook

        handles = []
        for i, head in enumerate(model.attn):
            handles.append(head.register_forward_hook(make_hook(i)))

        try:


            _ = self.agg_model.get_features(tile_embeddings)
        finally:
            for h in handles:
                h.remove()


        attn_list = []
        raw_list  = []
        for i in range(H):
            a = cache_attn[i]
            if a is None:
                continue

            if a.dim() == 3 and a.size(-1) == 1:
                a = a.squeeze(-1)
            attn_list.append(a)
            r = cache_raw[i]
            if r is not None:
                if r.dim() == 3 and r.size(-1) == 1:
                    r = r.squeeze(-1)
                raw_list.append(r)

        if not attn_list:
            return None if not return_raw else (None, None)


        attn_stack = torch.stack(attn_list, dim=0)
        attn_stack = attn_stack.transpose(0, 1)
        attn_mean  = attn_stack.mean(dim=1).squeeze(0)

        if not return_raw or not raw_list:
            return attn_mean





def make_adapter(fm: str, agg_model,  device) -> FMAdapter:
    fm = fm.lower()
    if fm == 'titan':
        return TitanAdapter( agg_model,  device)
    if fm == 'chief':
        return ChiefAdapter( agg_model,  device, tmp_z=13)
    if fm == 'gigapath':
        return GigaPathAdapter(agg_model,  device)
    if fm == 'prism':
        return PrismAdapter(agg_model,  device)
    if fm == 'feather':
        return FeatherAdapter(agg_model,  device)
    if fm == 'tangle':
        return TangleAdapter(agg_model,  device)
    raise ValueError(f"Unsupported foundation model: {fm}")
