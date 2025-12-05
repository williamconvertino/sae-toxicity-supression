from sae_lens import SAE
from transformer_lens import HookedTransformer
from sae_lens import HookedSAETransformer

def load_model_and_sae():

    sae = SAE.from_pretrained(
        release="gemma-2b-it-res-jb",
        sae_id="blocks.12.hook_resid_post",
        device="cuda"
    )

    model = HookedSAETransformer.from_pretrained(
        "gemma-2b-it",
        device="cuda",
        dtype="float32"
    )

    return model, sae

def decode_feature_tokens(sae, model, feature_idx, k=20):
    w_dec = sae.W_dec[feature_idx]
    W_U = model.W_U
    scores = w_dec @ W_U
    topk = scores.topk(k)

    tokens = [model.tokenizer.decode([tid.item()]) for tid in topk.indices]
    values = topk.values.tolist()

    return list(zip(tokens, values))