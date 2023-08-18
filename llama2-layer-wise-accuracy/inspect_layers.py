import torch
from transformers import LlamaTokenizer, LlamaForCausalLM

class AttnWrapper(torch.nn.Module):
    def __init__(self, attn):
        super().__init__()
        self.attn = attn
        self.activations = None
        self.add_tensor = None

    def forward(self, *args, **kwargs):
        output = self.attn(*args, **kwargs)
        if self.add_tensor is not None:
            output = (output[0] + self.add_tensor,)+output[1:]
        self.activations = output[0]
        return output

    def reset(self):
        self.activations = None
        self.add_tensor = None

class BlockOutputWrapper(torch.nn.Module):
    def __init__(self, block, unembed_matrix, norm, final_block, extension=False):
        super().__init__()
        self.block = block
        self.unembed_matrix = unembed_matrix
        self.norm = norm
        self.final_block = final_block
        self.extension = extension

        self.block.self_attn = AttnWrapper(self.block.self_attn)
        self.post_attention_layernorm = self.block.post_attention_layernorm

        self.attn_mech_output_unembedded = None
        self.intermediate_res_unembedded = None
        self.mlp_output_unembedded = None
        self.block_output_unembedded = None
        
    def _sqz(self, x):
        if isinstance(x, torch.Tensor):
            return x
        try:
            return x[0]
        except:
            return x
        
    def F_L(self, x):
        return self._sqz(self.final_block(x))
    
    def forward(self, *args, **kwargs):
        output = self.block(*args, **kwargs)
        h_l, input = output[0], args[0]
        assert(
            h_l.size(0) == 1
        ), "Make sure you're only running LogitLens on single generations only"
        self.block_output_unembedded = self.unembed_matrix(self.norm(h_l + self.F_L(h_l))) if self.extension else self.unembed_matrix(self.norm(h_l))
        attn_output = self.block.self_attn.activations
        self.attn_mech_output_unembedded = self.unembed_matrix(self.norm(attn_output))
        attn_output += input
        self.intermediate_res_unembedded = self.unembed_matrix(self.norm(attn_output))
        mlp_output = self.block.mlp(self.post_attention_layernorm(attn_output))
        self.mlp_output_unembedded = self.unembed_matrix(self.norm(mlp_output))
        return output

    def attn_add_tensor(self, tensor):
        self.block.self_attn.add_tensor = tensor

    def reset(self):
        self.block.self_attn.reset()

    def get_attn_activations(self):
        return self.block.self_attn.activations

class Llama7BHelper:
    def __init__(self, token, extension=False):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.tokenizer = LlamaTokenizer.from_pretrained("meta-llama/Llama-2-7b-hf", use_auth_token=token)
        self.model = LlamaForCausalLM.from_pretrained("meta-llama/Llama-2-7b-hf", use_auth_token=token).to(self.device)
        for i, layer in enumerate(self.model.model.layers):
            self.model.model.layers[i] = BlockOutputWrapper(layer, self.model.lm_head, self.model.model.norm, self.model.model.layers[-1], extension)

    def generate_text(self, prompt, max_length=100):
        inputs = self.tokenizer(prompt, return_tensors="pt")
        generate_ids = self.model.generate(inputs.input_ids.to(self.device), max_length=max_length)
        return self.tokenizer.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]

    def get_logits(self, prompt):
        inputs = self.tokenizer(prompt, return_tensors="pt")
        with torch.no_grad():
          logits = self.model(inputs.input_ids.to(self.device)).logits
          return logits

    def set_add_attn_output(self, layer, add_output):
        self.model.model.layers[layer].attn_add_tensor(add_output)

    def get_attn_activations(self, layer):
        return self.model.model.layers[layer].get_attn_activations()

    def reset_all(self):
        for layer in self.model.model.layers:
            layer.reset()

    def print_decoded_activations(self, decoded_activations, label, topk=10):
        softmaxed = torch.nn.functional.softmax(decoded_activations[0][-1], dim=-1)
        values, indices = torch.topk(softmaxed, topk)
        probs_percent = [int(v * 100) for v in values.tolist()]
        tokens = self.tokenizer.batch_decode(indices.unsqueeze(-1))
        print(label, list(zip(tokens, probs_percent)))
        
    def first_decoded_activations(self, decoded_activations):
        softmaxed = torch.nn.functional.softmax(decoded_activations[0][-1], dim=-1)
        values, indices = torch.topk(softmaxed, 1)
        probs_percent = [int(v * 100) for v in values.tolist()]
        tokens = self.tokenizer.batch_decode(indices.unsqueeze(-1))
        return (tokens, probs_percent)

    def decode_all_layers(self, text, answer, topk=10, start_layer=0, end_layer=31, print_attn_mech=True, print_intermediate_res=True, print_mlp=True, print_block=True):
        self.get_logits(text)
        layer_wise_attention = {'question': text, 'correct_ans': answer}
        layer_wise_residual = {'question': text, 'correct_ans': answer}
        layer_wise_mlp = {'question': text, 'correct_ans': answer}
        layer_wise_block = {'question': text, 'correct_ans': answer}
        
        for i, layer in enumerate(self.model.model.layers):
            if start_layer <= i <= end_layer:
                if print_attn_mech:
                    layer_wise_attention[f'layer_{i}'] = self.first_decoded_activations(layer.attn_mech_output_unembedded)
                if print_intermediate_res:
                    layer_wise_residual[f'layer_{i}'] = self.first_decoded_activations(layer.intermediate_res_unembedded)
                if print_mlp:
                    layer_wise_mlp[f'layer_{i}'] = self.first_decoded_activations(layer.mlp_output_unembedded)
                if print_block:
                    layer_wise_block[f'layer_{i}'] = self.first_decoded_activations(layer.block_output_unembedded)
                    
        return layer_wise_attention, layer_wise_residual, layer_wise_mlp, layer_wise_block
