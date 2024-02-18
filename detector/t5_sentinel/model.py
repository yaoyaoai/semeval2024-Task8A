import torch
import torch.nn as nn
from torch import Tensor
from typing import Optional, Tuple
from detector.t5_sentinel.__init__ import config
if config.backbone.name=='mt5-large' or config.backbone.name=='mt5-small' or config.backbone.name=='mt5-xl':
    from transformers import MT5ForConditionalGeneration as Backbone
elif config.backbone.name=='flan-t5-large':
    from transformers import AutoModelForSeq2SeqLM as Backbone
elif config.backbone.name=='umt5-base':
    from transformers import UMT5ForConditionalGeneration as Backbone
else:
    from transformers import T5ForConditionalGeneration as Backbone


from detector.t5_sentinel.t5types import SentinelOutput


class Sentinel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.backbone: Backbone = Backbone.from_pretrained(f"/home/yaoxy/T5-Sentinel-public-main/detector/t5_sentinel/{config.backbone.name}")
        self.config = config
    
    def forward(self, corpus_ids: Tensor, corpus_mask: Tensor, label_ids: Optional[Tensor] = None, selectedDataset: Tuple[str] = ('Human', 'ChatGPT', 'cohere', 'davinci','bloomz','dolly')) -> SentinelOutput:
        '''
        Args:
            corpus_ids (Tensor): The input corpus ids.
            corpus_mask (Tensor): The input attention mask.
            label_ids (Tensor): The input label ids.

        Returns:
            output (SentinelOutput): The output of the model.
        
        Example:
        >>> model = Sentinel()
        >>> model.eval()
        >>> with torch.no_grad():
        >>>     corpus_ids, corpus_mask, label_ids = next(iter(train_loader))
        >>>     model.forward(corpus_ids.cuda(), corpus_mask.cuda(), label_ids.cuda())
        huggingface=Seq2SeqLMOutput(
            loss=..., 
            logits=..., 
            past_key_values=..., 
            decoder_hidden_states=..., 
            decoder_attentions=..., 
            cross_attentions=..., 
            encoder_last_hidden_state=..., 
            encoder_hidden_states=..., 
            encoder_attentions=...
        ),
        probabilities=tensor([
            [1.0000e+00, 2.5421e-07, 1.8315e-07, 4.8886e-07],
            [1.0000e+00, 5.2608e-07, 1.0334e-06, 9.4020e-07],
            [9.9997e-01, 5.3097e-06, 8.8986e-06, 1.4712e-05],
            [9.9999e-01, 2.4895e-06, 1.7681e-06, 5.8721e-06],
            [9.9999e-01, 1.3558e-06, 1.1293e-06, 2.8045e-06],
            [1.0000e+00, 3.5004e-07, 3.6059e-07, 8.7667e-07],
            [9.9997e-01, 5.6359e-06, 7.8194e-06, 1.4346e-05],
            [9.9995e-01, 1.1463e-05, 1.2729e-05, 2.9505e-05]
        ], device='cuda:0')
        '''

        filteredDataset = [item for item in config.dataset if item.label in selectedDataset]
        
        if self.training:
            outputs = self.backbone.forward(
                input_ids=corpus_ids,
                attention_mask=corpus_mask,
                labels=label_ids,
                output_hidden_states=(self.config.mode == 'interpret'),
                output_attentions=(self.config.mode == 'interpret')
            )
            raw_scores = outputs.logits
            # 指定额外的特殊标记 0是human，1是mechine
            additional_special_tokens = ["<extra_id_0>", "<extra_id_1>"]
            filtered_scores = raw_scores[:, 0, [item.token_id for item in filteredDataset]]
            probabilities = torch.softmax(filtered_scores, dim=-1)
        else:
            outputs = self.backbone.generate(
                input_ids=corpus_ids,
                max_length=2, # one for label token, one for eos token
                output_scores=True, 
                return_dict_in_generate=True,
                output_hidden_states=True
            )
            raw_scores = torch.stack(outputs.scores)
            filtered_scores = raw_scores[0, :, [item.token_id for item in filteredDataset]]
            probabilities = torch.softmax(filtered_scores, dim=-1)
        
        return SentinelOutput.construct(huggingface=outputs, probabilities=probabilities)

    def interpretability_study_entry(self, corpus_ids: Tensor, corpus_mask: Tensor, label_ids: Tensor, selectedDataset: Tuple[str] = ('Human', 'ChatGPT', 'cohere', 'davinci','bloomz','dolly')):
        assert self.injected_embedder is not None, "Injected gradient collector did not found"

        filteredDataset = [item for item in config.dataset if item.label in selectedDataset]
        outputs = self.backbone(
            input_ids=corpus_ids,
            attention_mask=corpus_mask,
            labels=label_ids,
            output_hidden_states=False,
            output_attentions=False
        )
        raw_scores = outputs.logits
        loss = outputs.loss
        loss.backward()

        filtered_scores = raw_scores[:, 0, [item.token_id for item in filteredDataset]]
        probabilities = torch.softmax(filtered_scores, dim=-1)
        return probabilities
