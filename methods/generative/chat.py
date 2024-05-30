import torch
import torch.nn.functional as F
import transformers
from transformers import AutoTokenizer
from transformers import LogitsProcessor
from modeling_nort5_acd import NorT5ForConditionalGeneration
import json
import os
import re
import shutil



class Chat:
    def __init__(self):
        print(f"Starting to load the model to memory")

        self.tokenizer = AutoTokenizer.from_pretrained("chat_nort5_large")
        self.cls_index = self.tokenizer.convert_tokens_to_ids("[CLS]")
        self.sep_index = self.tokenizer.convert_tokens_to_ids("[SEP]")

        self.temperature = 0.3
        self.top_p = 0.95
        self.top_k = 64
        self.repetition_penalty = 0.8
        self.acd = True

        self.model = NorT5ForConditionalGeneration.from_pretrained("chat_nort5_large", ignore_mismatched_sizes=True)

        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"SYSTEM: Running on {self.device}", flush=True)

        self.model = self.model.to(self.device)
        self.model.eval()

        print(f"Sucessfully loaded the model to the memory")


    def __call__(self, prompt, initial_prompt=None, few_shots=None, response=None):

        class RepetitionPenaltyLogitsProcessor(LogitsProcessor):
            def __init__(self, penalty: float, model):
                last_bias = model.classifier.nonlinearity[-1].bias.data
                last_bias = F.log_softmax(last_bias, dim=-1)
                self.penalty = penalty * (last_bias - last_bias.max())

            def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor) -> torch.FloatTensor:
                penalized_score = torch.gather(scores + self.penalty.unsqueeze(0).to(input_ids.device), 1, input_ids)
                scores.scatter_(1, input_ids, penalized_score)
                scores = F.log_softmax(scores, dim=-1)
                return scores

        prefix = f"System: {initial_prompt.strip()}[SEP] " if initial_prompt is not None else ""
        if few_shots is not None:
            few_shots = "[SEP] ".join([
                f"brukeren: {item[0].strip()}[SEP] NorT5: {item[1].strip()}"
                for item in few_shots
            ]) + "[SEP] "
        else:
            few_shots = ""

        prompt = prompt.replace('\n', '<br>')
        message = f"{prefix}{few_shots}brukeren: {prompt.strip()}"

        # Tokenize the messages string
        prompt = self.tokenizer(message, add_special_tokens=False).input_ids
        prompt = prompt[-(768-2):]
        prompt = [self.cls_index] + prompt + [self.sep_index]
        prompt = torch.tensor([prompt], device=self.device)

        if response is None:

            def generate(model, **kwargs):
                with torch.inference_mode():
                    with torch.autocast(enabled=self.device != "cpu", device_type=self.device, dtype=torch.bfloat16):
                        return self.model.generate(**kwargs)

            generate_kwargs = dict(
                inputs=prompt,
                max_new_tokens=128,
                decoder_input_ids=torch.tensor([self.tokenizer("[BOS] NorT5:", add_special_tokens=False).input_ids], device=self.device),
                top_k=self.top_k,
                top_p=self.top_p,
                do_sample=True,
                temperature=self.temperature,
                num_beams=1,
                use_cache=True,
                use_acd=self.acd,
                logits_processor=[RepetitionPenaltyLogitsProcessor(self.repetition_penalty, self.model)]
            )
            output = generate(self.model, **generate_kwargs)

            return self.tokenizer.decode(output.squeeze()[1:-1])[len(" NorT5:"):].strip()

        else:

            response_tensor = torch.tensor([self.tokenizer(f"NorT5: {response.strip()}[EOS]", add_special_tokens=False).input_ids], device=self.device)
            cross_entropy = self.model(input_ids=prompt, labels=response_tensor, use_cache=False, return_dict=True, use_acd=False).loss

            summed_prob = (-cross_entropy).exp().item()
            mean_prob = (-cross_entropy / response_tensor.size(1)).exp().item()
            char_mean_prob = (-cross_entropy / len(f"NorT5: {response.strip()}[EOS]")).exp().item()

            return (summed_prob, mean_prob, char_mean_prob)


if __name__ == "__main__":
    chat = Chat()

    print(chat("Er denne setningen negativ eller positiv?\nJeg misliker fastlegen min"))
    print(chat("Er denne setningen negativ eller positiv? Jeg misliker fastlegen min", initial_prompt="Du er en hjelpsom språkmodell som heter ChatNorT5."))
    print(chat("Er denne setningen negativ eller positiv? Jeg misliker fastlegen min", few_shots=[
        ["Er denne setningen negativ eller positiv? Jeg hater fastlegen min", "negativ"],
        ["Er denne setningen negativ eller positiv? Jeg liker fastlegen min", "positiv"]
    ]))
    print(chat("Er denne setningen negativ eller positiv? Jeg misliker fastlegen min", initial_prompt="Du er en hjelpsom språkmodell som heter ChatNorT5.", few_shots=[
        ["Er denne setningen negativ eller positiv? Jeg hater fastlegen min", "negativ"],
        ["Er denne setningen negativ eller positiv? Jeg liker fastlegen min", "positiv"]
    ]))
    print(chat("Er denne setningen negativ eller positiv? Jeg misliker fastlegen min", response="Setningen er negativ."))
    print(chat("Er denne setningen negativ eller positiv? Jeg misliker fastlegen min", response="Setningen er positiv."))

