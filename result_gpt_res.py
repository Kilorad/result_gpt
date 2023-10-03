#сделали поддержку большого датасета
#
#здесь мы выгружаем GPT-2 и пытаемся дообучать на кроссдоменном датасете
import torch
from transformers import GPT2Tokenizer, PreTrainedTokenizer
import codecs
import numpy as np
import os, pickle, random
import psutil
import warnings
from typing import Dict, List, Optional
from torch.utils.data import DataLoader
from collections import deque

from trl import AutoModelForCausalLMWithValueHead
from transformers import AutoTokenizer, AutoModelForCausalLM
from transformers import Trainer, TrainingArguments
from transformers.trainer import (
    seed_worker
)
from torch.utils.data import DataLoader, IterableDataset, Dataset, RandomSampler, SequentialSampler
from typing import Dict, List, Optional

#!pip install transformers
#!pip install trl
#!pip install accelerate -U
#!pip install transformers[torch]

#вариант трейнера без рандомного семплирования
class TrainerUnsampled(Trainer):
    def get_train_dataloader(self) -> DataLoader:
            """
            Returns the training [`~torch.utils.data.DataLoader`].

            Will use no sampler if `train_dataset` does not implement `__len__`, a random sampler (adapted to distributed
            training if necessary) otherwise.

            Subclass and override this method if you want to inject some custom behavior.
            """
            if self.train_dataset is None:
                raise ValueError("Trainer: training requires a train_dataset.")

            train_dataset = self.train_dataset
            data_collator = self.data_collator

            dataloader_params = {
                "batch_size": self._train_batch_size,
                "collate_fn": data_collator,
                "num_workers": self.args.dataloader_num_workers,
                "pin_memory": self.args.dataloader_pin_memory,
            }

            if not isinstance(train_dataset, torch.utils.data.IterableDataset):
                #dataloader_params["sampler"] = self._get_train_sampler()
                dataloader_params["drop_last"] = self.args.dataloader_drop_last
                dataloader_params["worker_init_fn"] = seed_worker

            return self.accelerator.prepare(DataLoader(train_dataset, **dataloader_params))

def getmem():
    process = psutil.Process(os.getpid())
    mem_info = process.memory_info()
    byte = (mem_info.rss//8)
    mb = (mem_info.rss//8)//(1024*1024)
    gb = mb/1024
    return gb/1024


class AutoModelForCausalLMWithValueHeadCL(AutoModelForCausalLMWithValueHead):
    '''Модель для генерации следующего токена последовательности, с value_head, адаптированная под offline actor-critic rl'''        
    def __init__(self, pretrained_model, **kwargs):
        r"""
        Initializes the model.

        Args:
            pretrained_model (`transformers.PreTrainedModel`):
                The model to wrap. It should be a causal language model such as GPT2.
                or any model mapped inside the `AutoModelForCausalLM` class.
            kwargs (`dict`, `optional`):
                Additional keyword arguments, that are passed to the `ValueHead` class.
        """
        super().__init__(pretrained_model)

        if not any(hasattr(self.pretrained_model, attribute) for attribute in self.lm_head_namings):
            raise ValueError("The model does not have a language model head, please use a model that has one.")

        #self.v_head = ValueHead(self.pretrained_model.config, **v_head_kwargs)
        
        
        if hasattr(self.pretrained_model.config, "word_embed_proj_dim"):
            hidden_size = self.pretrained_model.config.word_embed_proj_dim
        else:
            hidden_size = self.pretrained_model.config.hidden_size
            
        
        if not hasattr(self.pretrained_model, "v_head"):
            #self.pretrained_model.v_head = torch.nn.Linear(hidden_size, 1)
            self.pretrained_model.v_head = torch.nn.Sequential(
                torch.nn.Linear(hidden_size, 512),
                torch.nn.ReLU(),
                torch.nn.Linear(512, 256),
                torch.nn.ReLU(),
                torch.nn.Linear(256, 1),
            )
        self.v_head = self.pretrained_model.v_head
        self.advance_coef = 2.#при единице там совсем мелкие градиенты
        self.advance_coef_negative = 0.25 #негативные advance домножить на него, иначе система выдаёт сигналы в неверном формате
        
        self.advances_dq = deque(maxlen=1000)
        self.pos_advance_success = deque(maxlen=1000)
        self.neg_advance_success = deque(maxlen=1000)
        self.pos_advance_success2 = deque(maxlen=1000)
        self.neg_advance_success2 = deque(maxlen=1000)
        self.use_out_in_past = True

    def forward(
        self,
        input_ids=None,
        past_key_values=None,
        attention_mask=None,
        **kwargs,
    ):
        r"""
        Applies a forward pass to the wrapped model and returns the logits of the value head.

        Args:
            input_ids (`torch.LongTensor` of shape `(batch_size, sequence_length)`):
                Indices of input sequence tokens in the vocabulary.
            past_key_values (`tuple(tuple(torch.FloatTensor))`, `optional`):
                Contains pre-computed hidden-states (key and values in the attention blocks) as computed by the model
                (see `past_key_values` input) to speed up sequential decoding.
            attention_mask (`torch.FloatTensor` of shape `(batch_size, sequence_length)`, `optional`):
                Mask to avoid performing attention on padding token indices. Mask values selected in ``[0, 1]``:
                - 1 for tokens that are **not masked**,
                - 0 for tokens that are **masked**.
            kwargs (`dict`, `optional`):
                Additional keyword arguments, that are passed to the wrapped model.
        """
        #print('mem before', getmem())
        kwargs["output_hidden_states"] = True  # this had already been set in the LORA / PEFT examples
        kwargs["past_key_values"] = past_key_values

        if self.is_peft_model and self.pretrained_model.active_peft_config.peft_type == "PREFIX_TUNING":
            kwargs.pop("past_key_values")
            
        device = input_ids.device
        rewards = torch.zeros((input_ids.shape[0]), device=device)
        reward_used = torch.zeros((input_ids.shape[0]), dtype=torch.bool, device=device)
        
        for r_token in self.r_mapping.keys():
            r_token_gpu = torch.tensor(r_token).to(device)
            r_value_gpu = torch.tensor(self.r_mapping[r_token]).to(device)
            idx = input_ids == r_token_gpu
            idx_any = torch.any(idx, axis=1)
            reward_used = reward_used | idx_any
            rewards += torch.sum(idx, axis=1) * r_value_gpu
        del r_value_gpu
        del r_token_gpu
        del idx
        del idx_any
        #reward_used - это важно, contrastive считай для них
        out_token_gpu = torch.tensor(self.out_token).to(device)
        idx = input_ids == out_token_gpu
        del out_token_gpu
        
        if self.use_out_in_past:
            #out_in_past - маска. До первого <out> она 0, после 1
            #нужна, чтобы считать loss только по токенам после out
            out_in_past = torch.cumsum(idx, dim=1).to(torch.bool)
            out_in_past_any = torch.any(out_in_past, dim=1)
            #если out токена нет, считаем loss по всем токенам
            out_in_past[~out_in_past_any] = 1
            del out_in_past_any
        else:
            out_in_past = torch.ones(idx.shape).to(torch.bool).to(idx.device)

        base_model_output = self.pretrained_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            **kwargs,
        )
        
        last_hidden_state = base_model_output.hidden_states[-1]
        #накроем маской, чтобы реворд считался по тому, что ДО действий и ДО <OUT>
        
        lm_logits = base_model_output.logits
        if last_hidden_state.device != self.v_head[0].weight.device:
            last_hidden_state = last_hidden_state.to(self.v_head[0].weight.device)
        value = self.v_head(last_hidden_state).squeeze(-1)
        value4train = torch.zeros((input_ids.shape[0]), device=device)
        value4advance = value.detach()
        advance_reward = torch.zeros((input_ids.shape[0]), device=device)
        #вынуть те номера, где out
        for i in range(value4advance.shape[0]):
            out_idx = torch.nonzero(out_in_past[i])[0]
            if (out_idx > 0) or (not self.use_out_in_past):
                value4train[i] = value[i, out_idx][0]
                advance_reward[i] = rewards[i] - value4advance[i, out_idx][0]
            else:
                value4train[i] = 0
                advance_reward[i] = 0
        reward_loss = torch.nn.MSELoss()(value4train, rewards)
        # Shift so that tokens < n predict n
        out_in_past_cont_m1 = out_in_past[..., :-1].contiguous()
        
        shift_logits = lm_logits[..., :-1, :].contiguous()
        shift_labels = input_ids[..., 1:].contiguous()
        #* out_in_past - это маска, то есть всё до этого занулить
        ##
        # Flatten the tokens
        loss_fct = torch.nn.CrossEntropyLoss(reduce=False)
        loss = 0
        #сделать, чтобы loss был contrastive
        #loss_logits = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
        
        self.advances_dq.extend(list(torch.ravel(advance_reward[reward_used]).detach()))
        advance_reward_normed = self.advance_coef * advance_reward / (torch.std(torch.stack(tuple(self.advances_dq))) + 1e-3) * (reward_used).to(torch.float32)
        #advance_reward_normed = 1 - это что-то типа 75% квантиля по эффективности, где 50% - это дефолт, а ниже 50% - это проигрыш
        #убери реворды для тех мест, где их нет
        #batch_size
        #batch_size x seq_size
        advance_reward_normed[~reward_used] = 1
        advance_reward_normed[advance_reward_normed<0] = advance_reward_normed[advance_reward_normed<0] * self.advance_coef_negative
        sing = torch.sign(advance_reward_normed)
        ampl = torch.abs(advance_reward_normed)
        advance_reward_normed2d = torch.stack([sing * ampl**4] * out_in_past_cont_m1.shape[1]).T
        advance_reward_normed2d = torch.ravel(advance_reward_normed2d)
        advance_reward_normed = torch.ravel(advance_reward_normed)
        
        loss_logits = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1)) * torch.ravel(out_in_past_cont_m1)
        
        loss += torch.mean(torch.clip(loss_logits * advance_reward_normed2d, -1.5))
        #loss_logits = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
        #loss += torch.mean(loss_logits)
        loss += 0.3 * reward_loss
        
        
        #записать метрики
        idx = advance_reward_normed > 0.6
        idx_nums = torch.ravel(torch.nonzero(idx))
        for i in idx_nums:
            tokens_pred = torch.max(shift_logits[i], axis=-1).indices
            tokens_fact = shift_labels[i]
            chars_pred = self.tokenizer.decode(tokens_pred)
            chars_fact = self.tokenizer.decode(tokens_fact)
            if ('<OUT>' in chars_pred) and ('<OUT>' in chars_fact) and ('plan' in chars_fact):
                chars_pred = chars_pred.split('<OUT>')[1][:15]
                chars_fact = chars_fact.split('<OUT>')[1][:15]
                self.pos_advance_success.append(chars_pred == chars_fact)
        idx = advance_reward_normed < -0.6
        idx_nums = torch.ravel(torch.nonzero(idx))
        for i in idx_nums:
            tokens_pred = torch.max(shift_logits[i], axis=-1).indices
            tokens_fact = shift_labels[i]
            chars_pred = self.tokenizer.decode(tokens_pred)
            chars_fact = self.tokenizer.decode(tokens_fact)
            if ('<OUT>' in chars_pred) and ('<OUT>' in chars_fact) and ('plan' in chars_fact):
                chars_pred = chars_pred.split('<OUT>')[1][:15]
                chars_fact = chars_fact.split('<OUT>')[1][:15]
                self.neg_advance_success.append(chars_pred == chars_fact)
                
                
        idx = advance_reward_normed < -1.8
        idx_nums = torch.ravel(torch.nonzero(idx))
        for i in idx_nums:
            tokens_pred = torch.max(shift_logits[i], axis=-1).indices
            tokens_fact = shift_labels[i]
            chars_pred = self.tokenizer.decode(tokens_pred)
            chars_fact = self.tokenizer.decode(tokens_fact)
            if ('<OUT>' in chars_pred) and ('<OUT>' in chars_fact) and ('plan' in chars_fact):
                chars_pred = chars_pred.split('<OUT>')[1][:15]
                chars_fact = chars_fact.split('<OUT>')[1][:15]
                self.neg_advance_success2.append(chars_pred == chars_fact)        
        idx = advance_reward_normed > 1.8
        idx_nums = torch.ravel(torch.nonzero(idx))
        for i in idx_nums:
            tokens_pred = torch.max(shift_logits[i], axis=-1).indices
            tokens_fact = shift_labels[i]
            chars_pred = self.tokenizer.decode(tokens_pred)
            chars_fact = self.tokenizer.decode(tokens_fact)
            if ('<OUT>' in chars_pred) and ('<OUT>' in chars_fact) and ('plan' in chars_fact):
                chars_pred = chars_pred.split('<OUT>')[1][:15]
                chars_fact = chars_fact.split('<OUT>')[1][:15]
                self.pos_advance_success2.append(chars_pred == chars_fact)
                
                
                
        if (len(self.neg_advance_success)>=400) and (len(self.pos_advance_success)>=400):
            print('advance +-0.6 neg_success share', np.mean(list(self.neg_advance_success)), 'pos_success share', np.mean(list(self.pos_advance_success)), '+-1.8 neg_success share', np.mean(list(self.neg_advance_success2)), 'pos_success share', np.mean(list(self.pos_advance_success2)) , 'std', torch.std(torch.stack(tuple(self.advances_dq))))
            self.neg_advance_success = deque(maxlen=1000)
            self.pos_advance_success = deque(maxlen=1000)
            self.neg_advance_success2 = deque(maxlen=1000)
            self.pos_advance_success2 = deque(maxlen=1000)
        
        verbose = False
        if verbose:
            print('v', value4train[reward_used], '\nr', rewards[reward_used])
            print('loss_logits', loss_logits)
            print('reward_loss', reward_loss)
            print('advance_reward_normed', advance_reward_normed)
            print('shift_labels', shift_labels.shape, 'shift_logits', shift_logits.shape)
            for i in range(int(torch.sum(reward_used).detach().cpu().numpy())):
                tokens_pred = torch.max(shift_logits[i], axis=-1).indices
                tokens_fact = shift_labels[i]
                chars_pred = self.tokenizer.decode(tokens_pred)
                chars_fact = self.tokenizer.decode(tokens_fact)
                print('chars_pred:', chars_pred)
                print('chars_fact:', chars_fact)
                adv = advance_reward_normed[i].detach().cpu().numpy()
                print('v', value4train[i].detach().cpu().numpy(), 'r', rewards[i].detach().cpu().numpy(), 'a', adv)
                
                print('logits_loss * advance_reward_normed2d max ',torch.max(torch.clip(loss_logits * advance_reward_normed2d, -1.5) ),'min',torch.min(torch.clip(loss_logits * advance_reward_normed2d, -1.5)), 'mean', torch.mean(torch.clip(loss_logits * advance_reward_normed2d, -1.5) ))
                
                if adv>0:
                    print('++++++++')
                else:
                    print('--------')
                if len(self.advances_dq)>200:
                    dq_trch = torch.ravel(torch.stack(list(self.advances_dq)))
                    print('self.advances_dq', len(self.advances_dq), 'positives count', torch.sum(dq_trch>0), 'negatives count', torch.sum(dq_trch<0), 'big positives count', torch.sum(dq_trch>0.2), 'big negatives count', torch.sum(dq_trch<-0.2), 'big positives agv', torch.mean(dq_trch[dq_trch>0.2]), 'big negatives avg', torch.mean(dq_trch[dq_trch<-0.2]), 'std', torch.std(torch.stack(tuple(self.advances_dq))))
                

        # force upcast in fp32 if logits are in half-precision
        if lm_logits.dtype != torch.float32:
            lm_logits = lm_logits.float()
        
        #return (lm_logits, loss, value)
        #print('loss', loss, 'base_model_output.loss', base_model_output.loss)
        if not torch.isnan(loss):
            base_model_output.loss += loss
        #ones = torch.ones([122, 122, 1024], device=self.v_head.weight.device)
        #print('ones', self.v_head(ones)[0,0,0:3])
        return base_model_output
    

class LineByLineTextDataset(Dataset):
    """
    Превращает текстовый файл в сэмплы, разделённые \n. Если в строке есть реворд-токен - пытаемся его включить, даже если начало строки потеряется
    """

    def __init__(self, tokenizer: PreTrainedTokenizer, file_path: str, block_size: int):
        if os.path.isfile(file_path) is False:
            raise ValueError(f"Input file path {file_path} not found")
        # Here, we do not cache the features, operating under the assumption
        # that we will soon use fast multithreaded tokenizers from the
        # `tokenizers` repo everywhere =)
        block_size_macro = 1200000
        path_cash = 'ints_cash.bin'
        with open(file_path, encoding="utf-8") as f:
            lines = f.readlines()
        with open(path_cash,'w') as file:
            file.write('')
        self.examples = []
        #for batch_start in range(0, len(lines), block_size_macro):
        if len(lines) < block_size_macro:
            block_size_macro = len(lines) - 1
        for batch_start in np.arange(len(lines) - block_size_macro, 0, -block_size_macro):
            batch_start = int(batch_start)
            if batch_start<0:
                batch_start = 0
            batch_encoding = tokenizer(lines[batch_start:batch_start + block_size_macro], add_special_tokens=True, truncation=True, max_length=block_size * 4)
                    
            del lines[batch_start:batch_start + block_size_macro]
            batch_encoding = batch_encoding["input_ids"]
            
            batch_encoding = [batch_encoding_local[-block_size:] for batch_encoding_local in batch_encoding]
                    
            s = str(batch_encoding)
            s = s.replace("],", "]\n").replace("]]", "]").replace("[[", "[")
            s = s.replace("],", "]\n").replace("]]", "]").replace("[[", "[")
            with open(path_cash,'a') as file:
                file.write(s)
            del batch_encoding
            del s
            
        del tokenizer
        del lines
        with open(path_cash, 'r', encoding="utf-8") as file:
            self.examples = file.readlines()
        
        self.examples_new = []
        for e in self.examples:
            e = e.replace(' ', '')
            if not ('][' in e):
                try:
                    self.examples_new.append({"input_ids": torch.tensor(eval(e), dtype=torch.long)})
                except Exception:
                    pass
        self.examples = self.examples_new
        random.shuffle(self.examples_new)

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, i) -> Dict[str, torch.tensor]:
        return self.examples[i]

class LineByLineTextDatasetCached(Dataset):
    """
    This will be superseded by a framework-agnostic approach soon.
    """

    def __init__(self, tokenizer: PreTrainedTokenizer, file_path: str, block_size: int):
        if os.path.isfile(file_path) is False:
            raise ValueError(f"Input file path {file_path} not found")
        # Here, we do not cache the features, operating under the assumption
        # that we will soon use fast multithreaded tokenizers from the
        # `tokenizers` repo everywhere =)
        T_OUT = tokenizer.encode('<OUT>')[0]
        block_size_macro = 1600000
        path_cash = 'ints_cash.bin'
        with open(file_path, encoding="utf-8") as f:
            lines = f.readlines()
        with open(path_cash,'w') as file:
            file.write('')
        self.examples = []
        #for batch_start in range(0, len(lines), block_size_macro):
        if len(lines) < block_size_macro:
            block_size_macro = len(lines) - 1
        for batch_start in np.arange(len(lines) - block_size_macro, 0, -block_size_macro):
            batch_start = int(batch_start)
            if batch_start<0:
                batch_start = 0
            batch_encoding = tokenizer(lines[batch_start:batch_start + block_size_macro], add_special_tokens=True, truncation=True, max_length=block_size * 4)
            
            print(lines[batch_start:batch_start + 100])
            print(batch_encoding[batch_start:batch_start + 100])
            print('T_PAD', T_PAD)
            paddings = batch_encoding == T_PAD
            for i in range(len(batch_encoding)):
                if torch.any(paddings[i]):
                    print(batch_encoding[i])
                    whr = torch.ravel(torch.nonzero(paddings[i]))[0]
                    batch_encoding[i] = batch_encoding[i][whr + 1:] + batch_encoding[i][:whr + 1]
                    print(batch_encoding[i])
                    1/0
            
            del lines[batch_start:batch_start + block_size_macro]
            batch_encoding = batch_encoding["input_ids"]
            s = str(batch_encoding)
            s = s.replace("],", "]\n").replace("]]", "]").replace("[[", "[")
            with open(path_cash,'a') as file:
                file.write(s)
            del batch_encoding
            del s
            
        del tokenizer
        del lines
        with open(path_cash, 'r', encoding="utf-8") as file:
            self.examples = file.readlines()
        
        self.examples_new = []
        for e in self.examples:
            e = e.replace(' ', '')
            if not ('][' in e):
                try:
                    self.examples_new.append({"input_ids": torch.tensor(eval(e), dtype=torch.long)})
                except Exception:
                    pass
        del self.examples        
        random.shuffle(self.examples_new)
        self.examples_count = len(self.examples_new)
        #закешировать
        self.path_dir = './cache_tokenized_data'
        os.system(f'mkdir {self.path_dir}')
        self.cache_batch_size = 5000
        self.current_cache_batch_num = -1
        for cache_batch_start in range(0, len(self.examples_new), self.cache_batch_size):
            ex_path = self.path_dir + f'/{cache_batch_start}.pkl'
            if os.path.isfile(ex_path):
                print(ex_path, 'exists: no writing')
            else:
                with open(ex_path, 'wb') as f:
                    slc = self.examples_new[cache_batch_start:cache_batch_start + self.cache_batch_size]
                    pickle.dump(slc, f,protocol=pickle.HIGHEST_PROTOCOL)
        

    def __len__(self):
        return self.examples_count

    def __getitem__(self, i) -> Dict[str, torch.tensor]:
        needed_cache_batch = i//self.cache_batch_size
        idx_in_batch = i - needed_cache_batch * self.cache_batch_size
        if needed_cache_batch != self.current_cache_batch_num:
            with open(f'{self.path_dir}/{needed_cache_batch}.pkl', 'rb') as f:
                self.cache_batch = pickle.load(f)
        return self.cache_batch[idx_in_batch]
