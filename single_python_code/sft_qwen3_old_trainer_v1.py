#!/usr/bin/env python
# coding: utf-8

# # Qwen3微调实战：医疗R1推理风格聊天
# 
# [![](https://raw.githubusercontent.com/SwanHubX/assets/main/badge1.svg)](https://swanlab.cn/@ZeyiLin/qwen3-sft-medical/overview)
# 
# - **Github**: [Qwen3-Medical-SFT](https://github.com/Zeyi-Lin/Qwen3-Medical-SFT)
# - **基础模型**：[Qwen3-1.7B](https://modelscope.cn/models/Qwen/Qwen3-1.7B/summary)
# - **微调后模型**：[Qwen3-1.7b-Medical-R1-sft](https://modelscope.cn/models/testUser/Qwen3-1.7b-Medical-R1-sft/summary)
# - **数据集**：[delicate_medical_r1_data](https://modelscope.cn/datasets/krisfu/delicate_medical_r1_data)
# - **SwanLab**：[qwen3-sft-medical](https://swanlab.cn/@ZeyiLin/qwen3-sft-medical/runs/agps0dkifth5l1xytcdyk/chart)
# - **微调方式**：全参数微调、LoRA微调
# - **推理风格**：R1推理风格
# - **算力要求**：
#   - **全参数微调**：32GB显存
#   - **LoRA微调**：28GB显存
# - **图文教程**：[Qwen3大模型微调入门实战（完整代码）](https://zhuanlan.zhihu.com/p/1903848838214705484)

# ## 1. 安装环境

# ## 3. 登录SwanLab
# 1. 前往[swanlab](https://swanlab.cn/space/~/settings)复制你的API Key，粘贴到下面的代码中
# 2. 如果你不希望将登录信息保存到该计算机中，可将`save=True`去掉（每次运行训练需要重新执行下面的代码块）

# In[1]:


import swanlab

swanlab.login(api_key="5Q7lBtLI6qBF8OoEu66Lk", save=True)


# ## 4. 开启全参数微调

# In[2]:


import json
import pandas as pd
import torch
from datasets import Dataset
from modelscope import snapshot_download, AutoTokenizer
from transformers import AutoModelForCausalLM, TrainingArguments, Trainer, DataCollatorForSeq2Seq
import os
import swanlab


# In[3]:


os.environ["SWANLAB_PROJECT"]="qwen3-sft-medical-old-v1-0920"
PROMPT = "你是一个医学专家，你需要根据用户的问题，给出带有思考的回答。"
MAX_LENGTH = 2048

# swanlab.config.update({
#     "model": "Qwen/Qwen3-1.7B",
#     "prompt": PROMPT,
#     "data_max_length": MAX_LENGTH,
#     })


# In[4]:


def dataset_jsonl_transfer(origin_path, new_path):
    """
    将原始数据集转换为大模型微调所需数据格式的新数据集
    """
    messages = []

    # 读取旧的JSONL文件
    with open(origin_path, "r") as file:
        for line in file:
            # 解析每一行的json数据
            data = json.loads(line)
            input = data["question"]
            think = data["think"]
            answer = data["answer"]
            output = f"<think>{think}</think> \n {answer}"
            message = {
                "instruction": PROMPT,
                "input": f"{input}",
                "output": output,
            }
            messages.append(message)

    # 保存重构后的JSONL文件
    with open(new_path, "w", encoding="utf-8") as file:
        for message in messages:
            file.write(json.dumps(message, ensure_ascii=False) + "\n")



# In[5]:


def process_func(example):
    """
    将数据集进行预处理
    """ 
    input_ids, attention_mask, labels = [], [], []
    instruction = tokenizer(
        f"<|im_start|>system\n{PROMPT}<|im_end|>\n<|im_start|>user\n{example['input']}<|im_end|>\n<|im_start|>assistant\n",
        add_special_tokens=False,
    )
    response = tokenizer(f"{example['output']}", add_special_tokens=False)
    input_ids = instruction["input_ids"] + response["input_ids"] + [tokenizer.pad_token_id]
    attention_mask = (
        instruction["attention_mask"] + response["attention_mask"] + [1]
    )
    labels = [-100] * len(instruction["input_ids"]) + response["input_ids"] + [tokenizer.pad_token_id]
    if len(input_ids) > MAX_LENGTH:  # 做一个截断
        input_ids = input_ids[:MAX_LENGTH]
        attention_mask = attention_mask[:MAX_LENGTH]
        labels = labels[:MAX_LENGTH]
    return {"input_ids": input_ids, "attention_mask": attention_mask, "labels": labels}   


# In[6]:


def predict(messages, model, tokenizer):
    device = "cuda"
    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )
    model_inputs = tokenizer([text], return_tensors="pt").to(device)

    generated_ids = model.generate(
        model_inputs.input_ids,
        max_new_tokens=MAX_LENGTH,
    )
    generated_ids = [
        output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
    ]

    response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]

    return response


# In[7]:


# # 在modelscope上下载Qwen模型到本地目录下
# model_dir = snapshot_download("Qwen/Qwen3-1.7B", cache_dir="./", revision="master")


# In[8]:


# Transformers加载模型权重
tokenizer = AutoTokenizer.from_pretrained("/ssd3/Qwen3-1.7B", use_fast=False, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained("/ssd3/Qwen3-1.7B", device_map="auto", torch_dtype=torch.bfloat16)
model.enable_input_require_grads()  # 开启梯度检查点时，要执行该方法


# In[10]:


# 加载、处理数据集和测试集
train_dataset_path = "../jupyter/train.jsonl"
test_dataset_path = "../jupyter/val.jsonl"

train_jsonl_new_path = "../jupyter/train_format.jsonl"
test_jsonl_new_path = "../jupyter/val_format.jsonl"

if not os.path.exists(train_jsonl_new_path):
    dataset_jsonl_transfer(train_dataset_path, train_jsonl_new_path)
if not os.path.exists(test_jsonl_new_path):
    dataset_jsonl_transfer(test_dataset_path, test_jsonl_new_path)


# In[11]:


# 得到训练集
train_df = pd.read_json(train_jsonl_new_path, lines=True)
train_ds = Dataset.from_pandas(train_df)

messages = [
    {"role": "user", "content": train_ds[0]['input']},
    {"role": "assistant", "content": train_ds[0]['output']},    
]
# 此处的token, 没有用chat_template模板，而是直接手动拼接，　模板函数的功能，也仅仅是把各个字段封装成 start, end的字段
# 

# In[16]:


train_dataset = train_ds.map(process_func, remove_columns=train_ds.column_names)


#

# In[18]:


# 得到验证集
eval_df = pd.read_json(test_jsonl_new_path, lines=True)
eval_ds = Dataset.from_pandas(eval_df)
eval_dataset = eval_ds.map(process_func, remove_columns=eval_ds.column_names)


# In[20]:


args = TrainingArguments(
    output_dir="./output/Qwen3-1.7B",
    per_device_train_batch_size=1,
    per_device_eval_batch_size=1,
    gradient_accumulation_steps=4,
    eval_strategy="steps",
    eval_steps=10,
    logging_steps=10,
    num_train_epochs=1,
#     save_steps=400,
    learning_rate=1e-4,
    save_on_each_node=True,
    gradient_checkpointing=True,
    report_to="swanlab",
    run_name="qwen3-1.7B",
)

trainer = Trainer(
    model=model,
    args=args,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    data_collator=DataCollatorForSeq2Seq(tokenizer=tokenizer, padding=True),
)



# args=TrainingArguments(
#     per_device_train_batch_size=1,
#     per_device_eval_batch_size=1,
#     gradient_accumulation_steps=4,
#     # Use num_train_epochs = 1, warmup_ratio for full training runs!
#     num_train_epochs=2,
#     warmup_steps=5,
#     max_steps=60,
#     learning_rate=1e-4,
#     gradient_checkpointing=True,
#     logging_steps=10,

#     seed=3407,
#     output_dir="outputs",
#     report_to="swanlab",
#     run_name="qwen3-1.7B"


#     #    optim="adamw_8bit",
#     #    weight_decay=0.01,
#     #    lr_scheduler_type="linear",
#     #     fp16=not is_bfloat16_supported(),
#     #     bf16=is_bfloat16_supported(),

# )


# trainer = SFTTrainer(
#     model=model,
#     args=args,

#     train_dataset=train_dataset,
#     eval_dataset=eval_dataset,
#     data_collator=DataCollatorForSeq2Seq(tokenizer=tokenizer, padding=True),
# #    dataset_text_field="text",
# #    max_seq_length=max_seq_length,
# #    tokenizer=tokenizer,    

# )





trainer.train()

# 用测试集的前3条，主观看模型


# In[21]:


# In[22]:


test_df = pd.read_json(test_jsonl_new_path, lines=True)[:3]

test_text_list = []

for index, row in test_df.iterrows():
    instruction = row['instruction']
    input_value = row['input']

    messages = [
        {"role": "system", "content": f"{instruction}"},
        {"role": "user", "content": f"{input_value}"}
    ]

    response = predict(messages, model, tokenizer)

    response_text = f"""
    Question: {input_value}

    LLM:{response}
    """

    test_text_list.append(swanlab.Text(response_text))
    print(response_text)

swanlab.log({"Prediction": test_text_list})

swanlab.finish()


# In[ ]:




