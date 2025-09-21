import transformers
import datasets
import sys
import os
from trl import (ModelConfig,
                 SFTTrainer,
                 TrlParser,
                 get_peft_config,
                 setup_chat_format)

from configs import ScriptArguments, SFTConfig
from transformers import set_seed
import swanlab
import logging
from transformers.trainer_utils import get_last_checkpoint
from utils.swanlab import init_swanlab_training
from utils.model_utils import get_tokenizer, get_model
import pandas as pd
from datasets import Dataset
from transformers import DataCollatorForLanguageModeling
from utils.callbacks import get_callbacks

logger = logging.getLogger(__name__)
swanlab.login(api_key="5Q7lBtLI6qBF8OoEu66Lk", save=True)
# micro definition
PROMPT = "你是一个医学专家，你需要根据用户的问题，给出带有思考的回答。"




def process_func(example, tokenizer, max_length):
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
    if len(input_ids) > max_length:  # 做一个截断
        input_ids = input_ids[:max_length]
        attention_mask = attention_mask[:max_length]
        labels = labels[:max_length]
    return {"input_ids": input_ids, "attention_mask": attention_mask, "labels": labels}


def main(script_args, training_args, model_args):
    set_seed(training_args.seed)

    ###############
    # Setup logging
    ###############
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
    )
    log_level = training_args.get_process_log_level()
    logger.setLevel(log_level)
    datasets.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.enable_default_handler()
    transformers.utils.logging.enable_explicit_format()

    logger.info(f"Model parameters {model_args}")
    logger.info(f"Script parameters {script_args}")
    logger.info(f"Training parameters {training_args}")


    # Check for last checkpoint
    last_checkpoint = None
    if os.path.isdir(training_args.output_dir):
        last_checkpoint = get_last_checkpoint(training_args.output_dir)
    if last_checkpoint is not None and training_args.resume_from_checkpoint is None:
        logger.info(f"Checkpoint detected, resuming training at {last_checkpoint=}.")


    if "swanlab" in training_args.report_to:
        init_swanlab_training(training_args)



    #model, tokenizer
    tokenizer = get_tokenizer(model_args, training_args)
    model = get_model(model_args, training_args)


    #dataset, from local file
    train_dataset_path = "/home/luke/distributed_machine_learning/luke_sft/jupyter/train.jsonl"
    test_dataset_path = "/home/luke/distributed_machine_learning/luke_sft/jupyter/val.jsonl"
    train_jsonl_new_path = "/home/luke/distributed_machine_learning/luke_sft/jupyter/train_format.jsonl"
    test_jsonl_new_path = "/home/luke/distributed_machine_learning/luke_sft/jupyter/val_format.jsonl"

    # 得到训练集
    train_df = pd.read_json(train_jsonl_new_path, lines=True)
    train_ds = Dataset.from_pandas(train_df)
    map_kwargs = {}
    map_kwargs["num_proc"] = 52  # here is the parallel process number
    map_kwargs["desc"] = f"Applying chat template to train dataset"

    train_dataset = train_ds.map(process_func, fn_kwargs={"tokenizer": tokenizer, "max_length":training_args.max_length}, remove_columns=train_ds.column_names, **map_kwargs)

    # 得到验证集
    eval_df = pd.read_json(test_jsonl_new_path, lines=True)
    eval_ds = Dataset.from_pandas(eval_df)
    map_kwargs = {}
    map_kwargs["num_proc"] = 52  # here is the parallel process number
    map_kwargs["desc"] = f"Applying chat template to eval dataset"
    eval_dataset = eval_ds.map(process_func, fn_kwargs={"tokenizer": tokenizer,"max_length":training_args.max_length},  remove_columns=eval_ds.column_names, **map_kwargs)


    if tokenizer.chat_template is None:
        logger.info("No chat template provided, defaulting to ChatML.")
        model, tokenizer = setup_chat_format(model, tokenizer, format="chatml")

    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False  # we're doing causal LM, not masked LM
    )

    ############################
    # Initialize the SFT Trainer
    ############################
    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        processing_class=tokenizer,
        peft_config=get_peft_config(model_args),
        callbacks=get_callbacks(training_args, model_args),
        data_collator=data_collator
    )
    ###############
    # Training loop
    ###############
    logger.info("*** Train ***")
    checkpoint = None
    if training_args.resume_from_checkpoint is not None:
        checkpoint = training_args.resume_from_checkpoint
    elif last_checkpoint is not None:
        checkpoint = last_checkpoint
    train_result = trainer.train(resume_from_checkpoint=checkpoint)

    metrics = train_result.metrics
    metrics["train_samples"] = len(train_dataset)
    trainer.log_metrics("train", metrics)
    trainer.save_metrics("train", metrics)
    trainer.save_state()

    ##################################
    # Save model and create model card
    ##################################
    logger.info("*** Save model ***")
    # Align the model's generation config with the tokenizer's eos token
    # to avoid unbounded generation in the transformers `pipeline()` function
    trainer.model.generation_config.eos_token_id = tokenizer.eos_token_id
    trainer.save_model(training_args.output_dir)
    logger.info(f"Model saved to {training_args.output_dir}")

    # Save everything else on main process
    kwargs = {
        "dataset_name": script_args.dataset_name,
        "tags": ["luke-sft"],
    }
    if trainer.accelerator.is_main_process:
        trainer.create_model_card(**kwargs)
        # Restore k,v cache for fast inference
        trainer.model.config.use_cache = True
        trainer.model.config.save_pretrained(training_args.output_dir)

    ##########
    # Evaluate
    ##########
    if training_args.do_eval:
        logger.info("*** Evaluate ***")
        metrics = trainer.evaluate()
        metrics["eval_samples"] = len(eval_dataset)
        trainer.log_metrics("eval", metrics)
        trainer.save_metrics("eval", metrics)

    #############
    # push to hub
    #############
    if training_args.push_to_hub:
        logger.info("Pushing to hub...")
        trainer.push_to_hub(**kwargs)



#
if __name__ == "__main__":
    parser = TrlParser((ScriptArguments, SFTConfig, ModelConfig))
    script_args, training_args, model_args = parser.parse_args_and_config()

    main(script_args, training_args, model_args)
