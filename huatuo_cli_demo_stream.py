import os
import platform
import torch
from transformers import AutoTokenizer
from transformers import AutoModelForCausalLM
import re
import argparse


def load_model(model_name, device, num_gpus):
    if device == "cuda":
        kwargs = {"torch_dtype": torch.float32}
        if num_gpus == "auto":
            kwargs["device_map"] = "auto"
        else:
            num_gpus = int(num_gpus)
            if num_gpus != 1:
                kwargs.update({
                    "device_map": "auto",
                    "max_memory": {i: "13GiB" for i in range(num_gpus)},
                })
    elif device == "cpu":
        kwargs = {}
    else:
        raise ValueError(f"Invalid device: {device}")

    tokenizer = AutoTokenizer.from_pretrained(model_name, padding_side="right", use_fast=True)
    model = AutoModelForCausalLM.from_pretrained(model_name, low_cpu_mem_usage=True, **kwargs)

    if device == "cuda" and num_gpus == 1:
        model.cuda()

    return model, tokenizer


@torch.inference_mode()
def chat_stream(model, tokenizer, query, history, max_new_tokens=512,
                temperature=0.2, repetition_penalty=1.2, context_len=1024, stream_interval=2):
    
    prompt = generate_prompt(query, history, tokenizer.eos_token)
    input_ids = tokenizer(prompt).input_ids
    output_ids = list(input_ids)

    device = model.device
    stop_str = tokenizer.eos_token
    stop_token_ids = [tokenizer.eos_token_id]

    l_prompt = len(tokenizer.decode(input_ids, skip_special_tokens=False))

    max_src_len = context_len - max_new_tokens - 8
    input_ids = input_ids[-max_src_len:]

    for i in range(max_new_tokens):
        if i == 0:
            out = model(torch.as_tensor([input_ids], device=device), use_cache=True)
            logits = out.logits
            past_key_values = out.past_key_values
        else:
            out = model(
                input_ids=torch.as_tensor([[token]], device=device),
                use_cache=True,
                past_key_values=past_key_values,
            )
            logits = out.logits
            past_key_values = out.past_key_values

        last_token_logits = logits[0][-1]

        if device == "mps":
            # Switch to CPU by avoiding some bugs in mps backend.
            last_token_logits = last_token_logits.float().to("cpu")

        if temperature < 1e-4:
            token = int(torch.argmax(last_token_logits))
        else:
            probs = torch.softmax(last_token_logits / temperature, dim=-1)
            token = int(torch.multinomial(probs, num_samples=1))

        output_ids.append(token)

        if token in stop_token_ids:
            stopped = True
        else:
            stopped = False

        if i % stream_interval == 0 or i == max_new_tokens - 1 or stopped:
            output = tokenizer.decode(output_ids, skip_special_tokens=False)
            if stop_str:
                pos = output.rfind(stop_str, l_prompt)
                if pos != -1:
                    output = output[l_prompt:pos]
                    stopped = True
                else:
                    output = output[l_prompt:]
                yield output
            else:
                raise NotImplementedError

        if stopped:
            break

    del past_key_values


def generate_prompt(query, history, eos):
    if not history:
        return f"""一位用户和智能医疗大模型HuatuoGPT之间的对话。对于用户的医疗问诊，HuatuoGPT给出准确的、详细的、温暖的指导建议。对于用户的指令问题，HuatuoGPT给出有益的、详细的、有礼貌的回答。<病人>：{query} <HuatuoGPT>："""
    else:
        prompt = '一位用户和智能医疗大模型HuatuoGPT之间的对话。对于用户的医疗问诊，HuatuoGPT给出准确的、详细的、温暖的指导建议。对于用户的指令问题，HuatuoGPT给出有益的、详细的、有礼貌的回答。'
        for i, (old_query, response) in enumerate(history):
            prompt += "<病人>：{} <HuatuoGPT>：{}".format(old_query, response) + eos
        prompt += "<病人>：{} <HuatuoGPT>：".format(query)
        return prompt


def main(args):
    model, tokenizer = load_model(args.model_name, args.device, args.num_gpus)

    model = model.eval()

    os_name = platform.system()
    clear_command = 'cls' if os_name == 'Windows' else 'clear'
    history = []
    print("HuatuoGPT: 你好，我是一个解答医疗健康问题的大模型，目前处于测试阶段，请以医嘱为准。请问有什么可以帮到您？输入 clear 清空对话历史，stop 终止程序")
    while True:
        query = input("\n用户：")
        if query == "stop":
            break
        if query == "clear":
            history = []
            os.system(clear_command)
            print("HuatuoGPT: 你好，我是一个解答医疗健康问题的大模型，目前处于测试阶段，请以医嘱为准。请问有什么可以帮到您？输入 clear 清空对话历史，stop 终止程序")
            continue
        
        print(f"HuatuoGPT: ", end="", flush=True)
        pre = 0
        for outputs in chat_stream(model, tokenizer, query, history, max_new_tokens=args.max_new_tokens,
                temperature=args.temperature, repetition_penalty=1.2, context_len=1024):
            outputs = outputs.strip()
            # outputs = outputs.split("")
            now = len(outputs)
            if now - 1 > pre:
                print(outputs[pre:now - 1], end="", flush=True)
                pre = now - 1
        print(outputs[pre:], flush=True)
        history = history + [(query, outputs)]



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-name", type=str, default="HuatuoGPT")
    parser.add_argument("--device", type=str, choices=["cuda", "cpu"], default="cuda")
    parser.add_argument("--num-gpus", type=str, default="1")
    # parser.add_argument("--load-8bit", action="store_true")
    parser.add_argument("--temperature", type=float, default=0.5)
    parser.add_argument("--max-new-tokens", type=int, default=512)
    args = parser.parse_args()

    main(args)

