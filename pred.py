import os, json
import argparse
import time
from tqdm import tqdm
from datasets import load_dataset
import re
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline, LlamaConfig
from models.llama_kivi import LlamaForCausalLM_KIVI
import torch.multiprocessing as mp
import torch

template_rag = open('prompts/0shot_rag.txt', encoding='utf-8').read()
template_no_context = open('prompts/0shot_no_context.txt', encoding='utf-8').read()
template_0shot = open('prompts/0shot.txt', encoding='utf-8').read()
template_0shot_cot = open('prompts/0shot_cot.txt', encoding='utf-8').read()
template_0shot_cot_ans = open('prompts/0shot_cot_ans.txt', encoding='utf-8').read()

def query_llm(prompt, tokenizer, pipe=None, temperature=0.5, max_new_tokens=128, stop=None):
    # truncate
    max_len = args.maxlen
    if args.model_path:
        input_ids = tokenizer.encode(prompt)
        if len(input_ids) > max_len:
            input_ids = input_ids[:max_len//2] + input_ids[-max_len//2:]
            prompt = tokenizer.decode(input_ids, skip_special_tokens=True)
    else:
        input_ids = tokenizer.encode(prompt, disallowed_special=())
        if len(input_ids) > max_len:
            input_ids = input_ids[:max_len//2] + input_ids[-max_len//2:]
            prompt = tokenizer.decode(input_ids)
    tries = 0
    while tries < 5:
        tries += 1
        try:
            messages = [{"role": "user", "content": prompt}]
            result = pipe(messages, temperature=temperature, max_new_tokens=max_new_tokens)
            return result[0]['generated_text']
        except KeyboardInterrupt as e:
            raise e
        except Exception as e:
            print("Error Occurs: \"%s\"        Retry ..."%(str(e)))
            time.sleep(1)
    else:
        print("Max tries. Failed.")
        return ''

def extract_answer(response):
    response = response.replace('*', '')
    match = re.search(r'The correct answer is \(([A-D])\)', response)
    if match:
        return match.group(1)
    else:
        match = re.search(r'The correct answer is ([A-D])', response)
        if match:
            return match.group(1)
        else:
            return None

def get_pred(data, args, fout):
    
    
    # load model locally
    if args.config_path:
        tokenizer = AutoTokenizer.from_pretrained(
            args.model_path, use_fast=False, trust_remote_code=True
        )
        config = LlamaConfig.from_pretrained(args.config_path)
        model_instance = LlamaForCausalLM_KIVI.from_pretrained(
            args.model_path,
            torch_dtype=torch.float16,
            device_map="auto",
            low_cpu_mem_usage=True,
            config=config,
        )
    else:
        tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)
        model_instance = AutoModelForCausalLM.from_pretrained(
            args.model_path, 
            torch_dtype=torch.float16, 
            device_map="auto",
            trust_remote_code=True
        )
    pipe = pipeline(
        "text-generation",
        model=model_instance,
        tokenizer=tokenizer,
        return_full_text=False
    )
    
    for item in tqdm(data):
        context = item['context']
        if args.rag > 0:
            template = template_rag
            retrieved = item["retrieved_context"][:args.rag]
            retrieved = sorted(retrieved, key=lambda x: x['c_idx'])
            context = '\n\n'.join([f"Retrieved chunk {idx+1}: {x['content']}" for idx, x in enumerate(retrieved)])
        elif args.no_context:
            template = template_no_context
        elif args.cot:
            template = template_0shot_cot
        else:
            template = template_0shot
        prompt = template.replace('$DOC$', context.strip()).replace('$Q$', item['question'].strip()).replace('$C_A$', item['choice_A'].strip()).replace('$C_B$', item['choice_B'].strip()).replace('$C_C$', item['choice_C'].strip()).replace('$C_D$', item['choice_D'].strip())
        if args.cot:
            output = query_llm(prompt, tokenizer, pipe, temperature=0.1, max_new_tokens=1024)
        else:
            output = query_llm(prompt, tokenizer, pipe, temperature=0.1, max_new_tokens=128)
        if output == '':
            continue
        if args.cot: # extract answer
            response = output.strip()
            item['response_cot'] = response
            prompt = template_0shot_cot_ans.replace('$DOC$', context.strip()).replace('$Q$', item['question'].strip()).replace('$C_A$', item['choice_A'].strip()).replace('$C_B$', item['choice_B'].strip()).replace('$C_C$', item['choice_C'].strip()).replace('$C_D$', item['choice_D'].strip()).replace('$COT$', response)
            output = query_llm(prompt, tokenizer, pipe, temperature=0.1, max_new_tokens=128)
            if output == '':
                continue
        response = output.strip()
        item['response'] = response
        item['pred'] = extract_answer(response)
        item['judge'] = item['pred'] == item['answer']
        item['context'] = context[:1000]
        fout.write(json.dumps(item, ensure_ascii=False) + '\n')
        fout.flush()

def main():
    os.makedirs(args.save_dir, exist_ok=True)
    print(args)
    if args.rag > 0:
        out_file = os.path.join(args.save_dir, args.model_path.split("/")[-1] + \
            f"_rag_{str(args.rag)}-{args.rank}_{args.total_rank}.jsonl")
    elif args.no_context:
        out_file = os.path.join(args.save_dir, args.model_path.split("/")[-1] + \
            f"_no_context-{args.rank}_{args.total_rank}.jsonl")
    elif args.cot:
        out_file = os.path.join(args.save_dir, args.model_path.split("/")[-1] + \
            f"_cot-{args.rank}_{args.total_rank}.jsonl")
    else:
        out_file = os.path.join(args.save_dir, args.model_path.split("/")[-1] + \
            f"-{args.rank}_{args.total_rank}.jsonl")

    dataset = load_dataset('THUDM/LongBench-v2', split='train') # dataset = json.load(open('data.json', 'r', encoding='utf-8'))
    data_all = [{"_id": item["_id"], "domain": item["domain"], "sub_domain": item["sub_domain"], "difficulty": item["difficulty"], "length": item["length"], "question": item["question"], "choice_A": item["choice_A"], "choice_B": item["choice_B"], "choice_C": item["choice_C"], "choice_D": item["choice_D"], "answer": item["answer"], "context": item["context"]} for item in dataset]

    # cache
    has_data = {}
    if os.path.exists(out_file):
        with open(out_file, encoding='utf-8') as f:
            has_data = {json.loads(line)["_id"]: 0 for line in f}
    fout = open(out_file, 'a', encoding='utf-8')
    data = []
    for item in data_all:
        if item["_id"] not in has_data:
            data.append(item)

    # data parallelism is handled outside of this script by --total_rank and --rank
    data_subsets = [data[i::args.total_rank] for i in range(args.total_rank)]
    get_pred(data_subsets[args.rank], args, fout)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--save_dir", "-s", type=str, default="results")
    parser.add_argument("--config_path", "-cp", type=str, default=None)
    parser.add_argument("--model_path", "-mp", type=str, required=True, help="path to the model, e.g. meta-llama/Llama-3.1-8B-Instruct")
    parser.add_argument("--maxlen", "-ml", type=int, default=120000, help="maximum token length for the model")
    parser.add_argument("--cot", "-cot", action='store_true') # set to true if using cot
    parser.add_argument("--no_context", "-nc", action='store_true') # set to true if using no context (directly measuring memorization)
    parser.add_argument("--rag", "-rag", type=int, default=0) # set to 0 if rag is not used, otherwise set to n when using top-n retrieved context
    parser.add_argument("--total_rank", type=int, default=1, help="total number of parallel processes")
    parser.add_argument("--rank", type=int, default=0, help="rank of current process")
    args = parser.parse_args()
    main()