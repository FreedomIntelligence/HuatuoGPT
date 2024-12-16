# HuatuoGPT (华佗GPT), Towards Taming Language Models To Be a Doctor.

## ✨ Latest News
- [12/11/2023]: 🎉🎉🎉 Our paper is accepted for EMNLP 2023! Check it out [here](https://aclanthology.org/2023.findings-emnlp.725/).
- [11/25/2023]: We realeased **[HuatuoGPT-II](https://github.com/FreedomIntelligence/HuatuoGPT-II)**, which achieved a new state-of-the-art in Chinese medical applications! See [here](https://github.com/FreedomIntelligence/HuatuoGPT-II).
- [09/26/2023]: Release [HuatuoGPT-reward-model](https://huggingface.co/FreedomIntelligence/HuatuoGPT-reward-model-7B).
- [06/30/2023]: Evaluation data of HuatuoGPT released in the `eval/` folder.
- [06/30/2023]: Release the code, model weights of [HuatuoGPT-7B](https://huggingface.co/FreedomIntelligence/HuatuoGPT-7B) and [HuatuoGPT-13B](https://huggingface.co/FreedomIntelligence/HuatuoGPT-13b-delta)
- [05/25/2023]: Release the [tech report](https://arxiv.org/pdf/2305.15075.pdf) and the HuatuoGPT [demo](https://www.huatuogpt.cn/).

## ⚡ Introduction
Welcome to the repository of HuatuoGPT, a large language model (LLM) trained on a vast Chinese medical corpus. Our objective with HuatuoGPT is to construct a more professional ‘ChatGPT’ for medical consultation scenarios. 

Here is a list of what has been released:

1. HuatuoGPT-SFT-data: A hybrid SFT data capitalizing on both strengths to endow the model with Doctor-like and Patient-friendly characteristics.
2. HuatuoGPT model: HuatuoGPT model weights(HuatuoGPT-7B and HuatuoGPT-13B) and the online demo. **HuatuoGPT-7B** is trained on **Baichuan-7B** and **HuatuoGPT-13B** is trained on **Ziya-LLaMA-13B-Pretrain-v1**.
3. Medical evaluation benchmark: an evaluation method used to evaluate LLMs in medical scenarios.

<div align=center>
<img src="assets/huatuo.png" width = "640" alt="HuatuoGPT" align=center/>
</div>


## 💭 Motivation
- To address the growing demand for quick medical consultations both online and in hospitals that do not necessarily require deep medical knowledge. We believe that LLMs like HuatuoGPT can be effectively utilized to meet these demands, freeing up physicians’ time and energy for more complex cases.
- To provide open data for training medical LLMs. Building high-quality instruction training data for LLMs is essential, but it can be also challenging. We have constructed medical instruction data using various methods and made it publicly available. This dataset can be combined with other datasets to train one's own medical 'ChatGPT'.
- To emphasize the importance of carefully evaluating the ability of medical LLMs before using them to offer medical assistance to patients. We recognize the potential benefits of LLMs in the medical field, but also acknowledge the need for thorough evaluation and testing to ensure patient safety and accurate diagnoses.

## 📚 Data

### Overview

To leverage the best of both distilled data (from ChatGPT) and real-world data (from Doctors), we firstly fine-tune  `HuatuoGPT`  using four types of data:

| Type                          | Scale | Descriptions                                                 | Examples                                                     |
| ------------------------------------ | ----- | ------------------------------------------------------------ | ------------------------------------------------------------ |
| Distilled Instructions from ChatGPT  | 61,400   | Medical instructions generated with self-instruct. Seed instructions are manually written by us. | 💬：现在该做什么准备手术前的身体检查？<br>👨‍⚕️：在准备手术前的身体检查中，需要进行以下几个步骤：<br>&emsp;&emsp;1. 了解病人的病史：了解病人的病史，包括既往病史、手术史、药物过敏史、家族病史等，以便评估手术风险和制定手术方案。<br>&emsp;&emsp;2. 进行体格检查：进行全面的体格检查，包括测量身高、体重、血压、心率、呼吸… |
| Real-world Instructions from Doctors | 69,768  | Medical single-round medical consultants in real-world scenarios, in which the doctor's response is more specific and can address the user's medical concerns. | 💬：我有糖尿病血压160-90.该吃什么降压药？空腹血糖8左右。餐后11左右。我有糖尿病血压160-90.该吃什么降压药？<br>👨‍⚕️：阅读你的提问描述，综合分析症状，体征，糖尿病，血压160-90.血压略高，空腹血糖8左右。餐后11左右。血糖略高。口服依那普利，二甲双胍片治疗，新近研究，二甲双胍片，可以抑制食欲，促进肠道蠕动，预防癌症，预防心脑血管病，延长寿命作用。 |
| Distilled Conversations from ChatGPT     | 68,888   | Medical multi-turn conversations are created using two ChatGPTs: one acting as the patient and the other as the doctor. The conversations are based on real-world patient cases and their corresponding treatment. | 💬：User: 我想问一下大夫，我发烧三天后，第四天全身起了疹子，是怎么回事呢？<br>👨‍⚕️：~您好！感谢您的提问。您的症状可能与幼儿急疹相关。幼儿急疹是一种常见的儿童病，病因为人类疱疹病毒6型或7型感染引起。它的症状包括：<br> 1. 发热 n2. 全身起红色疹子 <br> 3. 口腔内及咽喉部有小水泡<br> 4. 病程一般为3-5天左右...<br>💬：谢谢大夫！除了幼儿急疹还有其他可能吗？我之前有感冒，这和疹子有关系吗？<br>👨‍⚕️：除了幼儿急疹，病毒性感冒和其他病毒感染也有可能引起疹子，但这种情况相对较少见。在您的情况下，感冒可能是一种诱发因素，但主要还是由病毒感染引起的疹子... |
| Real-world Conversations with Doctors      | 25,986   | Medical multi-round conversations in real-world scenarios. Conversations include multiple rounds of questioning by doctors and continuous questioning by patients. | 💬：额头下巴下长满痘痘。就是不知道.<br>👨‍⚕️：你好，有什么困扰吗？我可以为你提供帮助。请问您平时的饮食习惯、睡眠和护肤习惯怎样？化妆品和防晒霜的使用频率如何？还有洁面的频次是怎样的呢？<br>💬：平时我基本上只是用水乳霜，然后洗面奶只用一种。日常防晒只有在夏天时会使用，我一般用悦诗风吟的洗面奶以及AHC的护肤产品，这些都比较温和。<br>👨‍⚕️：谢谢您的描述。首先，建议您停用化妆品和普通护肤品，使用特定的药妆品牌，如雅漾和理肤泉的洁面泡沫或洁面乳，并避免使用热水或肥皂清洗面部。建议您外用阿达帕林凝胶涂于面部痤疮，一般夜间使用，涂抹十五分钟之后外用医学护肤品比如雅漾、薇资、理肤泉清爽型的舒缓面霜，或者是维生素e乳膏… |

### Download
- [HuatuoGPT-sft-data-v1](https://huggingface.co/datasets/FreedomIntelligence/HuatuoGPT-sft-data-v1): The data used in the Supervised Fine-Tuning (SFT) stage of HuatuoGPT.

  

## 👨‍⚕️ Model

### Model Access
| Model                | Backbone      | Link                                                                          |
|----------------------|---------------|-------------------------------------------------------------------------------|
| HuatuoGPT-13B | Ziya-LLaMA-13B-Pretrain-v1 | [Delta](https://huggingface.co/FreedomIntelligence/HuatuoGPT-13b-delta) |
| HuatuoGPT-7B      | Baichuan-7B | [Model Weights](https://huggingface.co/FreedomIntelligence/HuatuoGPT-7B)      |


Note that due to that HuatuoGPT-13B-delta is a LLaMA based model, we only release the delta of weights. You can download LLaMA-13B weights and use apply_delta.py to convert:
```bash 
python apply_delta.py \
--base-model-path $LLaMA_Base_Path \
--target-model-path $Save_Path \
--delta-path $Delta_Path
```

### Deploy

Firstly, you should install all required packages
```bash
pip install -r requirements.txt
```

Please make sure you have download our model weights and run
```bash
python huatuo_cli_demo_stream.py --model-name $model_dir
```



## 🚀 Demo

Try our model in [https://www.huatuogpt.cn/](https://www.huatuogpt.cn/). Note that it is still in progressing.

<!-- ![demo_1](assets/demo_1.png) -->
<!-- ![demo_2](assets/demo_2.png) -->



## 🧐 Evaluations

### Evaluation by GPT-4 and Doctors
We invite GPT-4 and doctors to compare responses from HuatuoGPT(13B version) and other LLMs. Evaluation data is available in the `eval/` folder. Results are as below:

- Single turn evaluation

<div align=center>
<img src="assets/single_turn_compare.png"  alt="eval1" align=center/>
</div>

- Multi turn evaluation
<div align=center>
<img src="assets/multi_turn_compare.png"  alt="eval1" align=center/>
</div>


### Benchmark  Evaluation

| Dataset   | Model | BLEU-1 | BLEU-2 | BLEU-3 | BLEU-4 | GLEU | ROUGE-1 | ROUGE-2 | ROUGE | Distinct-1 | Distinct-2 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| cMedQA2 | T5-finetuned | 20.88 | 11.87 | 7.69 | 5.09 | 7.62 | 27.16 | 9.30 | 20.11 | 0.41 | 0.52 |
|  | HuatuoGPT | 27.39 | 14.38 | 8.06 | 4.55 | 8.52 | 29.26 | 8.02 | 15.46 | 0.74 | 0.93 |
| WebMedQA | T5-finetuned | 21.42 | 13.79 | 10.06 | 7.38 | 8.94 | 31.00 | 13.85 | 25.78 | 0.37 | 0.46 |
|  | HuatuoGPT | 24.85 | 13.42 | 7.72 | 4.51 | 7.50 | 28.30 | 7.72 | 14.50 | 0.73 | 0.93 |
| Huatuo-26M | T5-finetuned | 26.63 | 16.74 | 11.77 | 8.46 | 11.38 | 33.21 | 13.26 | 24.85 | 0.51 | 0.68 |
|  | HuatuoGPT | 27.42 | 14.84 | 8.54 | 4.96 | 8.01 | 29.16 | 8.29 | 15.84 | 0.74 | 0.93 |

## ⚒️ Training
### Prepare the Data
You can download the SFT data from [HuatuoGPT-sft-data-v1](https://huggingface.co/datasets/FreedomIntelligence/HuatuoGPT-sft-data-v1) or buld your SFT data as the same schema.

### Training
You can train the model by:
```bash
accelerate launch \
	--config_file scripts/sft.yaml \
	--num_processes 8 \
	--num_machines 1 \
	--machine_rank 0 \
	--deepspeed_multinode_launcher standard scripts/finetune.py \
    --experiment_name HuatuoGPT \
	--model_path /path/to/your/model \
    --gradient_accumulation_steps 8 \
    --max_ckpts 3 \
    --max_seq_len 2048 \
	--data_dir /path/to/your/data \
	--output_dir ./ckpts \
	--log_dir ./train_logs \
	--n_epochs 3 \
	--train_bsz_per_gpu 2 \
	--eval_bsz_per_gpu 2 \
	--learning_rate 5e-5 \
	--eval_step -1 \
	--save_step -1 \
    --gradient_checkpointing
```

## 🤖 Limitations

Our goal with HuatuoGPT is to address the need for quick medical consultations, rather than replace doctors or provide full medical support to patients. However, our model does have several limitations that must be taken into consideration:

- Misunderstandings: As with all language models, there is a risk of misunderstandings or misinterpretations, especially when dealing with medical jargon or complex conditions. In this scenario, our models may give wrong answers.
- Hallucinations: Large language models can sometimes generate responses that do not make sense or are completely unrelated to the given input. These "hallucinations" can be especially problematic when users are not familiar with the concepts being discussed, as they may not be able to easily recognize the errors in the model's output. These "hallucinations" can be a challenge to detect and avoid.
- Bias: LLMs are trained on large datasets, which can inadvertently introduce bias into the model's responses. Additionally, care should be taken to ensure that the model is not used to perpetuate biases in medical treatment.

## Acknowledgement

We are aware that our works are inspired by the following works, including but not limited to

- IDEA-CCNL/Ziya-LLaMA-13B-Pretrain-v1: https://huggingface.co/IDEA-CCNL/Ziya-LLaMA-13B-Pretrain-v1
- Baichuan-7B: https://huggingface.co/baichuan-inc/baichuan-7B
- LLaMA: https://arxiv.org/abs/2302.13971
- Self-instruct: https://github.com/yizhongw/self-instruct

Without these, nothing could happen in this repository.

## Citation
```angular2
@article{huatuogpt-2023,
  title={HuatuoGPT, Towards Taming Language Models To Be a Doctor},
  author={Hongbo Zhang and Junying Chen and Feng Jiang and Fei Yu and Zhihong Chen and Jianquan Li and Guiming Chen and Xiangbo Wu and Zhiyi Zhang and Qingying Xiao and Xiang Wan and Benyou Wang and Haizhou Li},
  journal={arXiv preprint arXiv:2305.15075},
  year={2023}
}
```

We are from the School of Data Science, the Chinese University of Hong Kong, Shenzhen (CUHKSZ) and the Shenzhen Rsearch
Institute of Big Data (SRIBD).

## Star History

<a href="https://star-history.com/#FreedomIntelligence/HuatuoGPT&Date">
  <picture>
    <source media="(prefers-color-scheme: dark)" srcset="https://api.star-history.com/svg?repos=FreedomIntelligence/HuatuoGPT&type=Date&theme=dark" />
    <source media="(prefers-color-scheme: light)" srcset="https://api.star-history.com/svg?repos=FreedomIntelligence/HuatuoGPT&type=Date" />
    <img alt="Star History Chart" src="https://api.star-history.com/svg?repos=FreedomIntelligence/HuatuoGPT&type=Date" />
  </picture>
</a>
