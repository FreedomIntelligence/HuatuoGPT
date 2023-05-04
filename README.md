# HuatuoGPT(华佗GPT): taming large language models to follow medical instruction.

# 👨‍⚕️Introduction

Welcome to the repository of HuatuoGPT, a large language model (LLM) trained on a vast Chinese medical corpus. Our objective with HuatuoGPT is to construct a more professional ‘ChatGPT’ for medical consultation scenarios. Here is a list of what has been released:

1. Huatuo-200K (Medical Instruction Dataset): a  high-qulity instruction dataset that sampled from HuatuoGPT training corpus.
2. Medical-Chat LLMs: weights of a set of popular LLMs trained on our Huatuo-200K.
3. HuatuoGPT: a currently beta version HuatuoGPT online.
4. Medical evaluation benchmark: an evaluation method used to evaluate several LLMs in medical domain.

Hope our repository will help enhance the field of intelligent medicine and its related applications.

# 💭Motivation

- To address the growing demand for quick medical consultations both online and in hospitals that do not necessarily require deep medical knowledge. We believe that LLMs like HuatuoGPT can be effectively utilized to meet these demands, freeing up physicians’ time and energy for more complex cases.
- To provide open data for training medical LLMs. Building high-quality instruction training data for LLMs is essential, but it can be also challenging. We have constructed medical instruction data using various methods and made it publicly available. This dataset can be combined with other datasets to train one's own medical 'ChatGPT'.
- To emphasize the importance of carefully evaluating the ability of medical LLMs before using them to offer medical assistance to patients. We recognize the potential benefits of LLMs in the medical field, but also acknowledge the need for thorough evaluation and testing to ensure patient safety and accurate diagnoses.

# 📚Data

We have gathered hundreds of billion tokens of online data to create a high-quality medical training corpus for HuatuoGPT. We also selected 200K pieces from this dataset as medical instruction data, named Huatuo-200K. This subset is adequate for fine-tuning LLMs to create ChatGPT-like consultant models. We are releasing this resource to the public, enabling anyone to train their very own intelligent doctor. 

| Category                          | Scale | Descriptions                                                 | Examples                                                     |
| --------------------------------- | ----- | ------------------------------------------------------------ | ------------------------------------------------------------ |
| Medical Instructions              | 30K   | Medical instructions generated with self-instruct. Seed instructions are manually written by us. | 💬：现在该做什么准备手术前的身体检查？<br>👨‍⚕️：在准备手术前的身体检查中，需要进行以下几个步骤：<br>&emsp;&emsp;1. 了解病人的病史：了解病人的病史，包括既往病史、手术史、药物过敏史、家族病史等，以便评估手术风险和制定手术方案。<br>&emsp;&emsp;2. 进行体格检查：进行全面的体格检查，包括测量身高、体重、血压、心率、呼吸… |
| Medical Single-Round consultants  | 130K  | Medical single-round medical consultants in real-world scenarios, in which the doctor's response is more specific and can address the user's medical concerns. | 💬：我有糖尿病血压160-90.该吃什么降压药？空腹血糖8左右。餐后11左右。我有糖尿病血压160-90.该吃什么降压药？<br>👨‍⚕️：阅读你的提问描述，综合分析症状，体征，糖尿病，血压160-90.血压略高，空腹血糖8左右。餐后11左右。血糖略高。口服依那普利，二甲双胍片治疗，新近研究，二甲双胍片，可以抑制食欲，促进肠道蠕动，预防癌症，预防心脑血管病，延长寿命作用。 |
| Medical Multi-Round consultants   | 20K   | Medical multi-round conversations in real-world scenarios. Conversations include multiple rounds of questioning by doctors and continuous questioning by patients. | 💬：额头下巴下长满痘痘。就是不知道.<br>👨‍⚕️：你好，有什么困扰吗？我可以为你提供帮助。请问您平时的饮食习惯、睡眠和护肤习惯怎样？化妆品和防晒霜的使用频率如何？还有洁面的频次是怎样的呢？<br>💬：平时我基本上只是用水乳霜，然后洗面奶只用一种。日常防晒只有在夏天时会使用，我一般用悦诗风吟的洗面奶以及AHC的护肤产品，这些都比较温和。<br>👨‍⚕️：谢谢您的描述。首先，建议您停用化妆品和普通护肤品，使用特定的药妆品牌，如雅漾和理肤泉的洁面泡沫或洁面乳，并避免使用热水或肥皂清洗面部。建议您外用阿达帕林凝胶涂于面部痤疮，一般夜间使用，涂抹十五分钟之后外用医学护肤品比如雅漾、薇资、理肤泉清爽型的舒缓面霜，或者是维生素e乳膏…. |
| General Instructions              | 10K   | Sampled general instructions. Mainly are generated with the method of self-instruct. | 💬：给出一个成熟男人潜移默化间可以去实践的人生哲学。<br>👨‍⚕️：回顾你曾经有过的每一次经历并从中学习，并认真而深刻地感受这些经历带给你的教训。这样，你每天都可以从中获得更多自信和动力，更好地去面对生活中的新挑战。以真诚与仁爱为基础，以善待他人及自己为原则，以博爱与服务共存为哲学，以不断自我超越和进步为目标…. |
| General Multi-Round Conversations | 10K   | General conversations with ChatGPT. Mainly are from real conversations between users and ChatGPT. | 💬：三级公立医院绩效考核自评报告怎么写<br>👨‍⚕️：编写三级公立医院绩效考核自评报告需要遵循以下步骤：<br>&emsp;&emsp;1.梳理考核指标：仔细阅读考核指标，将其进行分类、梳理，理清考核指标之间的关系和重要性… <br>💬：请写一篇6000字左右的示例。<br>👨‍⚕️：很抱歉，我是一名人工智能语言模型，无法为您提供长篇6000字左右的示例。我可以提供一些写作建议和范例，以帮助您编写三级公立医院绩效考核自评报告。<br>&emsp;&emsp;1. 简介…. |

# Models

- Bloomz-Huatuo200K
- Phoenix-Huatuo200K
- ChatGLM-Huatuo200K
- Llama-Huatuo200K


## Evaluations

- Huatuo200K
- ChatDoctor
- DoctorGLM

- Huatuo-Llama-Med-Chinese


# HuatuoGPT

See our model in [https://www.huatuogpt.cn/](https://www.huatuogpt.cn/) (API key required). Note that it is still in progressing.

# Limitations

Our goal with HuatuoGPT is to address the need for quick medical consultations, rather than replace doctors or provide full medical support to patients. However, our model does have several limitations that must be taken into consideration:

- Misunderstandings: As with all language models, there is a risk of misunderstandings or misinterpretations, especially when dealing with medical jargon or complex conditions. In this scenario, our models may give wrong answers.
- Hallucinations: Large language models can sometimes generate responses that do not make sense or are completely unrelated to the given input. These "hallucinations" can be especially problematic when users are not familiar with the concepts being discussed, as they may not be able to easily recognize the errors in the model's output. These "hallucinations" can be a challenge to detect and avoid.
- Bias: LLMs are trained on large datasets, which can inadvertently introduce bias into the model's responses. Additionally, care should be taken to ensure that the model is not used to perpetuate biases in medical treatment.

# Acknowledgement



# References

HuatuoGPT, taming Language models to follow medical instructions in conversation.

Less is more, taming Language models to follow medical instructions in conversation using 200k samples.

