## bLLaMa

bLLaMa is a [b1.58](https://arxiv.org/pdf/2402.17764v1.pdf) LLaMa model.

### Set up

Both the module configuration dataclass and the module itself are contained on `bllama`. By default, the configuration is a 1.7B model, which can be found on `config.py`.

```python
from bllama import bLlamaConfig, bLlama

config = bLlamaConfig()
bllm = bLlama(config)
```

#### Training

bLLaMa is built as a Lightning module, so you may pass `pl.Trainer`s and `pl.LightningDataModules` for training tasks. To faciliate, some examples of datasets the corresponding datamodules are given on `utils`.

```python
from transformers import LlamaTokenizer
from utils import ShakespeareDataset, TextDataModule, GepetoDataset

tokenizer = LlamaTokenizer.from_pretrained("fxmarty/tiny-llama-fast-tokenizer")
dataset = ShakespeareDataset.from_file("/path/to/shakespeare/input.txt", tokenizer=tokenizer, max_length=1024)
dataloader = TextDataModule(dataset, batch_size=config.batch_size, train_test_split=0.9)
```

To setup a trainer, you may pass a `pl.Trainer` or a manually setup a training run.

```
import pytorch_lightning as pl

bllm_trainer = pl.Trainer(
    accelerator="gpu",
    max_epochs=1,
)

bllm_trainer.fit(bllm, dataloader)
```

#### Inference

The BitLinear layers of bLLaMa have 2 modes, one for training (`fp32`) and one for quantized inference (`int8`). To perform quantized inference, the weights have to be offline-quantized. bLLaMa has a built-in method to quantize the BitLinear modules for inference:

![bLLaMa quantization](utils/images/bllama_quantization.png)

After quantization, the model can then generate with the `generate` method.

```python
bllm.generate(prompt="In a shocking turn of events,", tokenizer=tokenizer, max_len=200, do_sample=False, top_k=3, repetition_penalty=2)
```

Full precision inference is also allowed, but the model will promptly caution all the BitLinear layers that are not quantized.

