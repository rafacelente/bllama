## BitNet b1.58

[BitNet b1.58](https://arxiv.org/pdf/2402.17764v1.pdf) is a novel 1-bit LLM variant that introduces ternary {-1,0,1} parameter weights. Although only having 3 possible values for weights, it matches full-precision Transformers of the same size in performance.

What is incredibily interesting is that this concept allows models to be both performant and cost-effective.

### Tips on training BitNet

- Replace all nn.Linear in attention and SwiGLU with BitLinear
- Remove RMSNorm before attention and SwiGLU because BitLinear has built-in RMSNorm


## A review of LLaMa

### Datasets used

- 67% English CommonCrawl (CCNet pipeline with filtering of low quality content with n-gram model)
- 15% C4 with filtering
- 4.5% GitHub with removed boilerplates
- 4.5% Wikipedia with preprocess to remove hyperlinks, comments and formatting boilerplates.
- 4.5% Gutenberg and Books3
- 2.5% ArXiv with removed first sections and bibliography
- 2% Stack Exchange with removed HTML tags

### Notable differences from past architectures

- RMSNorm instead of LayerNorm and normalization of each sub-layer's input instead of output (already done on GPT-2)
- SwiGLU instead of ReLU (GPT-2 uses GeLU)
- Rotary encodings instead of absolute positional embeddings

### Optimizer

- Trained with AdamW with $\beta_1 = 0.9$, $\beta_2 = 0.95$ 
- Cosine learning rate schedule such that the final learning rate is equal to 10% of the maximal learning rate.
- Weight decay of 0.1 and gradient clipping of 1.0.
- 2000 warm up steps.   


### Optimizations and efficient implementation

- Efficient implementation of multi-head attention using the xformers library. -> Doesn't store attention weights and doesn't compute key/query scores that are masked.
- Manual iomplementation of the backward function of the transformer layers.
- Training efficiency of the 65B-parameter model: 380 tokens/sec/GPU on 2048 A100-80GB. -> 21 days to train 1.4T tokens.
- 