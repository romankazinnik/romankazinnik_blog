# Multimodal Conditioning Part-2: Text-Conditioning in Diffusion Transformer Models (DiT)

In this post I am going to show how DiT solve the problems of text conditioning in GANs as shown in my previous post (Multimodal Conditioning Part-1: Enhancing GAN with Text-Conditioning, https://medium.com/@roman-kazinnik/multimodal-conditioning-part-1-enhancing-gan-with-text-conditioning-7e556a95c78d).

Here is the summary of the major drawbacks of text conditioning in GANs and how DiT solves them:


GANsArchitectural Imbalance resulting in text conditioning not being able to influence all levels of the generation process. 
DiT:  cross-attention mechanisms at all the transformer blockss.

GANs: Limited Text Representation not being able to capture the complex semantic relationships of the text tokens and image patches adequately. 
DiT:  transformer-based text encoders with cross-attention mechanisms.

GANs: This results inLimited Conditioning Control:  projecting the text embedding to a fixed-size feature vector (ngf*4) loses fine-grained control over different aspects of the generated image. 
DiT: Improving conditioning strength by removing the classifier guidance.

GANs Discriminator: Late Feature Integration due to the text features being only integrated after all the convolutional layers have processed the image and do not influence the earlier feature extraction stages
DiT:  Early feature integration through cross-attention mechanisms at all the transformer blockss.


GANs Discriminator: Spatial Uniformity Problem that doesn't allow the discriminator to focus on specific image regions that might be more relevant to particular words in the text description.
DiT: cross-attention mechanisms for text tokens and image patches.

GANs Descriminator: Dimensionality Imbalance where text features have equal weight to the entire processed image representation, potentially overwhelming the visual features.
DiT: text tokens are projected to the same dimension as image patches.

GANs Descriminator: Limited Text-Image Correlation Learning of how well the discriminator can learn correlations between specific text concepts and visual elements.
DiT: cross-attention mechanisms for text tokens and image patches.

GANs Descriminator: Fixed Patch Size Limitation
DiT: cross-attention mechanisms for text tokens and image patches.

GANs Descriminator: Potential Feature Collapse with the discriminator learns to rely too heavily on either text or image features and ignores the other, it could lead to a form of feature collapse where only one modality influences the real/fake decision.
DiT:  Classifier-free guidance (CFG) in DiT aims to improve the quality of the generated images.
CFG uses Dual Prediction: during each denoising step, the model makes two noise predictions:

Conditional prediction (εθ(xt, t, c)): Predicts noise given the current noisy image, timestep, and text condition
Unconditional prediction (εθ(xt, t, ∅)): Predicts noise with no text conditioning (empty prompt)

Guidance Formula: The final noise prediction combines both:
ε_guided = ε_unconditional + w × (ε_conditional - ε_unconditional)
Where w is the guidance scale (typically 7-15 for Stable Diffusion).


GANs Descriminator: Lack of Multi-Scale Text Conditioning which conditions only at the final layer.
DiT: cross-attention mechanisms for text tokens and image patches at all the transformer blocks.
