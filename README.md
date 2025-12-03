# Diffusion Models in Computer Vision: Literature Survey

## About This Repository

This repository contains a curated collection of references from my literature survey on **Diffusion Models in Computer Vision**. The survey explores the theoretical foundations, architectures, acceleration techniques, and applications of diffusion models.
---

## Table of Contents

- [Foundational Papers](#foundational-papers)
- [Theoretical Foundations](#theoretical-foundations)
- [Network Architectures](#network-architectures)
- [Acceleration Techniques](#acceleration-techniques)
- [Applications](#applications)
- [Performance & Evaluation](#performance--evaluation)
- [Surveys & Reviews](#surveys--reviews)

---

## Foundational Papers

### Denoising Diffusion Probabilistic Models (DDPM)
**Ho, J., Jain, A., & Abbeel, P. (2020)**  
*Denoising diffusion probabilistic models*  
Advances in Neural Information Processing Systems 33:6840–6851  
[Paper (NeurIPS)](https://proceedings.neurips.cc/paper/2020/hash/4c5bcfec8584af0d967f1ab10179ca4b-Abstract.html) | [arXiv](https://arxiv.org/abs/2006.11239)

**Key Contribution:** Introduced the DDPM framework with forward and reverse diffusion processes, establishing diffusion models as competitive with GANs.

---

### Score-Based Generative Modeling
**Sohl-Dickstein, J., Weiss, E., Maheswaranathan, N., & Ganguli, S. (2015)**  
*Deep unsupervised learning using nonequilibrium thermodynamics*  
International Conference on Machine Learning, 2256–2265  
[Paper (PMLR)](http://proceedings.mlr.press/v37/sohl-dickstein15.html) | [arXiv](https://arxiv.org/abs/1503.03585)

**Key Contribution:** Early theoretical foundations connecting non-equilibrium thermodynamics to generative modeling.

---

**Song, Y., & Ermon, S. (2020)**  
*Generative modeling by estimating gradients of the data distribution*  
[arXiv:1907.05600](https://arxiv.org/abs/1907.05600)

**Key Contribution:** Score-based perspective using score matching and Langevin dynamics.

---

### Stochastic Differential Equations (SDE) Formulation
**Song, Y., Sohl-Dickstein, J., Kingma, D. P., Kumar, A., Ermon, S., & Poole, B. (2021)**  
*Score-based generative modeling through stochastic differential equations*  
[arXiv:2011.13456](https://arxiv.org/abs/2011.13456) | [ICLR 2021](https://openreview.net/forum?id=PxTIG12RRHS)

**Key Contribution:** Unified DDPM and score-based models through continuous-time SDE framework, enabling advanced sampling techniques.

---

## Theoretical Foundations

### Energy-Based Models
**LeCun, Y., Chopra, S., Hadsell, R., Ranzato, M., Huang, F., et al. (2006)**  
*A tutorial on energy-based learning*  
Predicting Structured Data 1(0)  
[Paper](http://yann.lecun.com/exdb/publis/pdf/lecun-06.pdf)

**Key Contribution:** Foundation for understanding the intractability problem in probabilistic models.

---

**Grathwohl, W., Wang, K.-C., Jacobsen, J.-H., Duvenaud, D., Norouzi, M., & Swersky, K. (2019)**  
*Your classifier is secretly an energy based model and you should treat it like one*  
[arXiv:1912.03263](https://arxiv.org/abs/1912.03263) | [ICLR 2020](https://openreview.net/forum?id=Hkxzx0NtDB)

---

## Network Architectures

### U-Net for Diffusion Models
**Ronneberger, O., Fischer, P., & Brox, T. (2015)**  
*U-Net: Convolutional networks for biomedical image segmentation*  
[arXiv:1505.04597](https://arxiv.org/abs/1505.04597) | [MICCAI 2015](https://link.springer.com/chapter/10.1007/978-3-319-24574-4_28)

**Key Contribution:** Original U-Net architecture with skip connections, later adopted as standard backbone for diffusion models.

---

### Vision Transformers (ViT)
**Dosovitskiy, A., Beyer, L., Kolesnikov, A., Weissenborn, D., Zhai, X., Unterthiner, T., Dehghani, M., Minderer, M., Heigold, G., Gelly, S., Uszkoreit, J., & Houlsby, N. (2021)**  
*An image is worth 16x16 words: Transformers for image recognition at scale*  
[arXiv:2010.11929](https://arxiv.org/abs/2010.11929) | [ICLR 2021](https://openreview.net/forum?id=YicbFdNTTy)

---

### U-ViT: Vision Transformer with U-Net Structure
**Bao, F., Nie, S., Xue, K., Cao, Y., Li, C., Su, H., & Zhu, J. (2023)**  
*All are worth words: A ViT backbone for diffusion models*  
[arXiv:2209.12152](https://arxiv.org/abs/2209.12152) | [CVPR 2023](https://openaccess.thecvf.com/content/CVPR2023/html/Bao_All_Are_Worth_Words_A_ViT_Backbone_for_Diffusion_Models_CVPR_2023_paper.html)

**Key Contribution:** Combined transformer architecture with U-Net-inspired long skip connections for diffusion models.

---

### Diffusion Transformers (DiT)
**Peebles, W., & Xie, S. (2023)**  
*Scalable diffusion models with transformers*  
[arXiv:2212.09748](https://arxiv.org/abs/2212.09748) | [ICCV 2023](https://openaccess.thecvf.com/content/ICCV2023/html/Peebles_Scalable_Diffusion_Models_with_Transformers_ICCV_2023_paper.html)

**Key Contribution:** Replaced U-Net with pure transformer architecture operating in latent space (DiT-XL/2 achieved FID = 2.27 on ImageNet 256×256).

---

## Acceleration Techniques

### Latent Diffusion Models (LDM)
**Rombach, R., Blattmann, A., Lorenz, D., Esser, P., & Ommer, B. (2022)**  
*High-resolution image synthesis with latent diffusion models*  
[arXiv:2112.10752](https://arxiv.org/abs/2112.10752) | [CVPR 2022](https://openaccess.thecvf.com/content/CVPR2022/html/Rombach_High-Resolution_Image_Synthesis_With_Latent_Diffusion_Models_CVPR_2022_paper.html)

**Key Contribution:** Train diffusion in compressed latent space using VAE, dramatically reducing computational requirements (basis for Stable Diffusion).

---

### Denoising Diffusion GANs
**Xiao, Z., Kreis, K., & Vahdat, A. (2022)**  
*Tackling the generative learning trilemma with denoising diffusion GANs*  
[arXiv:2112.07804](https://arxiv.org/abs/2112.07804) | [ICLR 2022](https://openreview.net/forum?id=JprM0p-q0Co)

**Key Contribution:** Addresses the generative learning trilemma (sample quality, mode coverage, fast sampling).

---

## Applications

### Text-to-Image Synthesis

#### DALL-E 2
**Ramesh, A., Dhariwal, P., Nichol, A., Chu, C., & Chen, M. (2022)**  
*Hierarchical text-conditional image generation with CLIP latents*  
[arXiv:2204.06125](https://arxiv.org/abs/2204.06125)

**Key Contribution:** Uses CLIP embeddings to guide diffusion process for text-to-image generation.

---

#### Imagen
**Saharia, C., Chan, W., Saxena, S., Li, L., Whang, J., Denton, E. L., Ghasemipour, K., Gontijo Lopes, R., Karagol Ayan, B., Salimans, T., et al. (2022)**  
*Photorealistic text-to-image diffusion models with deep language understanding*  
Advances in Neural Information Processing Systems 35:36479–36494  
[arXiv:2205.11487](https://arxiv.org/abs/2205.11487) | [NeurIPS 2022](https://proceedings.neurips.cc/paper_files/paper/2022/hash/ec795aeadae0b7d230fa35cbaf04c041-Abstract-Conference.html)

**Key Contribution:** Uses large language models (T5) for text encoding with cascaded diffusion for multi-resolution synthesis (1024×1024).

---

#### Stable Diffusion
**Rombach, R., Blattmann, A., Lorenz, D., Esser, P., & Ommer, B. (2021)**  
*High-resolution image synthesis with latent diffusion models*  
[arXiv:2112.10752](https://arxiv.org/abs/2112.10752)

**Key Contribution:** Open-source implementation of latent diffusion with CLIP text conditioning.

---

### Video Synthesis

#### Video Diffusion Models (VDM)
**Ho, J., Salimans, T., Gritsenko, A., Chan, W., Norouzi, M., & Fleet, D. J. (2022)**  
*Video diffusion models*  
[arXiv:2204.03458](https://arxiv.org/abs/2204.03458)

**Key Contribution:** Extends 2D U-Net to 3D (frames × height × width × channels) factorized over space and time.

---

#### Imagen Video
**Ho, J., Chan, W., Saharia, C., Whang, J., Gao, R., Gritsenko, A., Kingma, D. P., Poole, B., Norouzi, M., Fleet, D. J., & Salimans, T. (2022)**  
*Imagen video: High definition video generation with diffusion models*  
[arXiv:2210.02303](https://arxiv.org/abs/2210.02303)

**Key Contribution:** Cascade of diffusion models for high-quality video (1280×768, 24 fps).

---

#### Sora
**Liu, Y., Zhang, K., Li, Y., Yan, Z., Gao, C., Chen, R., Yuan, Z., Huang, Y., Sun, H., Gao, J., He, L., & Sun, L. (2024)**  
*Sora: A review on background, technology, limitations, and opportunities of large vision models*  
[arXiv:2402.17177](https://arxiv.org/abs/2402.17177)

**Key Contribution:** Diffusion Transformer operating on spacetime patches of video latent codes.

---

### Motion Generation
**Tevet, G., Raab, S., Gordon, B., Shafir, Y., Cohen-Or, D., & Bermano, A. H. (2022)**  
*Human motion diffusion model*  
[arXiv:2209.14916](https://arxiv.org/abs/2209.14916) | [ICLR 2023](https://openreview.net/forum?id=SJ1kSyO2jwu)

**Key Contribution:** Transformer encoder for variable-length human motion synthesis from text descriptions.

---

**Zhang, M., Guo, X., Pan, L., Cai, Z., Hong, F., Li, H., Yang, L., & Liu, Z. (2023)**  
*ReMoDiffuse: Retrieval-augmented motion diffusion model*  
[arXiv:2304.01116](https://arxiv.org/abs/2304.01116) | [ICCV 2023](https://openaccess.thecvf.com/content/ICCV2023/html/Zhang_ReMoDiffuse_Retrieval-Augmented_Motion_Diffusion_Model_ICCV_2023_paper.html)

---

## Performance & Evaluation

### Fréchet Inception Distance (FID)
**Dhariwal, P., & Nichol, A. (2021)**  
*Diffusion models beat GANs on image synthesis*  
[arXiv:2105.05233](https://arxiv.org/abs/2105.05233) | [NeurIPS 2021](https://proceedings.neurips.cc/paper/2021/hash/49ad23d1ec9fa4bd8d77d02681df5cfa-Abstract.html)

**Key Contribution:** Demonstrated diffusion models outperform GANs on multiple benchmarks:
- ADM-G achieved FID = 2.97 on ImageNet 128×128 vs BigGAN-deep's 6.02
- Better recall metrics: 0.51-0.69 (diffusion) vs 0.28-0.48 (GANs)

---

## Surveys & Reviews

**Chen, H., Xiang, Q., Hu, J., Ye, M., Yu, C., Cheng, H., & Zhang, L. (2025)**  
*Comprehensive exploration of diffusion models in image generation: A survey*  
Artificial Intelligence Review 58(4):99  
[Paper](https://link.springer.com/article/10.1007/s10462-024-11006-y)

---

**Yang, L., Zhang, Z., Song, Y., Hong, S., Xu, R., Zhao, Y., Zhang, W., Cui, B., & Yang, M.-H. (2025)**  
*Diffusion models: A comprehensive survey of methods and applications*  
[arXiv:2209.00796](https://arxiv.org/abs/2209.00796)

---

**Yazdani, S., Singh, A., Saxena, N., Wang, Z., Palikhe, A., Pan, D., Pal, U., Yang, J., & Zhang, W. (2025)**  
*Generative AI in depth: A survey of recent advances, model variants, and real-world applications*  
Journal of Big Data 12(1):1–43  
[Paper](https://journalofbigdata.springeropen.com/articles/10.1186/s40537-024-01021-8)

---

**Weng, L. (2021)**  
*What are diffusion models?*  
[Blog Post](https://lilianweng.github.io/posts/2021-07-11-diffusion-models/)

**Key Contribution:** Highly accessible blog post explaining diffusion models with clear visualizations.

---

**Wang, Z., Li, D., Wu, Y., He, T., Bian, J., & Jiang, R. (2025)**  
*Diffusion models in 3D vision: A survey*  
[arXiv:2310.07204](https://arxiv.org/abs/2310.07204)

---

## Privacy & Safety

**Carlini, N., Hayes, J., Nasr, M., Jagielski, M., Sehwag, V., Tramer, F., Balle, B., Ippolito, D., & Wallace, E. (2023)**  
*Extracting training data from diffusion models*  
32nd USENIX Security Symposium (USENIX Security 23), 5253–5270  
[Paper](https://www.usenix.org/conference/usenixsecurity23/presentation/carlini) | [arXiv](https://arxiv.org/abs/2301.13188)

**Key Contribution:** Demonstrated that diffusion models can memorize and extract training data, raising privacy and copyright concerns.

---

---