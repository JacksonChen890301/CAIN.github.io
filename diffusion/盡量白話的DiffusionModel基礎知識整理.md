layout: page
title: "盡量白話的Diffusion Model基礎知識整理  "
permalink: /diffusion/basic/

# 盡量白話的Diffusion Model基礎知識整理  
整理給自己看的Diffusion Model基本知識 
為了方便理解寫得很白話(不負責任版)
更新日期: 2024/10/16

## Denoising Diffusion Probabilistic Models (DDPM) 
<font  size=3>主要源自[Denoising Diffusion Probabilistic Models(2020)](https://arxiv.org/abs/2006.11239)</font>

<font  size=3>一般我們可以假設人眼認為合理的影像應該會滿足某種特定的分布，用數學來表達就是一張真實圖像 $\mathbf{x}_0$ 需要滿足 $\mathbf{x}_0 \sim q(\mathbf{x})$ 這個分布（這邊的$\mathbf{x}_0$不見得一定要是圖像，你想像得出來任何適用以上假設的case都可以）。當然想也知道像圖像這類複雜的分布是很難人為的用任何已知的數學形式去表達出來的，所以有人想到是不是可以用深度學習來模擬這樣的分布，藉此幫助我們做圖像生成。如果粗淺的理解，擴散這個概念就是不管你的初始狀態 $\mathbf{x}_0$ 長怎麼樣，反正經過長時間的擴散，各種不同的 $\mathbf{x}_0$ 他們原始的特徵都會被稀釋掉、分布上也變得趨近一致。所以如果神經網路能逆轉這個擴散行為，用大量的擴散數據去學習反擴散的可能性，是不是就可以反過來從擴散的盡頭推測出一個合理的 $\mathbf{x}_0$？透過這個概念，以下的流程就被設計了出來，能夠對Diffusion為基礎的圖像生成模型進行訓練及採樣：</font>  

<center><img src="https://lilianweng.github.io/posts/2021-07-11-diffusion-models/DDPM.png" width="90%" alt="img01"/></center><br>  

* **Forward Diffusion**
<font size=3>&emsp;在$\mathbf{x}_0 \sim q(\mathbf{x})$這個前提下，如果對$\mathbf{x}_0$添加Gaussian noise，並重複 $T$ 次，生成出 $\mathbf{x}_1, ..., \mathbf{x}_T$ 一系列的加噪圖像。在 $T$ 夠大的情況下，最後生成出來的$\mathbf{x}_T$應該會趨近於Gaussian noise。以上操作數學上表示為：<br>
$$\begin{aligned}
    q(\mathbf{x}_t|\mathbf{x}_{t-1})
    &=\mathcal{N}(\mathbf{x}_t;\sqrt{1-\beta_t}\mathbf{x}_{t-1},\beta_t\mathbf{I}) \\
    &=\sqrt{1-\beta_t}\mathbf{x}_{t-1}+\sqrt{\beta_t}\epsilon\qquad\epsilon\sim\mathcal{N}(0, \mathbf{I})
    \end{aligned}$$ 
    &emsp;其中，每一步加噪的Gaussian noise強度由 $\{ \beta_t \in (0, 1) \}_{t=0}^{T}$ 控制，$\beta_t$ 會隨著 $T$ 的上升也跟這越來越大，另外 $\beta_t$ 也有很多種不同的schedule設計，包含linear、quadratic、cosine等等，會影響在 $T$ 個時間步階中圖像被加噪或去噪的趨勢。 
    &emsp;以上這個計算過程有一個好處，就是要得出 $\mathbf{x}_t$ 時，不需要真的把中間過程的每一張圖都算出來，而是可以透過reparameterize的方式簡化（<font color=#800000>**對計算過程沒興趣可以直接跳到下個紅字**</font>）：<br>
    令$\alpha_t=1-\beta_t$ 且 $\bar{\alpha}_t=\prod_{i=1}^t \alpha_i$
$$\begin{aligned}
\mathbf{x}_t 
&= \sqrt{\alpha_t}\mathbf{x}_{t-1} + \sqrt{1 - \alpha_t}\boldsymbol{\epsilon}_{t-1}\quad\text{ ;where } \boldsymbol{\epsilon}_{t-1}, \boldsymbol{\epsilon}_{t-2}, \dots \sim \mathcal{N}(\mathbf{0}, \mathbf{I}) \\
&= \sqrt{\alpha_t \alpha_{t-1}} \mathbf{x}_{t-2} + \sqrt{1 - \alpha_t \alpha_{t-1}} \bar{\boldsymbol{\epsilon}}_{t-2}  \quad\text{ ;where } \bar{\boldsymbol{\epsilon}}_{t-2} \text{ merges two Gaussians (*).} \\
&= \dots \\
&= \sqrt{\bar{\alpha}_t}\mathbf{x}_0 + \sqrt{1 - \bar{\alpha}_t}\boldsymbol{\epsilon} \\
q(\mathbf{x}_t \vert \mathbf{x}_0) &= \mathcal{N}(\mathbf{x}_t; \sqrt{\bar{\alpha}_t} \mathbf{x}_0, (1 - \bar{\alpha}_t)\mathbf{I})
\end{aligned}$$ 
    這個簡化之所以可以成立是因為兩個分布的和是這樣計算的：
    $$\mathcal{N}(\mathbf{0}, \sigma_1^2\mathbf{I})+\mathcal{N}(\mathbf{0}, \sigma_2^2\mathbf{I})=\mathcal{N}(\mathbf{0}, (\sigma_1^2 + \sigma_2^2)\mathbf{I})$$ 這使得 $\epsilon$ 項前面的係數（也就是標準差）能夠在推導的時候輕易的被合併。
    $$\sqrt{(1 - \alpha_t) + \alpha_t (1-\alpha_{t-1})} = \sqrt{1 - \alpha_t\alpha_{t-1}}$$ &emsp;這邊講了那麼多<font color=#800000>**其實結論就是**</font>，我們要在給定 $\mathbf{x}_0$ 時，求出 $\mathbf{x}_t$ 只需要做一次以下的計算就夠了：$$q(\mathbf{x}_t \vert \mathbf{x}_0)=\sqrt{\bar{\alpha}_t}\mathbf{x}_0 + \sqrt{1 - \bar{\alpha}_t}\boldsymbol{\epsilon}\qquad\alpha_t=1-\beta_t\text{ ; }\bar{\alpha}_t=\prod_{i=1}^t \alpha_i$$ &emsp;我看到有些人之所以會誤會DDPM的訓練過程，以為訓練需要真的做數百次加噪的迭代，就是因為不清楚這個結論可以直接把迭代過程一步到位。
</font>
    
* **Reverse Diffusion**
<font size=3>&emsp;如果我們可以逆轉上述流程，反過來用 $q(\mathbf{x}_{t-1} \vert \mathbf{x}_t)$ 去推測 $\mathbf{x}_{t-1}$，理論上就能夠從純粹的Gaussian noise $\mathbf{x}_T \sim \mathcal{N}(\mathbf{0}, \mathbf{I})$ 中一步步還原出真實圖像 $\mathbf{x}_0 \sim q(\mathbf{x})$。然而，人類很難用現有的數學知識解出 $q(\mathbf{x}_{t-1} \vert \mathbf{x}_t)$，所以這裡引入了一個神經網路 $p_\theta$ 來預測逆向的分布（在影像問題上，$p_\theta$ 目前幾乎都是Unet+Attention的網路結構）： $$p_\theta(\mathbf{x}_{0:T}) = p(\mathbf{x}_T) \prod^T_{t=1} p_\theta(\mathbf{x}_{t-1} \vert \mathbf{x}_t) \quad p_\theta(\mathbf{x}_{t-1} \vert \mathbf{x}_t) = \mathcal{N}(\mathbf{x}_{t-1}; \boldsymbol{\mu}_\theta(\mathbf{x}_t, t), \boldsymbol{\Sigma}_\theta(\mathbf{x}_t, t))$$ &emsp;在Foward Diffusion中，每個條件機率中的均值及方差都是已知的（透過人為指定的 $\beta_T$ 和 $\mathbf{x}_0$）；而Reverse Diffusion中，條件機率的均值及方差則是透過神經網路來推測。雖然 $q(\mathbf{x}_{t-1} \vert \mathbf{x}_t)$ 人類解不出來，但是在已知 $\mathbf{x}_0$ 及 $\beta_T$ 的條件下，$q(\mathbf{x}_{t-1} \vert \mathbf{x}_t, \mathbf{x}_0)$ 的均值與方差是能根據貝式定理推導出來的，首先定義：  $$q(\mathbf{x}_{t-1} \vert \mathbf{x}_t, \mathbf{x}_0) = \mathcal{N}(\mathbf{x}_{t-1}; \tilde{\boldsymbol{\mu}}(\mathbf{x}_t, \mathbf{x}_0), \tilde{\beta}_t \mathbf{I})$$
進一步的推導成高斯函數會變成： 
$$\begin{aligned} 
q(\mathbf{x}_{t-1} \vert \mathbf{x}_t, \mathbf{x}_0)
&= q(\mathbf{x}_t \vert \mathbf{x}_{t-1}, \mathbf{x}_0) \frac{ q(\mathbf{x}_{t-1} \vert \mathbf{x}_0) }{ q(\mathbf{x}_t \vert \mathbf{x}_0) } \\
&= \exp\Big( -\frac{1}{2} \big({(\frac{\alpha_t}{\beta_t} + \frac{1}{1 - \bar{\alpha}_{t-1}})} \mathbf{x}_{t-1}^2 - {(\frac{2\sqrt{\alpha_t}}{\beta_t} \mathbf{x}_t + \frac{2\sqrt{\bar{\alpha}_{t-1}}}{1 - \bar{\alpha}_{t-1}} \mathbf{x}_0)} \mathbf{x}_{t-1} +C(\mathbf{x}_t, \mathbf{x}_0) \big) \Big)
\end{aligned}$$
跳過一些複雜的數學推導過程，最後得到的 $\tilde{\boldsymbol{\mu}}$ 和 $\tilde{\beta}_t$ 解析解為：
$$\begin{aligned}
\tilde{\boldsymbol{\mu}}_t
&= \frac{1}{\sqrt{\alpha_t}} \Big( \mathbf{x}_t - \frac{1 - \alpha_t}{\sqrt{1 - \bar{\alpha}_t}} \boldsymbol{\epsilon}_t \Big) \\
\tilde{\beta}_t 
&= \frac{1 - \bar{\alpha}_{t-1}}{1 - \bar{\alpha}_t} \cdot \beta_t \\
\end{aligned}$$ 寫累了，後面有空再補</font>

## Reference
1. [What are Diffusion Models?](https://lilianweng.github.io/posts/2021-07-11-diffusion-models/) (這篇很細，數學推導過程很完整)  
2. [Diffusion Models：生成扩散模型](https://yinglinzheng.netlify.app/diffusion-model-tutorial/) (簡中的，寫的也還行)
