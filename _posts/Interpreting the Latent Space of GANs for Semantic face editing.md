# Interpreting the Latent Space of GANs for Semantic Face Editing

Conference/Journal: CVPR
Link: https://arxiv.org/abs/1907.10786
Publishing/Release Date: 2020
Score /5: ⭐️⭐️⭐️
Status: Finished
Summary: interpreting the Latent space of GANs for Semantic Face Editing
Type: GAN
reference: Interpreting%20the%20Latent%20Space%20of%20GANs%20for%20Semantic%205161bfc4ef0443ada8d95ff1702cf9e3/Interpreting_the_Latent_Space_of_GANs_for_Semantic_Face_Editing.pdf

<aside>
📱 GAN의 Latent Space Understanding에 관한 논문임. 잘 Training된 generative model의 경우 latent code를 linear transformation을 거치면 disentangled representation을 확인할 수 있고, facial attributes에 대해서 각 attributes들을 잘 decouple해서 정교하게 조절하는 방법을 보여줌.

</aside>

# 1. Main Contribution

1. PGGAN, StyleGan 등 기존 GAN의 latent space에서 구체적인 Attribute들이  Linear transformation을 통해서 Deisentagle 되는것을 확인함. 
2. fixed Pre-trained GAN model에서 latent code만을 변경하여 Semantic face editing을 수행하는 InterFaceGAN을 선보임.
3.  Real image의 latend code를 GAN inversion 혹은 encoder-involved model을 통해서 뽑아내고 interFaceGAN을 이용해 Editing함.

# 2. Framework of InterFaceGAN

- Well-trained GAN model의 latent space에서 나타나는 semantic attributes에 대한 엄밀한 분석진행.
- latent code의 semantics를 이용하여 facial attribute  editing을 수행하는 manipulation pipeline을 구성함.

## 2-1. Semantics in the Latent Space

- Generator
    
    $g : Z → X$ 
    
    $Z ⊆ R^d$ , Gaussian distribution $N (0, I_d)$ 를 따름.
    
    $X$ 는 image space를 나타내고, 각 $x$는 특정 semantic information을 가짐(성별, 나이, 안경유무 등)
    
- Semantic scoring function
    
     $f_S : X → S$
    
    $S ⊆ R^m$ ,  $m$ 개의 semantics에 대한 semantics space를 나타냄.
    
    $s = f_S(g(z))$
    
- **Property 1.**
    
     $n ∈ R^d$,  ${z ∈R^d : n^T z = 0}$ 은 hyperplane $R^d$을 정의하고, n은 normal vector임.  $n^T z > 0$  을 만족하는 모든 z는 hyperplane의 같은 면에 위치함.
    
    ![Untitled](Interpreting%20the%20Latent%20Space%20of%20GANs%20for%20Semantic%205161bfc4ef0443ada8d95ff1702cf9e3/Untitled.png)
    
    ![Untitled](Interpreting%20the%20Latent%20Space%20of%20GANs%20for%20Semantic%205161bfc4ef0443ada8d95ff1702cf9e3/Untitled%201.png)
    
- **Property 2.**
    
    ![Untitled](Interpreting%20the%20Latent%20Space%20of%20GANs%20for%20Semantic%205161bfc4ef0443ada8d95ff1702cf9e3/Untitled%202.png)
    
    → random sample z는 hyperplane의 근처에 위치할 것임. 
    

### Single Semantic

![Untitled](Interpreting%20the%20Latent%20Space%20of%20GANs%20for%20Semantic%205161bfc4ef0443ada8d95ff1702cf9e3/Untitled%203.png)

- 위 (1) 식은 부호가 존재하는 distance의 개념임.
- z가 hyperplane을 지나서 n에 가까워질수록 semantic score가 해당 방향으로 변함.
- z가 hyperplane의 반대방향으로 가면 -로 바뀌고, semantic attribute는 reverse됨.

![Untitled](Interpreting%20the%20Latent%20Space%20of%20GANs%20for%20Semantic%205161bfc4ef0443ada8d95ff1702cf9e3/Untitled%204.png)

- (2)에서 λ는 양수이고, 이는 semantic score와 distance가 linearly dependent 함을 의미함.

### Multiple Semantics

- m개의 different semantics가 있다고 가정함.

![Untitled](Interpreting%20the%20Latent%20Space%20of%20GANs%20for%20Semantic%205161bfc4ef0443ada8d95ff1702cf9e3/Untitled%205.png)

- Λ = diag(λ1, . . . , λm)  (mxm)
- N = [n1,...,nm] (dxm)
- z → (dx1), ~$N (0, I_d)$
- s → (mx1)

![Untitled](Interpreting%20the%20Latent%20Space%20of%20GANs%20for%20Semantic%205161bfc4ef0443ada8d95ff1702cf9e3/Untitled%206.png)

→ $s ∼ N (0, Σs)$

$Σs$ 가 diagonal matrix라면 s의 각 요소들은 disentangle함.

→ 그러기 위해서는 {n1, ..., nm}이 서로 orthogonal이어야함.

→ orthogonal이 아니라면 m의 각 요소들이 correlate되어있는 것.

→  $n^T_i n_j$ 를 통해 entanglement를 측정가능

## 2-2. Manipulation in the Latent Space

### Single Attribute Manipulation

![Untitled](Interpreting%20the%20Latent%20Space%20of%20GANs%20for%20Semantic%205161bfc4ef0443ada8d95ff1702cf9e3/Untitled%207.png)

- $f(g(z _ e )) = f(g(z)) + λα$ , alpha가 양수일 경우 semantic을 더 positive하게 만들것이고, 음수일 경우 더 negative하게 만듦.

### Conditional Manipulation

- 각 n들이 서로 correlate되어있을 경우, 한 Semantic만 조작하는 것이 쉽지 않기 때문에, conditional manipulation을 가해서 $N^T N$ 을 diagonal하게 만듦.
    
    ![Untitled](Interpreting%20the%20Latent%20Space%20of%20GANs%20for%20Semantic%205161bfc4ef0443ada8d95ff1702cf9e3/Untitled%208.png)
    
- 각각 n1, n2의 normal을 가진 두 hyperplane이 있을때 attribute2에 영향 받지않고 attribute1만 바꿔주기 위해서는 $n1 − (n^T_1 n_2)n_2$방향으로 sample을 움직임.
- 2개 이상이 correlation되어있을땐 primal direction부터 correlate된 모든 것들을 재구성함.

### Real Image Manipulation

- 실제 이미지로부터 latent code를 구하기 위해서 reconstruction loss를 직접적으로 optimize하는 방법이 있고, image를 다시 latent space로 mapping해주는 encoder를 학습시키는 방법이 있음.

# 3. Experiments

## 3-1. Latent Space Separation

- 실제로 5개의 feature (m=5)에 대해서 labeling을 하고, latent space에서 svm으로 해당 feature를 만들어내는 latent code들을 separate했을 때 높은 성능으로 구분되는 것을 확인함.
    
    ![Untitled](Interpreting%20the%20Latent%20Space%20of%20GANs%20for%20Semantic%205161bfc4ef0443ada8d95ff1702cf9e3/Untitled%209.png)
    
    - 위 Figure 3에서 이상한 그림들이 보이는 것은, hyperplane 근처의 일반적인 사례가 아니라, normal direction의 방향으로 latent code를 infinite하게 보낸 경우임.

 

## 3-2. Latent Space Manipulation

- Pose 등에서 interpolation한것처럼 각 방향을 잘 학습한 것을 확인할 수 있고, 데이터 상에서 안보이는 각도도 represent할 수 있다는 것을 확인함.
- 육안상 보이는 Bad result등과 그렇지 않은 result 등을 구별하는 hyperplane도 만들 수 있었고, 실제로 Bad result에서 멀어지는 방향으로 이동하면 좋은 결과들이 많이 나옴.

## 3-3. Conditional Manipulation

### Correlation between Attributes

- Correlation을 확인하는 metrics으로 두가지를 사용.
1. Cosine Similarity
    
    ![Untitled](Interpreting%20the%20Latent%20Space%20of%20GANs%20for%20Semantic%205161bfc4ef0443ada8d95ff1702cf9e3/Untitled%2010.png)
    
    - n1과 n2는 unit vector임.
2. Correlation coefficient
    
    ![Untitled](Interpreting%20the%20Latent%20Space%20of%20GANs%20for%20Semantic%205161bfc4ef0443ada8d95ff1702cf9e3/Untitled%2011.png)
    
    ![Untitled](Interpreting%20the%20Latent%20Space%20of%20GANs%20for%20Semantic%205161bfc4ef0443ada8d95ff1702cf9e3/Untitled%2012.png)
    
    - Pose, Smile은 correlate가 잘 안되어있는데, Age, Gender, Eyeglasses는 상당히 Correlate 되어있음. 이는 Dataset의 분포 때문임.

![Untitled](Interpreting%20the%20Latent%20Space%20of%20GANs%20for%20Semantic%205161bfc4ef0443ada8d95ff1702cf9e3/Untitled%2013.png)

- Figure 7.은 실제로 correlate 되어있는 feature들을 위의 Figure 2.  방법으로 manipulation한 결과임.

## 3-4. Results on Style GAN

- Style GAN은 앞에서 보여준 PGGAN의 예시와 다르게, z가 Generator로 들어가기 전에 W로 한번더 mapping되는 과정이 있음.
- 이 과정에서 disentangled representation이 더 강해짐.

![Untitled](Interpreting%20the%20Latent%20Space%20of%20GANs%20for%20Semantic%205161bfc4ef0443ada8d95ff1702cf9e3/Untitled%2014.png)

- Figure 9.은 Style GAN에서 Age에 대해서 manipulation한 결과인데, Age와 eyeglasses는 크게 correlate되어있고, W space와 z space 각각을 보았을땐 W space의 manipulation결과가 더 좋은 것을 볼 수 있음.
- 마지막 row의 결과는 figure 2의 방법으로 z space에서 eyeglasses는 고정시키고 Age만 변경시킨 결과임.
- 위 Conditional Manipulation 실험을 W space에서 했을때는 결과가 좋지 않았는데, train Data의 age와 eyeglasses feature가 correlation이 높은 상황에서 z에서 W space로 mapping 될때 entanglement가 굉장히 강해져서 둘의 normal이 거의 동일한 방향이되고, 뺏을땐 거의 0이 되기 때문.

## 3-5. Real Image Manipulation

- 두가지 방향성 존재.
    - 첫번째는 latent code를 generator에 넣고 나온 image를 실제 image와 pixel-wise reconstruction error을 계산해 optimization하는 방법.
    - 두번째는 image에서 latent space로 가는 inverse mapping을 하는 encoder를 추가로 학습하는 방법.
- 이미 학습되어있는 PGGAN 을 기반으로 했을땐 원하는 방향으로 feature가 manipulation 되기는 했지만, reconstruction quality가 좋지는 않았음. style GAN의 경우는 각 layer에 들어가는 w를 optimization target으로 삼았고, 실제 editing 시에 모든 style codes를 smae direction으로 push하여 좋은 성능을 보이는 것을 확인함.
- encoder-decoder based generative model에서도 적용되는 것을 확인함.

# 4. Conclusion

- GAN의 latent space에 encode된 semantics를 이해할 수 있었고, unconditional GAN을 controllable GAN으로 바꿀수도 있다는 것을 보임.
- 또한 real Image에도 적용가능함을 보임.