# Gradient Descent: The Ultimate Optimizer
**概要（一言）**

- 勾配法のパラメタを勾配法で決定するとパラメタの初期依存性が解消していい感じ！

**なぜべす**

- ステップサイズなどのパラメタ調整における初期依存性の解消
- パラメタの勾配を取る方法はあったが、自動微分は利用しておらず、手で計算して求めていた。
        →勝手にパラメタをいい感じにしてくれる
- コードも公開してくれていて親切！

**本論文での提案とやったこと**

- 誤差逆伝播を用いてハイパーグラジエントを自動的に計算する方法
- ハイパラのハイパラの・・ハイパラに再帰的に適用可能。
    →繰り返すとハイパラの初期値依存性が低くなる。
- MLP、CNN、RNNで実験
- Pytorch実装



# イントロ
- 深層学習の学習では、勾配降下法を利用
    - 今まで（の多くの場面では）、ステップサイズ$\alpha$ を固定（大きすぎても小さすぎてもだめ）
        - 適切なステップサイズってどうやって計算すればよい？
            →Baydinら（2018）は、標準的なオプティマイザの更新ルールを手動で微分
        - Baydinら（2018）のいまいちポイント
            - オプティマイザーのバリエーションごとにやり直す必要がある
            - この方法はステップサイズのハイパーパラメーターだけを調整し、他のハイパーパラメーターを調整しない（できない）。
            - ハイパラのステップサイズはどうするの？？という疑問も。
    - 本論文では、Baydinら（2018）の手動による微分を自動微分（AD）に置き換えることで、
        - **人手をかけずに正しい微分を自動的に計算し、他のハイパーパラメータにも一般化できる。また、ハイパーハイパーパラメータ、ハイパーハイパーパラメータなどの最適化にも適用できる**
            - →ネストするとパラメタの初期値依存性の軽減


# 2 Implementing hyperoptimizers

**記号を定義**

- $f$
    - loss function
- $w_i$
    - step $i$におけるウェイトの初期値

**step** $i$**におけるSGD**
$w_{i+1}=w_i - \alpha \frac{\partial f(w_i)}{\partial w_i}$


$\alpha$も調整するならば以下の感じ。

![](https://paper-attachments.dropboxusercontent.com/s_476291B13A7F991FF322F070EB34167EB8257D8EA23E84B9C784FFBC10673CB4_1685263048698_image.png)


具体的にどのように$\alpha$を最適化するか？hyper-step $\kappa$ を用いて$\alpha$を更新。

![](https://paper-attachments.dropboxusercontent.com/s_476291B13A7F991FF322F070EB34167EB8257D8EA23E84B9C784FFBC10673CB4_1685263111459_image.png)


$\partial f(w_i)/\partial \alpha_i$を求めるのをどうやるか→先行研究Baydin et al. (2018)の方法を見てみる！

**2.1 Computing the step-size update rule by hand(Baydin et al., 2018)**
[chain rule](https://ja.wikipedia.org/wiki/%E9%80%A3%E9%8E%96%E5%BE%8B)を適用（$f \leftarrow g \leftarrow x$ という関係性で、$f$を$x$で微分するとき）

![](https://paper-attachments.dropboxusercontent.com/s_476291B13A7F991FF322F070EB34167EB8257D8EA23E84B9C784FFBC10673CB4_1685264254947_image.png)


同様に

![](https://paper-attachments.dropboxusercontent.com/s_476291B13A7F991FF322F070EB34167EB8257D8EA23E84B9C784FFBC10673CB4_1685264330683_image.png)


導出
$w_i = w_{i-1} - \alpha_i \frac{\partial f(w_{i-1})}{w_{i-1}}$。
上式より、$f \leftarrow w \leftarrow \alpha$となっていることがわかるので、そのままchain ruleを適用。(3)式成立。
(4)はそのままかっこの中身を微分するだけ。($w_{i-1}$ and $f(w_{i-1})$は$\alpha_i$には依らない）

より具体的に計算しようとするとかなり大変。
ADAMでは

![](https://paper-attachments.dropboxusercontent.com/s_476291B13A7F991FF322F070EB34167EB8257D8EA23E84B9C784FFBC10673CB4_1685265457368_image.png)


[ご参考](https://www.hello-statisticians.com/explain-terms-cat/adam1.html#Adam-3)


# 実験
- 先行研究（Maclaurin et al., 2015; Baydin et al., 2018）と同じようにMNISTで実験を行った。
- サイズ128の1つの完全連結隠れ層を持つニューラルネットワーク
-  tanhの活性化
- 256のバッチサイズ
- 30エポック学習させ、3回実行
- ベースラインとして、α=0.01のSGDを使用
![](https://paper-attachments.dropboxusercontent.com/s_476291B13A7F991FF322F070EB34167EB8257D8EA23E84B9C784FFBC10673CB4_1686492618148_image.png)


slashの後はパラメタ最適化のモデル。(赤字）はハイパーパラメタ最適化で得られたパラメタを固定パラメタとして与えた場合について。



## CNNでの実験
- ResNet-20をCIFAR-10で学習。
- He at al(2016)はdecayのスケジュールを実験的に求めたらしい
- ハイパーパラメタの初期値
    - 
![](https://paper-attachments.dropboxusercontent.com/s_476291B13A7F991FF322F070EB34167EB8257D8EA23E84B9C784FFBC10673CB4_1686493559241_image.png)


どの初期値でもハイパーオプティマイザーはbaselineを超えている。。また、He at elが見つけたdecayの方法と同じような減衰スケジュール


## RNN(省略）


## Higher-Order Hyperoptimization
- 「ハイパーパラメタの最適化」の「ハイパーパラメタの最適化」→height2
    - heightが高くなるとハイパーパラメタの初期値依存性が弱くなる。
    - 高さを1つ増やすのは、実行時間の1-2%の増加
![](https://paper-attachments.dropboxusercontent.com/s_476291B13A7F991FF322F070EB34167EB8257D8EA23E84B9C784FFBC10673CB4_1686494289118_image.png)



# 背景(ちょこっとフォーマルな記述）
## 損失最小化問題
- データ$z \in \mathbb{R}$
- $\mathbf{x} \in \mathbb{R}^d$
- 微分可能な非凸損失関数$l(\mathbf{x},z)$

**期待損失最小化問題**
目的関数：$f(\mathbf{x}):= \mathbb{E}_{z \sim \mathcal{D}}[l(\mathbf{x};z)]$
条件：$\mathbf{x} \in \mathbb{R}^d$

**経験損失最小化問題**
目的関数：$f(\mathbf{x};S):= \frac{1}{n}\sum_{i=1}^n l(\mathbf{x};z_i)= \frac{1}{n}\sum_{i=1}^n l_i(\mathbf{x})$
条件：$\mathbf{x} \in \mathbb{R}^d$
where $l_i$はi番目の訓練データ$z_i$に関する損失関数。

深層学習における最適化法について（多くの場合）以下を満たす損失関数$f$の$\mathbf{x} \in \mathbb{R}^d$における**確率的勾配**$G_{\xi}(\mathbf{x})$を利用する。（$G_{\xi_{k,i}}= \nabla l_{\xi_{k,i}}$）

1. 損失関数の微分可能
    1. 損失関数は連続的微分可能（つまり、$C_1$級）
2. 確率的勾配の不偏性
    1. 最適化ステップで生成される点列$(\mathbf{x}_k)_{k\in \mathbb{N}} \subset \mathbb{R}^d$とすると、任意の$k \in \mathbb{N}$に対して、$\mathbb{E}_{\xi_k}[G_{\xi_k}(\mathbf{x_k})]=\nabla f(\mathbf{x_k})$
        が成立。各$\xi_i$は独立標本。確率変数$\xi_k$と点列$(\mathbf{x}_k)_{l=0}^k$は独立。
3. 確率的勾配の分散
    1. ある非負実数$\sigma^2$が存在して、任意の自然数$k$に対して、
        $\mathbb{E}_{\xi_k}[ \| G_{\xi_k}(\mathbf{x_k})] - \nabla f(\mathbf{x_k})\|^2] \leq \sigma^2$
4. ミニバッチ確率的勾配の計算
    1. 各反復回数$k$に対して、大きさ$b$のミニバッチ$B_k$を用いて、勾配$\nabla f$を
        $\nabla f_{B_k}(\mathbf{x}_k):=\frac{1}{b}\sum_{i=1}^b G_{\xi_{k,i}}(\mathbf{x}_k)$
        ここで$\xi_{k,i}$は反復回数kでのi番目の標本によって生成される確率変数

ミニバッチ確率的勾配は以下のように表現できる。
$\nabla f_{B_k}(\mathbf{x}_k):=\frac{1}{b}\sum_{i=1}^b \nabla l_{\xi_{k,i}}(\mathbf{x}_k)$


## **確率的勾配降下法**
- 初期点 $\mathbf{x}_0 \in \mathbb{R}^d$
- ステップサイズ
    - $\alpha_k > 0$
    - mini batch size $b>0$
- 探索方向　$\mathbf{d}_k := - \nabla f_{B_k}(\mathbf{x}_k)$
- $\mathbf{x}_{k+1} = \mathbf{x}_{k} - \alpha_k \nabla f_{B_k}(\mathbf{x}_k) = \mathbf{x}_{k} - \frac{\alpha_k}{b}\sum_{i=1}^b G_{\xi_{k,i}}(\mathbf{x}_k)=\mathbf{x}_{k} - \frac{\alpha_k}{b}\sum_{i=1}^b\nabla l_{\xi_{k,i}}(\mathbf{x}_k)$ 


**定数ステップサイズを利用した確率的勾配降下法の収束解析**
$f \in C^1_L (\mathbb{R}^d)$は$\mathbb{R}^d$で下に有界であるとし、$\nabla f$のリプシッツ定数を$L>0$とする。定数ステップサイズ$\alpha \in (0, 2/L)$に対して、SGDで生成される点列$(\mathbf{x}_k)_{k\in \mathbb{N}}$は任意の整数$K \geq 1$に対して、
$\text{min}_{k\in [0:K-1]} \mathbb{E}[\| \nabla f(\mathbf{x}_k) \|^2] \leq \frac{2 \mathbb{E}[f(x_0) - f_\star]}{(2-L\alpha) \alpha K}+\frac{L \sigma^2 \alpha}{(2-L\alpha)b}$　を満たす。$f_{\star}$は$f$の有限な下限値。

証明しようと思いましたが時間不足・・・お許しを・・・


# Pytorchのdetachって何？
https://qiita.com/tttamaki/items/28f13a1507eb63387901


〇計算グラフを切る

- detach()
- with no_grad

〇勾配を計算しないだけ（計算グラフは作ってる）

- require_grad =True


こんな計算

    # a, b, c, d, e, f, x, y, z, wに対して、requires_grad=Trueとする。
    a = torch.tensor(2., requires_grad=True)
    ・・・
    x = torch.tensor(17., requires_grad=True)
    
    # 勾配を初期化（普通のNNならoptimizer.zero_grad()とか書いてる）
    a.grad.data = torch.torch(0.0)
    ・・・
    x.grad.data = torch.torch(0.0)
    
    ####式①#####
    y = a * x + b
    z = c * y + d
    w = e * z + f
    w.backward() # wを微分

$\frac{\partial w}{\partial e}=z, \cdots, \frac{\partial w}{\partial x}=aec$ のようになる。

yをdetach

    ####式②#####
    y = a * x + b
    z = c * y.detach() + d
    w = e * z + f

$\frac{\partial w}{\partial a}=0,\frac{\partial w}{\partial x}=0 , \frac{\partial w}{\partial b}=0$のように初期値となる。一方で$\frac{\partial w}{\partial c}=ey$のようになる。



    ####式③#####
    y = a * x + b
    with torch.no_grad():
        z = c * y + d
    w = e * z + f
    w.backward()

z以降の変数に対して計算グラフなし。
`e`と`f`までは勾配が計算されていますが，`c, d, a, b, x`の勾配は計算されない。


# コードどうやって書けばよいの！！

**普通のSGD(BEFORE)**

![](https://paper-attachments.dropboxusercontent.com/s_476291B13A7F991FF322F070EB34167EB8257D8EA23E84B9C784FFBC10673CB4_1686673237948_image.png)

- 1行目 
    - $dw$は定数的使い方。→detach()
- 2行目
    - w.detach()は定数的使い方→detach()
    - **alphaも定数的使い方→detach()**
        - ここが大事！ SDGのパラメタに対しては計算グラフを作成しない
- コードの外
    - loss.backward()はやっている。すなわち$\frac{\partial f(w)}{\partial w}$はやっている。


Hypergradient **SGD(AFTER)**

![](https://paper-attachments.dropboxusercontent.com/s_476291B13A7F991FF322F070EB34167EB8257D8EA23E84B9C784FFBC10673CB4_1686673437359_image.png)

- 「ハイパーハイパーパラメタ（ステップサイズ：d_alpha）」は定数的使い方。（階層的に深くしたいならばdetachをはずす）
- 
    - $dw$は定数的使い方。→detach()
- 2行目
    - w.detach()は定数的使い方→detach()
    - alphaは変数的な使い方で勾配計算を実施
        - $\frac{\partial f(w, \alpha)}{\partial \alpha}$を計算している。
![](https://paper-attachments.dropboxusercontent.com/s_476291B13A7F991FF322F070EB34167EB8257D8EA23E84B9C784FFBC10673CB4_1686674257212_image.png)


