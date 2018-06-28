[<h1 align = "center">:rocket: 第三届魔镜杯大赛：相似问匹配 :facepunch:</h1>][1]

---
## [1. Baseline][2]
- LSTM
- CNN

---
## 2. 特征工程
- bleu（机器翻译指标）：对两个句子的共现词频率计算
- n-grams

- question(leak): 
  - tf: q1/q2/q1+q2
  - tfidf: q1/q2/q1+q2
  
- words(chars): 针对字符串计算
  - 词数/重叠词数（去重）
  - 相同度（相异度 = 1 - 相同度）: com / (q1 + q2 - com)每个状态分量根据目标设置最优权重
  - 杰卡顿
  - simhash
  - 对目标影响大的词（lstm状态差等）
  - 编辑距离
    - fuzz.QRatio
    - fuzz.WRatio
    - fuzz.partial_ratio
    - fuzz.token_set_ratio
    - fuzz.token_sort_ratio
    - fuzz.partial_token_set_ratio
    - fuzz.partial_token_sort_ratio
    - ...
  
- tfidf(tf): 针对数值计算
  - cosine（修正）
  - 欧式距离
  - 雅克比

- WordVectors:
  - wmd
  - norm_wmd
  - cosine

- lda




---
[1]: https://ai.ppdai.com/mirror/goToMirrorDetail?mirrorId=1
[2]: https://github.com/Jie-Yuan/PpdaiQuestionPairsMatching/tree/master/Baseline
