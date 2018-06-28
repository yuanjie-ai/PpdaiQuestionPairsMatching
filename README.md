[<h1 align = "center">:rocket: 第三届魔镜杯大赛：相似问匹配 :facepunch:</h1>][1]

---
## [1. Baseline][2]
- LSTM
- CNN

---
## 2. 特征工程
> qq匹配
- bleu（机器翻译指标）：对两个句子的共现词频率计算
- n-grams

- chars(words)
> 针对字符串计算
  - 公共字数
  - 最大公共序列（长度）
  - 编辑距离
  - 杰卡顿
  - simhash
  - 相同度（相异度 = 1 - 相同度）: com / (q1 + q2 - com)每个状态分量根据任务设置最优权重
  

- tfidf(tf)
> 针对数值向量计算
  - cosine（修正）
  - 欧式距离
  - 雅克比
  - wmd
  
- lda


---
[1]: https://ai.ppdai.com/mirror/goToMirrorDetail?mirrorId=1
[2]: https://github.com/Jie-Yuan/PpdaiQuestionPairsMatching/tree/master/Baseline
