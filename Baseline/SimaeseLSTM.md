http://www.mit.edu/~jonasm/info/MuellerThyagarajan_AAAI16.pdf

```python
#相似度计算
l1 = lambda x: K.sum(K.abs(x), axis=-1, keepdims=True)

def manhattan(a, b, scale=True):
    dist = l1(a - b)
    if scale:
        dist = K.exp(-dist)
    return dist


#输入层
left_input = Input(shape=(None,))
right_input = Input(shape=(None,))

#对句子embedding
encoded_left = embedding_layer(left_input)
encoded_right = embedding_layer(right_input)

#两个LSTM共享参数
shared_lstm = LSTM(128)

left_output = shared_lstm(encoded_left)
right_output = shared_lstm(encoded_right)

dist = Lambda(lambda x: manhattan(*x), output_shape = lambda x: (x[0][0], 1))([left_output, right_output])  # lambda x: (None, 1)

# model
model = Model([left_input, right_input], [dist])
model.compile(loss='binary_crossentropy', optimizer=Adam(), metrics=['accuracy'])

model.summary()
model.fit([bow_x, bow_y], train.label.values, batch_size=1280, epochs=10, validation_split=0.25)
```
