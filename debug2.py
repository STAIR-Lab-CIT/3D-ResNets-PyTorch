batch_size=3
n_classes=5
outputs=torch.randint(0,9,(9*batch_size,5))
print(outputs)
print("-----------")
res = []
indx_ = [i*batch_size for i in range(9)]
indx = torch.LongTensor(indx_)
for ii in range(batch_size):
    pos = torch.argmax(outputs[indx,:])
    row = pos/n_classes
    print(ii+row*batch_size)
    res.append(ii+row*batch_size)
    indx=indx+1
print(res)

