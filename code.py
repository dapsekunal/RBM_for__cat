

import pandas as pd
import torch
import torch.nn as nn
import torch.nn.parallel

import torch.utils.data

x_dataset = pd.read_csv(r"C:\Users\Kunal Dapse\Downloads\AI-DataTrain.csv")
y_dataset = pd.read_csv(r"C:\Users\Kunal Dapse\Downloads\AI-DataTest.csv")
training_dataset = x_dataset.iloc[:900,1:11].values
test_dataset     =x_dataset.iloc[900:,1:11].values

training_dataset = torch.FloatTensor(training_dataset)
test_dataset = torch.FloatTensor(test_dataset)


class RBM(nn.Module):
    def __init__(self, nv, nh):
              super(RBM,self).__init__()
              
              self.W = nn.Parameter(torch.randn(nh,nv))
              self.a = nn.Parameter(torch.randn(1,nh))
              self.b = nn.Parameter(torch.zeros(1,nv))
    
    def forward(self,input):
              out = self.layer(input)
              return out
    def sample_h(self, x):
        wx = torch.mm(x, self.W.t())
        activation = wx + self.a.expand_as(wx)
        p_h_given_v = torch.sigmoid(activation)
        return p_h_given_v, torch.bernoulli(p_h_given_v)
    def sample_v(self, y):
        wy = torch.mm(y,self.W)
        activation = wy + self.b.expand_as(wy)
        p_v_given_h = torch.sigmoid(activation)
        return p_v_given_h, torch.bernoulli(p_v_given_h)
    def train(self, v0, vk, ph0, phk):
        with torch.no_grad():
            
            self.W += (torch.mm(v0.t(), ph0) - torch.mm(vk.t(), phk)).t()
            self.b += torch.sum((v0-vk), 0)
            self.a += torch.sum((ph0-phk), 0)
    
   
nv = len(training_dataset[0])
nh = 100
batch_size = 100
rbm = RBM(nv, nh)


#Training the RBM
nb_epoch = 10
for epoch in range(1, nb_epoch + 1):
    train_loss = 0
    s = 0.
    for id_examinee in range(0,1000 -  batch_size, batch_size):
        vk = training_dataset[id_examinee:id_examinee + batch_size]
        v0 = training_dataset[id_examinee:id_examinee + batch_size]
        ph0,_ =rbm.sample_h(v0)
        for k in range(10):
            _,hk = rbm.sample_h(vk)
            _,vk = rbm.sample_v(hk)
        phk,_ = rbm.sample_h(vk)
        rbm.train(v0, vk, ph0, phk)
        train_loss += torch.mean(torch.abs(v0 - vk))
        s += 1.
    print('epoch: '+ str(epoch)+' loss: '+str(train_loss/s))

#Testing the RBM

test_loss = 0
s = 0.
for id_examinee in range(0, 1000):
    v = training_dataset[id_examinee:id_examinee+1]
    vt = test_dataset[id_examinee:id_examinee +1]

    if len(vt[vt>=0]) > 0:
       _,h = rbm.sample_h(v)
       _,v = rbm.sample_v(h)

       test_loss += torch.mean(torch.abs(vt[vt>=0] - v[vt>=0]))
       s += 1.
print('***test loss: '+str(test_loss/s))

def get_weights(self):
    
            return self.W
  
    
Questions = ['Q1', 'Q2', 'Q3','Q4','Q5', 'Q6', 'Q7', 'Q8', 'Q9', 'Q10']
#model_1 = RBM(nv,nh) ## ERROR!!!

Weight = rbm.W.detach().numpy()
Weight = Weight.ravel()
df = pd.DataFrame.from_dict({'Questions':Questions,
                   'Weight': Weight
                   },orient='index').transpose()

writer = pd.ExcelWriter("output.xlsx", engine='xlsxwriter')

df.to_excel(writer, sheet_name = 'Sheet1', startrow=1, header=False)
workbook = writer.book
worksheet = writer.sheets['Sheet1']

header_format = workbook.add_format({
    'bold': True,
    'text_wrap': True,
    'valign': 'top',
    'fg_color': '#D7E4BC',
    'border': 1})
for col_num, value in enumerate(df.columns.values):
    worksheet.write(0, col_num + 1, value, header_format)

writer.save()