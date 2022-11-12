class encoder():

    def __init__(self):
        self.a = 1

class actor():

    def __init__(self, b):
        self.b = b

policy = []
for p in range(5):
    po = actor(p)
    policy.append(po)

en_model = encoder()

class trainer():

    def __init__(self, en, p):

        self.en = en
        self.policy = p

train = []
for i in range(5):
    tt = trainer(en_model, policy[i])
    train.append(tt)

print(train[0].en.a)
train[0].en.a = 100
print(train[0].en.a)
print(train[4].en.a)