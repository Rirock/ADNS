from custompackage.load_architecture import simple_fcnn, ktree_gen
import torch

ina = torch.rand(5, 10)
input_dim = ina.size(-1)
model = ktree_gen(ds='other', Activation="relu", Sparse=False,
                 Input_order=None, Repeats=1, Padded=False, Input_size=input_dim)

out = model(ina)
print(out)