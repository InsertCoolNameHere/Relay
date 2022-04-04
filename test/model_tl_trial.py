from model.DeepNNTrainer import DeepModel

old_model = DeepModel(5, 15)
modules=list(old_model.children())

print(modules)
print(modules[:-1])