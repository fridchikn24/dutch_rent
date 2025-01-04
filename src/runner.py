from model.model_service import Model_Service

model_svc = Model_Service()
model_svc.load_model()
print(model_svc.predict([85, 2015, 2, 2, 20, 1, 1, 0, 0, 1]))