from tsai.all import TSClassification,TSStandardize,TSClassifier


def model_train(X, Y, splits):
        tfms = [None, TSClassification()]
        batch_tfms = TSStandardize()
        fcst2 = TSClassifier(X, Y,splits=splits, path='models', arch="TSiTPlus", tfms=tfms, batch_tfms=batch_tfms, metrics=accuracy, cbs=ShowGraph())
        fcst2.fit_one_cycle(10, 1e-3)
        fcst2.export("TSiTPlus.pkl")
