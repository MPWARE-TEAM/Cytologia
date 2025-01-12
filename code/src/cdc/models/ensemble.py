import torch
import torch.nn as nn
import pytorch_lightning as L


# Load ensemble model
# TTA is a list of pairs [(tta, optional tta back), ...]
class EnsembleAverage(L.LightningModule):  # nn.Module
    def __init__(self, models, config=None, tta=None, verbose=True):
        super(EnsembleAverage, self).__init__()
        self.models = models
        self.config = config
        self.tta = tta
        if verbose is True:
            print("Ensemble of %d model(s), tta=%s" % (len(self.models), self.tta))

    def eval(self):
        for i, m in enumerate(self.models):
            m.eval()

    def to(self, device):
        for i, m in enumerate(self.models):
            m.to(device)

    def execute_model(self, model, data):
        logits_ = model(data)
        return logits_

    def forward_model_(self, model_, data):
        if self.tta is not None:
            outputs_ = [self.execute_model(model_, data)]
            for tta_ in self.tta:
                if len(tta_) == 2:
                    outputs_.append(tta_[1](self.execute_model(model_, tta_[0](data))))
                else:
                    outputs_.append(self.execute_model(model_, tta_[0](data)))
            output_ = torch.stack(outputs_)
            output_ = torch.mean(output_, dim=0)
        else:
            output_ = self.execute_model(model_, data)
        return output_

    def forward_models_mean_(self, data):
        outputs = [self.forward_model_(m, data) for m in self.models]
        output = torch.stack(outputs)
        output = torch.mean(output, dim=0)
        return output

    def forward_nn(self, data):
        if len(self.models) > 1:
            output = self.forward_models_mean_(data)
        else:
            output = self.forward_model_(self.models[0], data)

        return output

    def predict_step(self, batch, batch_idx, dataloader_idx=0):
        X = batch['image']
        logits = self.forward_nn(X)
        return logits
