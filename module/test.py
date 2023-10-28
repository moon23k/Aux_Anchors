import time, math, torch, evaluate



class Tester:
    def __init__(self, config, model, tokenizer, test_dataloader):
        super(Tester, self).__init__()
        
        self.model = model
        self.tokenizer = tokenizer
        self.dataloader = test_dataloader

        self.task = config.task
        self.bos_id = config.bos_id
        self.pad_id = config.pad_id
        self.device = config.device
        self.max_len = config.max_len
        
        self.metric_name = 'BLEU' if self.task == 'translation' else 'ROUGE'
        self.metric_module = evaluate.load(self.metric_name.lower())
        


    def test(self):
        score, aux_score = 0.0, 0.0         
        self.model.eval()

        with torch.no_grad():
            for batch in self.dataloader:
                x = batch['x'].to(self.device)
                y = batch['y'].to(self.device)

                pred = self.predict(x)
                scores = self.evaluate(pred, y)
                score += scores[0]
                aux_score += scores[1]

        txt = f"TEST Result on {self.task.upper()} Task\n"
        txt += f"-- {self.metric_name} Score: {round(score/len(self.dataloader), 2)}\n"
        txt += f"-- First Token Prediction Acc: {round(aux_score/len(self.dataloader), 2)}\n"
        print(txt)


    def tokenize(self, batch):
        return [self.tokenizer.decode(x) for x in batch.tolist()]


    def predict(self, x):

        batch_size = x.size(0)
        pred = torch.zeros((batch_size, self.max_len)).fill_(self.pad_id)
        pred = pred.type(torch.LongTensor).to(self.device)
        pred[:, 0] = self.bos_id

        e_mask = self.model.pad_mask(x)
        memory = self.model.encoder(x, e_mask)

        for idx in range(1, self.max_len):
            y = pred[:, :idx]
            d_mask = self.model.dec_mask(y)
            d_out = self.model.decoder(y, memory, e_mask, d_mask)

            logit = self.model.generator(d_out)
            pred[:, idx] = logit.argmax(dim=-1)[:, -1]

        return pred



    def evaluate(self, pred, label):
        aux_score = (pred[:, 1] == label[:, 1]).sum().item() / pred.size(0) * 100

        pred = self.tokenize(pred)
        label = self.tokenize(label)

        #For TRANSLATION Evaluation
        if self.task == 'translation':
            score = self.metric_module.compute(
                predictions=pred, 
                references =[[l] for l in label]
            )['bleu'] * 100
        #For DIALOGUE & SUMMARIZATION Evaluation
        else:
            score = self.metric_module.compute(
                predictions=pred, 
                references =[[l] for l in label]
            )['rouge2'] * 100

        return score, aux_score