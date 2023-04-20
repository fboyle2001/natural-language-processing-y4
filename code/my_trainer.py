import torch
import torch.utils.data.sampler
import evaluate
import numpy as np
import matplotlib.pyplot as plt
import random
import json 
import time
from datetime import datetime
import os
from sklearn import metrics
from tqdm.notebook import tqdm
from scipy.special import softmax
import transformers

class WeightedDatasetSampler(torch.utils.data.sampler.Sampler):
    def __init__(self, dataset, label_weights=None):
        self.sample_labels = [x.item() for x in dataset.labels.squeeze()]
        self.unique_labels = sorted(list(set(self.sample_labels)))
        
        print("Unique labels", self.unique_labels)
        
        self.label_indices = [[] for _ in self.unique_labels]
        
        for idx, sample_label in enumerate(self.sample_labels):
            self.label_indices[sample_label].append(idx)
        
        label_sample_distribution = label_weights
        
        if callable(label_weights):
            label_sample_distribution = [label_weights(len(self.label_indices[unique]), len(self.sample_labels)) for unique in self.unique_labels]
            print("Calculated balanced with lambda")
        elif label_weights is None:
            label_sample_distribution = [len(self.label_indices[unique]) / len(self.sample_labels) for unique in self.unique_labels]
            print("Calculated raw distribution")
        elif label_weights == "balanced":
            label_sample_distribution = [1 for unique in self.unique_labels]
            print("Calculated balanced")
        elif label_weights == "inverse":
            label_sample_distribution = [len(self.sample_labels) / len(self.label_indices[unique]) for unique in self.unique_labels]
            print("Calculated inverse")
        elif label_weights == "log_inverse":
            log_total = np.log(len(self.sample_labels))
            label_sample_distribution = [log_total - np.log(len(self.label_indices[unique])) for unique in self.unique_labels]
        
        print("Distribution", label_sample_distribution)
        
        total = sum(label_sample_distribution) # type: ignore
        label_sample_distribution = [x / total for x in label_sample_distribution] # type: ignore
        
        print("Normalised Distribution", label_sample_distribution)
        
        self.label_sample_distribution = label_sample_distribution

    def __iter__(self):
        selected_classes = random.choices(population=self.unique_labels, weights=self.label_sample_distribution, k=len(self))
        
        for selected_class in selected_classes:
            yield random.choice(self.label_indices[selected_class])
    
    def __len__(self):
        return len(self.sample_labels)

class TransformerBasedModelTrainer:
    def __init__(
        self, 
        model_class,
        model_transformer,
        model_freeze_transformer,
        model_config,
        optimiser_class,
        optimiser_config,
        hf_scheduler_name,
        hf_scheduler_config,
        loss_class,
        batch_size,
        max_epochs,
        train_dataset,
        val_dataset,
        test_dataset,
        label_names,
        sampling_weight_fn,
        no_improvement_epochs_stop=3,
        eval_batch_size=None,
        loss_weights=None,
        device="cuda:0"
    ):
        for name, param in model_transformer.named_parameters():   
            param.requires_grad = not model_freeze_transformer
        
        if model_freeze_transformer:
            print("Transformer Frozen")
        
        self.model = model_class(model_transformer, **model_config).to(device)
        print("Model Initialised")
        
        self.optimiser = optimiser_class(self.model.parameters(), **optimiser_config)
        print("Model Initialised")
        
        if loss_weights is not None:
            assert False, "Need to implement!"

        if hf_scheduler_name is not None:
            self.scheduler = transformers.get_scheduler(hf_scheduler_name, self.optimiser, **hf_scheduler_config)
        else:
            self.scheduler = None
        
        self.loss_fn = loss_class().to(device)
        print("Loss Initialised")
        
        self.batch_size = batch_size
        self.eval_batch_size = eval_batch_size if eval_batch_size is not None else batch_size
        
        self.max_epochs = max_epochs
        self.no_improvement_epochs_stop = no_improvement_epochs_stop
        
        self.train_dataset = train_dataset
        self.val_dataset = val_dataset
        self.test_dataset = test_dataset
        
        train_sampler = WeightedDatasetSampler(train_dataset, label_weights=sampling_weight_fn)
        self.sampling_weights = train_sampler.label_sample_distribution
        print("Train Data Sampler Initialised")
        
        self.train_dataloader = torch.utils.data.DataLoader(train_dataset, sampler=train_sampler, batch_size=self.batch_size, drop_last=True)
        print("Dataloaders Initialised")
        
        self.label_names = label_names
        self.device = device
        self.is_trained = False
        
        self.high_level_name = f"{model_class.__qualname__}/{time.time()}"
        self.save_folder = f"./runs/{self.high_level_name}"
        os.makedirs(self.save_folder)
        
        save_config = {
            "model": {
                "model_class": model_class.__qualname__,
                "model_transformer": model_transformer.__class__.__qualname__,
                "model_freeze_transformer": model_freeze_transformer,
                "model_config": model_config,
                "model_trainable_layers": [name for name, param in self.model.named_parameters() if param.requires_grad]
            },
            "loss": {
                "loss_class": loss_class.__qualname__,
                "loss_weights": loss_weights
            },
            "optimiser": {
                "optimiser_class": optimiser_class.__qualname__,
                "optimiser_config": optimiser_config,
                "scheduler_name": hf_scheduler_name,
                "scheduler_config": hf_scheduler_config
            },
            "datasets": {
                "train_size": len(train_dataset),
                "val_size": len(val_dataset),
                "test_size": len(test_dataset),
                "sampling_weights": self.sampling_weights
            },
            "train_settings": {
                "max_epochs": max_epochs,
                "batch_size": batch_size,
                "no_improvement_epochs_stop": no_improvement_epochs_stop,
                "label_names": label_names
            }
        }
        
        with open(f"{self.save_folder}/info.json", "w+") as fp:
            json.dump(save_config, fp, indent=2)
        
        print("Dumped Config")
        
        # wandb.init(
        #     project=self.high_level_name.replace("/", "_"),
        #     config=save_config
        # )
        
        self._stored_log = []
        
        self._best_validation_loss = 1e9
        self._best_validation_epoch = -1

    def _log(self, msg):
        message = f"[{datetime.today().strftime('%Y-%m-%d %H:%M:%S')}]"
        
        if msg.startswith("["):
            message += msg
        else:
            message = f"{message} {msg}"
        
        self._stored_log.append(message)
        print(message)
    
    def _save_log(self, file_name):
        loc = f"{self.save_folder}/{file_name}"
        
        with open(loc, "w+") as fp:
            fp.write("\n".join(self._stored_log))
        
        self._stored_log = []
    
    def train(self):
        if self.is_trained:
            assert False, "Model already trained"
        
        self.is_trained = True
        
        for epoch in range(1, self.max_epochs + 1):
            self._log(f"Starting epoch {epoch}")
            
            self._log("Starting epoch training")
            epoch_loss = self._train_epoch(epoch)
            self._log("Finished epoch training")

            self._log(f"Epoch Loss: {epoch_loss}")
            
            self._log("Starting validation evaluation")
            validation_results = self.evaluate_model(self.val_dataset, f"{epoch}_validation_set")
            self._log(f"Epoch Accuracy {validation_results['evaluate']['accuracy']}")
            self._log("Finished validation evaluation")
            
            if validation_results["avg_validation_loss"] < self._best_validation_loss:
                self._log(f"Beat best validation loss, new validation loss: {validation_results['avg_validation_loss']} (surpassed {self._best_validation_loss} from epoch {self._best_validation_epoch})")
                self._best_validation_epoch = epoch
                self._best_validation_loss = validation_results["avg_validation_loss"]
                
                torch.save(self.model.state_dict(), f"{self.save_folder}/{epoch}_model.pth")
                
                self._log(f"Saved best model")
            elif epoch - self._best_validation_epoch >= self.no_improvement_epochs_stop:
                self._log(f"Early stopping triggered, best validation loss achieved at {self._best_validation_epoch} (loss: {self._best_validation_loss})")
                self._save_log(f"log_epoch_{epoch}.txt")
                break
                
            # wandb.log({"val_acc": validation_results["accuracy"], "train_loss": epoch_loss, "val_loss": validation_results["avg_validation_loss"]})
            
            self._log("Logged to wandb")
            self._save_log(f"log_epoch_{epoch}.txt")
            
        self._log(f"Loading best model (epoch {self._best_validation_loss}) for evaluation")
        self.model.load_state_dict(torch.load(f"{self.save_folder}/{self._best_validation_epoch}_model.pth"))
        self.model.to(self.device)
        self.model.eval()
        
        self._log("Starting test evaluation")
        test_results = self.evaluate_model(self.test_dataset, f"final_test_set")
        self._log("Finished test evaluation")
        
        self._save_log(f"log_final_evaluation.txt")
        
    def _train_epoch(self, epoch):
        self.model.train()
        self._log("Set model to train mode")
        
        epoch_st = time.time()
        epoch_loss = 0

        batch_acc_loss = 0

        for batch_no, batch in enumerate(tqdm(self.train_dataloader)):
            tokens = batch["input_ids"].to(self.device)
            attention_masks = batch["attention_mask"].to(self.device)
            labels = batch["labels"].to(self.device).squeeze()

            predictions = self.model(tokens, attention_masks)
            loss = self.loss_fn(predictions, labels)

            self.optimiser.zero_grad()
            loss.backward()
            self.optimiser.step()

            if self.scheduler is not None:
                self.scheduler.step()

            epoch_loss += loss.item()
            batch_acc_loss += loss.item()

            if batch_no != 0 and batch_no % 40 == 0:
                self._log(f"[{epoch}:{batch_no}] Loss: {(batch_acc_loss / 40):.3f}")
                batch_acc_loss = 0

        epoch_dt = time.time() - epoch_st
        self._log(f"[{epoch}:END] Took {epoch_dt:.3f}s")
        self._log(f"[{epoch}:END] Training Loss: {(epoch_loss / len(self.train_dataloader)):.3f}")
        
        return epoch_loss / len(self.train_dataloader)
    
    def evaluate_model(self, dataset, base_save_name=None):
        results = self._evaluate_model(dataset) 
        
        metrics.ConfusionMatrixDisplay.from_predictions(results["y_true"], results["y_pred"], display_labels=self.label_names)
        plt.grid(False)
        
        if base_save_name is not None:
            plt.savefig(f"{self.save_folder}/{base_save_name}_confusion_matrix.png", bbox_inches="tight")
        
        plt.show()
        
        self._log("Classification Report:")
        self._log(metrics.classification_report(results["y_true"], results["y_pred"]))
        
        if base_save_name is not None:
            saveable_results = {k: v for k, v in results.items() if k not in ["y_true", "y_pred"]}
            
            saveable_results["y_pred"] = list(results["y_pred"])
            saveable_results["y_true"] = list(results["y_true"].detach().squeeze().cpu().numpy())
            
            with open(f"{self.save_folder}/{base_save_name}_results.json", "w+") as fp:
                json.dump(saveable_results, fp, indent=2, default=lambda o: str(o))
        
        return results
        
    def _evaluate_model(self, dataset):
        y_true = dataset.labels
        y_logits = np.empty((len(y_true), 3))
        y_pred = np.empty(len(y_true))
        dataloader = torch.utils.data.DataLoader(dataset, batch_size=self.eval_batch_size, shuffle=False)
        
        self._log("Using model to generate predictions")
        
        with torch.inference_mode():
            self.model.eval()
            self._log("Set model to eval mode")

            total_loss = 0

            for i, batch in enumerate(tqdm(dataloader)):
                upper = ((i + 1) if i + 1 < len(dataloader) else len(dataloader)) * self.eval_batch_size

                tokens = batch["input_ids"].to(self.device)
                attention_masks = batch["attention_mask"].to(self.device)
                labels = batch["labels"].to(self.device).squeeze()

                logits = self.model(tokens, attention_masks)
                y_logits[i * self.eval_batch_size : upper] = logits.cpu().numpy()
                
                loss = self.loss_fn(logits, labels)
                
                pred = torch.argmax(logits, 1)
                y_pred[i * self.eval_batch_size : upper] = pred.cpu().numpy()
                
                total_loss += loss.item()

            self.model.train()
            self._log("Set model to train mode")
        
        classification_report = metrics.classification_report(y_true, y_pred, output_dict=True)
        
        probs = softmax(y_logits, axis=-1)
        
        accuracy = evaluate.load("accuracy").compute(predictions=y_pred, references=y_true)
        precision = evaluate.load("precision").compute(predictions=y_pred, references=y_true, average="weighted")
        recall = evaluate.load("recall").compute(predictions=y_pred, references=y_true, average="weighted")
        f1 = evaluate.load("f1").compute(predictions=y_pred, references=y_true, average="weighted")
        roc_auc = evaluate.load("roc_auc", "multiclass").compute(prediction_scores=probs, references=y_true.squeeze(), average="weighted", multi_class="ovr")
        
        results = {
            "y_true": y_true,
            "y_pred": y_pred,
            "evaluate": {
                "accuracy": accuracy,
                "precision": precision,
                "recall": recall,
                "f1": f1,
                "roc_auc": roc_auc
            },
            "accuracy": (y_true.squeeze().numpy() == y_pred).mean() * 100,
            "avg_validation_loss": total_loss / len(dataloader),
            "classification_report": classification_report
        }

        return results