# -*- coding: utf-8 -*-
# Copyright (C) 2020 Unbabel
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

r"""
UniTE Metric
========================
    Implementation of the UniTE metric proposed in 
    [UniTE: Unified Translation Evaluation](https://arxiv.org/pdf/2204.13346.pdf)
"""
from typing import Dict, List, Optional, Tuple, Union

import torch
from comet.models.regression.regression_metric import RegressionMetric
from comet.modules import FeedForward


class UniTEMetric(RegressionMetric):
    """UniTEMetric:

    :param nr_frozen_epochs: Number of epochs (% of epoch) that the encoder is frozen.
    :param keep_embeddings_frozen: Keeps the encoder frozen during training.
    :param optimizer: Optimizer used during training.
    :param encoder_learning_rate: Learning rate used to fine-tune the encoder model.
    :param learning_rate: Learning rate used to fine-tune the top layers.
    :param layerwise_decay: Learning rate % decay from top-to-bottom encoder layers.
    :param encoder_model: Encoder model to be used.
    :param pretrained_model: Pretrained model from Hugging Face.
    :param pool: Pooling strategy to derive a sentence embedding ['cls', 'max', 'avg'].
    :param layer: Encoder layer to be used ('mix' for pooling info from all layers.)
    :param dropout: Dropout used in the top-layers.
    :param batch_size: Batch size used during training.
    :param train_data: Path to a csv file containing the training data.
    :param validation_data: Path to a csv file containing the validation data.
    :param hidden_sizes: Hidden sizes for the Feed Forward regression.
    :param activations: Feed Forward activation function.
    :param final_activation: Feed Forward final activation.
    :param input_segments: Input sequences used during training/inference.
        ["mt", "src"] for QE, ["mt", "ref"] for reference-base evaluation and ["mt", "src", "ref"]
        for full sequence evaluation.
    :param unite_training: If set to true the model is trained with UniTE loss that combines QE 
        with Metrics.
    :param load_weights_from_checkpoint: Path to a checkpoint file.
    """

    def __init__(
        self,
        nr_frozen_epochs: Union[float, int] = 0.3,
        keep_embeddings_frozen: bool = False,
        optimizer: str = "AdamW",
        encoder_learning_rate: float = 1e-05,
        learning_rate: float = 3e-05,
        layerwise_decay: float = 0.95,
        encoder_model: str = "XLM-RoBERTa",
        pretrained_model: str = "xlm-roberta-base",
        pool: str = "cls",
        layer: Union[str, int] = "mix",
        dropout: float = 0.1,
        batch_size: int = 4,
        train_data: Optional[str] = None,
        validation_data: Optional[List[str]] = None,
        hidden_sizes: List[int] = [2304, 768],
        activations: str = "Tanh",
        final_activation: Optional[str] = None,
        input_segments: Optional[List[str]] = ["mt", "src", "ref"],
        load_weights_from_checkpoint: Optional[str] = None,
        unite_training: Optional[bool] = False,
    ) -> None:
        super(RegressionMetric, self).__init__(
            nr_frozen_epochs,
            keep_embeddings_frozen,
            optimizer,
            encoder_learning_rate,
            learning_rate,
            layerwise_decay,
            encoder_model,
            pretrained_model,
            pool,
            layer,
            dropout,
            batch_size,
            train_data,
            validation_data,
            load_weights_from_checkpoint,
            "unite_metric",
        )
        self.save_hyperparameters()

        self.estimator = FeedForward(
            in_dim=self.encoder.output_units,
            hidden_sizes=self.hparams.hidden_sizes,
            activations=self.hparams.activations,
            dropout=self.hparams.dropout,
            final_activation=self.hparams.final_activation,
        )
        self.input_segments = input_segments
        self.unite_training = unite_training
    
    def is_referenceless(self) -> bool:
        return "ref" not in self.input_segments

    def set_input_segments(self, input_segments: List[str]):
        assert input_segments in [
            ["mt", "src"],
            ["mt", "ref"],
            ["mt", "src", "ref"],
        ], (
            "Input segments is ['mt', 'src'] for QE, ['mt', 'ref'] for reference-based evaluation"
            "and ['mt', 'src', 'ref'] for complete sequence evaluation."
        )
        self.input_segments = input_segments

    def prepare_sample(
        self, sample: List[Dict[str, Union[str, float]]], inference: bool = False
    ) -> Union[
        Tuple[Dict[str, torch.Tensor], Dict[str, torch.Tensor]], Dict[str, torch.Tensor]
    ]:
        """
        Function that prepares a sample to input the model.

        :param sample: list of dictionaries.
        :param inference: If set to true prepares only the model inputs.

        :returns: Tuple with 2 dictionaries (model inputs and targets).
            If `inference=True` returns only the model inputs.
        """
        sample = {k: [dic[k] for dic in sample] for k in sample[0]}
        inputs = [self.encoder.prepare_sample(sample["mt"])]
        if "src" in self.input_segments:
            assert (
                "src" in sample.keys()
            ), "UniTEMetric expects a source segment ('src') as input."
            inputs.append(self.encoder.prepare_sample(sample["src"]))

        if "ref" in self.input_segments:
            assert (
                "ref" in sample.keys()
            ), "UniTEMetric expects a source segment ('ref') as input."
            inputs.append(self.encoder.prepare_sample(sample["ref"]))

        contiguous_input = self.encoder.concat_sequences(inputs)
        if inference:
            return contiguous_input

        targets = {"score": torch.tensor(sample["score"], dtype=torch.float)}

        if self.unite_training:
            qe_inputs = self.encoder.concat_sequences([
                self.encoder.prepare_sample(sample["mt"]), 
                self.encoder.prepare_sample(sample["src"])
            ])
            metric_inputs = self.encoder.concat_sequences([
                self.encoder.prepare_sample(sample["mt"]), 
                self.encoder.prepare_sample(sample["ref"])
            ])
            inputs = (qe_inputs, metric_inputs, contiguous_input)

        else:
            inputs = contiguous_input

        return inputs, targets

    def forward(
        self, input_ids: torch.tensor, attention_mask: torch.tensor, **kwargs
    ) -> Dict[str, torch.Tensor]:
        sentemb = self.get_sentence_embedding(input_ids, attention_mask)
        return {"score": self.estimator(sentemb)}

    def training_step(
        self,
        batch: Tuple[Dict[str, torch.Tensor], Dict[str, torch.Tensor]],
        batch_nb: int,
    ) -> torch.Tensor:
        """
        Runs one training step and logs the training loss.

        :param batch: The output of your prepare_sample function.
        :param batch_nb: Integer displaying which batch this is.

        :returns: Loss value
        """
        batch_input, batch_target = batch
        if self.unite_training:
            # In UniTE training is made of 3 losses:
            #    Lsrc + Lref + Lsrc+ref
            # For that reason we have to perform 3 forward passes and sum 
            # the respective losses.
            qe_inputs, metric_inputs, unified_inputs = batch_input
            qe_prediction = self.forward(**qe_inputs)
            metric_prediction = self.forward(**metric_inputs)
            unified_prediction = self.forward(**unified_inputs)
            
            qe_loss = self.compute_loss(qe_prediction, batch_target)
            metric_loss = self.compute_loss(metric_prediction, batch_target)
            unified_loss = self.compute_loss(unified_prediction, batch_target)
            loss_value = qe_loss + metric_loss + unified_loss

        else:
            batch_prediction = self.forward(**batch_input)
            loss_value = self.compute_loss(batch_prediction, batch_target)

        if (
            self.nr_frozen_epochs < 1.0
            and self.nr_frozen_epochs > 0.0
            and batch_nb > self.epoch_total_steps * self.nr_frozen_epochs
        ):
            self.unfreeze_encoder()
            self._frozen = False

        self.log("train_loss", loss_value, on_step=True, on_epoch=True)
        return loss_value

    def validation_step(
        self,
        batch: Tuple[Dict[str, torch.Tensor], Dict[str, torch.Tensor]],
        batch_nb: int,
        dataloader_idx: int,
    ) -> None:
        """
        Runs one validation step and logs metrics.

        :param batch: The output of your prepare_sample function.
        :param batch_nb: Integer displaying which batch this is.
        :param dataloader_idx: Integer displaying which dataloader this is.
        """
        batch_input, batch_target = batch
        if self.unite_training:
            # In UniTE training is made of 3 losses:
            #    Lsrc + Lref + Lsrc+ref
            # For that reason we have to perform 3 forward passes and sum 
            # the respective losses.
            qe_inputs, metric_inputs, unified_inputs = batch_input
            qe_prediction = self.forward(**qe_inputs)
            metric_prediction = self.forward(**metric_inputs)
            unified_prediction = self.forward(**unified_inputs)
    
            qe_loss = self.compute_loss(qe_prediction, batch_target)
            metric_loss = self.compute_loss(metric_prediction, batch_target)
            unified_loss = self.compute_loss(unified_prediction, batch_target)
            scores = (
                qe_prediction["score"].view(-1) + 
                metric_prediction["score"].view(-1) + 
                unified_prediction["score"].view(-1)
            )/3
            loss_value = qe_loss + metric_loss + unified_loss

        else:
            batch_prediction = self.forward(**batch_input)
            scores = batch_prediction["score"].view(-1)
            loss_value = self.compute_loss(batch_prediction, batch_target)

        self.log("val_loss", loss_value, on_step=True, on_epoch=True)
        if dataloader_idx == 0:
            self.train_metrics.update(scores, batch_target["score"])

        elif dataloader_idx > 0:
            self.val_metrics[dataloader_idx-1].update(scores, batch_target["score"])