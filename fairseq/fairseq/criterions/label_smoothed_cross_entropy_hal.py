# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import math
from dataclasses import dataclass, field

import torch
import torch.nn.functional as F
from fairseq import metrics, utils
from fairseq.criterions import FairseqCriterion, register_criterion
from .label_smoothed_cross_entropy import label_smoothed_nll_loss
from fairseq.dataclass import FairseqDataclass
from omegaconf import II


@dataclass
class LabelSmoothedCrossEntropyHallucCriterionConfig(FairseqDataclass):
    label_smoothing: float = field(
        default=0.0,
        metadata={"help": "epsilon for label smoothing, 0 means no label smoothing"},
    )
    consistency_loss: str = field(
        default='kld',
        metadata={"help": "type of consistency loss"},
    )
    consistency_weight: float = field(
        default=0.0,
        metadata={"help": "weight of consistency loss relative to cross entropy"},
    )
    halluc_weight: float = field(
        default=0.0,
        metadata={"help": "weight of hallucination loss relative to cross entropy"},
    )
    report_accuracy: bool = field(
        default=False,
        metadata={"help": "report accuracy metric"},
    )
    ignore_prefix_size: int = field(
        default=0,
        metadata={"help": "Ignore first N tokens"},
    )
    sentence_avg: bool = II("optimization.sentence_avg")


def kl_div_loss(lprobs_p, lprobs_q, target, ignore_index=None, reduce=True):
    '''
    Kullback–Leibler divergence between probability distributions
    '''
    # if target.dim() == lprobs.dim() - 1:
    #     target = target.unsqueeze(-1)
    # loss = -lprobs.gather(dim=-1, index=target)
    loss = F.kl_div(lprobs_p, lprobs_q.exp(), reduction='none')
    if ignore_index is not None:
        pad_mask = target.eq(ignore_index)
        loss.masked_fill_(pad_mask, 0.0)
    else:
        loss = loss.squeeze(-1)
    if reduce:
        loss = loss.sum()
    return loss


def js_div_loss(lprobs, target):
    '''
    Jensen–Shannon divergence between probability distributions
    '''
    raise NotImplementedError


@register_criterion(
    "label_smoothed_cross_entropy_halluc", dataclass=LabelSmoothedCrossEntropyHallucCriterionConfig
)
class LabelSmoothedCrossEntropyHallucCriterion(FairseqCriterion):
    def __init__(
        self,
        task,
        sentence_avg,
        label_smoothing,
        consistency_loss,
        consistency_weight,
        halluc_weight,
        ignore_prefix_size=0,
        report_accuracy=False,
    ):
        super().__init__(task)
        self.sentence_avg = sentence_avg
        self.eps = label_smoothing
        self.consistency_loss = consistency_loss
        self.consistency_weight = consistency_weight
        self.halluc_weight = halluc_weight
        self.ignore_prefix_size = ignore_prefix_size
        self.report_accuracy = report_accuracy

    def forward(self, model, sample, reduce=True):
        """Compute the loss for the given sample.

        Returns a tuple with three elements:
        1) the loss
        2) the sample size, which is used as the denominator for the gradient
        3) logging outputs to display while training
        """
        net_output = model(**sample["net_input"])
        loss, nll_loss, loss_dict = self.compute_loss(model, net_output, sample, reduce=reduce)
        sample_size = (
            sample["target"].size(0) if self.sentence_avg else sample["ntokens"]
        )
        logging_output = {
            "loss": loss.data,
            "nll_loss": nll_loss.data,
            "vislang_loss": loss_dict['vislang'].data if 'vislang' in loss_dict else 0,
            "halluc_loss": loss_dict['halluc'].data if 'halluc' in loss_dict else 0,
            "consistency_loss": loss_dict['consistency'].data if 'consistency' in loss_dict else 0,
            "dalle_loss": loss_dict['dalle'].data if 'dalle' in loss_dict else 0,
            "ntokens": sample["ntokens"],
            "nsentences": sample["target"].size(0),
            "sample_size": sample_size,
        }
        if self.report_accuracy:
            n_correct, total = self.compute_accuracy(model, net_output, sample)
            logging_output["n_correct"] = utils.item(n_correct.data)
            logging_output["total"] = utils.item(total.data)
        return loss, sample_size, logging_output

    def get_lprobs_and_target(self, model, net_output, sample):
        lprobs = model.get_normalized_probs(net_output, log_probs=True)
        target = model.get_targets(sample, net_output)
        if self.ignore_prefix_size > 0:
            if getattr(lprobs, "batch_first", False):
                lprobs = lprobs[:, self.ignore_prefix_size :, :].contiguous()
                target = target[:, self.ignore_prefix_size :].contiguous()
            else:
                lprobs = lprobs[self.ignore_prefix_size :, :, :].contiguous()
                target = target[self.ignore_prefix_size :, :].contiguous()
        return lprobs.view(-1, lprobs.size(-1)), target.view(-1)

    def compute_loss(self, model, net_output, sample, reduce=True):
        loss = None
        nll_loss = None
        loss_dict = {}
        vis = 'vislang' in net_output and net_output['vislang'] is not None
        hal = 'halluc' in net_output and net_output['halluc'] is not None
        assert vis or hal, "Net output is empty"

        # label-smoothed cross entropy loss (translation)
        if vis:
            lprobs_vis, target = self.get_lprobs_and_target(model, net_output['vislang'], sample)
            loss_dict['vislang'], loss_dict['vislang_nll'] = label_smoothed_nll_loss(
                lprobs_vis,
                target,
                self.eps,
                ignore_index=self.padding_idx,
                reduce=reduce,
            )
            loss = loss_dict['vislang']
            nll_loss = loss_dict['vislang_nll']
        if hal:
            lprobs_hal, target = self.get_lprobs_and_target(model, net_output['halluc'], sample)
            loss_dict['halluc'], loss_dict['halluc_nll'] = label_smoothed_nll_loss(
                lprobs_hal,
                target,
                self.eps,
                ignore_index=self.padding_idx,
                reduce=reduce,
            )
            loss = (loss + loss_dict['halluc']) / 2 if loss is not None else loss_dict['halluc']
            nll_loss = (nll_loss + loss_dict['halluc_nll']) / 2 if nll_loss is not None else loss_dict['halluc_nll']
        
        # hallucination loss
        if 'loss' in net_output:
            if self.halluc_weight > 1e-6:
                hal_loss = net_output['loss']
                if hal_loss is not None:
                    loss_dict['dalle'] = hal_loss
                    loss += self.halluc_weight * hal_loss

        # consistency loss
        if vis and hal:
            if self.consistency_weight > 1e-6:
                if self.consistency_loss == 'kld':
                    con_loss = kl_div_loss(lprobs_hal, lprobs_vis, target)
                elif self.consistency_loss == 'jsd':
                    con_loss = js_div_loss(lprobs_hal, lprobs_vis, target)
                else:
                    raise Exception('Consistency loss {} not supported'.format(self.consistency_loss))
                loss_dict['consistency'] = con_loss
                loss += self.consistency_weight * con_loss

        return loss, nll_loss, loss_dict

    def compute_accuracy(self, model, net_output, sample):
        lprobs, target = self.get_lprobs_and_target(model, net_output, sample)
        mask = target.ne(self.padding_idx)
        n_correct = torch.sum(
            lprobs.argmax(1).masked_select(mask).eq(target.masked_select(mask))
        )
        total = torch.sum(mask)
        return n_correct, total

    @classmethod
    def reduce_metrics(cls, logging_outputs) -> None:
        """Aggregate logging outputs from data parallel training."""
        loss_sum = sum(log.get("loss", 0) for log in logging_outputs)
        nll_loss_sum = sum(log.get("nll_loss", 0) for log in logging_outputs)
        vislang_loss_sum = sum(log.get("vislang_loss", 0) for log in logging_outputs)
        halluc_loss_sum = sum(log.get("halluc_loss", 0) for log in logging_outputs)
        consistency_loss_sum = sum(log.get("consistency_loss", 0) for log in logging_outputs)
        dalle_loss_sum = sum(log.get("dalle_loss", 0) for log in logging_outputs)
        ntokens = sum(log.get("ntokens", 0) for log in logging_outputs)
        sample_size = sum(log.get("sample_size", 0) for log in logging_outputs)

        metrics.log_scalar(
            "loss", loss_sum / sample_size / math.log(2), sample_size, round=3
        )
        metrics.log_scalar(
            "nll_loss", nll_loss_sum / ntokens / math.log(2), ntokens, round=3
        )
        metrics.log_scalar(
            "vislang_loss", vislang_loss_sum / ntokens / math.log(2), ntokens, round=3
        )
        metrics.log_scalar(
            "halluc_loss", halluc_loss_sum / ntokens / math.log(2), ntokens, round=3
        )
        metrics.log_scalar(
            "consistency_loss", consistency_loss_sum / ntokens / math.log(2), ntokens, round=3
        )
        metrics.log_scalar(
            "dalle_loss", dalle_loss_sum / ntokens / math.log(2), ntokens, round=3
        )
        metrics.log_derived(
            "ppl", lambda meters: utils.get_perplexity(meters["nll_loss"].avg)
        )

        total = utils.item(sum(log.get("total", 0) for log in logging_outputs))
        if total > 0:
            metrics.log_scalar("total", total)
            n_correct = utils.item(
                sum(log.get("n_correct", 0) for log in logging_outputs)
            )
            metrics.log_scalar("n_correct", n_correct)
            metrics.log_derived(
                "accuracy",
                lambda meters: round(
                    meters["n_correct"].sum * 100.0 / meters["total"].sum, 3
                )
                if meters["total"].sum > 0
                else float("nan"),
            )

    @staticmethod
    def logging_outputs_can_be_summed() -> bool:
        """
        Whether the logging outputs returned by `forward` can be summed
        across workers prior to calling `reduce_metrics`. Setting this
        to True will improves distributed training speed.
        """
        return True
