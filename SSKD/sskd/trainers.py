from __future__ import print_function, absolute_import
import time

import torch
import torch.nn as nn
from torch.nn import functional as F

from .evaluation_metrics import accuracy
from .loss import TripletLoss, CrossEntropyLabelSmooth, SoftTripletLoss, SoftEntropy, MultiSimilarityLoss
from .utils.meters import AverageMeter


class PreTrainer(object):
    def __init__(self, model, num_classes, margin=0.0):
        super(PreTrainer, self).__init__()
        self.model = model
        self.criterion_ce = CrossEntropyLabelSmooth(num_classes).cuda()
        self.criterion_triple = SoftTripletLoss(margin=margin).cuda()
        self.criterion_ms = MultiSimilarityLoss().cuda()

    def train(self, epoch, data_loader_source, data_loader_target, optimizer, train_iters=200, print_freq=1):
        self.model.train()

        batch_time = AverageMeter()
        data_time = AverageMeter()
        losses_ce = AverageMeter()
        losses_ms = AverageMeter()
        transfer_loss_ = AverageMeter()
        precisions = AverageMeter()

        end = time.time()

        for i in range(train_iters):
            source_inputs = data_loader_source.next()
            target_inputs = data_loader_target.next()
            data_time.update(time.time() - end)

            s_inputs, targets = self._parse_data(source_inputs)
            t_inputs, _ = self._parse_data(target_inputs)
            s_features, s_cls_out = self.model(s_inputs)
            # target samples: only forward
            t_features, _ = self.model(t_inputs)

            ############################################
            target_softmax = F.softmax(s_cls_out, dim=1)
            transfer_loss = -torch.norm(target_softmax,'nuc')/target_softmax.shape[0]
            ############################################


            # backward main #
            loss_ce, loss_ms, prec1 = self._forward(s_features, s_cls_out, targets)
            loss = loss_ce + loss_ms + 2 * transfer_loss

            losses_ce.update(loss_ce.item())
            losses_ms.update(loss_ms.item())
            transfer_loss_.update(transfer_loss.item())
            precisions.update(prec1)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            batch_time.update(time.time() - end)
            end = time.time()

            if ((i + 1) % print_freq == 0):
                print('Epoch: [{}][{}/{}]\t'
                      'Time {:.3f} ({:.3f})\t'
                      'Data {:.3f} ({:.3f})\t'
                      'Loss_ce {:.3f} ({:.3f})\t'
                      'Loss_ms {:.3f} ({:.3f})\t'
                      'Prec {:.2%} ({:.2%})\t'
                      'transfer_loss_ {:.3f}'
                      .format(epoch, i + 1, train_iters,
                              batch_time.val, batch_time.avg,
                              data_time.val, data_time.avg,
                              losses_ce.val, losses_ce.avg,
                              losses_ms.val, losses_ms.avg,
                              precisions.val, precisions.avg, transfer_loss_.avg))

    def _parse_data(self, inputs):
        imgs, _, pids, _ = inputs
        inputs = imgs.cuda()
        targets = pids.cuda()
        return inputs, targets

    def _forward(self, s_features, s_outputs, targets):
        loss_ce = self.criterion_ce(s_outputs, targets)
        loss_ms = self.criterion_ms(s_features, targets)


        prec, = accuracy(s_outputs.data, targets.data)
        prec = prec[0]

        return loss_ce, loss_ms, prec

class ClusterBaseTrainer(object):
    def __init__(self, model, num_cluster=500):
        super(ClusterBaseTrainer, self).__init__()
        self.model = model
        self.num_cluster = num_cluster

        self.criterion_ce = CrossEntropyLabelSmooth(num_cluster).cuda()
        self.criterion_tri = SoftTripletLoss(margin=0.0).cuda()
        self.criterion_ms = MultiSimilarityLoss().cuda()

    def train(self, epoch, data_loader_target, optimizer, print_freq=1, train_iters=200):
        self.model.train()

        batch_time = AverageMeter()
        data_time = AverageMeter()

        losses_ce = AverageMeter()
        losses_tri = AverageMeter()
        precisions = AverageMeter()

        end = time.time()
        for i in range(train_iters):
            target_inputs = data_loader_target.next()
            data_time.update(time.time() - end)

            # process inputs
            inputs, targets = self._parse_data(target_inputs)

            # forward
            f_out_t, p_out_t = self.model(inputs)
            p_out_t = p_out_t[:,:self.num_cluster]

            loss_ce = self.criterion_ce(p_out_t, targets)
            loss_tri = self.criterion_tri(f_out_t, f_out_t, targets)
            loss = loss_ce + loss_tri

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            prec, = accuracy(p_out_t.data, targets.data)

            losses_ce.update(loss_ce.item())
            losses_tri.update(loss_tri.item())
            precisions.update(prec[0])

            # print log #
            batch_time.update(time.time() - end)
            end = time.time()

            if (i + 1) % print_freq == 0:
                print('Epoch: [{}][{}/{}]\t'
                      'Time {:.3f} ({:.3f})\t'
                      'Data {:.3f} ({:.3f})\t'
                      'Loss_ce {:.3f} ({:.3f})\t'
                      'Loss_tri {:.3f} ({:.3f})\t'
                      'Prec {:.2%} ({:.2%})\t'
                      .format(epoch, i + 1, len(data_loader_target),
                              batch_time.val, batch_time.avg,
                              data_time.val, data_time.avg,
                              losses_ce.val, losses_ce.avg,
                              losses_tri.val, losses_tri.avg,
                              precisions.val, precisions.avg))

    def _parse_data(self, inputs):
        imgs, _, pids, _ = inputs
        inputs = imgs.cuda()
        targets = pids.cuda()
        return inputs, targets

class MMTTrainer(object):
    def __init__(self, model_1, model_2,
                       model_1_ema, model_2_ema, num_cluster=500, alpha=0.999):
        super(MMTTrainer, self).__init__()
        self.model_1 = model_1
        self.model_2 = model_2
        self.num_cluster = num_cluster

        self.model_1_ema = model_1_ema
        self.model_2_ema = model_2_ema
        self.alpha = alpha

        self.criterion_ce = CrossEntropyLabelSmooth(num_cluster).cuda()
        self.criterion_ce_soft = SoftEntropy().cuda()
        self.criterion_tri = SoftTripletLoss(margin=0.0).cuda()
        self.criterion_tri_soft = SoftTripletLoss(margin=None).cuda()

    def train(self, epoch, data_loader_target,
            optimizer, ce_soft_weight=0.5, tri_soft_weight=0.5, print_freq=1, train_iters=200):
        self.model_1.train()
        self.model_2.train()
        self.model_1_ema.train()
        self.model_2_ema.train()

        batch_time = AverageMeter()
        data_time = AverageMeter()

        losses_ce = [AverageMeter(),AverageMeter()]
        losses_tri = [AverageMeter(),AverageMeter()]
        losses_ce_soft = AverageMeter()
        losses_tri_soft = AverageMeter()
        precisions = [AverageMeter(),AverageMeter()]

        end = time.time()
        for i in range(train_iters):
            target_inputs = data_loader_target.next()
            data_time.update(time.time() - end)

            # process inputs
            inputs_1, inputs_2, targets = self._parse_data(target_inputs)

            # forward
            f_out_t1, p_out_t1 = self.model_1(inputs_1)
            f_out_t2, p_out_t2 = self.model_2(inputs_2)
            p_out_t1 = p_out_t1[:,:self.num_cluster]
            p_out_t2 = p_out_t2[:,:self.num_cluster]

            f_out_t1_ema, p_out_t1_ema = self.model_1_ema(inputs_1)
            f_out_t2_ema, p_out_t2_ema = self.model_2_ema(inputs_2)
            p_out_t1_ema = p_out_t1_ema[:,:self.num_cluster]
            p_out_t2_ema = p_out_t2_ema[:,:self.num_cluster]

            loss_ce_1 = self.criterion_ce(p_out_t1, targets)
            loss_ce_2 = self.criterion_ce(p_out_t2, targets)

            loss_tri_1 = self.criterion_tri(f_out_t1, f_out_t1, targets)
            loss_tri_2 = self.criterion_tri(f_out_t2, f_out_t2, targets)

            loss_ce_soft = self.criterion_ce_soft(p_out_t1, p_out_t2_ema) + self.criterion_ce_soft(p_out_t2, p_out_t1_ema)
            loss_tri_soft = self.criterion_tri_soft(f_out_t1, f_out_t2_ema, targets) + \
                            self.criterion_tri_soft(f_out_t2, f_out_t1_ema, targets)

            loss = (loss_ce_1 + loss_ce_2)*(1-ce_soft_weight) + \
                     (loss_tri_1 + loss_tri_2)*(1-tri_soft_weight) + \
                     loss_ce_soft*ce_soft_weight + loss_tri_soft*tri_soft_weight

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            self._update_ema_variables(self.model_1, self.model_1_ema, self.alpha, epoch*len(data_loader_target)+i)
            self._update_ema_variables(self.model_2, self.model_2_ema, self.alpha, epoch*len(data_loader_target)+i)

            prec_1, = accuracy(p_out_t1.data, targets.data)
            prec_2, = accuracy(p_out_t2.data, targets.data)

            losses_ce[0].update(loss_ce_1.item())
            losses_ce[1].update(loss_ce_2.item())
            losses_tri[0].update(loss_tri_1.item())
            losses_tri[1].update(loss_tri_2.item())
            losses_ce_soft.update(loss_ce_soft.item())
            losses_tri_soft.update(loss_tri_soft.item())
            precisions[0].update(prec_1[0])
            precisions[1].update(prec_2[0])

            # print log #
            batch_time.update(time.time() - end)
            end = time.time()

            if (i + 1) % print_freq == 0:
                print('Epoch: [{}][{}/{}]\t'
                      'Time {:.3f} ({:.3f})\t'
                      'Data {:.3f} ({:.3f})\t'
                      'Loss_ce {:.3f} / {:.3f}\t'
                      'Loss_tri {:.3f} / {:.3f}\t'
                      'Loss_ce_soft {:.3f}\t'
                      'Loss_tri_soft {:.3f}\t'
                      'Prec {:.2%} / {:.2%}\t'
                      .format(epoch, i + 1, len(data_loader_target),
                              batch_time.val, batch_time.avg,
                              data_time.val, data_time.avg,
                              losses_ce[0].avg, losses_ce[1].avg,
                              losses_tri[0].avg, losses_tri[1].avg,
                              losses_ce_soft.avg, losses_tri_soft.avg,
                              precisions[0].avg, precisions[1].avg))

    def _update_ema_variables(self, model, ema_model, alpha, global_step):
        alpha = min(1 - 1 / (global_step + 1), alpha)
        for ema_param, param in zip(ema_model.parameters(), model.parameters()):
            ema_param.data.mul_(alpha).add_(param.data, alpha=1 - alpha)

    def _parse_data(self, inputs):
        imgs_1, imgs_2, pids = inputs
        inputs_1 = imgs_1.cuda()
        inputs_2 = imgs_2.cuda()
        targets = pids.cuda()
        return inputs_1, inputs_2, targets

class SSKDTrainer(object):
    def __init__(self, model_1, model_2, model_3,
                       model_1_ema, model_2_ema, model_3_ema, num_cluster=500, alpha=0.999):
        super(SSKDTrainer, self).__init__()
        self.model_1 = model_1
        self.model_2 = model_2
        self.model_3 = model_3
        self.num_cluster = num_cluster


        self.model_1_ema = model_1_ema
        self.model_2_ema = model_2_ema
        self.model_3_ema = model_3_ema
        self.alpha = alpha

        self.criterion_ce = CrossEntropyLabelSmooth(num_cluster).cuda()
        self.criterion_ce_soft = SoftEntropy().cuda()
        self.criterion_tri = SoftTripletLoss(margin=0.0).cuda()
        self.criterion_tri_soft = SoftTripletLoss(margin=None).cuda()
        self.criterion_ms = MultiSimilarityLoss().cuda()

    def train(self, epoch, data_loader_target,
            optimizer, weight=0.5, weight_ms=3, weight_tf=1.5, print_freq=1, train_iters=200):
        self.model_1.train()
        self.model_2.train()
        self.model_3.train()
        self.model_1_ema.train()
        self.model_2_ema.train()
        self.model_3_ema.train()

        batch_time = AverageMeter()
        data_time = AverageMeter()

        losses_ce = [AverageMeter(),AverageMeter(),AverageMeter()]
        losses_tri = [AverageMeter(),AverageMeter(),AverageMeter()]
        losses_ce_soft = AverageMeter()
        losses_tri_soft = AverageMeter()
        precisions = [AverageMeter(),AverageMeter(),AverageMeter()]
        loss_ms_ = AverageMeter()
        transfer_loss_ = AverageMeter()

        end = time.time()
        for i in range(train_iters):
            target_inputs = data_loader_target.next()
            data_time.update(time.time() - end)

            # process inputs
            inputs_1, inputs_2, inputs_3, targets = self._parse_data(target_inputs)
            

            # forward
            f_out_t1, p_out_t1 = self.model_1(inputs_1)
            f_out_t2, p_out_t2 = self.model_2(inputs_2)
            f_out_t3, p_out_t3 = self.model_3(inputs_3)
            p_out_t1 = p_out_t1[:,:self.num_cluster]
            p_out_t2 = p_out_t2[:,:self.num_cluster]
            p_out_t3 = p_out_t3[:,:self.num_cluster]

            f_out_t1_ema, p_out_t1_ema = self.model_1_ema(inputs_1)
            f_out_t2_ema, p_out_t2_ema = self.model_2_ema(inputs_2)
            f_out_t3_ema, p_out_t3_ema = self.model_3_ema(inputs_3)
            p_out_t1_ema = p_out_t1_ema[:,:self.num_cluster]
            p_out_t2_ema = p_out_t2_ema[:,:self.num_cluster]
            p_out_t3_ema = p_out_t3_ema[:,:self.num_cluster]


            target_softmax_1 = F.softmax(p_out_t1, dim=1)
            target_softmax_2 = F.softmax(p_out_t2, dim=1)
            target_softmax_3 = F.softmax(p_out_t3, dim=1)
            target_softmax_1_ema = F.softmax(p_out_t1_ema, dim=1)
            target_softmax_2_ema = F.softmax(p_out_t2_ema, dim=1)
            target_softmax_3_ema = F.softmax(p_out_t3_ema, dim=1)
            transfer_loss = -torch.norm(target_softmax_1,'nuc')/target_softmax_1.shape[0] + \
                                    -torch.norm(target_softmax_2,'nuc')/target_softmax_2.shape[0] + \
                                    -torch.norm(target_softmax_3,'nuc')/target_softmax_3.shape[0] + \
                                    -torch.norm(target_softmax_1_ema,'nuc')/target_softmax_1_ema.shape[0] + \
                                    -torch.norm(target_softmax_2_ema,'nuc')/target_softmax_2_ema.shape[0] + \
                                    -torch.norm(target_softmax_3_ema,'nuc')/target_softmax_3_ema.shape[0] 


            loss_ce_1 = self.criterion_ce(p_out_t1, targets)
            loss_ce_2 = self.criterion_ce(p_out_t2, targets)
            loss_ce_3 = self.criterion_ce(p_out_t3, targets)

            loss_ce_soft = self.criterion_ce_soft(p_out_t1, p_out_t2_ema) + self.criterion_ce_soft(p_out_t2, p_out_t3_ema) + \
                            self.criterion_ce_soft(p_out_t3, p_out_t1_ema)

            loss_tri_soft = self.criterion_tri_soft(f_out_t1, f_out_t2_ema, targets) + \
                            self.criterion_tri_soft(f_out_t2, f_out_t3_ema, targets) + \
                            self.criterion_tri_soft(f_out_t3, f_out_t1_ema, targets)


            ###################
            loss_ms = self.criterion_ms(f_out_t1, targets) + self.criterion_ms(f_out_t2, targets) + self.criterion_ms(f_out_t3, targets)
            #0.5 3 1.5
            loss = (loss_ce_1 + loss_ce_2 + loss_ce_3)*(1-weight) + \
                     loss_ce_soft*(weight) + weight_ms*loss_ms + weight_tf * transfer_loss
                               
                                 

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            self._update_ema_variables(self.model_1, self.model_1_ema, self.alpha, epoch*len(data_loader_target)+i)
            self._update_ema_variables(self.model_2, self.model_2_ema, self.alpha, epoch*len(data_loader_target)+i)
            self._update_ema_variables(self.model_3, self.model_3_ema, self.alpha, epoch*len(data_loader_target)+i)

            prec_1, = accuracy(p_out_t1.data, targets.data)
            prec_2, = accuracy(p_out_t2.data, targets.data)
            prec_3, = accuracy(p_out_t3.data, targets.data)

            losses_ce[0].update(loss_ce_1.item())
            losses_ce[1].update(loss_ce_2.item())
            losses_ce[2].update(loss_ce_3.item())
            losses_ce_soft.update(loss_ce_soft.item())

            precisions[0].update(prec_1[0])
            precisions[1].update(prec_2[0])
            precisions[2].update(prec_3[0])
            loss_ms_.update(loss_ms.item())
            transfer_loss_.update(transfer_loss.item())

            # print log #
            batch_time.update(time.time() - end)
            end = time.time()

            if (i + 1) % print_freq == 0:
                print('Epoch: [{}][{}/{}]\t'
                      'Time {:.3f} ({:.3f})\t'
                      'Data {:.3f} ({:.3f})\t'
                      'Loss_ce {:.3f} / {:.3f} / {:.3f}\t'
                      'Loss_ce_soft {:.3f}\t'
                      'Prec {:.2%} / {:.2%} / {:.2%}\t'
                      'loss_ms_ {:.3f}\t'
                      'transfer_loss_ {:.3f}\t'
                      .format(epoch, i + 1, len(data_loader_target),
                              batch_time.val, batch_time.avg,
                              data_time.val, data_time.avg,
                              losses_ce[0].avg, losses_ce[1].avg, losses_ce[2].avg,
                              losses_ce_soft.avg, 
                              precisions[0].avg, precisions[1].avg, precisions[2].avg, loss_ms_.avg, transfer_loss_.avg))

    def _update_ema_variables(self, model, ema_model, alpha, global_step):
        alpha = min(1 - 1 / (global_step + 1), alpha)
        for ema_param, param in zip(ema_model.parameters(), model.parameters()):
            ema_param.data.mul_(alpha).add_(param.data, alpha=1 - alpha)

    def _parse_data(self, inputs):
        imgs_1, imgs_2, imgs_3, pids = inputs
        inputs_1 = imgs_1.cuda()
        inputs_2 = imgs_2.cuda()
        inputs_3 = imgs_3.cuda()
        targets = pids.cuda()
        return inputs_1, inputs_2, inputs_3, targets


class ATMMTTrainer(object):
    '''
    Adversarial training with mutual mean teaching.
    Camera and pose discriminator is conducted in this framework.
    '''
    def __init__(self, model_1, model_2,
                       model_1_ema, model_2_ema, cam_disc, pose_disc, args, num_cluster=500, alpha=0.999):
        super(ATMMTTrainer, self).__init__()
        self.model_1 = model_1
        self.model_2 = model_2
        self.num_cluster = num_cluster

        self.model_1_ema = model_1_ema
        self.model_2_ema = model_2_ema
        self.alpha = alpha

        # create both camera and pose discriminator
        self.cam_disc = cam_disc
        self.pose_disc = pose_disc
        self.args = args

        self.criterion_ce = CrossEntropyLabelSmooth(num_cluster).cuda()
        self.criterion_ce_soft = SoftEntropy().cuda()
        self.criterion_tri = SoftTripletLoss(margin=0.0).cuda()
        self.criterion_tri_soft = SoftTripletLoss(margin=None).cuda()
        self.criterion_disc = nn.CrossEntropyLoss()

    def update_cam_disc(self, f_out, c_org, disc_optimizer):
        logit = self.cam_disc(f_out.detach())
        cam_loss = self.criterion_disc(logit, c_org)
        disc_optimizer[0].zero_grad()
        cam_loss.backward()
        disc_optimizer[0].step()
        return cam_loss

    def update_pose_disc(self, f_out, p_org, disc_optimizer):
        logit = self.pose_disc(f_out.detach())
        pose_loss = self.criterion_disc(logit, p_org)
        disc_optimizer[1].zero_grad()
        pose_loss.backward()
        disc_optimizer[1].step()
        return pose_loss

    def train(self, epoch, data_loader_target, optimizer, disc_optimizer, 
            ce_soft_weight=0.5, tri_soft_weight=0.5, pose_reid_weight=0.5, cam_reid_weight=0.5,  print_freq=1, train_iters=200):
        self.model_1.train()
        self.model_2.train()
        self.model_1_ema.train()
        self.model_2_ema.train()

        batch_time = AverageMeter()
        data_time = AverageMeter()

        losses_ce = [AverageMeter(),AverageMeter()]
        losses_tri = [AverageMeter(),AverageMeter()]
        losses_ce_soft = AverageMeter()
        losses_tri_soft = AverageMeter()
        losses_disc = [AverageMeter(),AverageMeter()]
        losses_disc_reid = [AverageMeter(),AverageMeter()]
        precisions = [AverageMeter(),AverageMeter()]

        end = time.time()
        for i in range(train_iters):
            target_inputs = data_loader_target.next()
            data_time.update(time.time() - end)

            # process inputs
            inputs_1, inputs_2, targets, c_org, p_org = self._parse_data(target_inputs)

            # forward
            f_out_t1, p_out_t1 = self.model_1(inputs_1)
            f_out_t2, p_out_t2 = self.model_2(inputs_2)
            p_out_t1 = p_out_t1[:,:self.num_cluster]
            p_out_t2 = p_out_t2[:,:self.num_cluster]

            f_out_t1_ema, p_out_t1_ema = self.model_1_ema(inputs_1)
            f_out_t2_ema, p_out_t2_ema = self.model_2_ema(inputs_2)
            p_out_t1_ema = p_out_t1_ema[:,:self.num_cluster]
            p_out_t2_ema = p_out_t2_ema[:,:self.num_cluster]

            # update cam discriminator
            if not self.args.wo_cat:
                loss_cam_disc  = (self.update_cam_disc(f_out_t1, c_org, disc_optimizer) + \
                                self.update_cam_disc(f_out_t2, c_org, disc_optimizer)) / 2
            else:
                loss_cam_disc = torch.zeros((1,), device=self.args.device)
            losses_disc[0].update(loss_cam_disc.item())

            # update pose discriminator
            if not self.args.wo_pat:
                loss_pose_disc = (self.update_pose_disc(f_out_t1, p_org, disc_optimizer) + \
                                self.update_pose_disc(f_out_t2, p_org, disc_optimizer)) / 2
            else:
                loss_pose_disc = torch.zeros((1,), device=self.args.device)
            losses_disc[1].update(loss_pose_disc.item())

            # ReID training
            loss_ce_1 = self.criterion_ce(p_out_t1, targets)
            loss_ce_2 = self.criterion_ce(p_out_t2, targets)

            loss_tri_1 = self.criterion_tri(f_out_t1, f_out_t1, targets)
            loss_tri_2 = self.criterion_tri(f_out_t2, f_out_t2, targets)

            loss_ce_soft = self.criterion_ce_soft(p_out_t1, p_out_t2_ema) + self.criterion_ce_soft(p_out_t2, p_out_t1_ema)
            loss_tri_soft = self.criterion_tri_soft(f_out_t1, f_out_t2_ema, targets) + \
                            self.criterion_tri_soft(f_out_t2, f_out_t1_ema, targets)

            # compute camera adversarial training loss
            if not self.args.wo_cat:
                logit_t1 = self.cam_disc(f_out_t1)
                logit_t2 = self.cam_disc(f_out_t2)
                loss_cam_reid = - (self.criterion_disc(logit_t1, c_org) + self.criterion_disc(logit_t2, c_org)) / 2
            else:
                loss_cam_reid = torch.zeros((1,), device=self.args.device)

            # compute pose adversarial training loss
            if not self.args.wo_pat:
                logit_t1 = self.pose_disc(f_out_t1)
                logit_t2 = self.pose_disc(f_out_t2)
                loss_pose_reid = - (self.criterion_disc(logit_t1, p_org) + self.criterion_disc(logit_t2, p_org)) / 2
            else:
                loss_pose_reid = torch.zeros((1,), device=self.args.device)

            loss = (loss_ce_1 + loss_ce_2)*(1-ce_soft_weight) + \
                     (loss_tri_1 + loss_tri_2)*(1-tri_soft_weight) + \
                     loss_ce_soft*ce_soft_weight + loss_tri_soft*tri_soft_weight + \
                         loss_cam_reid * cam_reid_weight + loss_pose_reid * pose_reid_weight

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            self._update_ema_variables(self.model_1, self.model_1_ema, self.alpha, epoch*len(data_loader_target)+i)
            self._update_ema_variables(self.model_2, self.model_2_ema, self.alpha, epoch*len(data_loader_target)+i)

            prec_1, = accuracy(p_out_t1.data, targets.data)
            prec_2, = accuracy(p_out_t2.data, targets.data)

            losses_ce[0].update(loss_ce_1.item())
            losses_ce[1].update(loss_ce_2.item())
            losses_tri[0].update(loss_tri_1.item())
            losses_tri[1].update(loss_tri_2.item())
            losses_disc_reid[0].update(loss_cam_reid.item())
            losses_disc_reid[1].update(loss_pose_reid.item())
            losses_ce_soft.update(loss_ce_soft.item())
            losses_tri_soft.update(loss_tri_soft.item())
            precisions[0].update(prec_1[0])
            precisions[1].update(prec_2[0])

            # print log #
            batch_time.update(time.time() - end)
            end = time.time()

            if (i + 1) % print_freq == 0:
                print('Epoch: [{}][{}/{}]\t'
                      'Time {:.3f} ({:.3f})\t'
                      'Data {:.3f} ({:.3f})\t'
                      'Loss_ce {:.3f} / {:.3f}\t'
                      'Loss_tri {:.3f} / {:.3f}\t'
                      'Loss_ce_soft {:.3f}\t'
                      'Loss_tri_soft {:.3f}\t'
                      'Loss_cam_disc {:.3f} \t'
                      'Loss_cam_reid {:.3f} \t'
                      'Loss_pose_disc {:.3f} \t'
                      'Loss_pose_reid {:.3f} \t'
                      'Prec {:.2%} / {:.2%}\t'
                      .format(epoch, i + 1, len(data_loader_target),
                              batch_time.val, batch_time.avg,
                              data_time.val, data_time.avg,
                              losses_ce[0].avg, losses_ce[1].avg,
                              losses_tri[0].avg, losses_tri[1].avg,
                              losses_ce_soft.avg, losses_tri_soft.avg,
                              losses_disc[0].avg, losses_disc_reid[0].avg, 
                              losses_disc[1].avg, losses_disc_reid[1].avg, 
                              precisions[0].avg, precisions[1].avg))

    def _update_ema_variables(self, model, ema_model, alpha, global_step):
        alpha = min(1 - 1 / (global_step + 1), alpha)
        for ema_param, param in zip(ema_model.parameters(), model.parameters()):
            ema_param.data.mul_(alpha).add_(param.data, alpha=1 - alpha)
            
    def _parse_data(self, inputs):
        imgs_1, imgs_2, pids, c_org, p_org = inputs
        inputs_1 = imgs_1.cuda()
        inputs_2 = imgs_2.cuda()
        targets = pids.cuda()
        c_org = c_org.cuda()
        p_org = p_org.cuda()
        return inputs_1, inputs_2, targets, c_org, p_org
