from __future__ import print_function
import os
import torch.optim as optim
import torch.backends.cudnn as cudnn
from torch.autograd import Variable
import visual_utils
import sys
import random
from visual_utils import ImageFilelist, compute_per_class_acc, compute_per_class_acc_gzsl, \
    prepare_attri_label, add_glasso, add_dim_glasso
from logger import Logger
from utils import init_log
from model_proto import resnet_proto_IoU
import numpy as np
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.utils.data
import json
from main_utils import test_zsl, calibrated_stacking, test_gzsl,test_gzsl_loss, \
    calculate_average_IoU, test_with_IoU
from main_utils import set_randomseed, get_loader, get_middle_graph, Loss_fn, Result, SupConLoss_clear, \
    mse_loss, update_ema_variables, get_current_consistency_weight
from opt import get_opt
from setproctitle import setproctitle
setproctitle('wanggerong')


cudnn.benchmark = True

opt = get_opt()
# set random seed
if opt.manualSeed is None:
    opt.manualSeed = random.randint(1, 10000)
print("Random Seed: ", opt.manualSeed)
random.seed(opt.manualSeed)
torch.manual_seed(opt.manualSeed)
np.random.seed(opt.manualSeed)
if opt.cuda:
    torch.cuda.manual_seed_all(opt.manualSeed)


def main():
    # load data
    data = visual_utils.DATA_LOADER(opt)
    opt.test_seen_label = data.test_seen_label  # weird

    # define test_classes
    if opt.image_type == 'test_unseen_small_loc':
        test_loc = data.test_unseen_small_loc
        test_classes = data.unseenclasses
    elif opt.image_type == 'test_unseen_loc':
        test_loc = data.test_unseen_loc
        test_classes = data.unseenclasses
    elif opt.image_type == 'test_seen_loc':
        test_loc = data.test_seen_loc
        test_classes = data.seenclasses
    else:
        try:
            sys.exit(0)
        except:
            print("choose the image_type in ImageFileList")

    # prepare the attribute labels
    class_attribute = data.attribute
    attribute_zsl = prepare_attri_label(class_attribute, data.unseenclasses).cuda()
    attribute_seen = prepare_attri_label(class_attribute, data.seenclasses).cuda()
    attribute_gzsl = torch.transpose(class_attribute, 1, 0).cuda()

    # Dataloader for train, test, visual
    trainloader, testloader_unseen, testloader_seen, visloader = get_loader(opt, data)

    #experiment name
    experiment_name = 'pretrainepoch{}_pretrainlr{}_classifierlr{}_{}ways-{}shots-{}'.format(opt.pretrain_epoch, opt.pretrain_lr, opt.classifier_lr, opt.ways, opt.shots, opt.contrative_loss_weight)
    #visual_loss_path
    visual_loss_path = './out/visual/alpha/' + experiment_name + '/'
    logger = Logger(visual_loss_path)

    #store_result_path
    save_dir ='./out/result/alpha/{}.txt'.format(experiment_name)
    logging = init_log(save_dir)
    _print = logging.info

    weight_path = './out/weight/alpha/' + experiment_name + '/'
    if not os.path.exists(weight_path):
        os.makedirs(weight_path)

    # define attribute groups
    if opt.dataset == 'CUB':
        parts = ['head', 'belly', 'breast', 'belly', 'wing', 'tail', 'leg', 'others']
        group_dic = json.load(open(os.path.join(opt.root, 'data', opt.dataset, 'attri_groups_8.json')))
        sub_group_dic = json.load(open(os.path.join(opt.root, 'data', opt.dataset, 'attri_groups_8_layer.json')))
        opt.resnet_path = '/data/wgr/code/APN-ZSL-master/pretrained_models/resnet101_c.pth.tar'
    elif opt.dataset == 'AWA2':
        parts = ['color', 'texture', 'shape', 'body_parts', 'behaviour', 'nutrition', 'activativity', 'habitat',
                 'character']
        group_dic = json.load(open(os.path.join(opt.root, 'data', opt.dataset, 'attri_groups_9.json')))
        sub_group_dic = {}
        if opt.awa_finetune:
            opt.resnet_path = './pretrained_models/resnet101_awa2.pth.tar'
        else:
            opt.resnet_path = './pretrained_models/resnet101-5d3b4d8f.pth'
    elif opt.dataset == 'SUN':
        parts = ['functions', 'materials', 'surface_properties', 'spatial_envelope']
        group_dic = json.load(open(os.path.join(opt.root, 'data', opt.dataset, 'attri_groups_4.json')))
        sub_group_dic = {}
        opt.resnet_path = './pretrained_models/resnet101_sun.pth.tar'
    else:
        opt.resnet_path = './pretrained_models/resnet101-5d3b4d8f.pth'
    # initialize model
    print('Create Model...')

    def create_model(ema=False):

        model = resnet_proto_IoU(opt)

        if ema:
            for param in model.parameters():
                param.detach_()

        return model

    model = create_model()
    ema_model = create_model(ema=True)


    criterion = nn.CrossEntropyLoss()
    criterion_regre = nn.MSELoss()
    criterion_contras = SupConLoss_clear(opt.ins_temp)
    consistency_criterion = mse_loss

    # optimzation weight, only ['final'] + model.extract are used.
    reg_weight = {'final': {'xe': opt.xe, 'attri': opt.attri, 'regular': opt.regular},
                  'layer4': {'l_xe': opt.l_xe, 'attri': opt.l_attri, 'regular': opt.l_regular,
                             'cpt': opt.cpt},  # l denotes layer
                  }
    reg_lambdas = {}
    for name in ['final'] + model.extract:
        reg_lambdas[name] = reg_weight[name]
    # print('reg_lambdas:', reg_lambdas)

    if torch.cuda.is_available():
        model.cuda()
        ema_model.cuda()
        attribute_zsl = attribute_zsl.cuda()
        attribute_seen = attribute_seen.cuda()
        attribute_gzsl = attribute_gzsl.cuda()

    layer_name = model.extract[0]  # only use one layer currently
    # compact loss configuration, define middle_graph
    middle_graph = get_middle_graph(reg_weight[layer_name]['cpt'], model)

    # train and test
    result_zsl_student = Result()
    result_zsl_teacher = Result()
    result_gzsl_student = Result()
    result_gzsl_teacher = Result()


    if opt.only_evaluate:
        print('Evaluate ...')

        model.load_state_dict(torch.load(opt.resume))
        model.eval()
        # test zsl
        if not opt.gzsl:
            acc_ZSL = test_zsl(opt, model, testloader_unseen, attribute_zsl, data.unseenclasses)
            print('ZSL test accuracy is {:.1f}%'.format(acc_ZSL))
        else:
            # test gzsl
            acc_GZSL_unseen = test_gzsl(opt, model, testloader_unseen, attribute_gzsl, data.unseenclasses)
            acc_GZSL_seen = test_gzsl(opt, model, testloader_seen, attribute_gzsl, data.seenclasses)

            if (acc_GZSL_unseen + acc_GZSL_seen) == 0:
                acc_GZSL_H = 0
            else:
                acc_GZSL_H = 2 * acc_GZSL_unseen * acc_GZSL_seen / (
                        acc_GZSL_unseen + acc_GZSL_seen)

            print('GZSL test accuracy is Unseen: {:.1f} Seen: {:.1f} H:{:.1f}'.format(acc_GZSL_unseen, acc_GZSL_seen, acc_GZSL_H))
    else:
        print('Train and test...')



        global_step = 0
        for epoch in range(opt.nepoch):
            # print("training")
            model.train()
            ema_model.train()

            current_lr = opt.classifier_lr * (0.8 ** (epoch // 10))
            realtrain = epoch > opt.pretrain_epoch
            # if epoch <= opt.pretrain_epoch:   # pretrain ALE for the first several epoches
            if epoch < opt.pretrain_epoch:   # pretrain ALE for the first several epoches
                optimizer = optim.Adam(params=[model.prototype_vectors[layer_name], model.ALE_vector],
                                       lr=opt.pretrain_lr, betas=(opt.beta1, 0.999))
            else:
                #过滤掉requires_grad=false的层
                optimizer = optim.Adam(params=filter(lambda p: p.requires_grad, model.parameters()),
                                       lr=current_lr, betas=(opt.beta1, 0.999))
                # optimizer = optim.Adam(params=filter(lambda p: p.requires_grad, model.parameters()),
                #         lr=opt.classifier_lr, betas=(opt.beta1, 0.999))
            # loss for print
            loss_log = {'ave_loss': 0, 'contrastive_loss': 0, 'consistency_loss': 0, 'l_xe_final': 0, 'l_attri_final': 0, 'l_regular_final': 0,
                        'l_xe_layer': 0, 'l_attri_layer': 0, 'l_regular_layer': 0, 'l_cpt': 0}

            batch = len(trainloader) #### 设置的 n_batch = 1000

            for i, (batch_input, batch_input2, batch_target, impath) in enumerate(trainloader):
                model.zero_grad()
                # map target labels

                batch_target = visual_utils.map_label(batch_target, data.seenclasses)
                input_v = Variable(batch_input)
                ema_input_v = Variable(batch_input2, requires_grad=False)
                label_v = Variable(batch_target)

                if opt.cuda:
                    input_v = input_v.cuda()
                    ema_input_v = ema_input_v.cuda()
                    label_v = label_v.cuda()
                output, pre_attri, attention, pre_class = model(input_v, attribute_seen)
                if hasattr(torch.cuda, 'empty_cache'):
                    torch.cuda.empty_cache()

                output2, pre_attri2, attention2, pre_class2 = ema_model(ema_input_v, attribute_seen)

                ema_logit = Variable(output2.detach().data, requires_grad=False)

                label_a = attribute_seen[:, label_v].t()

                loss = Loss_fn(opt, loss_log, reg_weight, criterion, criterion_regre, model,
                               output, pre_attri, attention, pre_class, label_a, label_v,
                               realtrain, middle_graph, parts, group_dic, sub_group_dic)

                loss_contrastive = criterion_contras(pre_attri[layer_name],label_v)
                loss += loss_contrastive

                if opt.consistency:
                    # consistency_weight = get_current_consistency_weight(opt, epoch)
                    # logger.scalar_summary('consistency_weight', consistency_weight, epoch)
                    # consistency_loss = consistency_weight * consistency_criterion(output, ema_logit) 

                    consistency_loss = opt.con * consistency_criterion(output, ema_logit) 

                else:
                    consistency_loss = 0
                loss += consistency_loss

                loss_log['ave_loss'] += loss.item()
                loss_log['contrastive_loss'] += loss_contrastive.item()
                loss_log['consistency_loss'] += consistency_loss.item()

                loss.backward()
                optimizer.step()

                global_step += 1
                update_ema_variables(model, ema_model, opt.ema_decay, global_step)


            # print('\nLoss log: {}'.format({key: loss_log[key] / batch for key in loss_log}))
            print('\n[Epoch %d, Batch %5d] Train loss: %.3f '
                  % (epoch+1, batch, loss_log['ave_loss'] / batch))
            logger.scalar_summary('Train loss', loss_log['ave_loss'] / batch, epoch+1)
            logger.scalar_summary('Train contrastive loss', loss_log['contrastive_loss'] / batch, epoch+1)
            logger.scalar_summary('Train consistency loss', loss_log['consistency_loss'] / batch, epoch+1)

            if (i + 1) == batch or (i + 1) % 200 == 0:
                ###### test #######
                # print("testing")
                model.eval()
                ema_model.eval()
                # test zsl
                
                #### test zsl student
                student_acc_ZSL = test_zsl(opt, model, testloader_unseen, attribute_zsl, data.unseenclasses)
                student_loss_ZSL, student_loss_contrastive_ZSL = test_gzsl_loss(opt, model, testloader_unseen, attribute_gzsl,reg_weight,criterion, criterion_regre,realtrain,middle_graph,parts,group_dic,sub_group_dic)
                
                print('Test student ZSL loss: %.3f'% student_loss_ZSL)
                logger.scalar_summary('Test ZSL loss', student_loss_ZSL, epoch+1)
                if student_acc_ZSL > result_zsl_student.best_acc:
                    # save model state
                    model_save_path = os.path.join(weight_path+'/{}_student_ZSL_id_{}.pth'.format(opt.dataset, opt.train_id))
                    torch.save(model.state_dict(), model_save_path)
                    print('model saved to:', model_save_path)
                result_zsl_student.update(epoch+1, student_acc_ZSL)
                print('\n[Epoch {}] student ZSL test accuracy is {:.1f}%, Best_acc [{:.1f}% | Epoch-{}]'.format(epoch+1, student_acc_ZSL, result_zsl_student.best_acc, result_zsl_student.best_iter))
            
                # test gzsl student model
                student_acc_GZSL_unseen = test_gzsl(opt, model, testloader_unseen, attribute_gzsl, data.unseenclasses)
                student_acc_GZSL_seen = test_gzsl(opt, model, testloader_seen, attribute_gzsl, data.seenclasses)

                student_loss_GZSL_seen , student_loss_contrastive_seen=  test_gzsl_loss(opt, model, testloader_seen, attribute_gzsl,reg_weight,criterion, criterion_regre,realtrain,middle_graph,parts,group_dic,sub_group_dic)
                student_loss_GZSL_unseen, student_loss_contrastive_unseen =  test_gzsl_loss(opt, model, testloader_unseen, attribute_gzsl,reg_weight,criterion, criterion_regre,realtrain,middle_graph,parts,group_dic,sub_group_dic)


                print('Test Seen student loss: %.3f, Test Unseen student loss: %.3f '% (student_loss_GZSL_seen, student_loss_GZSL_unseen))
                logger.scalar_summary('Test Seen student loss', student_loss_GZSL_seen, epoch+1)
                logger.scalar_summary('Test Unseen student loss', student_loss_GZSL_unseen, epoch+1)
                logger.scalar_summary('Test Seen contrastive student loss', student_loss_contrastive_seen, epoch+1)
                logger.scalar_summary('Test Unseen contrastive student loss', student_loss_contrastive_unseen, epoch+1)


                if (student_acc_GZSL_unseen + student_acc_GZSL_seen) == 0:
                    student_acc_GZSL_H = 0
                else:
                    student_acc_GZSL_H = 2 * student_acc_GZSL_unseen * student_acc_GZSL_seen / (
                            student_acc_GZSL_unseen + student_acc_GZSL_seen)
                if student_acc_GZSL_H > result_gzsl_student.best_acc:
                    # save model state
                    # model_save_path = os.path.join('./out/{}_GZSL_id_{}.pth'.format(opt.dataset, opt.train_id))
                    model_save_path = os.path.join(weight_path+'{}_student_GZSL_id_{}.pth'.format(opt.dataset, opt.train_id))
                    torch.save(model.state_dict(), model_save_path)
                    print('model saved to:', model_save_path)

                result_gzsl_student.update_gzsl(epoch+1, student_acc_GZSL_unseen, student_acc_GZSL_seen, student_acc_GZSL_H)

                print('\n[Epoch {}] GZSL test student accuracy is Unseen: {:.1f} Seen: {:.1f} H:{:.1f}'
                        '\n           Best_H student [Unseen: {:.1f}% Seen: {:.1f}% H: {:.1f}% | Epoch-{}]'.
                        format(epoch+1, student_acc_GZSL_unseen, student_acc_GZSL_seen, student_acc_GZSL_H, result_gzsl_student.best_acc_U, result_gzsl_student.best_acc_S,
                result_gzsl_student.best_acc, result_gzsl_student.best_iter))

                #存储.txt
                _print('--' * 60) #打印50个'--'
                _print('student') 
                _print('epoch:{} - Train loss: {:.3f}    Train contrastive loss: {:.3f} Train consistency loss: {:.3f}'.format(epoch+1,loss_log['ave_loss'] / batch, loss_log['contrastive_loss'] / batch, loss_log['consistency_loss'] / batch))
                _print('epoch:{} - Test student Seen loss: {:.3f}    Test student Seen contrastive loss: {:.3f}'.format(epoch+1,student_loss_GZSL_seen, student_loss_contrastive_seen))
                _print('epoch:{} - Test student Unseen loss: {:.3f}    Test student Unseen contrastive loss: {:.3f}'.format(epoch+1,student_loss_GZSL_unseen, student_loss_contrastive_unseen))
                _print('epoch:{} - student_acc_seen: {:.1f}          student_acc_novel: {:.1f}            student_H: {:.1f}'.format(epoch+1,student_acc_GZSL_seen,student_acc_GZSL_unseen,student_acc_GZSL_H))
                _print('epoch:{} - Test student ZSL loss: {:.3f}    Test student ZSL contrastive loss: {:.3f} student_T1: {:.1f}'.format(epoch+1,student_loss_ZSL, student_loss_contrastive_ZSL, student_acc_ZSL))
            

                #### test zsl teacher
                teacher_acc_ZSL = test_zsl(opt, ema_model, testloader_unseen, attribute_zsl, data.unseenclasses)
                teacher_loss_ZSL, teacher_loss_contrastive_ZSL = test_gzsl_loss(opt, ema_model, testloader_unseen, attribute_gzsl,reg_weight,criterion, criterion_regre,realtrain,middle_graph,parts,group_dic,sub_group_dic)
                
                print('Test teacher ZSL loss: %.3f'% teacher_loss_ZSL)
                logger.scalar_summary('Test ZSL loss', teacher_loss_ZSL, epoch+1)
                if teacher_acc_ZSL > result_zsl_teacher.best_acc:
                    # save model state
                    model_save_path = os.path.join(weight_path+'/{}_teacher_ZSL_id_{}.pth'.format(opt.dataset, opt.train_id))
                    torch.save(ema_model.state_dict(), model_save_path)
                    print('model saved to:', model_save_path)
                result_zsl_teacher.update(epoch+1, teacher_acc_ZSL)
                print('\n[Epoch {}] teacher ZSL test accuracy is {:.1f}%, Best_acc [{:.1f}% | Epoch-{}]'.format(epoch+1, teacher_acc_ZSL, result_zsl_teacher.best_acc, result_zsl_teacher.best_iter))
            

                # test gzsl teacher model
                teacher_acc_GZSL_unseen = test_gzsl(opt, ema_model, testloader_unseen, attribute_gzsl, data.unseenclasses)
                teacher_acc_GZSL_seen = test_gzsl(opt, ema_model, testloader_seen, attribute_gzsl, data.seenclasses)

                teacher_loss_GZSL_seen , teacher_loss_contrastive_seen=  test_gzsl_loss(opt, ema_model, testloader_seen, attribute_gzsl,reg_weight,criterion, criterion_regre,realtrain,middle_graph,parts,group_dic,sub_group_dic)
                teacher_loss_GZSL_unseen, teacher_loss_contrastive_unseen =  test_gzsl_loss(opt, ema_model, testloader_unseen, attribute_gzsl,reg_weight,criterion, criterion_regre,realtrain,middle_graph,parts,group_dic,sub_group_dic)


                print('Test Seen teacher loss: %.3f, Test Unseen teacher loss: %.3f '% (teacher_loss_GZSL_seen, teacher_loss_GZSL_unseen))
                logger.scalar_summary('Test Seen teacher loss', teacher_loss_GZSL_seen, epoch+1)
                logger.scalar_summary('Test Unseen teacher loss', teacher_loss_GZSL_unseen, epoch+1)
                logger.scalar_summary('Test Seen contrastive teacher loss', teacher_loss_contrastive_seen, epoch+1)
                logger.scalar_summary('Test Unseen contrastive teacher loss', teacher_loss_contrastive_unseen, epoch+1)


                if (teacher_acc_GZSL_unseen + teacher_acc_GZSL_seen) == 0:
                    teacher_acc_GZSL_H = 0
                else:
                    teacher_acc_GZSL_H = 2 * teacher_acc_GZSL_unseen * teacher_acc_GZSL_seen / (
                            teacher_acc_GZSL_unseen + teacher_acc_GZSL_seen)
                if teacher_acc_GZSL_H > result_gzsl_teacher.best_acc:
                    # save model state
                    # model_save_path = os.path.join('./out/{}_GZSL_id_{}.pth'.format(opt.dataset, opt.train_id))
                    model_save_path = os.path.join(weight_path+'{}_teacher_GZSL_id_{}.pth'.format(opt.dataset, opt.train_id))
                    torch.save(ema_model.state_dict(), model_save_path)
                    print('model saved to:', model_save_path)

                result_gzsl_teacher.update_gzsl(epoch+1, teacher_acc_GZSL_unseen, teacher_acc_GZSL_seen, teacher_acc_GZSL_H)

                print('\n[Epoch {}] GZSL teacher test accuracy is Unseen: {:.1f} Seen: {:.1f} H:{:.1f}'
                        '\n           Best_H teacher [Unseen: {:.1f}% Seen: {:.1f}% H: {:.1f}% | Epoch-{}]'.
                        format(epoch+1, teacher_acc_GZSL_unseen, teacher_acc_GZSL_seen, teacher_acc_GZSL_H, result_gzsl_teacher.best_acc_U, result_gzsl_teacher.best_acc_S,
                result_gzsl_teacher.best_acc, result_gzsl_teacher.best_iter))

                #存储.txt
                _print('--' * 10) #打印50个'--'
                _print('teacher') 
                _print('epoch:{} - Test teacher Seen loss: {:.3f}    Test teacher Seen contrastive loss: {:.3f}'.format(epoch+1,teacher_loss_GZSL_seen, teacher_loss_contrastive_seen))
                _print('epoch:{} - Test teacher Unseen loss: {:.3f}    Test teacher Unseen contrastive loss: {:.3f}'.format(epoch+1,teacher_loss_GZSL_unseen, teacher_loss_contrastive_unseen))
                _print('epoch:{} - teacher_acc_seen: {:.1f}          teacher_acc_novel: {:.1f}            teacher_H: {:.1f}'.format(epoch+1,teacher_acc_GZSL_seen,teacher_acc_GZSL_unseen,teacher_acc_GZSL_H))
                _print('epoch:{} - Test teacher ZSL loss: {:.3f}    Test teacher ZSL contrastive loss: {:.3f} teacher_T1: {:.1f}'.format(epoch+1,teacher_loss_ZSL, teacher_loss_contrastive_ZSL, teacher_acc_ZSL))
            
            
            
        _print('\nBest_student_H [Unseen: {:.1f}% Seen: {:.1f}% H: {:.1f}% | Epoch-{}]'.format(result_gzsl_student.best_acc_U, result_gzsl_student.best_acc_S,
        result_gzsl_student.best_acc, result_gzsl_student.best_iter))
        _print('\nBest_teacher_H [Unseen: {:.1f}% Seen: {:.1f}% H: {:.1f}% | Epoch-{}]'.format(result_gzsl_teacher.best_acc_U, result_gzsl_teacher.best_acc_S,
        result_gzsl_teacher.best_acc, result_gzsl_teacher.best_iter))
        _print('\nBest_student__T1 [T1: {:.1f}%  | Epoch-{}]'.format(result_zsl_student.best_acc, result_zsl_student.best_iter))
        _print('\nBest_teacher__T1 [T1: {:.1f}%  | Epoch-{}]'.format(result_zsl_teacher.best_acc, result_zsl_teacher.best_iter))

if __name__ == '__main__':
    main()

