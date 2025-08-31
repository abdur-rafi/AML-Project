import os
import time
import shutil
import logging
from datetime import datetime
from collections import OrderedDict
from contextlib import suppress
from itertools import islice
import numpy as np
from random import random
import torch
import torch.nn as nn
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as NativeDDP
from utils.snn_model import SEW
import math
from utils.data import population_info as pop_info
from utils.models import create_model, load_checkpoint
from utils.tools import resume
from utils.tools.utility import *
from utils.tools.option_de import args, args_text, amp_autocast, obtain_loader
from utils.tools import spe
from utils.tools.de import de
from utils.tools.de_multi_objective import (
    de_multi_objective, initialize_population_objectives, 
    calculate_generation_stats, save_pareto_front, log_multi_objective_progress
)
from utils.tools.multi_objective import (
    print_pareto_front_summary, print_pareto_front_summary_validation, 
    select_diverse_solutions, get_pareto_front_indices
)
from utils.tools.spe import model_dict_to_vector, model_vector_to_dict
# from utils.tools.plot_utils import plot_loss, plot_paras
# from torch.utils.tensorboard import SummaryWriter
# import wandb
from utils.tools.plot_utils_ import plot_top1_vs_baseline
from utils.tools import val
from utils.tools.greedy_soup_ann import greedy_soup,test_ood,test_single_model_ood
from spikingjelly.clock_driven import functional

_logger = logging.getLogger('train')
def main():
    os.environ['WANDB_MODE'] = 'offline'
    # wandb.init(
    #     project='spe',
    #     name='gd',
    #     entity = 'spe_gd',
    #     config = args,
    # )
    setup_default_logging(log_path=args.log_dir)
    logging.basicConfig(level=logging.DEBUG, filename=args.log_dir, filemode='a')
    random_seed(args.seed, args.rank)
    
    # dataloader setting;    
    _, loader_eval, loader_de = obtain_loader(args)
    
    # model setting;
    # model = create_model(args.model, pretrained=False, drop_rate=0.)
    model = SEW.resnet34(num_classes=100, g="add", down='max', T=4)
    n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    if args.local_rank == 0:
        _logger.info(f"Creating model...{args.model},\n number of params: {n_parameters}")
    model.cuda()
    model = model.to(memory_format=torch.channels_last)

    load_score = False
    if os.path.basename(args.pop_init).split('_')[-1] == 'score.txt':
        load_score = True
        score, acc1, acc5, val_loss, en_metrics, models_path = pop_info.get_path_with_acc(args.pop_init)
    else:
        models_path = pop_info.get_path(args.pop_init)
     # ---------- Methods of choosing parent-------------# 
    # optionally resume from a checkpoint
    population = []
    for resume_path in models_path:
        print(resume_path)
        resume.load_checkpoint(model, resume_path, log_info=args.local_rank == 0)
        solution = model_dict_to_vector(model).detach()
        population.append(solution)

    if args.distributed:
       model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
       model = NativeDDP(model, device_ids=[args.local_rank])  # can use device str in Torch >= 1.1

    # output_dir = ''
    # tb_writer = None
    if args.local_rank == 0:
        output_base = args.output if args.output else './output'
        exp_name = '-'.join([datetime.now().strftime("%Y%m%d-%H%M%S"), args.exp_name])
        output_dir = get_outdir(output_base, 'train', exp_name, inc=True)
        args.output_dir = output_dir
        current_dir = os.path.dirname(os.path.abspath(__file__))
        shutil.copytree(os.path.join(current_dir, 'utils'), os.path.join(output_dir, 'utils'))
        for filename in os.listdir(current_dir):
            if filename.endswith('.py') or filename.endswith('.sh'):
                src_path = os.path.join(current_dir, filename)
                dst_path = os.path.join(output_dir, filename)
                shutil.copy(src_path, dst_path)
        with open(os.path.join(output_dir, 'args.yaml'), 'w') as f:
            f.write(args_text)
        # tb_writer = SummaryWriter(output_dir + '/_logs')

    args.popsize = len(models_path)
    if not load_score:
        score = score_func(model, population, loader_de, args) 
        if args.test_ood:
            greedy_model_dict = greedy_soup(population, score, model, loader_de, args,amp_autocast = amp_autocast)
            model.load_state_dict(greedy_model_dict)
            greedy_metrics = val.validate(model, loader_eval, args, amp_autocast=amp_autocast)
            ood_metrics=test_ood(greedy_model_dict,args,model, population, args.popsize, amp_autocast=amp_autocast)
        acc1, acc5, val_loss, f1 = [torch.zeros(args.popsize).tolist() for _ in range(4)]
        for i in range(args.popsize): 
            solution = population[i]
            model_weights_dict = model_vector_to_dict(model=model, weights_vector=solution)
            model.load_state_dict(model_weights_dict)
            temp = val.validate(model, loader_eval, args, amp_autocast=amp_autocast)
            acc1[i], acc5[i], val_loss[i], f1[i] = temp['top1'], temp['top5'], temp['loss'], temp['f1']      
        en_metrics = val.validate_ensemble(model, population, args.popsize, loader_eval, args, amp_autocast=amp_autocast)
        # import pdb; pdb.set_trace()
        # print(score, acc1, acc5, val_loss, en_metrics, models_path)
        pop_info.write_path_with_acc(score, acc1, acc5, val_loss, en_metrics, models_path, args.pop_init, f1=f1)

    # print(score, acc1, acc5, val_loss, en_metrics, models_path)
    if args.local_rank == 0:
        update_summary('baselines:', OrderedDict(en_metrics), os.path.join(output_dir, 'summary.csv'), write_header=True)
        if args.test_ood:
            update_summary('greedy_val:', greedy_metrics, os.path.join(output_dir, 'summary.csv'), write_header=True)
            update_summary('greedy_and_ensemble_ood:', ood_metrics, os.path.join(output_dir, 'summary.csv'), write_header=True)
    rowd = OrderedDict([('score', score), ('top1', acc1), ('top5', acc5), ('val_loss', val_loss), ('f1', f1)])
    # print(score:)[tensor(1.1641, device='cuda:0', dtype=torch.float16), ,,,]
    if args.local_rank == 0:
        bestidx = score.index(max(score))
        _logger.info('epoch:{}, best_score:{:>7.4f}, best_idx:{}, \
                     score: {}'.format(0, max(score), bestidx, score))
        update_summary(0, rowd, os.path.join(output_dir, 'summary.csv'), write_header=True)

    # eval_metrics_ensemble_temp = val.validate_ensemble(model, population, args.popsize, loader_eval, args, amp_autocast=amp_autocast)
    # ***********************************************************************************************************
    # need to initialize in the main
    # population_init = population#copy.deepcopy(population)
    popsize = args.popsize
    max_iters = args.de_epochs
#     memory_size, lp, cr_init, f_init, k_ls = args.shade_mem, args.shade_lp, args.cr_init, args.f_init, [0,0,0,0]
#     dim = len(model_dict_to_vector(model))
#     # Initialize memory of control settings
#     u_f = np.ones((memory_size, 4)) * f_init
#     u_cr = np.ones((memory_size, 4)) * cr_init
#     u_freq = np.ones((memory_size, 4)) * args.freq_init
#     ns_1, nf_1, ns_2, nf_2, dyn_list_nsf = [], [], [], [], []
#     stra_perc = (1-args.trig_perc)/4
# #     p1_c, p2_c, p3_c, p4_c, p5_c = 1,0,0,0,0
#     p1_c, p2_c, p3_c, p4_c, p5_c = stra_perc, stra_perc, stra_perc, stra_perc, args.trig_perc
#     succ_ls = np.zeros([4, 13])
#     # ---------- set the vector for trainable parameter in DE-------------#   
#     train_bool = torch.from_numpy(np.array([True]*population[0].numel())).cuda()
#     paras1 = [lp, cr_init, f_init, dim, popsize, max_iters, train_bool]
#     paras2 = [p1_c, p2_c, p3_c, p4_c, p5_c, ns_1, nf_1, ns_2, nf_2, u_freq, u_f, u_cr, k_ls, dyn_list_nsf, succ_ls] 
    # plot
    #____________store the inital result in different file_________
    # ***********************************************************************************************************
    print("score",score)
    f = args.f_init
    cr = args.cr_init
    f_cr_threshold = 0
    max_acc_train=0
    max_acc_val=0
    
    # Initialize multi-objective variables if enabled
    if args.multi_objective:
        print("Multi-objective optimization enabled: optimizing both accuracy and F1 score")
        population_objectives = initialize_population_objectives(population, model, loader_de, args)
        print_pareto_front_summary(population_objectives, generation=0)
        print("\n--- VALIDATION OBJECTIVES ---")
        print_pareto_front_summary_validation(population, model, loader_eval, args, generation=0)
        
        # Create multi-objective results directory
        multi_obj_dir = os.path.join(output_dir, 'multi_objective')
        if args.local_rank == 0:
            os.makedirs(multi_obj_dir, exist_ok=True)
    
    for epoch in range(1, args.de_epochs):
        epoch_time = AverageMeter()
        end = time.time()
        print("cr f",cr,f)
        
        if args.multi_objective:
            # Multi-objective evolution
            population, population_objectives, update_label, gen_stats = de_multi_objective(
                popsize, f, cr, population, population_objectives, model, loader_de, args
            )
            # Update score list for compatibility with existing code (use accuracy as primary metric)
            score = [obj[0] for obj in population_objectives]
            
            if args.local_rank == 0:
                print_pareto_front_summary(population_objectives, generation=epoch)
                print("\n--- VALIDATION OBJECTIVES ---")
                print_pareto_front_summary_validation(population, model, loader_eval, args, generation=epoch)
                log_multi_objective_progress(epoch, gen_stats, multi_obj_dir)
                
                # Save Pareto front every 10 generations
                if epoch % 10 == 0:
                    pareto_indices = get_pareto_front_indices(population_objectives)
                    pareto_pop = [population[i] for i in pareto_indices]
                    pareto_obj = [population_objectives[i] for i in pareto_indices]
                    save_pareto_front(pareto_pop, pareto_obj, epoch, multi_obj_dir)
        else:
            # Single-objective evolution (original)
            population,update_label,score = de(popsize, f, cr, population,model,loader_de,args)
        # # -------- cr f change stratgy 1
        # if update_label.count(1) > 0:
        #     f_cr_threshold == 3
        # if update_label.count(1) == 0:
        #     if f_cr_threshold<=0:            
        #         f = args.fcr_min + (args.f - 0.000001) *(1 + math.cos(math.pi * epoch / 8)) / 2 
        #         cr = args.fcr_min + (args.cr - 0.000001) *(1 + math.cos(math.pi * epoch / 8)) / 2 
        #     else:
        #         f_cr_threshold-=1

        # # -------- cr f change 

        # -------- cr f change stratgy 2
        # fcr_min = 1e-9
        # f = fcr_min + (args.f_init - fcr_min) *(1 + math.cos(math.pi * epoch / 40)) / 2 
        # cr =fcr_min + (args.cr_init - fcr_min) *(1 + math.cos(math.pi * epoch / 40)) / 2 


        # -------- cr f change 
        # # -------- cr f change stratgy 3
        fcr_min = 1e-6
        f = fcr_min + (args.f_init - fcr_min) *(1 + math.cos(math.pi * epoch / 40)) / 2 +random.uniform(args.f_init, fcr_min)
        cr =fcr_min + (args.cr_init - fcr_min) *(1 + math.cos(math.pi * epoch / 40)) / 2 +random.uniform(args.cr_init, fcr_min)

        # # -------- cr f change 
        # # -------- cr f change stratgy 4
        # if update_label.count(1) >1:
        #     f_cr_threshold+=1
        # else:
        #     fcr_min = 0.000000001
        #     f = fcr_min + (args.f_init - fcr_min) *(1 + math.cos(math.pi * (epoch-f_cr_threshold) / 5)) / 2 +random.uniform(args.f_init, fcr_min)
        #     cr =fcr_min + (args.cr_init - fcr_min) *(1 + math.cos(math.pi * (epoch-f_cr_threshold) / 5)) / 2 +random.uniform(args.cr_init, fcr_min)
        #     f_cr_threshold=0
        # # -------- cr f change 
        # # -------- cr f change stratgy 5
        # fcr_min = 1e-9
        # f = fcr_min + (args.f_init - fcr_min) *(math.cos(math.pi * epoch / 5)) / 2 # +random.uniform(args.f_init, fcr_min)
        # cr =fcr_min + (args.cr_init - fcr_min) *(math.cos(math.pi * epoch / 5)) / 2 # +random.uniform(args.cr_init, fcr_min)

        # # -------- cr f change 
        if args.local_rank == 0:
            # population, score, bestidx, worstidx, dist_matrics, paras2, update_label = evolve_out
            # p1_c, p2_c, p3_c, p4_c, p5_c, ns_1, nf_1, ns_2, nf_2, u_freq, u_f, u_cr, k_ls, dyn_list_nsf, succ_ls = paras2
            bestidx = score.index(max(score))

            _logger.info('epoch:{}, best_score:{:>7.4f}, best_idx:{}, \
                     score: {}'.format(0, max(score), bestidx, score))
            pop_tensor = torch.stack(population)
            if max(score)>max_acc_train:
                max_acc_train = max(score)
                print("Best in train_set update and train acc = ",max_acc_train)
                model_path = os.path.join(output_dir, f'train_best_{args.model}.pt')
                print('Saving model to', model_path)
                torch.save(model.state_dict(), model_path)


        if args.local_rank != 0: 
            update_label = list(range(popsize))
            pop_tensor = torch.stack(population)

        if args.distributed: 
            torch.cuda.synchronize()
            dist.barrier() 
            torch.distributed.broadcast_object_list(update_label, src=0)
            torch.distributed.broadcast(pop_tensor, src=0)

        population = list(pop_tensor)
        
        # Validation handling for both single and multi-objective cases
        if args.multi_objective:
            # Multi-objective validation: validate updated individuals and update their objectives
            for i in range(popsize):
                if update_label[i] == 1:
                    solution = population[i]
                    model_weights_dict = model_vector_to_dict(model=model, weights_vector=solution)
                    model.load_state_dict(model_weights_dict)
                    temp = val.validate(model, loader_eval, args, amp_autocast=amp_autocast)
                    acc1[i], acc5[i], val_loss[i] = temp['top1'], temp['top5'], temp['loss']
                    
                    # Update population objectives with validation results
                    population_objectives[i] = (acc1[i], temp['f1'])
            
            # For multi-objective, save Pareto optimal solutions instead of single best
            if args.local_rank == 0:
                pareto_indices = get_pareto_front_indices(population_objectives)
                if pareto_indices:
                    # Save the solution with highest accuracy from Pareto front
                    pareto_accs = [population_objectives[i][0] for i in pareto_indices]
                    best_acc_idx = pareto_indices[pareto_accs.index(max(pareto_accs))]
                    
                    if population_objectives[best_acc_idx][0] > max_acc_val:
                        max_acc_val = population_objectives[best_acc_idx][0]
                        print(f"Best Pareto solution update - Acc: {max_acc_val:.3f}, F1: {population_objectives[best_acc_idx][1]:.3f}")
                        
                        solution = population[best_acc_idx]
                        model_weights_dict = model_vector_to_dict(model=model, weights_vector=solution)
                        model.load_state_dict(model_weights_dict)
                        
                        model_path = os.path.join(output_dir, f'val_best_{args.model}.pt')
                        print('Saving best Pareto model to', model_path)
                        torch.save(model.state_dict(), model_path)
                        print("Saving best Pareto solution to", os.path.join(output_dir, f'val_best_{args.model}_solution.pt'))
                        torch.save(solution, os.path.join(output_dir, f'val_best_{args.model}_solution.pt'))
        else:
            # Single-objective validation (original behavior)
            for i in range(popsize):
                if update_label[i] == 1:
                    solution = population[i]
                    model_weights_dict = model_vector_to_dict(model=model, weights_vector=solution)
                    model.load_state_dict(model_weights_dict)
                    temp = val.validate(model, loader_eval, args, amp_autocast=amp_autocast)
                    acc1[i], acc5[i], val_loss[i] = temp['top1'], temp['top5'], temp['loss'] 
                    if acc1[i]>max_acc_val:
                        max_acc_val = acc1[i]
                        print("Best in train_set update and val acc = ",max_acc_val)
                        model_path = os.path.join(output_dir, f'val_best_{args.model}.pt')
                        print('Saving best val model to', model_path)
                        torch.save(model.state_dict(), model_path)
                        print("Saving best val solution to", os.path.join(output_dir, f'val_best_{args.model}_solution.pt'))
                        torch.save(solution, os.path.join(output_dir, f'val_best_{args.model}_solution.pt'))

        if args.distributed: 
            torch.cuda.synchronize()
        epoch_time.update(time.time() - end)
        end = time.time()

        if args.local_rank == 0:
            if args.multi_objective:
                # Multi-objective logging
                pareto_indices = get_pareto_front_indices(population_objectives)
                if pareto_indices:
                    pareto_accs = [population_objectives[i][0] for i in pareto_indices]
                    pareto_f1s = [population_objectives[i][1] for i in pareto_indices]
                    best_acc_idx = pareto_indices[pareto_accs.index(max(pareto_accs))]
                    best_f1_idx = pareto_indices[pareto_f1s.index(max(pareto_f1s))]
                    
                    _logger.info(f'Pareto Front Size: {len(pareto_indices)}, '
                               f'Best Acc: {population_objectives[best_acc_idx][0]:.4f} (F1: {population_objectives[best_acc_idx][1]:.4f}), '
                               f'Best F1: {population_objectives[best_f1_idx][1]:.4f} (Acc: {population_objectives[best_f1_idx][0]:.4f})')
                    
                    # Use best accuracy solution for compatibility with existing logging
                    bestidx = best_acc_idx
                else:
                    bestidx = score.index(max(score))
                
                # Update rowd to include F1 scores for multi-objective
                f1_scores = [obj[1] for obj in population_objectives]
                rowd = OrderedDict([('best_idx', bestidx), ('score', score), ('top1', acc1), ('top5', acc5), 
                                  ('val_loss', val_loss), ('f1_scores', f1_scores)])
            else:
                # Single-objective logging (original)
                _logger.info('score: {}'.format(rowd['score']))
                bestidx = score.index(max(score))
                rowd = OrderedDict([('best_idx', bestidx), ('score', score), ('top1', acc1), ('top5', acc5), ('val_loss', val_loss)])
            
            _logger.info('DE:{} Acc@1: {top1:>7.4f} Acc@5: {top5:>7.4f} \
                         Epoch_time: {epoch_time.val:.3f}s'.format(
                            epoch,
                            top1 = rowd['top1'][bestidx] if len(rowd['top1']) > bestidx else 0.0,
                            top5 = rowd['top5'][bestidx] if len(rowd['top5']) > bestidx else 0.0,
                            epoch_time=epoch_time))
        
            update_summary(epoch, rowd, os.path.join(output_dir, 'summary.csv'), write_header=True)
            bestidx_tensor = torch.tensor(bestidx).cuda()
            if args.de_epochs-popsize <= epoch <= args.de_epochs-1:
                model_path = os.path.join(args.output_dir, f'{args.exp_name}_{epoch}.pt')
                print('Saving model to', model_path)
                torch.save(model.state_dict(), model_path)
            # if not args.test_ood:
            #     plot_top1_vs_baseline(output_dir,'./result',exp_name,None)
        # if args.test_ood:
        #     if args.local_rank != 0: 
        #         bestidx_tensor=torch.tensor(0).cuda()
        #     if args.distributed: 
        #         torch.distributed.broadcast(bestidx_tensor, src=0)
        #     solution = population[bestidx_tensor]
        #     model_weights_dict = model_vector_to_dict(model=model, weights_vector=solution)
        #     model.load_state_dict(model_weights_dict)
        #     de_ood_metric = test_single_model_ood(args,model)
        #     if args.local_rank == 0:
        #         update_summary('de_ood:', de_ood_metric, os.path.join(output_dir, 'summary.csv'), write_header=True)

            # plot_loss(output_dir, popsize, wandb)

    # Final multi-objective results summary
    if args.multi_objective and args.local_rank == 0:
        print("\n" + "="*80)
        print("FINAL MULTI-OBJECTIVE OPTIMIZATION RESULTS")
        print("="*80)
        
        # Get final Pareto front
        pareto_indices = get_pareto_front_indices(population_objectives)
        pareto_pop = [population[i] for i in pareto_indices]
        pareto_obj = [population_objectives[i] for i in pareto_indices]
        
        print_pareto_front_summary(population_objectives, generation="Final")
        print("\n--- FINAL VALIDATION OBJECTIVES ---")
        print_pareto_front_summary_validation(population, model, loader_eval, args, generation="Final")
        
        # Save final Pareto front
        save_pareto_front(pareto_pop, pareto_obj, "final", multi_obj_dir)
        
        # Select diverse solutions and save them
        diverse_solutions = select_diverse_solutions(pareto_pop, pareto_obj, min(5, len(pareto_obj)))
        
        print(f"\nSaving {len(diverse_solutions)} diverse Pareto optimal solutions:")
        for i, (solution, (acc, f1)) in enumerate(diverse_solutions):
            model_weights_dict = model_vector_to_dict(model=model, weights_vector=solution)
            model.load_state_dict(model_weights_dict)
            
            save_path = os.path.join(multi_obj_dir, f'pareto_solution_{i+1}_acc_{acc:.3f}_f1_{f1:.3f}.pt')
            torch.save(model.state_dict(), save_path)
            
            solution_path = os.path.join(multi_obj_dir, f'pareto_solution_{i+1}_vector.pt')
            torch.save(solution, solution_path)
            
            print(f"  Solution {i+1}: Accuracy={acc:.4f}, F1={f1:.4f} -> {save_path}")
        
        print("="*80)

    wandb.finish()
    return

def score_func(model, population, loader_de, args):
    popsize = len(population)
    batch_time_m = AverageMeter()
    data_time_m = AverageMeter()
    acc1_all = torch.zeros(popsize).tolist()
    acc5_all = torch.zeros(popsize).tolist()
    end = time.time()
    model.eval()
    torch.set_grad_enabled(False)
    slice_len = args.de_slice_len or len(loader_de)
    for batch_idx, (input, target) in enumerate(islice(loader_de, slice_len)):
        data_time_m.update(time.time() - end)
        for i in range(0, popsize):
            solution = population[i]
            model_weights_dict = model_vector_to_dict(model=model, weights_vector=solution)
            model.load_state_dict(model_weights_dict)
            input, target = input.cuda(), target.cuda()
            input = input.contiguous(memory_format=torch.channels_last)
            with amp_autocast():
                output,_ = model(input)
            # if batch_idx==0 and args.local_rank < 2 and i == 0:
            #     _logger.info('Checking Data >>>> pop: {} input: {}'.format(i, input.flatten()[6000:6005]))
            functional.reset_net(model)
            acc1, acc5 = accuracy(output, target, topk=(1, 5))
            if args.distributed:
                acc1 = reduce_tensor(acc1, args.world_size)
                acc5 = reduce_tensor(acc5, args.world_size)
            acc1_all[i] += acc1
            acc5_all[i] += acc5
        batch_time_m.update(time.time() - end)
        end = time.time()

    if args.local_rank == 0:
        print('data_time: {time1.val:.3f} ({time1.avg:.3f})  '
            'batch_time: {time2.val:.3f} ({time2.avg:.3f})  '.format(time1=data_time_m, time2=batch_time_m)) 

    score = [i.cpu()/slice_len for i in acc1_all]#!!!
    return score

def score_func_multi_objective(model, population, loader_de, args):
    """
    Multi-objective score function that computes both accuracy and F1 score.
    Returns a list of tuples where each tuple contains (accuracy, f1_score) for each individual.
    
    This function is used for:
    1. Initial population evaluation in multi-objective mode
    2. Batch evaluation of the entire population when needed
    
    Note: During DE evolution, score_func_de_multi_objective is used for pairwise comparisons.
    """
    popsize = len(population)
    batch_time_m = AverageMeter()
    data_time_m = AverageMeter()
    acc1_all = torch.zeros(popsize).tolist()
    f1_all = torch.zeros(popsize).tolist()
    end = time.time()
    model.eval()
    torch.set_grad_enabled(False)
    slice_len = args.de_slice_len or len(loader_de)
    
    for batch_idx, (input, target) in enumerate(islice(loader_de, slice_len)):
        data_time_m.update(time.time() - end)
        for i in range(0, popsize):
            solution = population[i]
            model_weights_dict = model_vector_to_dict(model=model, weights_vector=solution)
            model.load_state_dict(model_weights_dict)
            input, target = input.cuda(), target.cuda()
            input = input.contiguous(memory_format=torch.channels_last)
            with amp_autocast():
                output, _ = model(input)
            
            functional.reset_net(model)
            
            # Compute accuracy
            acc1, _ = accuracy(output, target, topk=(1, 5))
            if args.distributed:
                acc1 = reduce_tensor(acc1, args.world_size)
            acc1_all[i] += acc1
            
            # Compute F1 score
            f1 = f1_score(output, target, average='macro', num_classes=args.num_classes)
            if args.distributed:
                f1 = reduce_tensor(f1, args.world_size)
            f1_all[i] += f1
            
        batch_time_m.update(time.time() - end)
        end = time.time()

    if args.local_rank == 0:
        print('data_time: {time1.val:.3f} ({time1.avg:.3f})  '
            'batch_time: {time2.val:.3f} ({time2.avg:.3f})  '.format(time1=data_time_m, time2=batch_time_m)) 

    # Return list of tuples (accuracy, f1_score) for each individual
    scores = [(acc1_all[i].cpu() / slice_len, f1_all[i].cpu() / slice_len) for i in range(popsize)]
    return scores

if __name__ == '__main__':
    main()